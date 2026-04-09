#include "Scene/SceneDefines.slangh"
#include "Utils/Math/MathConstants.slangh"

import Utils.Math.PackedFormats;
import Utils.Math.FormatConversion;

import Scene.RaytracingInline;
import Scene.Material.ShadingUtils;
import Utils.Sampling.SampleGenerator;
import Utils.Geometry.GeometryHelpers;
import Utils.Math.MathHelpers;
import Utils.Color.ColorHelpers;

import Reservoir;
import SurfaceDataFG;
import FinalGatherData;

#ifndef USE_WORLD_SPACE_HASH_GRID
    #define USE_WORLD_SPACE_HASH_GRID 0
#endif

#if USE_WORLD_SPACE_HASH_GRID
import WorldSpaceHashGrid;
#endif

#if USE_REDUCED_RESERVOIR_FORMAT
    RWTexture2D<uint2> gReservoir;
    Texture2D<uint2> gReservoirPrev;
#else
    RWTexture2D<uint4> gReservoir;
    Texture2D<uint4> gReservoirPrev;
#endif

RWStructuredBuffer<FGData> gFGSampleData;
StructuredBuffer<FGData> gFGSampleDataPrev;
StructuredBuffer<PackedSurfaceDataFG> gSurface;
StructuredBuffer<PackedSurfaceDataFG>gSurfacePrev;
Texture2D<float2> gMVec;
Texture2D<float4> gView;
Texture2D<float4> gPrevView;
RWStructuredBuffer<SampleGenerator> gSampleGenState;

// World Space Hash Grid buffers（仅在启用时使用）
#if USE_WORLD_SPACE_HASH_GRID
StructuredBuffer<uint> gHashCellStorage;    // Cell 存储（像素索引数组）
ByteAddressBuffer gHashIndexBuffer;         // PrefixSum 后的 cell 起始偏移
ByteAddressBuffer gHashCheckSumBuffer;      // Hash table checksum
ByteAddressBuffer gHashCellCounters;        // Cell 计数器

cbuffer HashGridParams
{
    float3 gHashCameraPos;
    float gHashFov;
    float3 gHashSceneBBMin;
    float gHashMinCellSize;
    uint gHashTableSize;
};
#endif

#define BIAS_MODE_BASIC 1
#define BIAS_MODE_RAYTRACE 2
#define BIAS_MODE_PAIRWISE 3
#define BIAS_MODE_PAIRWISE_STOCHASTIC 4
//This is for syntax highlighting
#ifndef BIAS_CORRECTION_MODE
    #define BIAS_CORRECTION_MODE 2
#endif

// Pairwise MIS 辅助函数（参考 Bitterli 2020 / MiniSPMIS）
float safeDivide(float a, float b)
{
    return abs(b) > 0 ? a / b : 0;
}

// Non-defensive pairwise MIS: canonical 的 beta 项
// c_i: 邻居 i 的 confidence (M_i)
// c_c: canonical 的 confidence (M_c)
// c_sum: 所有邻居的 confidence 之和
// phat_from_i: canonical 样本在邻居 i 处的 target function
// phat_c: canonical 样本在当前像素处的 target function
float computePairwiseMIS_canonical_beta(float c_i, float c_c, float c_sum, float phat_from_i, float phat_c)
{
    return (c_i / max(c_sum, 1e-8)) * safeDivide(
        c_c * phat_c,
        c_c * phat_c + c_sum * phat_from_i);
}

// Non-defensive pairwise MIS: non-canonical 的权重
// c_i: 邻居 i 的 confidence (M_i)
// c_c: canonical 的 confidence (M_c)
// c_sum: 所有邻居的 confidence 之和
// phat_from_i: 邻居样本在邻居处的 target function
// phat_c: 邻居样本在当前像素处的 target function
float computePairwiseMIS_noncanonical(float c_i, float c_c, float c_sum, float phat_from_i, float phat_c)
{
    return safeDivide(
        c_i * phat_from_i,
        c_sum * phat_from_i + c_c * phat_c);
}

cbuffer PerFrame
{
    uint gFrameCount;                   //Current it for sample gen
}

cbuffer Constant
{
    uint2 gFrameDim;                //Frame Dimensions
    uint gMaxAge;                   //M limit            
    uint gSpatialSamples;           //Number of spatial samples
    float gSamplingRadius;          //Sampling radius
    float gDepthThreshold;          //Relative depth offset
    float gNormalThreshold;         //Maximum cosine between both Normals that is allowed
    uint gDisocclusionBoostSamples; //Number of spatial samples if no temporal surface was found
    float gAttenuationRadius;       //Radius for the attenuation
    float2 gJacobianMinMax;         //Minimum and Maximum values for the jacobian
    uint gPairwiseMIS_M;            //Pairwise MIS: 虚拟总候选数 M (覆盖 c_sum), 0 = 使用实际值
    uint gPairwiseMIS_N;            //Pairwise MIS: 实际采样邻居数 N (覆盖 numSamples), 0 = 使用 gSpatialSamples
}

struct LightSample{
    float3 dir;
    float Li;
};

struct PhotonLight{
    float4 pos;
    float4 flux;
};

static const float kRayEpsilon = 0.99f;
static const float kRayNearEpsilon = 0.01f;


SurfaceFG getSurface(uint2 idx, bool prev)
{
    float3 viewDir;
    if (prev)
        viewDir = gPrevView[idx].xyz;
    else
        viewDir = gView[idx].xyz;

    PackedSurfaceDataFG surfaceData;
    if (prev)
        surfaceData = gSurfacePrev[index2Dto1D(idx, gFrameDim.x)];
    else
        surfaceData = gSurface[index2Dto1D(idx, gFrameDim.x)];
    
    return SurfaceFG(surfaceData, viewDir);
}

bool getLightSample(in FGData fgData, in float3 worldPos, out LightSample ls, out float sampleDistance)
{
    //Get light position    
    ls.dir = fgData.posW - worldPos;
    sampleDistance = length(ls.dir);
    ls.dir = ls.dir / sampleDistance; //normalize
    
    ls.Li = luminance(fgData.flux);
    float distAttenuation = vplAttenuation(sampleDistance, gAttenuationRadius);
    float cosTheta = dot(normalize(fgData.normal), -ls.dir);
    float jacobian = (cosTheta * distAttenuation) / fgData.jacobianDeterminant;

    //Jacobian is outside of valid values (Solid angle ratio too different)
    if(jacobian < gJacobianMinMax.x || jacobian > gJacobianMinMax.y || cosTheta < kMinCosTheta)
        return false;

    ls.Li *= jacobian;
        
    return ls.Li >= 0; //true if sample is valid
}

void temporalResampling(uint2 pixel, inout SampleGenerator sg, inout Reservoir r, inout FGData fgSample, in SurfaceFG surface)
{
    uint M_canonical = r.M; // 保存 canonical 的 M 值
    float W_canonical = r.weightSum; // 保存 canonical 的 W 值（finalize 后的）
    float p_hat_canonical = r.targetFunc; // 保存 canonical 的 target function
    uint historyLimit = gMaxAge * r.M;

    //Reproject to the previous pixel using the motion vectors
    float2 motionOffset = gMVec[pixel] * float2(gFrameDim);
    float2 prevIdxF = motionOffset + float2(pixel);
    float2 rndOffset = sampleNext2D(sg) - 0.5f;
    prevIdxF += rndOffset;
    
    uint2 prevIdx = uint2(round(prevIdxF));

    //Check if a neighbor surface matches up
    //Try to find a matching surface in the neighborhood of the reprojected pixel
    SurfaceFG temporalSurface;
    bool selectPrevSample = false;
    uint previousM = 0;
    bool foundTemporalSurface = false;
    int2 idx = (0, 0);
    float temporalSearchRadius = 4.f; //actual radius is 2 due to random value in [-0.5,0.5]
    
    for (uint i = 0; i < 9; i++)
    {
        int2 offset = int2(0, 0);
        if (i > 0)
        {
            offset = int2((sampleNext2D(sg) - 0.5f) * temporalSearchRadius);
        }

        idx = prevIdx + offset;

        //Return if index is invalid
        if(any(idx < 0) || any(idx >= gFrameDim))
            continue;
        
        //Get Hit sample from last frame
        temporalSurface = getSurface(idx, true);
        if (!temporalSurface.valid)
            continue;

        //Test surface similarity and discard if too different
        if (!surface.isValidNeighbor(temporalSurface, gDepthThreshold, gNormalThreshold))
            continue;

        //Valid hit was found
        foundTemporalSurface = true;
        break;
    }

    //If temporal surface was found, resample the previous sample
    if (foundTemporalSurface)
    {
        Reservoir prev = Reservoir(gReservoirPrev[idx]);
        prev.M = min(prev.M, historyLimit);
        previousM = prev.M;

        bool sampleValid = true;
        FGData fgPrev = gFGSampleDataPrev[index2Dto1D(idx, gFrameDim.x)];
        //Kill the reservoir if the sample is no longer valid or was replaced
        if (!fgPrev.valid)
        {
            prev.weightSum = 0;
            prev.M = 0;
            sampleValid = false;
        }

        //Get target pdf at current location if reservoir and sample are valid
        float targetFunc = 0;
        bool combine = true;
        if (prev.M > 0 && sampleValid)
        {
            LightSample ls = { };
            float dis = 0;
            if (!getLightSample(fgPrev, surface.posW, ls, dis))
                combine = false;
            targetFunc = surface.getPdf(ls.dir, ls.Li);
        }

#if BIAS_CORRECTION_MODE == BIAS_MODE_PAIRWISE || BIAS_CORRECTION_MODE == BIAS_MODE_PAIRWISE_STOCHASTIC
        // Pairwise MIS: 计算邻居样本在邻居处的 target function
        float p_hat_neighbor = 0.0;
        if (combine && prev.M > 0 && sampleValid)
        {
            LightSample lsNeigh = { };
            float disNeigh = 0;
            if (getLightSample(fgPrev, temporalSurface.posW, lsNeigh, disNeigh))
                p_hat_neighbor = temporalSurface.getPdf(lsNeigh.dir, lsNeigh.Li);
        }

        // 计算 canonical 样本在邻居位置的 target function
        float p_hat_canonical_at_neighbor = 0.0;
        if (foundTemporalSurface && prev.M > 0)
        {
            LightSample lsCanonical = { };
            float disCanonical = 0;
            if (getLightSample(fgSample, temporalSurface.posW, lsCanonical, disCanonical))
                p_hat_canonical_at_neighbor = temporalSurface.getPdf(lsCanonical.dir, lsCanonical.Li);
        }

        // c_c = M_canonical, c_i = prev.M, c_sum = prev.M (只有一个邻居)
        float c_c = float(M_canonical);
        float c_i = float(prev.M);
        float c_sum = gPairwiseMIS_M > 0 ? float(gPairwiseMIS_M) : c_i;

        // Canonical MIS 权重
        float m_c = computePairwiseMIS_canonical_beta(c_i, c_c, c_sum, p_hat_canonical_at_neighbor, p_hat_canonical);

        // Non-canonical MIS 权重
        float m_i = combine ? computePairwiseMIS_noncanonical(c_i, c_c, c_sum, p_hat_neighbor, targetFunc) : 0.0;

        // 手动计算 w_sum（不使用 combineReservoir）
        float w_sum = m_c * p_hat_canonical * W_canonical;
        float w_i = m_i * targetFunc * prev.weightSum;
        w_sum += w_i;

        // RIS 选择
        r.M += prev.M;
        r.weightSum = w_sum;
        if (w_sum > 0 && sampleNext1D(sg) * w_sum < w_i)
        {
            selectPrevSample = true;
            r.targetFunc = targetFunc;
        }
#else
        if(combine)
            selectPrevSample = r.combineReservoir(prev, targetFunc, sampleNext1D(sg));
#endif
        if (selectPrevSample)
            fgSample = fgPrev;
    }


    //Bias correction via MIS
#if BIAS_CORRECTION_MODE == BIAS_MODE_PAIRWISE || BIAS_CORRECTION_MODE == BIAS_MODE_PAIRWISE_STOCHASTIC
    // Pairwise MIS: 直接 finalize，分母为 1（MIS 权重已内联处理）
    r.finalizeSample(1.0f, 1.0f);
#elif BIAS_CORRECTION_MODE >= BIAS_MODE_BASIC
    float pi = r.targetFunc;
    float piSum = r.targetFunc; //piSum = target * M_initial ; M_initial is always 1 for us

    if (foundTemporalSurface)
    {
        LightSample ls = { };
        float dist = 0;
        bool valid = getLightSample(fgSample, temporalSurface.posW, ls, dist);
        float ps = valid ? temporalSurface.getPdf(ls.dir, ls.Li) : 0.f;

#if BIAS_CORRECTION_MODE >= BIAS_MODE_RAYTRACE
        if (ps > 0)
        {
            SceneRayQuery < 0 > rayQuery;
            dist *= kRayEpsilon;
            float rayNear = dist * kRayNearEpsilon;
            dist -= rayNear;
            const Ray ray = Ray(computeRayOrigin(temporalSurface.posW, temporalSurface.normal), ls.dir, rayNear, dist);
            if (!rayQuery.traceVisibilityRay(ray, RAY_FLAG_NONE, 0xff))
            {
                ps = 0;
            }
        }
#endif // BIAS MODE RAYTRACE
        
        pi = selectPrevSample ? ps : pi;
        piSum += ps * previousM;
    }
    r.finalizeSample(pi, piSum);
    
#else    
    r.finalizeSample(1.0f, float(r.M));
#endif //BIAS_MODE
}

void spatialResampling(uint2 pixel, inout SampleGenerator sg, inout Reservoir r, inout FGData fgSample, in SurfaceFG surface)
{    
    uint M_canonical = r.M; // 保存 canonical 的 M 值
    float W_canonical = r.weightSum; // 保存 canonical 的 W 值
    float p_hat_canonical = r.targetFunc; // 保存 canonical 的 target function
    FGData fgCanonical = fgSample; // 保存原始 canonical 样本
    uint spatialCount = gPairwiseMIS_N > 0 ? gPairwiseMIS_N : gSpatialSamples;
    uint numSamples = clamp(spatialCount + 1, 1, 8); //Cap at 8 samples for the later bias correction
    uint cachedIndices[8]; //Store all indeces. 32 Bits per 2D index (16 per element)
    uint cachedM[4];
    int selectedSample = -1; //Index of the selected sample

    // Pairwise MIS 缓存数组
    float cachedTargetFunc[8];   // 邻居样本在当前像素处的 target function
    float cachedPhatNeighbor[8]; // 邻居样本在邻居处的 target function
    float cachedW[8];            // 邻居 reservoir 的 W 值
    FGData cachedFG[8];          // 邻居的 FGData
    SurfaceFG cachedNeighSurface[8]; // 邻居的表面数据
    uint cachedReservoirM[8];    // 邻居 reservoir 的 M 值
    float c_sum = 0.0;           // 所有邻居的 confidence 之和
    uint validCount = 0;         // 有效邻居数量

    //Go through all neighbors
    // Hash Grid 模式下的邻居查找变量
#if USE_WORLD_SPACE_HASH_GRID
    float cellSize = CalculateCellSize(surface.posW, gHashCameraPos, gHashFov, float2(gFrameDim), gHashMinCellSize);
    float3 jitteredPos = surface.posW + (sampleNext3D(sg) * 2.0f - 1.0f) * 0.1f * cellSize;
    int cellIdx = FindCell(jitteredPos, surface.normal, cellSize, gHashSceneBBMin, gHashTableSize, gHashCheckSumBuffer);
    uint cellBaseIdx = 0;
    uint cellSampleCount = 0;
    if (cellIdx != -1)
    {
        cellBaseIdx = gHashIndexBuffer.Load(uint(cellIdx) * 4);
        cellSampleCount = gHashCellCounters.Load(uint(cellIdx) * 4);
    }
    uint hashIncrement = max(1u, (cellSampleCount + numSamples - 1) / numSamples);
    uint hashOffset = uint(round(sampleNext1D(sg) * float(max(1u, hashIncrement) - 1)));
#endif

    for (uint i = 0; i < numSamples; i++)
    {
        cachedIndices[i] = 0xFFFFFFFF; //Set to max int
        cachedReservoirM[i] = 0;
        cachedTargetFunc[i] = 0;
        cachedPhatNeighbor[i] = 0;
        cachedW[i] = 0;
        if (i % 2 == 0)
            cachedM[i / 2] = 0;
        
        int2 idx = int2(pixel);

#if USE_WORLD_SPACE_HASH_GRID
        // World Space Hash Grid 邻居查找
        if (cellIdx != -1 && cellSampleCount > 0)
        {
            uint storageIdx = (hashOffset + i * hashIncrement) % cellSampleCount;
            uint neighborLinearIdx = gHashCellStorage[cellBaseIdx + storageIdx];
            idx = int2(neighborLinearIdx % gFrameDim.x, neighborLinearIdx / gFrameDim.x);

            // 跳过自身
            if (all(idx == int2(pixel)))
                continue;

            // 边界检查
            if(any(idx < 0) || any(idx >= gFrameDim))
                continue;

            // 获取 surface 并检查有效性
            SurfaceFG neighSurface = getSurface(idx, false);
            if (!neighSurface.valid)
                continue;

            if (!surface.isValidNeighbor(neighSurface, gDepthThreshold, gNormalThreshold))
                continue;
        }
        else
        {
            continue; // 没有找到 cell，跳过
        }
#else
        float2 randSamp = (sampleNext2D(sg) * 2.f) - 1.f; //random sample in [-1,1]
        int2 spatialOffset = int2(round(randSamp * gSamplingRadius));
        idx += spatialOffset;

        //Outside index is invalid
        if(any(idx < 0) || any(idx >= gFrameDim))
            continue;

        //Get surface and check if valid
        SurfaceFG neighSurface = getSurface(idx, false); //Get surface
        if (!neighSurface.valid)
            continue;
        
        if (!surface.isValidNeighbor(neighSurface, gDepthThreshold, gNormalThreshold))
            continue;
#endif

        //Neighbor passed the test, start resampling
        Reservoir sampledReservoir = Reservoir(gReservoirPrev[idx]);    //Current reservoir is stored in prev
        
        bool sampleValid = true;
        FGData fgNeigh = gFGSampleDataPrev[index2Dto1D(idx, gFrameDim.x)];
        //Kill the reservoir if the sample is no longer valid or was replaced
        if (!fgNeigh.valid)
        {
            sampledReservoir.weightSum = 0;
            sampledReservoir.M = 0;
            sampleValid = false;
        }
        //Store indices
        cachedIndices[i] = (idx.x & 0xFFFF) << 16 | (idx.y & 0xFFFF);
        uint M = 0xFFFF & sampledReservoir.M;
        if (i % 2 != 0)
            M = M << 16;
        cachedM[i / 2] |= M;

        //Get neightbor target PDF for current surface
        float sampletargetFunc = 0.f;
        if (sampledReservoir.M > 0 && sampleValid)
        {
            LightSample ls = { };
            float dis;
            if (!getLightSample(fgNeigh, surface.posW, ls, dis))
                continue;
            sampletargetFunc = surface.getPdf(ls.dir, ls.Li);
        }

#if BIAS_CORRECTION_MODE == BIAS_MODE_PAIRWISE || BIAS_CORRECTION_MODE == BIAS_MODE_PAIRWISE_STOCHASTIC
        // Pairwise MIS: 缓存邻居数据，第二遍循环中计算权重
        // 计算邻居样本在其原始像素处的 target function
        float p_hat_neighbor = 0.0;
        if (sampledReservoir.M > 0 && sampleValid)
        {
            LightSample lsNeigh = { };
            float disNeigh;
            if (getLightSample(fgNeigh, neighSurface.posW, lsNeigh, disNeigh))
                p_hat_neighbor = neighSurface.getPdf(lsNeigh.dir, lsNeigh.Li);
        }
        cachedTargetFunc[i] = sampletargetFunc;
        cachedPhatNeighbor[i] = p_hat_neighbor;
        cachedW[i] = sampledReservoir.weightSum;
        cachedFG[i] = fgNeigh;
        cachedNeighSurface[i] = neighSurface;
        cachedReservoirM[i] = sampledReservoir.M;
        c_sum += float(sampledReservoir.M);
        validCount++;
        r.M += sampledReservoir.M;
#else
        //Combine Reservoirs
        if (r.combineReservoir(sampledReservoir, sampletargetFunc, sampleNext1D(sg)))
        {
            selectedSample = i;
            fgSample = fgNeigh;
        }
#endif
    }

    //Bias correction
    // 当 gPairwiseMIS_M > 0 时，用虚拟总候选数 M 覆盖实际的 c_sum
    float final_c_sum = gPairwiseMIS_M > 0 ? float(gPairwiseMIS_M) : c_sum;
    #if BIAS_CORRECTION_MODE == BIAS_MODE_PAIRWISE
    {
        float c_c = float(M_canonical);

        // 标准 Pairwise MIS: 遍历所有邻居计算 canonical MIS 权重 m_c
        float m_c = 0.0;
        if (p_hat_canonical > 0 && validCount > 0)
        {
            for (uint j = 0; j < numSamples; j++)
            {
                if (cachedReservoirM[j] == 0) continue;
                // canonical 样本在邻居 j 位置的 target function
                float p_hat_canonical_at_j = 0.0;
                if (cachedNeighSurface[j].valid)
                {
                    LightSample lsCanonical = { };
                    float disCanonical;
                    if (getLightSample(fgCanonical, cachedNeighSurface[j].posW, lsCanonical, disCanonical))
                        p_hat_canonical_at_j = cachedNeighSurface[j].getPdf(lsCanonical.dir, lsCanonical.Li);
                }
                m_c += computePairwiseMIS_canonical_beta(float(cachedReservoirM[j]), c_c, final_c_sum, p_hat_canonical_at_j, p_hat_canonical);
            }
        }

        // w_sum 从 canonical 开始
        float w_sum = m_c * p_hat_canonical * W_canonical;

        // 累加每个邻居的贡献
        for (uint j = 0; j < numSamples; j++)
        {
            if (cachedReservoirM[j] == 0) continue;
            float m_i = computePairwiseMIS_noncanonical(float(cachedReservoirM[j]), c_c, final_c_sum, cachedPhatNeighbor[j], cachedTargetFunc[j]);
            float w_i = m_i * cachedTargetFunc[j] * cachedW[j];
            w_sum += w_i;
            if (w_sum > 0 && sampleNext1D(sg) * w_sum < w_i)
            {
                selectedSample = j;
                fgSample = cachedFG[j];
                r.targetFunc = cachedTargetFunc[j];
            }
        }

        r.weightSum = w_sum;
        r.finalizeSample(1.0f, 1.0f);
    }
    #elif BIAS_CORRECTION_MODE == BIAS_MODE_PAIRWISE_STOCHASTIC
    {
        float c_c = float(M_canonical);

        // Stochastic Pairwise MIS: 用单个随机邻居估计 canonical MIS 权重 m_c
        // 参考 MiniSPMIS: selectionW * computePairwiseMIS_canonical_beta(...)
        float m_c = 0.0;
        if (p_hat_canonical > 0 && validCount > 0)
        {
            // 随机选一个有效邻居来估计 canonical 权重
            uint randomIdx = min(uint(sampleNext1D(sg) * numSamples), numSamples - 1);
            // 找到一个有效邻居
            uint attempts = 0;
            while (cachedReservoirM[randomIdx] == 0 && attempts < numSamples)
            {
                randomIdx = (randomIdx + 1) % numSamples;
                attempts++;
            }
            if (cachedReservoirM[randomIdx] > 0)
            {
                float p_hat_canonical_at_j = 0.0;
                if (cachedNeighSurface[randomIdx].valid)
                {
                    LightSample lsCanonical = { };
                    float disCanonical;
                    if (getLightSample(fgCanonical, cachedNeighSurface[randomIdx].posW, lsCanonical, disCanonical))
                        p_hat_canonical_at_j = cachedNeighSurface[randomIdx].getPdf(lsCanonical.dir, lsCanonical.Li);
                }
                // selectionW = validCount（均匀采样的无偏估计因子）
                float selectionW = float(validCount);
                m_c = selectionW * computePairwiseMIS_canonical_beta(float(cachedReservoirM[randomIdx]), c_c, final_c_sum, p_hat_canonical_at_j, p_hat_canonical);
            }
        }

        // w_sum 从 canonical 开始
        float w_sum = m_c * p_hat_canonical * W_canonical;

        // 累加每个邻居的贡献（non-canonical 部分与标准版相同）
        for (uint j = 0; j < numSamples; j++)
        {
            if (cachedReservoirM[j] == 0) continue;
            float m_i = computePairwiseMIS_noncanonical(float(cachedReservoirM[j]), c_c, final_c_sum, cachedPhatNeighbor[j], cachedTargetFunc[j]);
            float w_i = m_i * cachedTargetFunc[j] * cachedW[j];
            w_sum += w_i;
            if (w_sum > 0 && sampleNext1D(sg) * w_sum < w_i)
            {
                selectedSample = j;
                fgSample = cachedFG[j];
                r.targetFunc = cachedTargetFunc[j];
            }
        }

        r.weightSum = w_sum;
        r.finalizeSample(1.0f, 1.0f);
    }
    #elif BIAS_CORRECTION_MODE >= BIAS_MODE_BASIC
        
        float pi = r.targetFunc;
        float piSum = r.targetFunc; //piSum = target * M_initial ; M_initial is always 1 for us
        
        //go through all neighbors
        for (int i = 0; i < numSamples; i++)
        {
            //Skip if we skipped in last loop
            if ((cachedIndices[i]) == 0xFFFFFFFF)
                continue;
            //Get Index
            uint2 idx = uint2(cachedIndices[i] >> 16, (cachedIndices[i] & 0xFFFF));
            //Get Surface
            SurfaceFG biasSurface = getSurface(idx, false);
            if (!biasSurface.valid)
                continue;
        
            //Get light pdf
            float ps = 0.0f;
            LightSample ls = { };
            float dist = 0;

            //Skip Sample if it was rejected by jacobian
            if (!getLightSample(fgSample, biasSurface.posW, ls, dist))
                continue;
            ps = biasSurface.getPdf(ls.dir, ls.Li);
        
            //Trace ray
    #if BIAS_CORRECTION_MODE >= BIAS_MODE_RAYTRACE
            if (ps > 0)
            {
                SceneRayQuery < 0 > rayQuery;
                dist *= kRayEpsilon;
                float rayNear = dist * kRayNearEpsilon;
                dist -= rayNear;
                const Ray ray = Ray(computeRayOrigin(biasSurface.posW, biasSurface.normal), ls.dir, rayNear, dist);
                if (!rayQuery.traceVisibilityRay(ray, RAY_FLAG_CULL_NON_OPAQUE, 0xff))
                {
                    ps = 0.0f;
                }
            }
    #endif // BIAS MODE RAYTRACE
        
            uint M = cachedM[i / 2];
            if (i % 2 == 0)
                M = M & 0xFFFF;
            else
                M = (M >> 16) & 0xFFFF;
        
            pi = selectedSample == i ? ps : pi;
            piSum += ps * M;
        }
        r.finalizeSample(pi, piSum);
        
    #else //BIAS MODE BASIC
        r.finalizeSample(1.0f, r.M);
    #endif //BIAS MODE    
}

void spatiotemporalResampling(uint2 pixel, inout SampleGenerator sg, inout Reservoir r, inout FGData fgSample, in SurfaceFG surface)
{
    uint M_canonical = r.M; // 保存 canonical 的 M 值
    float W_canonical = r.weightSum; // 保存 canonical 的 W 值
    float p_hat_canonical = r.targetFunc; // 保存 canonical 的 target function
    FGData fgCanonical = fgSample; // 保存原始 canonical 样本
    uint historyLimit = gMaxAge;

    //Check if motion vectors are inside of the frame
    float2 motionOffset = gMVec[pixel] * float2(gFrameDim);
    float2 prevIdxF = motionOffset + float2(pixel);
    float2 rndOffset = sampleNext2D(sg) - 0.5f;
    prevIdxF += rndOffset;
    
    int2 prevIdx = int2(round(prevIdxF));

    //Get surface from last frame
    bool foundTemporalSurface = false;
    int2 temporalOffset = int2(0, 0);
    const float temporalSearchRadius = 4.f; //actual radius is 2 due to random value in [-0.5,0.5]

    //Try to find a matching surface in the neighborhood of the reprojected pixel
    for (uint i = 0; i < 9; i++)
    {
        int2 offset = int2(0, 0);
        if (i > 0)
        {
            offset = int2((sampleNext2D(sg) - 0.5f) * temporalSearchRadius);
        }

        int2 idx = prevIdx + offset;

        //Return if idx is not valid
        if(any(idx < 0) || any(idx >= gFrameDim))
            continue;
        
        //Get Hit sample from last frame
        SurfaceFG temporalSurface = getSurface(idx, true);
        if (!temporalSurface.valid)
            continue;

        //Test surface similarity and discard if too different
        if (!surface.isValidNeighbor(temporalSurface, gDepthThreshold, gNormalThreshold))
            continue;
        
        //Valid hit was found
        temporalOffset = offset;
        foundTemporalSurface = true;
        break;
    }

    //
    //Spartio Temporal Resampling
    //
    
    uint spatialCountST = gPairwiseMIS_N > 0 ? gPairwiseMIS_N : gSpatialSamples;
    uint numSamples = foundTemporalSurface ? spatialCountST + 1 : max(spatialCountST + 1, gDisocclusionBoostSamples);
    numSamples = clamp(numSamples, 1, 8); //Cap at 8 samples for the later bias correction
    uint cachedIndices[8]; //Store all indeces. 32 Bits per 2D index (16 per element)
    uint cachedM[4];
    
    int selectedSample = -1;
    bool validSample = true;

    // Pairwise MIS 缓存数组
    float cachedTargetFunc[8];
    float cachedPhatNeighbor[8];
    float cachedW[8];
    FGData cachedFG[8];
    uint cachedReservoirM[8];
    float c_sum = 0.0;
    uint validCount = 0;

    //Go through all neighbors
    // Hash Grid 模式下的邻居查找变量（用于 spatial 部分）
#if USE_WORLD_SPACE_HASH_GRID
    float stCellSize = CalculateCellSize(surface.posW, gHashCameraPos, gHashFov, float2(gFrameDim), gHashMinCellSize);
    float3 stJitteredPos = surface.posW + (sampleNext3D(sg) * 2.0f - 1.0f) * 0.1f * stCellSize;
    int stCellIdx = FindCell(stJitteredPos, surface.normal, stCellSize, gHashSceneBBMin, gHashTableSize, gHashCheckSumBuffer);
    uint stCellBaseIdx = 0;
    uint stCellSampleCount = 0;
    if (stCellIdx != -1)
    {
        stCellBaseIdx = gHashIndexBuffer.Load(uint(stCellIdx) * 4);
        stCellSampleCount = gHashCellCounters.Load(uint(stCellIdx) * 4);
    }
    uint stHashIncrement = max(1u, (stCellSampleCount + numSamples - 1) / numSamples);
    uint stHashOffset = uint(round(sampleNext1D(sg) * float(max(1u, stHashIncrement) - 1)));
    uint stHashSpatialIdx = 0; // spatial 样本的计数器
#endif

    for (uint i = 0; i < numSamples; i++)
    {
        cachedIndices[i] = 0xFFFFFFFF; //Set to max int
        cachedReservoirM[i] = 0;
        cachedTargetFunc[i] = 0;
        cachedPhatNeighbor[i] = 0;
        cachedW[i] = 0;
        if (i % 2 == 0)
            cachedM[i / 2] = 0;
        int2 idx = int2(0);

        //Take temporal sample or generate a new neighbor sample
        if (i == 0 && foundTemporalSurface)
        {
            idx = prevIdx + temporalOffset;
        }
        else
        {
#if USE_WORLD_SPACE_HASH_GRID
            // World Space Hash Grid 邻居查找（spatial 部分）
            if (stCellIdx != -1 && stCellSampleCount > 0)
            {
                uint storageIdx = (stHashOffset + stHashSpatialIdx * stHashIncrement) % stCellSampleCount;
                stHashSpatialIdx++;
                uint neighborLinearIdx = gHashCellStorage[stCellBaseIdx + storageIdx];
                idx = int2(neighborLinearIdx % gFrameDim.x, neighborLinearIdx / gFrameDim.x);

                // 跳过自身
                if (all(idx == int2(pixel)))
                    continue;

                // 边界检查
                if(any(idx < 0) || any(idx >= gFrameDim))
                    continue;

                // 获取上一帧的 surface 并检查有效性
                SurfaceFG neighSurface = getSurface(idx, true);
                if (!neighSurface.valid)
                    continue;

                if (!surface.isValidNeighbor(neighSurface, gDepthThreshold, gNormalThreshold))
                    continue;
            }
            else
            {
                continue; // 没有找到 cell，跳过
            }
#else
            //Get sample from sampling mask
            float2 randSamp = (sampleNext2D(sg) * 2.f) - 1.f; //random sample in [-1,1]
            int2 spatialOffset = int2(round(randSamp * gSamplingRadius));
            idx = prevIdx + spatialOffset;

            //Outside index is invalid
            if(any(idx < 0) || any(idx >= gFrameDim))
                continue;
            
             //Get surface and check if valid
            SurfaceFG neighSurface = getSurface(idx, true);
            if (!neighSurface.valid)
                continue;
        
            if (!surface.isValidNeighbor(neighSurface, gDepthThreshold, gNormalThreshold))
                continue;
#endif
        }
            
        //If neighbor surface has passed the tests get reservoir and compair them
        Reservoir sampledReservoir = Reservoir(gReservoirPrev[idx]);
        sampledReservoir.M = min(sampledReservoir.M, historyLimit);

        FGData fgPrev = gFGSampleDataPrev[index2Dto1D(idx, gFrameDim.x)];

        //Kill the reservoir if it is not valid
        bool sampleValid = true;
        if (!fgPrev.valid)
        {
            sampledReservoir.weightSum = 0;
            sampledReservoir.M = 0;
            sampleValid = false;
        }

        //Store indices
        cachedIndices[i] = (idx.x & 0xFFFF) << 16 | (idx.y & 0xFFFF);
        uint M = 0xFFFF & sampledReservoir.M;
        if (i % 2 != 0)
            M = M << 16;
        cachedM[i / 2] |= M;
        
        //Get neightbor light target PDF for current surface
        float sampletargetFunc = 0.f;
        if (sampledReservoir.M > 0 && sampleValid)
        {
            LightSample ls = { };
            float dis;
            if (!getLightSample(fgPrev, surface.posW, ls, dis))
                continue;
            sampletargetFunc = surface.getPdf(ls.dir, ls.Li);
        }

#if BIAS_CORRECTION_MODE == BIAS_MODE_PAIRWISE || BIAS_CORRECTION_MODE == BIAS_MODE_PAIRWISE_STOCHASTIC
        // Pairwise MIS: 缓存邻居数据
        float p_hat_neighbor = 0.0;
        if (sampledReservoir.M > 0 && sampleValid)
        {
            uint2 neighIdx = uint2((cachedIndices[i] >> 16) & 0xFFFF, (cachedIndices[i] & 0xFFFF));
            SurfaceFG neighSurface = getSurface(neighIdx, true);
            if (neighSurface.valid)
            {
                LightSample lsNeigh = { };
                float disNeigh;
                if (getLightSample(fgPrev, neighSurface.posW, lsNeigh, disNeigh))
                    p_hat_neighbor = neighSurface.getPdf(lsNeigh.dir, lsNeigh.Li);
            }
        }
        cachedTargetFunc[i] = sampletargetFunc;
        cachedPhatNeighbor[i] = p_hat_neighbor;
        cachedW[i] = sampledReservoir.weightSum;
        cachedFG[i] = fgPrev;
        cachedReservoirM[i] = sampledReservoir.M;
        c_sum += float(sampledReservoir.M);
        validCount++;
        r.M += sampledReservoir.M;
#else
        //Combine Reservoirs
        if (r.combineReservoir(sampledReservoir, sampletargetFunc, sampleNext1D(sg)))
        {
            selectedSample = i;
            fgSample = fgPrev;
        }
#endif
    }

  //Bias correction
    // 当 gPairwiseMIS_M > 0 时，用虚拟总候选数 M 覆盖实际的 c_sum
    float final_c_sum = gPairwiseMIS_M > 0 ? float(gPairwiseMIS_M) : c_sum;
    #if BIAS_CORRECTION_MODE == BIAS_MODE_PAIRWISE
    {
        float c_c = float(M_canonical);

        // 标准 Pairwise MIS: 遍历所有邻居计算 canonical MIS 权重 m_c
        float m_c = 0.0;
        if (p_hat_canonical > 0 && validCount > 0)
        {
            for (uint j = 0; j < numSamples; j++)
            {
                if (cachedReservoirM[j] == 0) continue;
                // canonical 样本在邻居 j 位置的 target function
                uint2 neighIdx = uint2((cachedIndices[j] >> 16) & 0xFFFF, (cachedIndices[j] & 0xFFFF));
                SurfaceFG neighSurface = getSurface(neighIdx, true);
                float p_hat_canonical_at_j = 0.0;
                if (neighSurface.valid)
                {
                    LightSample lsCanonical = { };
                    float disCanonical;
                    if (getLightSample(fgCanonical, neighSurface.posW, lsCanonical, disCanonical))
                        p_hat_canonical_at_j = neighSurface.getPdf(lsCanonical.dir, lsCanonical.Li);
                }
                m_c += computePairwiseMIS_canonical_beta(float(cachedReservoirM[j]), c_c, final_c_sum, p_hat_canonical_at_j, p_hat_canonical);
            }
        }

        // w_sum 从 canonical 开始
        float w_sum = m_c * p_hat_canonical * W_canonical;

        // 累加每个邻居的贡献
        for (uint j = 0; j < numSamples; j++)
        {
            if (cachedReservoirM[j] == 0) continue;
            float m_i = computePairwiseMIS_noncanonical(float(cachedReservoirM[j]), c_c, final_c_sum, cachedPhatNeighbor[j], cachedTargetFunc[j]);
            float w_i = m_i * cachedTargetFunc[j] * cachedW[j];
            w_sum += w_i;
            if (w_sum > 0 && sampleNext1D(sg) * w_sum < w_i)
            {
                selectedSample = j;
                fgSample = cachedFG[j];
                r.targetFunc = cachedTargetFunc[j];
            }
        }

        r.weightSum = w_sum;
        r.finalizeSample(1.0f, 1.0f);
    }
    #elif BIAS_CORRECTION_MODE == BIAS_MODE_PAIRWISE_STOCHASTIC
    {
        float c_c = float(M_canonical);

        // Stochastic Pairwise MIS: 用单个随机邻居估计 canonical MIS 权重 m_c
        float m_c = 0.0;
        if (p_hat_canonical > 0 && validCount > 0)
        {
            // 随机选一个有效邻居来估计 canonical 权重
            uint randomIdx = min(uint(sampleNext1D(sg) * numSamples), numSamples - 1);
            uint attempts = 0;
            while (cachedReservoirM[randomIdx] == 0 && attempts < numSamples)
            {
                randomIdx = (randomIdx + 1) % numSamples;
                attempts++;
            }
            if (cachedReservoirM[randomIdx] > 0)
            {
                uint2 neighIdx = uint2((cachedIndices[randomIdx] >> 16) & 0xFFFF, (cachedIndices[randomIdx] & 0xFFFF));
                SurfaceFG neighSurface = getSurface(neighIdx, true);
                float p_hat_canonical_at_j = 0.0;
                if (neighSurface.valid)
                {
                    LightSample lsCanonical = { };
                    float disCanonical;
                    if (getLightSample(fgCanonical, neighSurface.posW, lsCanonical, disCanonical))
                        p_hat_canonical_at_j = neighSurface.getPdf(lsCanonical.dir, lsCanonical.Li);
                }
                // selectionW = validCount（均匀采样的无偏估计因子）
                float selectionW = float(validCount);
                m_c = selectionW * computePairwiseMIS_canonical_beta(float(cachedReservoirM[randomIdx]), c_c, final_c_sum, p_hat_canonical_at_j, p_hat_canonical);
            }
        }

        // w_sum 从 canonical 开始
        float w_sum = m_c * p_hat_canonical * W_canonical;

        // 累加每个邻居的贡献（non-canonical 部分与标准版相同）
        for (uint j = 0; j < numSamples; j++)
        {
            if (cachedReservoirM[j] == 0) continue;
            float m_i = computePairwiseMIS_noncanonical(float(cachedReservoirM[j]), c_c, final_c_sum, cachedPhatNeighbor[j], cachedTargetFunc[j]);
            float w_i = m_i * cachedTargetFunc[j] * cachedW[j];
            w_sum += w_i;
            if (w_sum > 0 && sampleNext1D(sg) * w_sum < w_i)
            {
                selectedSample = j;
                fgSample = cachedFG[j];
                r.targetFunc = cachedTargetFunc[j];
            }
        }

        r.weightSum = w_sum;
        r.finalizeSample(1.0f, 1.0f);
    }
    #elif BIAS_CORRECTION_MODE >= BIAS_MODE_BASIC
        
        float pi = r.targetFunc;
        float piSum = r.targetFunc; //piSum = target * M_initial
        
        //go through all neighbors again
        for (int i = 0; i < numSamples; i++)
        {
            //Skip if we skipped in last loop
            if ((cachedIndices[i]) == 0xFFFFFFFF)
                continue;
            //Get Index
            uint2 idx = uint2((cachedIndices[i] >> 16) & 0xFFFF, (cachedIndices[i] & 0xFFFF));
            //Get Surface
            SurfaceFG biasSurface = getSurface(idx, true);
        
            //Get light pdf
            float ps = 0.0f;
            LightSample ls = { };
            float dist = 0;

            //Skip Sample if it was rejected by jacobian
            if (biasSurface.valid)
            {
                if(getLightSample(fgSample, biasSurface.posW, ls, dist))
                    ps = biasSurface.getPdf(ls.dir, ls.Li);
            }
                    
            //Trace ray
    #if BIAS_CORRECTION_MODE >= BIAS_MODE_RAYTRACE
            if (ps > 0)
            {
                SceneRayQuery < 0 > rayQuery;
                dist *= kRayEpsilon;
                float rayNear = dist * kRayNearEpsilon;
                dist -= rayNear;
                const Ray ray = Ray(computeRayOrigin(biasSurface.posW, biasSurface.normal), ls.dir, rayNear, dist);
                if (!rayQuery.traceVisibilityRay(ray, RAY_FLAG_CULL_NON_OPAQUE, 0xff))
                {
                    ps = 0.0f;
                }
            }
    #endif // BIAS MODE RAYTRACE
    
            uint M = cachedM[i / 2];
            if (i % 2 == 0)
                M = M & 0xFFFF;
            else
                M = (M >> 16) & 0xFFFF;
        
            pi = selectedSample == i ? ps : pi;
            piSum += ps * M;
        }
        r.finalizeSample(pi, piSum);
        
    #else //BIAS MODE BASIC
        r.finalizeSample(1.0f, r.M);
    #endif //BIAS MODE  
}

[numthreads(16, 16, 1)]
void main(uint2 dTid : SV_DispatchThreadID)
{    
    //Check if surface is valid
    SurfaceFG surface = getSurface(dTid, false);
    if (!surface.valid)
        return;

    #if STORE_SAMPLE_GEN_STATE
        SampleGenerator sg = gSampleGenState[index2Dto1D(dTid,gFrameDim.x)];
    #else
        SampleGenerator sg = SampleGenerator(dTid, gFrameCount + 5 * 32);
    #endif
    
    //Load reservoir from current iteration
    Reservoir r = { };
    FGData fgData = FGData();

    //Load current reservoir in 
#if MODE_SPATIAL
    Reservoir current = Reservoir(gReservoirPrev[dTid]);
    if (current.weightSum > 0)
    {
        fgData = gFGSampleDataPrev[index2Dto1D(dTid, gFrameDim.x)];
    }
#else //Mode Temporal or Spatiotemporal
    Reservoir current = Reservoir(gReservoir[dTid]);
    if (current.weightSum > 0)
    {
        fgData = gFGSampleData[index2Dto1D(dTid, gFrameDim.x)];
    }
#endif //MODE_SPATIAL

    r.combineReservoir(current, current.targetFunc, 0.5 /* random */);
    //Do resampling
#if MODE_SPATIOTEMPORAL
    spatiotemporalResampling(dTid, sg, r, fgData, surface);
#elif MODE_TEMPORAL
    temporalResampling(dTid, sg, r, fgData, surface);
#elif MODE_SPATIAL
    spatialResampling(dTid, sg, r, fgData, surface);
#endif //Resampling Mode
    
    //Store reservoir and sample data    
#if USE_REDUCED_RESERVOIR_FORMAT
    gReservoir[dTid] = r.packReduced();
#else
    gReservoir[dTid] = r.pack();
#endif

    gFGSampleData[index2Dto1D(dTid, gFrameDim.x)] = fgData;

    #if STORE_SAMPLE_GEN_STATE
        gSampleGenState[index2Dto1D(dTid, gFrameDim.x)] = sg;
    #endif //STORE_SAMPLE_GEN_STATE
}
