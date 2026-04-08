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

#define BIAS_CORRECTION 1
#define BIAS_MODE_PAIRWISE 3
#define BIAS_MODE_PAIRWISE_STOCHASTIC 4

#ifndef BIAS_CORRECTION_MODE
    #define BIAS_CORRECTION_MODE 1
#endif

// Pairwise MIS 辅助函数（参考 Bitterli 2020 / MiniSPMIS）
float safeDivide(float a, float b)
{
    return abs(b) > 0 ? a / b : 0;
}

float computePairwiseMIS_canonical_beta(float c_i, float c_c, float c_sum, float phat_from_i, float phat_c)
{
    return (c_i / max(c_sum, 1e-8)) * safeDivide(
        c_c * phat_c,
        c_c * phat_c + c_sum * phat_from_i);
}

float computePairwiseMIS_noncanonical(float c_i, float c_c, float c_sum, float phat_from_i, float phat_c)
{
    return safeDivide(
        c_i * phat_from_i,
        c_sum * phat_from_i + c_c * phat_c);
}

#if USE_REDUCED_RESERVOIR_FORMAT
    RWTexture2D<uint2> gReservoir[2];
    Texture2D<uint2> gReservoirPrev[2];
#else
    RWTexture2D<uint4> gReservoir[2];
    Texture2D<uint4> gReservoirPrev[2];
#endif

RWStructuredBuffer<CausticSample> gCausticSample[2];
StructuredBuffer<CausticSample> gCausticSamplePrev[2];
StructuredBuffer<PackedSurfaceDataFG> gSurface;
StructuredBuffer<PackedSurfaceDataFG>gSurfacePrev;
Texture2D<float2> gMVec;
Texture2D<float4> gView;
Texture2D<float4> gPrevView;
RWStructuredBuffer<SampleGenerator> gSampleGenState;

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
    uint gDisocclusionBoostSamples; // Number of spatial samples if no temporal surface was found
    float2 gPhotonRadius;            //Caustic Photon Radius
    uint gPairwiseMIS_M;            //Pairwise MIS: 虚拟总候选数 M (覆盖 c_sum), 0 = 使用实际值
    uint gPairwiseMIS_N;            //Pairwise MIS: 实际采样邻居数 N (覆盖 numSamples), 0 = 使用 gSpatialSamples
}

static const float kRayEpsilon = 0.99f;
static const float kRayNearEpsilon = 0.01f;
static const bool kCollectDirectPhotons = RESERVOIR_PHOTON_DIRECT;

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

void temporalResampling(uint2 pixel, inout SampleGenerator sg, inout Reservoir r, inout CausticSample causticSample, in SurfaceFG surface, uint buffIdx = 0)
{
    uint M_canonical = r.M; // 保存 canonical 的 M 值
    float W_canonical = r.weightSum; // 保存 canonical 的 W 值
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
    float temporalSearchRadius = 1.f; //actual radius is 2 due to random value in [-0.5,0.5]
    
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

        //Test if a sample from the temporal surface could be in the domain of the current surface
        if (distance(surface.posW, temporalSurface.posW) > (gPhotonRadius[buffIdx] * 2))
            continue;

        //Valid hit was found
        foundTemporalSurface = true;
        break;
    }

    //If temporal surface was found, resample the previous sample
    if (foundTemporalSurface)
    {
        Reservoir prev = Reservoir(gReservoirPrev[buffIdx][idx]);
        prev.M = min(prev.M, historyLimit);
        previousM = prev.M;

        CausticSample cauPrev = gCausticSamplePrev[buffIdx][index2Dto1D(idx, gFrameDim.x)];
       
        //Get target pdf at current location if reservoir and sample are valid
        float targetFunc = 0;
        if (prev.M > 0)
        {
            //Target Function
            float NdotL = dot(cauPrev.dir, surface.normal);
            if (NdotL > kMinCosTheta)
                targetFunc = surface.getPdf(cauPrev.dir, luminance(cauPrev.flux)) / NdotL;

            // Shift into the domain
            if (distance(cauPrev.pos, surface.posW) > gPhotonRadius[buffIdx])
                targetFunc = 0;

#if BIAS_CORRECTION_MODE == BIAS_MODE_PAIRWISE || BIAS_CORRECTION_MODE == BIAS_MODE_PAIRWISE_STOCHASTIC
            // Pairwise MIS: 计算邻居样本在邻居处的 target function
            float p_hat_neighbor = 0.0;
            {
                float NdotL_n = dot(cauPrev.dir, temporalSurface.normal);
                if (NdotL_n > kMinCosTheta)
                    p_hat_neighbor = temporalSurface.getPdf(cauPrev.dir, luminance(cauPrev.flux)) / NdotL_n;
                if (distance(cauPrev.pos, temporalSurface.posW) > gPhotonRadius[buffIdx])
                    p_hat_neighbor = 0.0;
            }

            // 计算 canonical 样本在邻居位置的 target function
            float p_hat_canonical_at_neighbor = 0.0;
            {
                float NdotL_cn = dot(causticSample.dir, temporalSurface.normal);
                if (NdotL_cn > kMinCosTheta)
                    p_hat_canonical_at_neighbor = temporalSurface.getPdf(causticSample.dir, luminance(causticSample.flux)) / NdotL_cn;
                if (distance(causticSample.pos, temporalSurface.posW) > gPhotonRadius[buffIdx])
                    p_hat_canonical_at_neighbor = 0.0;
            }

            // c_c = M_canonical, c_i = prev.M, c_sum = prev.M (只有一个邻居)
            float c_c = float(M_canonical);
            float c_i = float(prev.M);
            float c_sum = gPairwiseMIS_M > 0 ? float(gPairwiseMIS_M) : c_i;

            // Canonical MIS 权重
            float m_c = computePairwiseMIS_canonical_beta(c_i, c_c, c_sum, p_hat_canonical_at_neighbor, p_hat_canonical);

            // Non-canonical MIS 权重
            float m_i = computePairwiseMIS_noncanonical(c_i, c_c, c_sum, p_hat_neighbor, targetFunc);

            // 手动计算 w_sum
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
            selectPrevSample = r.combineReservoir(prev, targetFunc, sampleNext1D(sg));
#endif
        }

        if (selectPrevSample)
            causticSample = cauPrev;
    }


    //Bias correction via MIS
#if BIAS_CORRECTION_MODE == BIAS_MODE_PAIRWISE || BIAS_CORRECTION_MODE == BIAS_MODE_PAIRWISE_STOCHASTIC
    // Pairwise MIS: 直接 finalize，MIS 权重已内联处理
    r.finalizeSample(1.0f, 1.0f);
#elif BIAS_CORRECTION
    float pi = r.targetFunc;
    float piSum = r.targetFunc; //piSum = target * M_initial ; M_initial is always 1 for us

    if (foundTemporalSurface)
    {
        float ps = 0.0;
        float NdotL = dot(causticSample.dir, temporalSurface.normal);
        if (NdotL > kMinCosTheta)
            ps = temporalSurface.getPdf(causticSample.dir, luminance(causticSample.flux)) / NdotL;
        if (distance(causticSample.pos, temporalSurface.posW) > gPhotonRadius[buffIdx])
            ps = 0.0;

        pi = selectPrevSample ? ps : pi;
        piSum += ps * previousM;
    }
    r.finalizeSample(pi, piSum);
#else  // BIAS_CORRECTION 
    r.finalizeSample(1.0f, float(r.M));
#endif // BIAS_CORRECTION
}

void spatiotemporalResampling(uint2 pixel, inout SampleGenerator sg, inout Reservoir r, inout CausticSample causticSample, in SurfaceFG surface, uint buffIdx = 0)
{
    uint M_canonical = r.M; // 保存 canonical 的 M 值
    float W_canonical = r.weightSum; // 保存 canonical 的 W 值
    float p_hat_canonical = r.targetFunc; // 保存 canonical 的 target function
    CausticSample causticCanonical = causticSample; // 保存原始 canonical 样本
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
    const float temporalSearchRadius = 1.f; //actual radius is 1 due to random value in [-0.5,0.5]

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

        //Test if a sample from the temporal surface could be in the domain of the current surface
        if (distance(surface.posW, temporalSurface.posW) > (gPhotonRadius[buffIdx] * 2))
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
    CausticSample cachedCaustic[8];
    uint cachedReservoirM[8];
    float c_sum = 0.0;
    uint validCount = 0;

    //Go through all neighbors
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
            //Get sample from sampling mask
            float2 randSamp = (sampleNext2D(sg) * 2.f) - 1.f; //random sample in [-1,1]
            int2 spatialOffset = int2(round(randSamp * gSamplingRadius));
            idx = prevIdx + spatialOffset;

            //Outside index is invalid
            if(any(idx < 0) || any(idx >= gFrameDim))
                continue;

            //Reject if the spatial sample is the same as the temporal sample
            if(all(spatialOffset == temporalOffset))
                continue;

             //Get surface and check if valid
            SurfaceFG neighSurface = getSurface(idx, true);
            if (!neighSurface.valid)
                continue;
        
            if (!surface.isValidNeighbor(neighSurface, gDepthThreshold, gNormalThreshold))
                continue;

            //Test if a sample from the temporal surface could be in the domain of the current surface
            if (distance(surface.posW, neighSurface.posW) > (gPhotonRadius[buffIdx] * 2))
                continue;
        }
            
        //If neighbor surface has passed the tests get reservoir and compair them
        Reservoir sampledReservoir = Reservoir(gReservoirPrev[buffIdx][idx]);
        sampledReservoir.M = min(sampledReservoir.M, historyLimit);

        CausticSample cauPrev = gCausticSamplePrev[buffIdx][index2Dto1D(idx, gFrameDim.x)];

        //Store indices
        cachedIndices[i] = (idx.x & 0xFFFF) << 16 | (idx.y & 0xFFFF);
        uint M = 0xFFFF & sampledReservoir.M;
        if (i % 2 != 0)
            M = M << 16;
        cachedM[i / 2] |= M;
        
        //Get neightbor light target PDF for current surface
        float targetFunc = 0.f;
        if (sampledReservoir.M > 0)
        {
            //Target Function
            float NdotL = dot(cauPrev.dir, surface.normal);
            if (NdotL > kMinCosTheta)
                targetFunc = surface.getPdf(cauPrev.dir, luminance(cauPrev.flux)) / NdotL;

            // Shift into the domain
            if (distance(cauPrev.pos, surface.posW) > gPhotonRadius[buffIdx])
                targetFunc = 0;
        }

#if BIAS_CORRECTION_MODE == BIAS_MODE_PAIRWISE || BIAS_CORRECTION_MODE == BIAS_MODE_PAIRWISE_STOCHASTIC
        // Pairwise MIS: 缓存邻居数据
        float p_hat_neighbor = 0.0;
        if (sampledReservoir.M > 0)
        {
            uint2 neighIdx = uint2((cachedIndices[i] >> 16) & 0xFFFF, (cachedIndices[i] & 0xFFFF));
            SurfaceFG neighSurface = getSurface(neighIdx, true);
            if (neighSurface.valid)
            {
                float NdotL_n = dot(cauPrev.dir, neighSurface.normal);
                if (NdotL_n > kMinCosTheta)
                    p_hat_neighbor = neighSurface.getPdf(cauPrev.dir, luminance(cauPrev.flux)) / NdotL_n;
                if (distance(cauPrev.pos, neighSurface.posW) > gPhotonRadius[buffIdx])
                    p_hat_neighbor = 0.0;
            }
        }
        cachedTargetFunc[i] = targetFunc;
        cachedPhatNeighbor[i] = p_hat_neighbor;
        cachedW[i] = sampledReservoir.weightSum;
        cachedCaustic[i] = cauPrev;
        cachedReservoirM[i] = sampledReservoir.M;
        c_sum += float(sampledReservoir.M);
        validCount++;
        r.M += sampledReservoir.M;
#else
        //Combine Reservoirs
        if (r.combineReservoir(sampledReservoir, targetFunc, sampleNext1D(sg)))
        {
            selectedSample = i;
            causticSample = cauPrev;
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
                    float NdotL_cn = dot(causticCanonical.dir, neighSurface.normal);
                    if (NdotL_cn > kMinCosTheta)
                        p_hat_canonical_at_j = neighSurface.getPdf(causticCanonical.dir, luminance(causticCanonical.flux)) / NdotL_cn;
                    if (distance(causticCanonical.pos, neighSurface.posW) > gPhotonRadius[buffIdx])
                        p_hat_canonical_at_j = 0.0;
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
                causticSample = cachedCaustic[j];
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
                    float NdotL_cn = dot(causticCanonical.dir, neighSurface.normal);
                    if (NdotL_cn > kMinCosTheta)
                        p_hat_canonical_at_j = neighSurface.getPdf(causticCanonical.dir, luminance(causticCanonical.flux)) / NdotL_cn;
                    if (distance(causticCanonical.pos, neighSurface.posW) > gPhotonRadius[buffIdx])
                        p_hat_canonical_at_j = 0.0;
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
                causticSample = cachedCaustic[j];
                r.targetFunc = cachedTargetFunc[j];
            }
        }

        r.weightSum = w_sum;
        r.finalizeSample(1.0f, 1.0f);
    }
    #elif BIAS_CORRECTION
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
            //Check if sample could be generated at the bias surface
            if (biasSurface.valid)
            {
                float NdotL = dot(causticSample.dir, biasSurface.normal);
                if (NdotL > kMinCosTheta)
                    ps = biasSurface.getPdf(causticSample.dir, luminance(causticSample.flux)) / NdotL;
                if (distance(causticSample.pos, biasSurface.posW) > gPhotonRadius[buffIdx])
                    ps = 0.0;
            }
    
            uint M = cachedM[i / 2];
            if (i % 2 == 0)
                M = M & 0xFFFF;
            else
                M = (M >> 16) & 0xFFFF;
        
            pi = selectedSample == i ? ps : pi;
            piSum += ps * M;
        }
        r.finalizeSample(pi, piSum);
        
    #else //BIAS_CORRECTION
        r.finalizeSample(1.0f, r.M);
    #endif //BIAS_CORRECTION
}

[numthreads(16, 16, 1)]
void main(uint2 dTid : SV_DispatchThreadID)
{    
    //Check if surface is valid
    float4 viewDir = gView[dTid];
    SurfaceFG surface = SurfaceFG(gSurface[index2Dto1D(dTid, gFrameDim.x)],viewDir.xyz);

    bool collectDirect = (viewDir.w > 0.1) && kCollectDirectPhotons; // Check for direct photon collection
    if (!surface.valid)
        return;

    #if STORE_SAMPLE_GEN_STATE
        SampleGenerator sg = gSampleGenState[index2Dto1D(dTid, gFrameDim.x)];
    #else
        SampleGenerator sg = SampleGenerator(dTid, gFrameCount + 8 * 32);
    #endif

    //Load reservoir from current iteration
    Reservoir r = {};
    CausticSample causticData = CausticSample();

    //Load current reservoir
    Reservoir current = Reservoir(gReservoir[0][dTid]);
    if (current.M > 0)
    {
        causticData = gCausticSample[0][index2Dto1D(dTid, gFrameDim.x)];
    }
    
    r.combineReservoir(current, current.targetFunc, 0.5 /* random */);

    //Resampling
#if MODE_SPATIOTEMPORAL
    spatiotemporalResampling(dTid, sg, r, causticData, surface);
#elif MODE_TEMPORAL
    temporalResampling(dTid, sg, r, causticData, surface);
#endif //Resampling Mode


    //Store reservoir and sample data    
#if USE_REDUCED_RESERVOIR_FORMAT
    gReservoir[0][dTid] = r.packReduced();
#else
    gReservoir[0][dTid] = r.pack();
#endif
    gCausticSample[0][index2Dto1D(dTid, gFrameDim.x)] = causticData;

    //Second Resample pass for directly collected photons on some metal surfaces
    if(collectDirect)
    {
        r = {};
        causticData = CausticSample();

        //Load current reservoir 
        current = Reservoir(gReservoir[1][dTid]);
        if (current.M > 0)
        {
            causticData = gCausticSample[1][index2Dto1D(dTid, gFrameDim.x)];
        }
        r.combineReservoir(current, current.targetFunc, 0.5 /* random */);

        //Resampling
        #if MODE_SPATIOTEMPORAL
            spatiotemporalResampling(dTid, sg, r, causticData, surface, 1);
        #elif MODE_TEMPORAL
            temporalResampling(dTid, sg, r, causticData, surface, 1);
        #endif //Resampling Mode

    
            //Store reservoir and sample data    
        #if USE_REDUCED_RESERVOIR_FORMAT
            gReservoir[1][dTid] = r.packReduced();
        #else
            gReservoir[1][dTid] = r.pack();
        #endif
            gCausticSample[1][index2Dto1D(dTid, gFrameDim.x)] = causticData;
    }
    
    #if STORE_SAMPLE_GEN_STATE
        gSampleGenState[index2Dto1D(dTid, gFrameDim.x)] = sg;
    #endif //STORE_SAMPLE_GEN_STATE
}
