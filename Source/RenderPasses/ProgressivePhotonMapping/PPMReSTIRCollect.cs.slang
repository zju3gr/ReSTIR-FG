/**
 * PPM ReSTIR Collect Shader (Compute)
 * 用 Reservoir-based Importance Sampling (RIS) 替代传统的全量光子累加。
 *
 * 核心思路：
 *   1. Initial Sampling: 遍历 hash grid 中的光子，用流式 RIS 选出 1 个高质量样本
 *   2. Temporal Resampling: 复用上一帧的 Reservoir，等效增加样本数
 *   3. Spatial Resampling: 复用空间邻居的 Reservoir，进一步增加有效样本
 *   4. Final Evaluation: 用最终 Reservoir 的单样本计算间接光
 *
 * 优势：
 *   - 每帧可以用更少的光子（搜索量减少 10~30 倍）
 *   - 通过时空复用等效看到大量光子
 *   - 消除 MAX_PHOTONS_PER_CELL 的归一化偏差问题
 */

#include "Scene/SceneDefines.slangh"
#include "Utils/Math/MathConstants.slangh"

import Scene.Raytracing;
import Scene.RaytracingInline;
import Scene.Intersection;
import Utils.Math.MathHelpers;
import Utils.Geometry.GeometryHelpers;
import Utils.Sampling.SampleGenerator;
import Utils.Color.ColorHelpers;
import Rendering.Lights.LightHelpers;
import PPMParams;

#if USE_EMISSIVE_LIGHTS
import Rendering.Lights.EmissiveLightSampler;
import Rendering.Lights.EmissiveLightSamplerHelpers;
#endif

cbuffer CB
{
    uint gFrameCount;
    uint2 gFrameDim;
    float gInitialRadius;       // Initial search radius
    uint gNumPhotonsPerFrame;   // Number of photons emitted per frame
    float gAlpha;               // Radius reduction parameter alpha
    float gCellSize;            // Hash grid cell size
    float gCurrentRadius;       // 当前帧的搜索半径
    uint gDisplayMode;          // Display mode
    uint gTemporalMaxM;         // 时间重采样的最大 M 值限制（防止过度复用）
    uint gSpatialNeighbors;     // 空间重采样的邻居数量
    float gSpatialRadius;       // 空间重采样的搜索半径（像素）
}

#if USE_EMISSIVE_LIGHTS
cbuffer EmissiveLight
{
    EmissiveLightSampler gEmissiveSampler;
}
#endif

// Input
Texture2D<PackedHitInfo> gVBuffer;
Texture2D<float4> gViewW;

StructuredBuffer<PPMPhotonSorted> gSortedPhotonBuffer;
Buffer<uint> gPhotonCounter;
ByteAddressBuffer gCellOffset;
Buffer<uint> gCellCounter;

// Reservoir buffers（双缓冲用于时间重采样）
RWStructuredBuffer<PhotonReservoir> gReservoirBuffer;       // 当前帧输出
StructuredBuffer<PhotonReservoir> gPrevReservoirBuffer;     // 上一帧输入

// Per-pixel hit info（用于时空重采样的有效性检查）
RWTexture2D<float4> gPositionBuffer;    // xyz = world pos, w = 1 if valid
RWTexture2D<float4> gNormalBuffer;      // xyz = normal, w = roughness
Texture2D<float4> gPrevPositionBuffer;  // 上一帧的 position
Texture2D<float4> gPrevNormalBuffer;    // 上一帧的 normal

// Output
RWTexture2D<float4> gOutputColor;

// Per-pixel persistent statistics（用于累积输出）
RWStructuredBuffer<PPMPixelStats> gPixelStats;

static const uint kPhotonBufferSize = PHOTON_BUFFER_SIZE;
static const float kRoughnessThreshold = ROUGHNESS_THRESHOLD;
static const uint kMaxCameraBounces = MAX_CAMERA_BOUNCES;

#define is_valid(name) (is_valid_##name != 0)

/** Get primary ray direction */
float3 getPrimaryRayDir(uint2 pixel, uint2 frameDim)
{
    if (is_valid(gViewW))
    {
        return -gViewW[pixel].xyz;
    }
    else
    {
        return gScene.camera.computeRayPinhole(pixel, frameDim).dir;
    }
}

/** Load shading data from hit info */
ShadingData loadShadingDataFromHit(const HitInfo hit, const float3 rayOrigin, const float3 rayDir)
{
    let lod = ExplicitLodTextureSampler(0.f);
    const TriangleHit triangleHit = hit.getTriangleHit();
    VertexData v = gScene.getVertexData(triangleHit);
    uint materialID = gScene.getMaterialID(triangleHit.instanceID);
    ShadingData sd = gScene.materials.prepareShadingData(v, materialID, -rayDir, lod);
    return sd;
}

/** Evaluate direct lighting from analytic lights */
float3 evalDirectAnalytic(const ShadingData sd, const IMaterialInstance mi, inout SampleGenerator sg)
{
#if USE_ANALYTIC_LIGHTS
    const uint lightCount = gScene.getLightCount();
    if (lightCount == 0) return float3(0.f);

    const uint lightIndex = min(uint(sampleNext1D(sg) * lightCount), lightCount - 1);
    float invPdf = lightCount;

    AnalyticLightSample ls;
    if (!sampleLight(sd.posW, gScene.getLight(lightIndex), sg, ls)) return float3(0.f);

    const uint lobeTypes = mi.getLobeTypes(sd);
    const bool hasReflection = lobeTypes & uint(LobeType::Reflection);
    const bool hasTransmission = lobeTypes & uint(LobeType::Transmission);
    float NdotL = dot(sd.getOrientedFaceNormal(), ls.dir);
    if ((NdotL <= kMinCosTheta && !hasTransmission) || (NdotL >= -kMinCosTheta && !hasReflection))
        return float3(0.f);

    const float3 origin = computeRayOrigin(sd.posW, dot(sd.faceN, ls.dir) >= 0.f ? sd.faceN : -sd.faceN);

    SceneRayQuery<1> rayQuery;
    const Ray shadowRay = Ray(origin, ls.dir, 0.f, ls.distance * 0.999f);
    bool visible = rayQuery.traceVisibilityRay(shadowRay, RAY_FLAG_NONE, 0xff);
    if (!visible) return float3(0.f);

    return mi.eval(sd, ls.dir, sg) * ls.Li * invPdf;
#else
    return float3(0.f);
#endif
}

/** Evaluate direct lighting from emissive lights (NEE) */
float3 evalDirectEmissive(const ShadingData sd, const IMaterialInstance mi, inout SampleGenerator sg)
{
#if USE_EMISSIVE_LIGHTS
    TriangleLightSample tls;
    if (!gEmissiveSampler.sampleLight(sd.posW, sd.getOrientedFaceNormal(), true, sg, tls))
        return float3(0.f);

    if (tls.pdf <= 0.f)
        return float3(0.f);

    float3 Li = tls.Le / tls.pdf;

    float NdotL = dot(sd.getOrientedFaceNormal(), tls.dir);
    if (NdotL <= kMinCosTheta)
        return float3(0.f);

    float3 lightPos = computeRayOrigin(tls.posW, tls.normalW);
    float3 origin = computeRayOrigin(sd.posW, dot(sd.faceN, lightPos - sd.posW) >= 0.f ? sd.faceN : -sd.faceN);
    float3 toLight = lightPos - origin;
    float distance = length(toLight);
    float3 dir = toLight / distance;

    SceneRayQuery<1> rayQuery;
    const Ray shadowRay = Ray(origin, dir, 0.f, distance * 0.999f);
    bool visible = rayQuery.traceVisibilityRay(shadowRay, RAY_FLAG_NONE, 0xff);
    if (!visible) return float3(0.f);

    return mi.eval(sd, dir, sg) * Li;
#else
    return float3(0.f);
#endif
}

/** 检查两个像素的几何是否兼容（用于时空重采样有效性检查） */
bool isGeometryCompatible(float3 pos1, float3 normal1, float3 pos2, float3 normal2, float radius)
{
    // 位置距离检查
    float dist = length(pos1 - pos2);
    if (dist > radius * 2.0) return false;

    // 法线一致性检查
    float normalDot = dot(normal1, normal2);
    if (normalDot < 0.5) return false;

    return true;
}

[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint2 pixel = dispatchThreadId.xy;
    if (any(pixel >= gFrameDim))
        return;

    uint pixelIdx = pixel.x + pixel.y * gFrameDim.x;

    // Load first hit from VBuffer
    const HitInfo hit = HitInfo(gVBuffer[pixel]);

    if (!hit.isValid())
    {
        // Background pixel
        float3 finalColor = float3(0.f);
#if USE_ENV_BACKGROUND
        float3 primaryRayDir = getPrimaryRayDir(pixel, gFrameDim);
        finalColor = gScene.envMap.eval(primaryRayDir);
#endif
        gOutputColor[pixel] = float4(finalColor, 1.f);
        gPositionBuffer[pixel] = float4(0, 0, 0, 0);
        gNormalBuffer[pixel] = float4(0, 0, 0, 0);
        gReservoirBuffer[pixelIdx] = PhotonReservoir();
        return;
    }

    // Create sample generator
    SampleGenerator sg = SampleGenerator(pixel, gFrameCount);

    // Load shading data
    float3 cameraPos = gScene.camera.getPosition();
    float3 primaryRayDir = getPrimaryRayDir(pixel, gFrameDim);

    let lod = ExplicitLodTextureSampler(0.f);
    ShadingData sd = loadShadingDataFromHit(hit, cameraPos, primaryRayDir);

    float3 throughput = float3(1.0);
    float3 emission = float3(0.0);
    bool isDiffuse = false;
    float3 hitPos = sd.posW;
    float3 hitNormal = sd.getOrientedFaceNormal();

    // Evaluate first hit
    {
        let mi = gScene.materials.getMaterialInstance(sd, lod);
        let bsdfProperties = mi.getProperties(sd);

        emission += bsdfProperties.emission;

        isDiffuse = bsdfProperties.roughness > kRoughnessThreshold;
        hitPos = sd.posW;
        hitNormal = sd.getOrientedFaceNormal();

        if (isDiffuse && gDisplayMode != 3)
        {
            emission += evalDirectAnalytic(sd, mi, sg);
            emission += evalDirectEmissive(sd, mi, sg);
        }
    }

    // If specular, trace camera path to find diffuse surface
    if (!isDiffuse)
    {
        SceneRayQuery<1> rayQuery;

        for (uint bounce = 0; bounce < kMaxCameraBounces; bounce++)
        {
            let curMi = gScene.materials.getMaterialInstance(sd, lod);

            BSDFSample bsdfSample;
            if (!curMi.sample(sd, sg, bsdfSample, true))
            {
                // 无法继续追踪
                gPositionBuffer[pixel] = float4(0, 0, 0, 0);
                gNormalBuffer[pixel] = float4(0, 0, 0, 0);
                gReservoirBuffer[pixelIdx] = PhotonReservoir();
                // 累积 emission
                PPMPixelStats ps = gPixelStats[pixelIdx];
                if (gFrameCount == 0) { ps.tau = float3(0); ps.emissionSum = float3(0); }
                ps.emissionSum += emission;
                gPixelStats[pixelIdx] = ps;
                float iter = float(gFrameCount + 1);
                float3 result = ps.emissionSum / iter;
                gOutputColor[pixel] = float4(result, 1.f);
                return;
            }

            throughput *= bsdfSample.weight;

            bool transmission = bsdfSample.isLobe(LobeType::Transmission);
            float3 rayOrigin = sd.computeNewRayOrigin(!transmission);
            float3 rayDir = bsdfSample.wo;

            const Ray ray = Ray(rayOrigin, rayDir, 0.f, FLT_MAX);
            HitInfo nextHit;
            float hitT;
            if (!rayQuery.traceRay(ray, nextHit, hitT, RAY_FLAG_NONE, 0xff))
            {
#if USE_ENV_LIGHT
                emission += throughput * gScene.envMap.eval(rayDir);
#endif
                gPositionBuffer[pixel] = float4(0, 0, 0, 0);
                gNormalBuffer[pixel] = float4(0, 0, 0, 0);
                gReservoirBuffer[pixelIdx] = PhotonReservoir();
                PPMPixelStats ps = gPixelStats[pixelIdx];
                if (gFrameCount == 0) { ps.tau = float3(0); ps.emissionSum = float3(0); }
                ps.emissionSum += emission;
                gPixelStats[pixelIdx] = ps;
                float iter = float(gFrameCount + 1);
                float3 result = ps.emissionSum / iter;
                gOutputColor[pixel] = float4(result, 1.f);
                return;
            }

            sd = loadShadingDataFromHit(nextHit, rayOrigin, rayDir);
            let nextMi = gScene.materials.getMaterialInstance(sd, lod);
            let nextProps = nextMi.getProperties(sd);

            emission += throughput * nextProps.emission;

            isDiffuse = nextProps.roughness > kRoughnessThreshold;
            hitPos = sd.posW;
            hitNormal = sd.getOrientedFaceNormal();

            if (isDiffuse)
            {
                if (gDisplayMode != 3)
                {
                    emission += throughput * evalDirectAnalytic(sd, nextMi, sg);
                    emission += throughput * evalDirectEmissive(sd, nextMi, sg);
                }
                break;
            }

            // Russian Roulette
            if (bounce >= 3)
            {
                float rrVal = min((throughput.x + throughput.y + throughput.z) / 3.0, 0.95f);
                if (sampleNext1D(sg) >= rrVal)
                {
                    gPositionBuffer[pixel] = float4(0, 0, 0, 0);
                    gNormalBuffer[pixel] = float4(0, 0, 0, 0);
                    gReservoirBuffer[pixelIdx] = PhotonReservoir();
                    PPMPixelStats ps = gPixelStats[pixelIdx];
                    if (gFrameCount == 0) { ps.tau = float3(0); ps.emissionSum = float3(0); }
                    ps.emissionSum += emission;
                    gPixelStats[pixelIdx] = ps;
                    float iter = float(gFrameCount + 1);
                    float3 result = ps.emissionSum / iter;
                    gOutputColor[pixel] = float4(result, 1.f);
                    return;
                }
                throughput /= rrVal;
            }
        }
    }

    if (!isDiffuse)
    {
        gPositionBuffer[pixel] = float4(0, 0, 0, 0);
        gNormalBuffer[pixel] = float4(0, 0, 0, 0);
        gReservoirBuffer[pixelIdx] = PhotonReservoir();
        PPMPixelStats ps = gPixelStats[pixelIdx];
        if (gFrameCount == 0) { ps.tau = float3(0); ps.emissionSum = float3(0); }
        ps.emissionSum += emission;
        gPixelStats[pixelIdx] = ps;
        float iter = float(gFrameCount + 1);
        float3 result = ps.emissionSum / iter;
        gOutputColor[pixel] = float4(result, 1.f);
        return;
    }

    // Re-acquire material instance
    let finalMi = gScene.materials.getMaterialInstance(sd, lod);

    // 存储当前像素的几何信息（用于时空重采样）
    gPositionBuffer[pixel] = float4(hitPos, 1.0);
    let finalProps = finalMi.getProperties(sd);
    gNormalBuffer[pixel] = float4(hitNormal, finalProps.roughness);

    // ========================================
    // Stage 1: Initial Reservoir Generation (流式 RIS)
    // 遍历 hash grid 中的光子，用 streaming RIS 选出 1 个
    // ========================================
    float currentRadius = gCurrentRadius;
    float currentRadius2 = currentRadius * currentRadius;

    PhotonReservoir reservoir = PhotonReservoir();

    int3 minCell = worldToCell(hitPos - float3(currentRadius), gCellSize);
    int3 maxCell = worldToCell(hitPos + float3(currentRadius), gCellSize);

    for (int cz = minCell.z; cz <= maxCell.z; cz++)
    {
        for (int cy = minCell.y; cy <= maxCell.y; cy++)
        {
            for (int cx = minCell.x; cx <= maxCell.x; cx++)
            {
                int3 cell = int3(cx, cy, cz);
                uint cellIdx = hashGridCellIndex(cell);

                uint cellStart = gCellOffset.Load(cellIdx * 4);
                uint cellPhotonCount = min(gCellCounter[cellIdx], MAX_PHOTONS_PER_CELL);

                for (uint i = 0; i < cellPhotonCount; i++)
                {
                    PPMPhotonSorted photon = gSortedPhotonBuffer[cellStart + i];

                    float3 diff = photon.posW - hitPos;
                    float distSq = dot(diff, diff);

                    if (distSq < currentRadius2)
                    {
                        // 法线方向一致性检查
                        float NdotD = dot(hitNormal, photon.dirW);
                        if (NdotD > 0.01f)
                        {
                            // 计算 target function: luminance(BRDF * flux)
                            float3 bsdfVal = finalMi.eval(sd, photon.dirW, sg);
                            float cosTheta = abs(dot(sd.frame.N, photon.dirW));
                            if (cosTheta > 1e-4f)
                                bsdfVal /= cosTheta;

                            float3 contribution = throughput * bsdfVal * photon.flux;
                            float targetP = luminance(contribution);

                            if (targetP > 0)
                            {
                                // RIS 权重 = targetP / sourcePdf
                                // sourcePdf 对所有光子均匀，所以 w = targetP
                                float w = targetP;
                                float rand = sampleNext1D(sg);
                                reservoir.update(photon.posW, photon.flux, photon.dirW, w, targetP, contribution, rand);
                            }
                        }
                    }
                }
            }
        }
    }

    // ========================================
    // Stage 2: Temporal Resampling（时间重采样）
    // 复用上一帧的 Reservoir
    // ========================================
    if (gFrameCount > 0 && gTemporalMaxM > 0)
    {
        // 简单的 reprojection：使用相同像素位置（静态场景）
        // TODO: 对于动态场景，需要 motion vector 做 reprojection
        float4 prevPosData = gPrevPositionBuffer[pixel];
        float4 prevNormalData = gPrevNormalBuffer[pixel];

        if (prevPosData.w > 0)
        {
            float3 prevPos = prevPosData.xyz;
            float3 prevNormal = prevNormalData.xyz;

            // 几何兼容性检查
            if (isGeometryCompatible(hitPos, hitNormal, prevPos, prevNormal, currentRadius))
            {
                PhotonReservoir prevReservoir = gPrevReservoirBuffer[pixelIdx];

                // 限制历史 M 值，防止过度复用导致 temporal lag
                if (prevReservoir.M > gTemporalMaxM)
                {
                    float scale = float(gTemporalMaxM) / float(prevReservoir.M);
                    prevReservoir.wSum *= scale;
                    prevReservoir.M = gTemporalMaxM;
                }

                // 重新评估 prev reservoir 的样本在当前着色点的 target function
                if (prevReservoir.M > 0 && prevReservoir.targetPdf > 0)
                {
                    float3 bsdfVal = finalMi.eval(sd, prevReservoir.photonDirW, sg);
                    float cosTheta = abs(dot(sd.frame.N, prevReservoir.photonDirW));
                    if (cosTheta > 1e-4f)
                        bsdfVal /= cosTheta;

                    float3 prevContrib = throughput * bsdfVal * prevReservoir.photonFlux;
                    float targetPdfAtCurrent = luminance(prevContrib);

                    float rand = sampleNext1D(sg);
                    if (reservoir.merge(prevReservoir, targetPdfAtCurrent, rand))
                    {
                        // 更新 F 为当前着色点的贡献
                        reservoir.F = prevContrib;
                    }
                }
            }
        }
    }

    // ========================================
    // Stage 3: Spatial Resampling（空间重采样）
    // 复用邻居像素的 Reservoir
    // ========================================
    if (gSpatialNeighbors > 0)
    {
        for (uint n = 0; n < gSpatialNeighbors; n++)
        {
            // 随机选择邻居像素
            float angle = sampleNext1D(sg) * 2.0 * M_PI;
            float radius = sampleNext1D(sg) * gSpatialRadius;
            int2 neighborPixel = int2(pixel) + int2(int(cos(angle) * radius), int(sin(angle) * radius));

            // 边界检查
            if (any(neighborPixel < int2(0)) || any(neighborPixel >= int2(gFrameDim)))
                continue;

            uint2 np = uint2(neighborPixel);
            uint neighborIdx = np.x + np.y * gFrameDim.x;

            // 读取邻居的几何信息（当前帧已写入的）
            float4 neighborPosData = gPositionBuffer[np];
            float4 neighborNormalData = gNormalBuffer[np];

            if (neighborPosData.w <= 0) continue;

            float3 neighborPos = neighborPosData.xyz;
            float3 neighborNormal = neighborNormalData.xyz;

            // 几何兼容性检查
            if (!isGeometryCompatible(hitPos, hitNormal, neighborPos, neighborNormal, currentRadius))
                continue;

            // 读取邻居的 Reservoir（注意：空间重采样读取的是当前帧已生成的 reservoir）
            // 这里我们读取上一帧的 reservoir 作为空间邻居（避免读写冲突）
            PhotonReservoir neighborReservoir = gPrevReservoirBuffer[neighborIdx];

            if (neighborReservoir.M == 0 || neighborReservoir.targetPdf <= 0)
                continue;

            // 重新评估邻居样本在当前着色点的 target function
            float3 bsdfVal = finalMi.eval(sd, neighborReservoir.photonDirW, sg);
            float cosTheta = abs(dot(sd.frame.N, neighborReservoir.photonDirW));
            if (cosTheta > 1e-4f)
                bsdfVal /= cosTheta;

            float3 neighborContrib = throughput * bsdfVal * neighborReservoir.photonFlux;
            float targetPdfAtCurrent = luminance(neighborContrib);

            float rand = sampleNext1D(sg);
            if (reservoir.merge(neighborReservoir, targetPdfAtCurrent, rand))
            {
                reservoir.F = neighborContrib;
            }
        }
    }

    // ========================================
    // Stage 4: Final Evaluation
    // 用 Reservoir 的单样本计算间接光
    // ========================================
    float3 indirectRadiance = float3(0);

    if (reservoir.M > 0 && reservoir.targetPdf > 0)
    {
        // RIS 估计: F(x) * W, 其中 W = wSum / (targetPdf * M)
        float W = reservoir.getInvPdf();
        indirectRadiance = reservoir.F * W;

        // 归一化：除以 (π * r2) 得到 radiance 估计
        indirectRadiance /= (M_PI * currentRadius2);
    }

    // 存储当前帧的 Reservoir（用于下一帧的时间重采样）
    gReservoirBuffer[pixelIdx] = reservoir;

    // ========================================
    // 累积输出（与 Probabilistic PPM 相同的累积方式）
    // ========================================
    PPMPixelStats ps = gPixelStats[pixelIdx];
    if (gFrameCount == 0)
    {
        ps.tau = float3(0);
        ps.emissionSum = float3(0);
    }

    ps.tau += indirectRadiance;
    ps.emissionSum += emission;
    gPixelStats[pixelIdx] = ps;

    // Output: running average
    float iter = float(gFrameCount + 1);
    float3 avgEmission = ps.emissionSum / iter;
    float3 avgRadiance = ps.tau / iter;

    float3 finalColor;
    if (gDisplayMode == 1) finalColor = avgEmission;           // Direct Only
    else if (gDisplayMode == 2) finalColor = avgRadiance;      // Indirect Only
    else finalColor = avgEmission + avgRadiance;               // Full

    gOutputColor[pixel] = float4(finalColor, 1.f);
}
