/**
 * PPM Photon Collection Shader (Compute)
 * Supports two modes via USE_PROBABILISTIC_PPM define:
 *
 * Mode 1 (USE_PROBABILISTIC_PPM = 1): Probabilistic PPM (Knaus & Zwicker, TOG 2011)
 *   - Each frame independently estimates radiance: L_i = photonContrib / (pi * r_i^2)
 *   - Running average across frames: L = (1/N) * sum L_i
 *   - Radius reduction: r_{i+1}^2 = r_i^2 * (i + alpha) / (i + 1)
 *   - Supports random camera paths each frame (glossy, DOF, motion blur)
 *
 * Mode 2 (USE_PROBABILISTIC_PPM = 0): Standard SPPM (Hachisuka et al. 2009)
 *   - Accumulates tau: tau_new = (tau_old + Phi) * (r_new^2 / r_old^2)
 *   - Radius reduction: Nnew = N + alpha*M, r_new^2 = r_old^2 * Nnew/(N+M)
 *   - Output: direct/iter + tau / (pi * r^2 * iter)
 *   - Better for fixed camera paths with high photon counts
 */

#include "Scene/SceneDefines.slangh"
#include "Utils/Math/MathConstants.slangh"

import Scene.Raytracing;
import Scene.RaytracingInline;
import Scene.Intersection;
import Utils.Math.MathHelpers;
import Utils.Geometry.GeometryHelpers;
import Utils.Sampling.SampleGenerator;
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
    float gAlpha;               // Radius reduction parameter alpha (0.6~0.9)
    float gCellSize;            // Hash grid cell size (= initial radius, for photon lookup)
    uint _pad0;
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

StructuredBuffer<PPMPhoton> gPhotonBuffer;
Buffer<uint> gPhotonCounter;
Buffer<uint> gHashGrid;  // Hash grid: each cell stores the index of the first photon

// Input/Output: per-pixel persistent statistics
RWStructuredBuffer<PPMPixelStats> gPixelStats;

// Output
RWTexture2D<float4> gOutputColor;

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

    // Visibility test using inline ray query
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
    // Use EmissiveLightSampler to sample a point on an emissive triangle
    TriangleLightSample tls;
    if (!gEmissiveSampler.sampleLight(sd.posW, sd.getOrientedFaceNormal(), true, sg, tls))
        return float3(0.f);

    if (tls.pdf <= 0.f)
        return float3(0.f);

    // Compute light contribution Li / pdf
    float3 Li = tls.Le / tls.pdf;

    // Check if direction is in the correct hemisphere
    float NdotL = dot(sd.getOrientedFaceNormal(), tls.dir);
    if (NdotL <= kMinCosTheta)
        return float3(0.f);

    // Compute shadow ray origin (offset to avoid self-intersection)
    float3 lightPos = computeRayOrigin(tls.posW, tls.normalW);
    float3 origin = computeRayOrigin(sd.posW, dot(sd.faceN, lightPos - sd.posW) >= 0.f ? sd.faceN : -sd.faceN);
    float3 toLight = lightPos - origin;
    float distance = length(toLight);
    float3 dir = toLight / distance;

    // Visibility test
    SceneRayQuery<1> rayQuery;
    const Ray shadowRay = Ray(origin, dir, 0.f, distance * 0.999f);
    bool visible = rayQuery.traceVisibilityRay(shadowRay, RAY_FLAG_NONE, 0xff);
    if (!visible) return float3(0.f);

    // Evaluate BSDF value
    return mi.eval(sd, dir, sg) * Li;
#else
    return float3(0.f);
#endif
}

/** Helper: write early exit result and return */
void writeEarlyExit(uint2 pixel, uint pixelIdx, float3 emission)
{
    PPMPixelStats ps = gPixelStats[pixelIdx];
    if (gFrameCount == 0)
    {
        ps.radius2 = gInitialRadius * gInitialRadius;
        ps.N = 0;
        ps.tau = float3(0);
        ps.emissionSum = float3(0);
    }
    ps.emissionSum += emission;

#if USE_PROBABILISTIC_PPM
    // Probabilistic PPM: increment iteration count and reduce radius
    ps.N += 1.0;
    float iterN = ps.N;
    ps.radius2 = ps.radius2 * (iterN - 1.0 + gAlpha) / iterN;
    ps.radius2 = max(ps.radius2, gInitialRadius * gInitialRadius * 0.0001f);
    gPixelStats[pixelIdx] = ps;

    // Output: running average
    float3 avgEmission = ps.emissionSum / ps.N;
    float3 avgRadiance = ps.tau / max(ps.N, 1.0);
    gOutputColor[pixel] = float4(avgEmission + avgRadiance, 1.f);
#else
    // Standard SPPM: no radius change when no photons found (M=0), no N change
    ps.radius2 = max(ps.radius2, gInitialRadius * gInitialRadius * 0.0001f);
    gPixelStats[pixelIdx] = ps;

    // Output: direct/iter + tau/(pi*r^2*iter)
    float iter = float(gFrameCount + 1);
    float3 direct = ps.emissionSum / iter;
    float3 indirect = ps.tau / (M_PI * ps.radius2 * iter);
    gOutputColor[pixel] = float4(direct + indirect, 1.f);
#endif
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

    float3 finalColor = float3(0.f);

    if (!hit.isValid())
    {
        // Background pixel
#if USE_ENV_BACKGROUND
        float3 primaryRayDir = getPrimaryRayDir(pixel, gFrameDim);
        finalColor = gScene.envMap.eval(primaryRayDir);
#endif
        gOutputColor[pixel] = float4(finalColor, 1.f);
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

        // Self-emission
        emission += bsdfProperties.emission;

        isDiffuse = bsdfProperties.roughness > kRoughnessThreshold;
        hitPos = sd.posW;
        hitNormal = sd.getOrientedFaceNormal();
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
                writeEarlyExit(pixel, pixelIdx, emission);
                return;
            }

            throughput *= bsdfSample.weight;

            bool transmission = bsdfSample.isLobe(LobeType::Transmission);
            float3 rayOrigin = sd.computeNewRayOrigin(!transmission);
            float3 rayDir = bsdfSample.wo;

            // Inline ray trace
            const Ray ray = Ray(rayOrigin, rayDir, 0.f, FLT_MAX);
            HitInfo nextHit;
            float hitT;
            if (!rayQuery.traceRay(ray, nextHit, hitT, RAY_FLAG_NONE, 0xff))
            {
#if USE_ENV_LIGHT
                emission += throughput * gScene.envMap.eval(rayDir);
#endif
                writeEarlyExit(pixel, pixelIdx, emission);
                return;
            }

            // Load new shading data
            sd = loadShadingDataFromHit(nextHit, rayOrigin, rayDir);
            let nextMi = gScene.materials.getMaterialInstance(sd, lod);
            let nextProps = nextMi.getProperties(sd);

            emission += throughput * nextProps.emission;

            isDiffuse = nextProps.roughness > kRoughnessThreshold;
            hitPos = sd.posW;
            hitNormal = sd.getOrientedFaceNormal();

            if (isDiffuse)
                break;

            // Russian Roulette for camera path
            if (bounce >= 3)
            {
                float rrVal = min((throughput.x + throughput.y + throughput.z) / 3.0, 0.95f);
                if (sampleNext1D(sg) >= rrVal)
                {
                    writeEarlyExit(pixel, pixelIdx, emission);
                    return;
                }
                throughput /= rrVal;
            }
        }
    }

    // No diffuse surface found after max bounces
    if (!isDiffuse)
    {
        writeEarlyExit(pixel, pixelIdx, emission);
        return;
    }

    // ========================================
    // Load per-pixel statistics
    // ========================================
    PPMPixelStats ps = gPixelStats[pixelIdx];

    // Initialize on first frame
    if (gFrameCount == 0)
    {
        ps.radius2 = gInitialRadius * gInitialRadius;
        ps.N = 0;
        ps.tau = float3(0);
        ps.emissionSum = float3(0);
    }

    float currentRadius = sqrt(ps.radius2);

    // Re-acquire material instance for the final diffuse hit point
    let finalMi = gScene.materials.getMaterialInstance(sd, lod);

    // ========================================
    // Photon density estimation using hash grid
    // ========================================
    float3 photonContrib = float3(0.f);
    uint photonCount = 0;  // M: number of photons found this frame

    // Determine which hash grid cells to search
    int3 minCell = worldToCell(hitPos - float3(currentRadius), gCellSize);
    int3 maxCell = worldToCell(hitPos + float3(currentRadius), gCellSize);

    // Iterate over neighboring cells
    for (int cz = minCell.z; cz <= maxCell.z; cz++)
    {
        for (int cy = minCell.y; cy <= maxCell.y; cy++)
        {
            for (int cx = minCell.x; cx <= maxCell.x; cx++)
            {
                int3 cell = int3(cx, cy, cz);
                uint cellIdx = hashGridCellIndex(cell);

                // Traverse linked list of photons in this cell
                uint photonIdx = gHashGrid[cellIdx];
                uint maxIter = 512; // Safety limit to avoid infinite loops
                while (photonIdx != HASH_GRID_INVALID && maxIter > 0)
                {
                    maxIter--;
                    PPMPhoton photon = gPhotonBuffer[photonIdx];

                    float3 diff = photon.posW - hitPos;
                    float distSq = dot(diff, diff);

                    if (distSq < ps.radius2)
                    {
                        // Check normal direction consistency
                        float NdotD = dot(hitNormal, photon.dirW);
                        if (NdotD > 0.01f)
                        {
                            // Evaluate BSDF: eval returns f_r * cos(theta)
                            // Photon mapping density estimation needs f_r only, divide by cos
                            float3 bsdfVal = finalMi.eval(sd, photon.dirW, sg);
                            float cosShading = abs(dot(sd.frame.N, photon.dirW));
                            if (cosShading > 1e-4f)
                                bsdfVal /= cosShading;

                            // Accumulate: beta * f_r * photon_flux
                            photonContrib += throughput * bsdfVal * photon.flux;
                            photonCount++;
                        }
                    }

                    photonIdx = photon.nextInCell;
                }
            }
        }
    }

    // ========================================
    // Progressive update - two modes
    // ========================================

#if USE_PROBABILISTIC_PPM
    // ========================================
    // Mode 1: Probabilistic PPM (Knaus & Zwicker 2011)
    // Per-frame radiance estimate: L_i = photonContrib / (pi * r_i^2)
    // Running average: L = (1/N) * sum L_i
    // Radius reduction: r_{i+1}^2 = r_i^2 * (N-1+alpha) / N
    // ========================================
    float3 frameRadiance = photonContrib / (M_PI * ps.radius2);

    ps.tau += frameRadiance;  // tau stores radianceSum in probabilistic mode
    ps.emissionSum += emission;
    ps.N += 1.0;

    // Radius reduction for next iteration
    float iterN = ps.N;
    ps.radius2 = ps.radius2 * (iterN - 1.0 + gAlpha) / iterN;

    // Clamp minimum radius
    ps.radius2 = max(ps.radius2, gInitialRadius * gInitialRadius * 0.0001f);

    // Write back
    gPixelStats[pixelIdx] = ps;

    // Output: running average
    float3 avgEmission = ps.emissionSum / ps.N;
    float3 avgRadiance = ps.tau / ps.N;
    finalColor = avgEmission + avgRadiance;

#else
    // ========================================
    // Mode 2: Standard SPPM (Hachisuka et al. 2009)
    // Nnew = N + alpha * M
    // r_new^2 = r_old^2 * Nnew / (N + M)
    // tau = (tau + Phi) * (r_new^2 / r_old^2)
    // Output: direct/iter + tau / (pi * r^2 * iter)
    // ========================================
    if (photonCount > 0)
    {
        float M = float(photonCount);
        float Nnew = ps.N + gAlpha * M;
        float NplusM = ps.N + M;
        float radius2New = ps.radius2 * (Nnew / NplusM);

        // Scale tau: old tau + new Phi, then scale by area ratio
        float ratioScale = radius2New / ps.radius2;
        ps.tau = (ps.tau + photonContrib) * ratioScale;

        ps.N = Nnew;
        ps.radius2 = radius2New;

        // Clamp minimum radius
        ps.radius2 = max(ps.radius2, gInitialRadius * gInitialRadius * 0.0001f);
    }

    // Accumulate emission
    ps.emissionSum += emission;

    // Write back
    gPixelStats[pixelIdx] = ps;

    // Output: direct average + tau / (pi * r^2 * iter)
    float iter = float(gFrameCount + 1);
    float3 direct = ps.emissionSum / iter;
    float3 indirect = ps.tau / (M_PI * ps.radius2 * iter);
    finalColor = direct + indirect;

#endif

    gOutputColor[pixel] = float4(finalColor, 1.f);
}
