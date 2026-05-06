/**
 * PPM Photon Tracing Shader
 *
 * Emits photons from light sources and stores them at diffuse surfaces.
 * Supports analytic lights (point/spot lights) and emissive triangles.
 */

#include "Scene/SceneDefines.slangh"
#include "Utils/Math/MathConstants.slangh"

import Scene.Raytracing;
import Utils.Math.MathHelpers;
import Utils.Geometry.GeometryHelpers;
import Utils.Sampling.SampleGenerator;
import Rendering.Lights.LightHelpers;
import PPMParams;

#if USE_EMISSIVE_LIGHT
import Rendering.Lights.EmissiveLightSampler;
import Rendering.Lights.EmissiveLightSamplerHelpers;
#endif

cbuffer CB
{
    uint gFrameCount;
    uint gMaxBounces;
    bool gUseAnalyticLights;
    float gCellSize;
    float gAnalyticLightRatio;  // Ratio of analytic light when using mixed light sources
}

#if USE_EMISSIVE_LIGHT
cbuffer Light
{
    EmissiveLightSampler gEmissiveSampler;
}
#endif

RWStructuredBuffer<PPMPhoton> gPhotonBuffer;
RWBuffer<uint> gPhotonCounter;
RWBuffer<uint> gHashGrid;  // Hash grid: each cell stores the index of the first photon in that cell

static const uint kPhotonBufferSize = PHOTON_BUFFER_SIZE;
static const float kRoughnessThreshold = ROUGHNESS_THRESHOLD;

// Ray payload
struct RayPayload
{
    PackedHitInfo packedHitInfo;

    __init()
    {
        this.packedHitInfo = {};
    }
};

// Photon light sample result
struct PhotonLightSample
{
    float3 posW;
    float3 normal;
    float3 flux;
};

// Transform direction from local to world coordinates
float3 fromLocalToWorld(const float3 normal, float3 w)
{
    float3 bitangent = perp_stark(normal);
    float3 tangent = cross(bitangent, normal);
    return w.x * tangent + w.y * bitangent + w.z * normal;
}

#if USE_EMISSIVE_LIGHT
/** Sample a photon from emissive triangles */
bool createEmissiveLightSample(inout SampleGenerator sg, out PhotonLightSample pls)
{
    pls = {};
    if (gScene.lightCollection.isEmpty())
        return false;

    const uint triangleCount = gScene.lightCollection.triangleCount;
    uint triangleIndex = min((uint)(sampleNext1D(sg) * triangleCount), triangleCount - 1);

    // Use alias table for power-weighted sampling
    uint2 packed = gEmissiveSampler._emissivePower.triangleAliasTable[triangleIndex];
    float threshold = f16tof32(packed.x >> 16u);
    uint selectAbove = ((packed.x & 0xFFFFu) << 8u) | ((packed.y >> 24u) & 0xFFu);
    uint selectBelow = packed.y & 0xFFFFFFu;
    triangleIndex = (sampleNext1D(sg) >= threshold) ? selectAbove : selectBelow;

    float triangleSelectionPdf = gScene.lightCollection.fluxData[triangleIndex].flux * gEmissiveSampler._emissivePower.invWeightsSum;

    const EmissiveTriangle tri = gScene.lightCollection.getTriangle(triangleIndex);
    const float3 barycentrics = sample_triangle(sampleNext2D(sg));
    pls.posW = tri.getPosition(barycentrics);
    pls.normal = tri.normal;

    float2 uv = tri.getTexCoord(barycentrics);
    pls.flux = gScene.materials.evalEmissive(tri.materialID, uv) * tri.area;

    pls.posW = computeRayOrigin(pls.posW, pls.normal);
    pls.flux /= triangleSelectionPdf;

    return true;
}
#endif

/** Sample a photon from analytic lights (point/spot/directional lights) */
bool createAnalyticLightSample(inout SampleGenerator sg, out PhotonLightSample pls, out float3 sampledDir)
{
    pls = {};
    sampledDir = float3(0);
    const uint lightCount = gScene.getLightCount();
    if (lightCount == 0)
        return false;

    uint lightIdx = min(uint(floor(sampleNext1D(sg) * lightCount)), lightCount - 1);
    LightData lightData = gScene.getLight(lightIdx);

    float invLightPdf;

    if (lightData.type == uint(LightType::Point))
    {
        float3 woLocal;
        float3 rnd = sampleNext3D(sg);
        float spotAngle = lightData.cosOpeningAngle;
        if (lightData.openingAngle < M_PI)
        {
            // Spot light: sample within cone
            woLocal = sample_cone(rnd.xy, spotAngle);
            invLightPdf = M_2PI * (1.f - spotAngle);
        }
        else
        {
            // Point light: sample full sphere
            woLocal = sample_sphere(rnd.xy);
            invLightPdf = 4.f * M_PI;
        }

        if (invLightPdf <= 0.f)
            return false;

        invLightPdf *= lightCount;

        pls.posW = lightData.posW;
        pls.normal = lightData.dirW;
        pls.flux = lightData.intensity * invLightPdf;

        sampledDir = fromLocalToWorld(pls.normal, woLocal);
    }
    else if (lightData.type == uint(LightType::Directional))
    {
        // Directional light: all photons travel in the same direction
        // Place photon origin far away in the light direction
        float3 sceneCenter = float3(0, 0, 0); // Approximate scene center
        float sceneRadius = 10.0f; // Approximate scene radius

        // Random position on a disk perpendicular to light direction
        float2 diskSample = sampleNext2D(sg);
        float r = sqrt(diskSample.x) * sceneRadius;
        float angle = diskSample.y * M_2PI;
        float3 bitangent = perp_stark(lightData.dirW);
        float3 tangent = cross(bitangent, lightData.dirW);

        pls.posW = sceneCenter - lightData.dirW * sceneRadius + tangent * r * cos(angle) + bitangent * r * sin(angle);
        pls.normal = -lightData.dirW;
        pls.flux = lightData.intensity * M_PI * sceneRadius * sceneRadius * lightCount;

        sampledDir = lightData.dirW;
    }
    else
    {
        return false;
    }

    return true;
}

[shader("miss")]
void miss(inout RayPayload rayData)
{
    HitInfo hitInfo = {};
    rayData.packedHitInfo = hitInfo.getData();
}

[shader("anyhit")]
void anyHit(inout RayPayload rayData, BuiltInTriangleIntersectionAttributes attribs)
{
    GeometryInstanceID instanceID = getGeometryInstanceID();
    VertexData v = getVertexData(instanceID, PrimitiveIndex(), attribs);
    uint materialID = gScene.getMaterialID(instanceID);
    if (gScene.materials.alphaTest(v, materialID, 0.f))
        IgnoreHit();
}

[shader("closesthit")]
void closestHit(inout RayPayload rayData, BuiltInTriangleIntersectionAttributes attribs)
{
    TriangleHit triangleHit;
    triangleHit.instanceID = getGeometryInstanceID();
    triangleHit.primitiveIndex = PrimitiveIndex();
    triangleHit.barycentrics = attribs.barycentrics;
    rayData.packedHitInfo = HitInfo(triangleHit).getData();
}

[shader("raygeneration")]
void rayGen()
{
    uint2 pixel = DispatchRaysIndex().xy;
    uint2 dispatchDims = DispatchRaysDimensions().xy;

    SampleGenerator sg = SampleGenerator(pixel, gFrameCount * 7 + 1);
    RayDesc ray;
    PhotonLightSample pls;

    // Sample photon from light source
#if USE_EMISSIVE_LIGHT
    // Mixed lights: randomly choose analytic or emissive light based on ratio
    bool useAnalytic = gUseAnalyticLights;
    if (!useAnalytic)
    {
        // Pure emissive light scene
        useAnalytic = false;
    }
    else if (gAnalyticLightRatio < 1.0)
    {
        // Mixed lights: random selection
        useAnalytic = (sampleNext1D(sg) < gAnalyticLightRatio);
    }

    if (useAnalytic)
#endif
    {
        float3 sampleDir;
        if (!createAnalyticLightSample(sg, pls, sampleDir))
            return;
        ray.Direction = sampleDir;
#if USE_EMISSIVE_LIGHT
        // If mixed lights, divide by selection probability
        if (gAnalyticLightRatio < 1.0)
            pls.flux /= gAnalyticLightRatio;
#endif
    }
#if USE_EMISSIVE_LIGHT
    else
    {
        if (!createEmissiveLightSample(sg, pls))
            return;

        float lightDirPDF = 1.0;
        float3 woLocal = sample_cosine_hemisphere_concentric(sampleNext2D(sg), lightDirPDF);
        if (lightDirPDF <= 0.f)
            return;

        ray.Direction = fromLocalToWorld(pls.normal, woLocal);
        // For Lambertian emitter with cosine-weighted hemisphere sampling:
        // weight = cos(theta) / pdf = cos(theta) / (cos(theta)/pi) = pi
        pls.flux *= M_PI;

        // If mixed lights, divide by selection probability
        if (gAnalyticLightRatio > 0.0)
            pls.flux /= (1.0 - gAnalyticLightRatio);
    }
#endif

    // Divide by total number of emitted photons
    pls.flux /= (dispatchDims.x * dispatchDims.y);

    ray.Origin = pls.posW;
    ray.TMin = 0.0;
    ray.TMax = FLT_MAX;
    RayPayload rayPayload = RayPayload();

    float3 thp = float3(1.0);
    float3 photonFlux = pls.flux;

    // Photon tracing main loop
    for (uint i = 0; i < gMaxBounces; i++)
    {
        // Early exit if photon buffer is already full
        if (gPhotonCounter[0] >= kPhotonBufferSize)
            return;

        TraceRay(gScene.rtAccel, 0, 0xff, 0, rayTypeCount, 0, ray, rayPayload);

        HitInfo hit = HitInfo(rayPayload.packedHitInfo);
        if (!hit.isValid())
            return;

        let lod = ExplicitLodTextureSampler(0);
        const TriangleHit triangleHit = hit.getTriangleHit();
        VertexData v = gScene.getVertexData(triangleHit);
        uint materialID = gScene.getMaterialID(triangleHit.instanceID);
        ShadingData sd = gScene.materials.prepareShadingData(v, materialID, -ray.Direction, lod);
        let mi = gScene.materials.getMaterialInstance(sd, lod);
        let bsdfProperties = mi.getProperties(sd);

        bool isDiffuse = bsdfProperties.roughness > kRoughnessThreshold;

        // Store photon at diffuse surface
        // Store all photons reaching diffuse surfaces (including direct illumination photons)
        // NEE in collect pass has been removed, all lighting is handled by photon density estimation
        if (isDiffuse)
        {
            uint photonIdx = 0;
            InterlockedAdd(gPhotonCounter[0], 1u, photonIdx);
            if (photonIdx < kPhotonBufferSize)
            {
                // Compute hash grid cell
                int3 cell = worldToCell(sd.posW, gCellSize);
                uint cellIdx = hashGridCellIndex(cell);

                // Insert into hash grid linked list (atomic exchange)
                uint prevHead;
                InterlockedExchange(gHashGrid[cellIdx], photonIdx, prevHead);

                PPMPhoton photon;
                photon.posW = sd.posW;
                photon.flux = photonFlux * thp;
                photon.dirW = -ray.Direction;
                photon.nextInCell = prevHead;

                gPhotonBuffer[photonIdx] = photon;
            }
        }

        // Sample next bounce direction
        BSDFSample bsdfSample;
        if (!mi.sample(sd, sg, bsdfSample, true))
            return;

        bool transmission = bsdfSample.isLobe(LobeType::Transmission);
        ray.Origin = sd.computeNewRayOrigin(!transmission);
        ray.Direction = bsdfSample.wo;
        thp *= bsdfSample.weight;

        // Russian Roulette (only apply after a few bounces)
        if (i >= 3)
        {
            float rrVal = min((thp.x + thp.y + thp.z) / 3.0, 0.95f);
            if (sampleNext1D(sg) >= rrVal)
                return;
            thp /= rrVal;
        }
    }
}
