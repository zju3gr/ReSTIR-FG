#include "ProgressivePhotonMapping.h"
#include "RenderGraph/RenderPassHelpers.h"
#include "RenderGraph/RenderPassStandardFlags.h"
#include "Rendering/Lights/EmissivePowerSampler.h"

extern "C" FALCOR_API_EXPORT void registerPlugin(Falcor::PluginRegistry& registry)
{
    registry.registerClass<RenderPass, ProgressivePhotonMapping>();
}

namespace
{
    // Shader file paths
    const std::string kShaderFolder = "RenderPasses/ProgressivePhotonMapping/";
    const std::string kShaderTracePhotons = kShaderFolder + "PPMTracePhotons.rt.slang";
    const std::string kShaderCollectPhotons = kShaderFolder + "PPMCollectPhotons.cs.slang";

    const std::string kShaderModel = "6_5";

    // Input channels
    const std::string kInputVBuffer = "vbuffer";
    const std::string kInputViewDir = "viewW";

    const ChannelList kInputChannels = {
        {kInputVBuffer, "gVBuffer", "Visibility buffer in packed format"},
        {kInputViewDir, "gViewW", "World-space view direction (xyz float format)", true /* optional */},
    };

    // Output channels
    const ChannelList kOutputChannels = {
        {"color", "gOutputColor", "Output color (progressive photon mapping result)", false, ResourceFormat::RGBA32Float},
    };

    // Property names
    const char kMaxCameraBounces[] = "maxCameraBounces";
    const char kPhotonMaxBounces[] = "photonMaxBounces";
    const char kNumPhotonsPerFrame[] = "numPhotonsPerFrame";
    const char kInitialRadius[] = "initialRadius";
    const char kAlpha[] = "alpha";
    const char kUseProbabilisticPPM[] = "useProbabilisticPPM";

    // Ray tracing configuration
    const uint32_t kMaxPayloadSizeBytes = 64u;
    const uint32_t kMaxRecursionDepth = 2u;
}

ProgressivePhotonMapping::ProgressivePhotonMapping(ref<Device> pDevice, const Properties& props)
    : RenderPass(pDevice)
{
    parseProperties(props);
    mpSampleGenerator = SampleGenerator::create(mpDevice, SAMPLE_GENERATOR_UNIFORM);
    FALCOR_ASSERT(mpSampleGenerator);
}

void ProgressivePhotonMapping::parseProperties(const Properties& props)
{
    for (const auto& [key, value] : props)
    {
        if (key == kMaxCameraBounces) mMaxCameraBounces = value;
        else if (key == kPhotonMaxBounces) mPhotonMaxBounces = value;
        else if (key == kNumPhotonsPerFrame) mNumPhotonsPerFrame = value;
        else if (key == kInitialRadius) mInitialRadius = value;
        else if (key == kAlpha) mAlpha = value;
        else if (key == kUseProbabilisticPPM) mUseProbabilisticPPM = value;
        else logWarning("Unknown property '{}' in ProgressivePhotonMapping properties.", key);
    }
}

Properties ProgressivePhotonMapping::getProperties() const
{
    Properties props;
    props[kMaxCameraBounces] = mMaxCameraBounces;
    props[kPhotonMaxBounces] = mPhotonMaxBounces;
    props[kNumPhotonsPerFrame] = mNumPhotonsPerFrame;
    props[kInitialRadius] = mInitialRadius;
    props[kAlpha] = mAlpha;
    props[kUseProbabilisticPPM] = mUseProbabilisticPPM;
    return props;
}

RenderPassReflection ProgressivePhotonMapping::reflect(const CompileData& compileData)
{
    RenderPassReflection reflector;
    addRenderPassInputs(reflector, kInputChannels);
    addRenderPassOutputs(reflector, kOutputChannels);
    return reflector;
}

void ProgressivePhotonMapping::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    // Handle options change
    auto& dict = renderData.getDictionary();
    if (mOptionsChanged)
    {
        auto flags = dict.getValue(kRenderPassRefreshFlags, RenderPassRefreshFlags::None);
        dict[Falcor::kRenderPassRefreshFlags] = flags | Falcor::RenderPassRefreshFlags::RenderOptionsChanged;
        mOptionsChanged = false;
        mResetAccumulation = true;
    }

    // Detect camera movement or scene changes and reset accumulation
    if (mpScene)
    {
        auto sceneUpdates = mpScene->getUpdates();
        if (is_set(sceneUpdates, Scene::UpdateFlags::CameraMoved) ||
            is_set(sceneUpdates, Scene::UpdateFlags::GeometryChanged) ||
            is_set(sceneUpdates, Scene::UpdateFlags::MaterialsChanged) ||
            is_set(sceneUpdates, Scene::UpdateFlags::LightsMoved) ||
            is_set(sceneUpdates, Scene::UpdateFlags::RenderSettingsChanged))
        {
            mResetAccumulation = true;
        }
    }

    // Clear output when no scene is loaded
    if (!mpScene)
    {
        for (auto it : kOutputChannels)
        {
            Texture* pDst = renderData.getTexture(it.name).get();
            if (pDst) pRenderContext->clearTexture(pDst);
        }
        return;
    }

    // Prepare lighting and resources
    prepareLighting(pRenderContext);
    if (!mHasLights)
    {
        for (auto it : kOutputChannels)
        {
            Texture* pDst = renderData.getTexture(it.name).get();
            if (pDst) pRenderContext->clearTexture(pDst);
        }
        return;
    }
    prepareResources(pRenderContext, renderData);

    // Reset accumulation (after prepareResources, since we need to know buffer size)
    // Make sure the buffer is re-created
    if (mResetAccumulation)
    {
        mFrameCount = 0;
        // Clear per-pixel statistics buffer (reset N, radius2, tau, emissionSum)
        // Note: StructuredBuffer cannot use clearUAV, so we re-create the buffer to zero it
        uint32_t pixelCount = mScreenRes.x * mScreenRes.y;
        mpPixelStatsBuffer = Buffer::createStructured(
            mpDevice, 48u, pixelCount,
            ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
            Buffer::CpuAccess::None, nullptr, false
        );
        mpPixelStatsBuffer->setName("PPMPixelStats");
        mResetAccumulation = false;
    }

    // Request light collection
    if (mpScene->getRenderSettings().useEmissiveLights)
    {
        mpScene->getLightCollection(pRenderContext);
    }

    // Clear photon counter and hash grid
    pRenderContext->clearUAV(mpPhotonCounter->getUAV().get(), uint4(0));
    pRenderContext->clearUAV(mpHashGrid->getUAV().get(), uint4(0xFFFFFFFF));

    // Step 1: Trace camera paths, record hit points
    // In PPM, camera paths are re-traced every frame (photons are progressively accumulated)
    traceCameraPass(pRenderContext, renderData);

    // Step 2: Emit photons from light sources
    tracePhotonsPass(pRenderContext, renderData);

    // Step 3: Collect photons for density estimation, accumulate results
    // Standard SPPM: per-pixel radius reduction is done in the shader
    collectPhotonsPass(pRenderContext, renderData);

    mFrameCount++;
}

void ProgressivePhotonMapping::renderUI(Gui::Widgets& widget)
{
    bool dirty = false;

    if (auto group = widget.group("Camera Path"))
    {
        dirty |= group.var("Max Camera Bounces", mMaxCameraBounces, 0u, 32u);
        group.tooltip("Max bounces for camera path before reaching a diffuse surface");
        dirty |= group.var("Specular Roughness Threshold", mSpecularRoughnessThreshold, 0.f, 1.f, 0.01f);
        group.tooltip("Surfaces below this roughness are treated as specular");
    }

    if (auto group = widget.group("Photon Tracing"))
    {
        dirty |= group.var("Photons Per Frame", mNumPhotonsPerFrame, 1024u, 10000000u);
        group.tooltip("Number of photons emitted per frame");
        dirty |= group.var("Max Photon Bounces", mPhotonMaxBounces, 1u, 32u);
        group.tooltip("Max bounces for photon tracing");
        dirty |= group.var("Max Photon Buffer", mMaxPhotonBufferSize, 10000u, 20000000u);
        group.tooltip("Max capacity of the photon buffer");
        if (mMixedLights)
        {
            dirty |= group.var("Analytic Light Ratio", mPhotonAnalyticRatio, 0.f, 1.f, 0.01f);
            group.tooltip("Ratio of analytic lights when using mixed light sources");
        }
    }

    if (auto group = widget.group("PPM Parameters"))
    {
        dirty |= group.checkbox("Probabilistic PPM (Knaus 2011)", mUseProbabilisticPPM);
        group.tooltip("Toggle between Probabilistic PPM (Knaus & Zwicker 2011) and Standard SPPM (Hachisuka 2009).\n"
                      "Probabilistic: per-frame independent estimation + running average.\n"
                      "Standard SPPM: tau accumulation with per-pixel radius reduction.");
        bool radiusDirty = group.var("Initial Radius", mInitialRadius, 0.001f, 10.f, 0.001f, false, "%.4f");
        group.tooltip("Initial search radius for photon gathering");
        dirty |= radiusDirty;
        dirty |= group.var("Alpha", mAlpha, 0.1f, 1.f, 0.01f);
        if (mUseProbabilisticPPM)
            group.tooltip("Radius reduction rate (Knaus 2011). r^2_{i+1} = r^2_i * (i+alpha)/(i+1). Smaller = faster convergence.");
        else
            group.tooltip("SPPM radius reduction parameter. Nnew = N + alpha*M. Larger = slower reduction.");
        group.text("Frame Count: " + std::to_string(mFrameCount));
        group.text(std::string("Mode: ") + (mUseProbabilisticPPM ? "Probabilistic PPM" : "Standard SPPM"));
    }

    if (widget.button("Reset Accumulation"))
    {
        mResetAccumulation = true;
    }

    if (dirty)
    {
        mOptionsChanged = true;
    }
}

void ProgressivePhotonMapping::setScene(RenderContext* pRenderContext, const ref<Scene>& pScene)
{
    // Clear old scene data
    mTraceCameraProgram = RayTraceProgramHelper::create();
    mTracePhotonProgram = RayTraceProgramHelper::create();
    mpCollectPhotonsPass.reset();
    mpEmissiveLightSampler.reset();
    mpPixelStatsBuffer.reset();
    mpPhotonBuffer.reset();
    mpPhotonCounter.reset();
    mpHashGrid.reset();
    mFrameCount = 0;

    mpScene = pScene;

    if (mpScene)
    {
        if (mpScene->hasGeometryType(Scene::GeometryType::Custom))
        {
            logWarning("ProgressivePhotonMapping: This render pass only supports triangles.");
        }
    }
}

void ProgressivePhotonMapping::prepareLighting(RenderContext* pRenderContext)
{
    auto& pLights = mpScene->getLightCollection(pRenderContext);

    bool emissiveUsed = mpScene->useEmissiveLights();
    bool analyticUsed = mpScene->useAnalyticLights();

    mHasLights = analyticUsed || emissiveUsed;
    mHasAnalyticLights = analyticUsed;
    mMixedLights = emissiveUsed && analyticUsed;

    if (emissiveUsed)
    {
        if (!mpEmissiveLightSampler)
        {
            FALCOR_ASSERT(pLights && pLights->getActiveLightCount(pRenderContext) > 0);
            mpEmissiveLightSampler = std::make_unique<EmissivePowerSampler>(pRenderContext, mpScene);
        }
        mpEmissiveLightSampler->update(pRenderContext);
    }
    else
    {
        if (mpEmissiveLightSampler)
        {
            mpEmissiveLightSampler.reset();
            mTracePhotonProgram.pVars.reset();
        }
    }
}

void ProgressivePhotonMapping::prepareResources(RenderContext* pRenderContext, const RenderData& renderData)
{
    auto& screenDims = renderData.getDefaultTextureDims();
    bool screenChanged = (screenDims.x != mScreenRes.x || screenDims.y != mScreenRes.y);
    if (screenChanged)
    {
        mScreenRes = screenDims;
        mResetAccumulation = true;
        mpPixelStatsBuffer.reset();
    }

    uint32_t pixelCount = mScreenRes.x * mScreenRes.y;

    // Per-pixel statistics buffer (PPMPixelStats)
    // PPMPixelStats: radius2(1)+N(1)+pad(2) + tau(3)+pad(1) + emissionSum(3)+pad(1) = 12 floats = 48 bytes
    if (!mpPixelStatsBuffer)
    {
        mpPixelStatsBuffer = Buffer::createStructured(
            mpDevice, 48u, pixelCount,
            ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
            Buffer::CpuAccess::None, nullptr, false
        );
        mpPixelStatsBuffer->setName("PPMPixelStats");
    }

    // Photon buffer
    if (!mpPhotonBuffer)
    {
        // PPMPhoton: posW(3)+pad(1)+flux(3)+pad(1)+dirW(3)+nextInCell(1) = 12 * 4 = 48 bytes
        mpPhotonBuffer = Buffer::createStructured(
            mpDevice, 48u, mMaxPhotonBufferSize,
            ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
            Buffer::CpuAccess::None, nullptr, false
        );
        mpPhotonBuffer->setName("PPMPhotons");
    }

    // Photon counter (typed buffer, not structured, so clearUAV works)
    if (!mpPhotonCounter)
    {
        mpPhotonCounter = Buffer::createTyped(
            mpDevice, ResourceFormat::R32Uint, 1,
            ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
            Buffer::CpuAccess::None, nullptr
        );
        mpPhotonCounter->setName("PPMPhotonCounter");
    }

    // Hash grid buffer for spatial photon lookup (must match HASH_GRID_SIZE in PPMParams.slang)
    static const uint32_t kHashGridSize = 2000000;
    if (!mpHashGrid)
    {
        mpHashGrid = Buffer::createTyped(
            mpDevice, ResourceFormat::R32Uint, kHashGridSize,
            ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
            Buffer::CpuAccess::None, nullptr
        );
        mpHashGrid->setName("PPMHashGrid");
    }
}

void ProgressivePhotonMapping::traceCameraPass(RenderContext* pRenderContext, const RenderData& renderData)
{
    FALCOR_PROFILE(pRenderContext, "PPM Camera Pass");

    // Camera pass reuses the photon tracing RT shader framework,
    // but actually only needs to trace from VBuffer to diffuse surface.
    // Here we handle camera path in the collectPhotons compute shader
    // since VBuffer already provides the first hit, we just continue tracing from there.

    // For simple PPM, we read the first hit from VBuffer in the collect pass.
    // If it's a specular surface, continue tracing until a diffuse surface is found.
    // This logic is implemented in PPMCollectPhotons.cs.slang.
}

void ProgressivePhotonMapping::tracePhotonsPass(RenderContext* pRenderContext, const RenderData& renderData)
{
    FALCOR_PROFILE(pRenderContext, "PPM Trace Photons");

    // Initialize shader
    if (!mTracePhotonProgram.pProgram)
    {
        RtProgram::Desc desc;
        desc.addShaderModules(mpScene->getShaderModules());
        desc.addShaderLibrary(kShaderTracePhotons);
        desc.setMaxPayloadSize(kMaxPayloadSizeBytes);
        desc.setMaxAttributeSize(mpScene->getRaytracingMaxAttributeSize());
        desc.setMaxTraceRecursionDepth(1);
        if (!mpScene->hasProceduralGeometry())
            desc.setPipelineFlags(RtPipelineFlags::SkipProceduralPrimitives);

        mTracePhotonProgram.pBindingTable = RtBindingTable::create(1, 1, mpScene->getGeometryCount());
        auto& sbt = mTracePhotonProgram.pBindingTable;
        sbt->setRayGen(desc.addRayGen("rayGen", mpScene->getTypeConformances()));
        sbt->setMiss(0, desc.addMiss("miss"));

        if (mpScene->hasGeometryType(Scene::GeometryType::TriangleMesh))
        {
            sbt->setHitGroup(
                0, mpScene->getGeometryIDs(Scene::GeometryType::TriangleMesh),
                desc.addHitGroup("closestHit", "anyHit")
            );
        }

        DefineList defines;
        defines.add("USE_EMISSIVE_LIGHT", mpScene->useEmissiveLights() ? "1" : "0");
        defines.add(mpScene->getSceneDefines());

        mTracePhotonProgram.pProgram = RtProgram::create(mpDevice, desc, defines);
    }

    // Runtime defines
    mTracePhotonProgram.pProgram->addDefine("PHOTON_BUFFER_SIZE", std::to_string(mMaxPhotonBufferSize));
    mTracePhotonProgram.pProgram->addDefine("ROUGHNESS_THRESHOLD", std::to_string(mSpecularRoughnessThreshold));
    if (mpEmissiveLightSampler)
        mTracePhotonProgram.pProgram->addDefines(mpEmissiveLightSampler->getDefines());

    // Program variables
    if (!mTracePhotonProgram.pVars)
        mTracePhotonProgram.initProgramVars(mpDevice, mpScene, mpSampleGenerator);

    FALCOR_ASSERT(mTracePhotonProgram.pVars);
    auto var = mTracePhotonProgram.pVars->getRootVar();
    mpScene->setRaytracingShaderData(pRenderContext, var);

    // Compute dispatch dimensions
    uint dispatchDim = static_cast<uint>(std::floor(std::sqrt(float(mNumPhotonsPerFrame))));
    dispatchDim = std::max(32u, dispatchDim);

    // Constant buffer
    var["CB"]["gFrameCount"] = mFrameCount;
    var["CB"]["gMaxBounces"] = mPhotonMaxBounces;
    var["CB"]["gUseAnalyticLights"] = mHasAnalyticLights;
    var["CB"]["gCellSize"] = mInitialRadius;  // Hash grid cell size = initial radius (fixed)
    var["CB"]["gAnalyticLightRatio"] = mMixedLights ? mPhotonAnalyticRatio : (mHasAnalyticLights ? 1.0f : 0.0f);

    // Light sampler
    if (mpEmissiveLightSampler)
        mpEmissiveLightSampler->setShaderData(var["Light"]["gEmissiveSampler"]);

    // Output buffers
    var["gPhotonBuffer"] = mpPhotonBuffer;
    var["gPhotonCounter"] = mpPhotonCounter;
    var["gHashGrid"] = mpHashGrid;

    // Emit photons
    mpScene->raytrace(pRenderContext, mTracePhotonProgram.pProgram.get(), mTracePhotonProgram.pVars, uint3(dispatchDim, dispatchDim, 1));

    // UAV barrier
    pRenderContext->uavBarrier(mpPhotonBuffer.get());
    pRenderContext->uavBarrier(mpPhotonCounter.get());
    pRenderContext->uavBarrier(mpHashGrid.get());
}

void ProgressivePhotonMapping::collectPhotonsPass(RenderContext* pRenderContext, const RenderData& renderData)
{
    FALCOR_PROFILE(pRenderContext, "PPM Collect Photons");

    // Initialize compute pass
    if (!mpCollectPhotonsPass)
    {
        Program::Desc desc;
        desc.addShaderModules(mpScene->getShaderModules());
        desc.addShaderLibrary(kShaderCollectPhotons).csEntry("main").setShaderModel(kShaderModel);
        desc.addTypeConformances(mpScene->getTypeConformances());

        DefineList defines;
        defines.add(mpScene->getSceneDefines());
        defines.add(mpSampleGenerator->getDefines());
        defines.add("USE_ENV_LIGHT", mpScene->useEnvLight() ? "1" : "0");
        defines.add("USE_ENV_BACKGROUND", mpScene->useEnvBackground() ? "1" : "0");
        defines.add("USE_ANALYTIC_LIGHTS", mpScene->useAnalyticLights() ? "1" : "0");
        defines.add("USE_EMISSIVE_LIGHTS", mpScene->useEmissiveLights() ? "1" : "0");
        defines.add("PHOTON_BUFFER_SIZE", std::to_string(mMaxPhotonBufferSize));
        defines.add("ROUGHNESS_THRESHOLD", std::to_string(mSpecularRoughnessThreshold));
        defines.add("MAX_CAMERA_BOUNCES", std::to_string(mMaxCameraBounces));
        defines.add("USE_PROBABILISTIC_PPM", mUseProbabilisticPPM ? "1" : "0");
        if (mpEmissiveLightSampler)
            defines.add(mpEmissiveLightSampler->getDefines());
        defines.add(getValidResourceDefines(kInputChannels, renderData));
        defines.add(getValidResourceDefines(kOutputChannels, renderData));

        mpCollectPhotonsPass = ComputePass::create(mpDevice, desc, defines, true);
    }
    FALCOR_ASSERT(mpCollectPhotonsPass);

    // Runtime defines (update if parameters changed)
    mpCollectPhotonsPass->getProgram()->addDefine("PHOTON_BUFFER_SIZE", std::to_string(mMaxPhotonBufferSize));
    mpCollectPhotonsPass->getProgram()->addDefine("ROUGHNESS_THRESHOLD", std::to_string(mSpecularRoughnessThreshold));
    mpCollectPhotonsPass->getProgram()->addDefine("MAX_CAMERA_BOUNCES", std::to_string(mMaxCameraBounces));
    mpCollectPhotonsPass->getProgram()->addDefine("USE_PROBABILISTIC_PPM", mUseProbabilisticPPM ? "1" : "0");
    if (mpEmissiveLightSampler)
        mpCollectPhotonsPass->getProgram()->addDefines(mpEmissiveLightSampler->getDefines());

    // Handle optional I/O resource defines
    mpCollectPhotonsPass->getProgram()->addDefines(getValidResourceDefines(kInputChannels, renderData));
    mpCollectPhotonsPass->getProgram()->addDefines(getValidResourceDefines(kOutputChannels, renderData));

    // Set variables
    auto var = mpCollectPhotonsPass->getRootVar();
    mpScene->setRaytracingShaderData(pRenderContext, var);
    mpSampleGenerator->setShaderData(var);

    // Constant buffer
    var["CB"]["gFrameCount"] = mFrameCount;
    var["CB"]["gFrameDim"] = mScreenRes;
    var["CB"]["gInitialRadius"] = mInitialRadius;
    var["CB"]["gNumPhotonsPerFrame"] = mNumPhotonsPerFrame;
    var["CB"]["gAlpha"] = mAlpha;
    var["CB"]["gCellSize"] = mInitialRadius;  // Hash grid cell size = initial radius

    // Emissive light sampler
    if (mpEmissiveLightSampler)
        mpEmissiveLightSampler->setShaderData(var["EmissiveLight"]["gEmissiveSampler"]);

    // Input
    var["gVBuffer"] = renderData[kInputVBuffer]->asTexture();
    auto pViewW = renderData[kInputViewDir];
    if (pViewW) var["gViewW"] = pViewW->asTexture();

    var["gPhotonBuffer"] = mpPhotonBuffer;
    var["gPhotonCounter"] = mpPhotonCounter;
    var["gHashGrid"] = mpHashGrid;

    // Input/Output: per-pixel statistics
    var["gPixelStats"] = mpPixelStatsBuffer;

    // Output
    var["gOutputColor"] = renderData.getTexture("color");

    // Execute
    const uint2 targetDim = renderData.getDefaultTextureDims();
    FALCOR_ASSERT(targetDim.x > 0 && targetDim.y > 0);
    mpCollectPhotonsPass->execute(pRenderContext, uint3(targetDim, 1));
}

DefineList ProgressivePhotonMapping::getMaterialDefines()
{
    DefineList defines;
    defines.add("DiffuseBrdf", "DiffuseBrdfLambert");
    return defines;
}

void ProgressivePhotonMapping::RayTraceProgramHelper::initProgramVars(
    ref<Device> pDevice, ref<Scene> pScene, ref<SampleGenerator> pSampleGenerator)
{
    FALCOR_ASSERT(pProgram);
    pProgram->addDefines(pSampleGenerator->getDefines());
    pProgram->setTypeConformances(pScene->getTypeConformances());
    pVars = RtProgramVars::create(pDevice, pProgram, pBindingTable);
    auto var = pVars->getRootVar();
    pSampleGenerator->setShaderData(var);
}
