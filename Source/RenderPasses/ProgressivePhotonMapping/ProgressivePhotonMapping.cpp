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
    const std::string kShaderScatterPhotons = kShaderFolder + "PPMScatterPhotons.cs.slang";
    const std::string kShaderCollectPhotons = kShaderFolder + "PPMCollectPhotons.cs.slang";
    const std::string kShaderReSTIRCollect = kShaderFolder + "PPMReSTIRCollect.cs.slang";

    const std::string kShaderModel = "6_5";

    // Hash grid size (must match HASH_GRID_SIZE in PPMParams.slang)
    static const uint32_t kHashGridSize = 2000000;

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
    const char kDisplayMode[] = "displayMode";

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
        else if (key == kDisplayMode) mDisplayMode = static_cast<PPMDisplayMode>((uint32_t)value);
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
    props[kDisplayMode] = static_cast<uint32_t>(mDisplayMode);
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
        mReservoirFrameIndex = 0;
        // Clear per-pixel statistics buffer (reset N, radius2, tau, emissionSum)
        // Note: StructuredBuffer cannot use clearUAV, so we re-create the buffer to zero it
        uint32_t pixelCount = mScreenRes.x * mScreenRes.y;
        mpPixelStatsBuffer = Buffer::createStructured(
            mpDevice, 48u, pixelCount,
            ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
            Buffer::CpuAccess::None, nullptr, false
        );
        mpPixelStatsBuffer->setName("PPMPixelStats");

        // Clear reservoir buffers if ReSTIR is enabled
        if (mUseReSTIR)
        {
            uint32_t reservoirSize = 64u;
            for (int i = 0; i < 2; i++)
            {
                mpReservoirBuffer[i] = Buffer::createStructured(
                    mpDevice, reservoirSize, pixelCount,
                    ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
                    Buffer::CpuAccess::None, nullptr, false
                );
                mpReservoirBuffer[i]->setName("PPMReservoir" + std::to_string(i));
            }
        }

        mResetAccumulation = false;
    }

    // Request light collection
    if (mpScene->getRenderSettings().useEmissiveLights)
    {
        mpScene->getLightCollection(pRenderContext);
    }

    // Clear photon counter and cell count buffer
    pRenderContext->clearUAV(mpPhotonCounter->getUAV().get(), uint4(0));
    pRenderContext->clearUAV(mpCellCount->getUAV().get(), uint4(0));

    // Step 1: Trace camera paths, record hit points
    // In PPM, camera paths are re-traced every frame (photons are progressively accumulated)
    traceCameraPass(pRenderContext, renderData);

    // Step 2: Emit photons from light sources (writes to unsorted buffer + cell counts)
    tracePhotonsPass(pRenderContext, renderData);

    // Step 3: Sort photons into contiguous memory by hash cell
    sortPhotonsPass(pRenderContext);

    // Step 4: Collect photons for density estimation, accumulate results
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
        // Display mode dropdown
        {
            Gui::DropdownList displayModeList;
            displayModeList.push_back({(uint32_t)PPMDisplayMode::Full, "Full (Direct + Indirect)"});
            displayModeList.push_back({(uint32_t)PPMDisplayMode::DirectOnly, "Direct Only"});
            displayModeList.push_back({(uint32_t)PPMDisplayMode::IndirectOnly, "Indirect Only"});
            displayModeList.push_back({(uint32_t)PPMDisplayMode::PhotonDirectTest, "Photon Direct Test (Sanity)"});
            uint32_t mode = (uint32_t)mDisplayMode;
            if (group.dropdown("Display Mode", displayModeList, mode))
            {
                mDisplayMode = (PPMDisplayMode)mode;
                dirty = true;
            }
            group.tooltip("Select which lighting component to display.\n"
                          "Full: Direct + Indirect (default)\n"
                          "Direct Only: Only NEE direct lighting\n"
                          "Indirect Only: Only photon density estimation (indirect)\n"
                          "Photon Direct Test: Sanity test - use photon map for direct lighting (no NEE)");
        }

        dirty |= group.checkbox("Probabilistic PPM (Knaus 2011)", mUseProbabilisticPPM);
        group.tooltip("Toggle between Probabilistic PPM (Knaus & Zwicker 2011) and Standard SPPM (Hachisuka 2009).\n"
                      "Probabilistic: Memoryless - global radius schedule, no per-pixel statistics.\n"
                      "Standard SPPM: Per-pixel radius reduction with local photon count tracking.");
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
        group.text("Photons Stored: " + std::to_string(mLastPhotonCount) + " / " + std::to_string(mMaxPhotonBufferSize));
        if (mLastPhotonCount >= mMaxPhotonBufferSize)
            group.text("WARNING: Photon buffer OVERFLOW! Increase buffer size or reduce bounces.");
    }

    if (auto group = widget.group("ReSTIR Optimization"))
    {
        dirty |= group.checkbox("Use ReSTIR Collect", mUseReSTIR);
        group.tooltip("Enable ReSTIR-based photon collection.\n"
                      "Uses Reservoir Importance Sampling to select high-quality photon samples,\n"
                      "with temporal and spatial resampling to reduce variance.\n"
                      "Allows using fewer photons per frame while maintaining quality.");

        if (mUseReSTIR)
        {
            dirty |= group.var("Temporal Max M", mTemporalMaxM, 0u, 100u);
            group.tooltip("Maximum M value for temporal resampling.\n"
                          "Higher = more history reuse = less noise but more temporal lag.\n"
                          "Set to 0 to disable temporal resampling.");
            dirty |= group.var("Spatial Neighbors", mSpatialNeighbors, 0u, 10u);
            group.tooltip("Number of spatial neighbors to resample from.\n"
                          "Higher = less noise but more computation.\n"
                          "Set to 0 to disable spatial resampling.");
            dirty |= group.var("Spatial Radius (px)", mSpatialRadius, 1.0f, 50.0f, 1.0f);
            group.tooltip("Spatial resampling search radius in pixels.");
        }
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
    mpScatterPhotonsPass.reset();
    mpCollectPhotonsPass.reset();
    mpReSTIRCollectPass.reset();
    mpEmissiveLightSampler.reset();
    mpPixelStatsBuffer.reset();
    mpPhotonBuffer.reset();
    mpPhotonCounter.reset();
    mpPhotonCounterStaging.reset();
    mpCellCount.reset();
    mpCellCounter.reset();
    mpSortedPhotonBuffer.reset();
    mpPrefixSum.reset();
    mpReservoirBuffer[0].reset();
    mpReservoirBuffer[1].reset();
    mpPositionBuffer[0].reset();
    mpPositionBuffer[1].reset();
    mpNormalBuffer[0].reset();
    mpNormalBuffer[1].reset();
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
        // Reset ReSTIR textures on screen resize
        for (int i = 0; i < 2; i++)
        {
            mpPositionBuffer[i].reset();
            mpNormalBuffer[i].reset();
            mpReservoirBuffer[i].reset();
        }
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

    // Unsorted photon buffer (trace pass 写入)
    // PPMPhoton: posW(3)+cellIndex(1)+flux(3)+pad(1)+dirW(3)+pad(1) = 12 * 4 = 48 bytes
    if (!mpPhotonBuffer || mpPhotonBuffer->getElementCount() != mMaxPhotonBufferSize)
    {
        mpPhotonBuffer = Buffer::createStructured(
            mpDevice, 48u, mMaxPhotonBufferSize,
            ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
            Buffer::CpuAccess::None, nullptr, false
        );
        mpPhotonBuffer->setName("PPMPhotonsUnsorted");
    }

    // Sorted photon buffer (按 cell 连续存储，collect pass 读取)
    // PPMPhotonSorted: posW(3)+pad(1)+flux(3)+pad(1)+dirW(3)+pad(1) = 12 * 4 = 48 bytes
    if (!mpSortedPhotonBuffer || mpSortedPhotonBuffer->getElementCount() != mMaxPhotonBufferSize)
    {
        mpSortedPhotonBuffer = Buffer::createStructured(
            mpDevice, 48u, mMaxPhotonBufferSize,
            ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
            Buffer::CpuAccess::None, nullptr, false
        );
        mpSortedPhotonBuffer->setName("PPMPhotonsSorted");
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

    // Staging buffer for reading back photon count (for overflow detection)
    if (!mpPhotonCounterStaging)
    {
        mpPhotonCounterStaging = Buffer::createTyped(
            mpDevice, ResourceFormat::R32Uint, 1,
            ResourceBindFlags::None,
            Buffer::CpuAccess::Read, nullptr
        );
        mpPhotonCounterStaging->setName("PPMPhotonCounterStaging");
    }

    // Cell count / offset buffer: 每个 hash cell 中的光子计数
    // trace pass 写入 count，然后 in-place prefix sum 变成 offset table
    // 使用 raw buffer 以兼容 PrefixSum 工具类
    if (!mpCellCount)
    {
        mpCellCount = Buffer::create(
            mpDevice, kHashGridSize * sizeof(uint32_t),
            ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
            Buffer::CpuAccess::None, nullptr
        );
        mpCellCount->setName("PPMCellCount");
    }

    // mpCellOffset is no longer needed (we do in-place prefix sum on mpCellCount)

    // Cell counter buffer: scatter pass 中的 per-cell atomic counter
    if (!mpCellCounter)
    {
        mpCellCounter = Buffer::createTyped(
            mpDevice, ResourceFormat::R32Uint, kHashGridSize,
            ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
            Buffer::CpuAccess::None, nullptr
        );
        mpCellCounter->setName("PPMCellCounter");
    }

    // PrefixSum utility
    if (!mpPrefixSum)
    {
        mpPrefixSum = std::make_unique<PrefixSum>(mpDevice);
    }

    // ReSTIR resources (only allocate when ReSTIR is enabled)
    if (mUseReSTIR)
    {
        // PhotonReservoir: 16 floats = 64 bytes
        // photonPosW(3) + wSum(1) + photonFlux(3) + M(1) + photonDirW(3) + targetPdf(1) + F(3) + pad(1) = 16 * 4 = 64 bytes
        uint32_t reservoirSize = 64u;
        for (int i = 0; i < 2; i++)
        {
            if (!mpReservoirBuffer[i] || mpReservoirBuffer[i]->getElementCount() != pixelCount)
            {
                mpReservoirBuffer[i] = Buffer::createStructured(
                    mpDevice, reservoirSize, pixelCount,
                    ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
                    Buffer::CpuAccess::None, nullptr, false
                );
                mpReservoirBuffer[i]->setName("PPMReservoir" + std::to_string(i));
            }

            if (!mpPositionBuffer[i])
            {
                mpPositionBuffer[i] = Texture::create2D(
                    mpDevice, mScreenRes.x, mScreenRes.y,
                    ResourceFormat::RGBA32Float, 1, 1,
                    nullptr, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess
                );
                mpPositionBuffer[i]->setName("PPMPosition" + std::to_string(i));
            }

            if (!mpNormalBuffer[i])
            {
                mpNormalBuffer[i] = Texture::create2D(
                    mpDevice, mScreenRes.x, mScreenRes.y,
                    ResourceFormat::RGBA32Float, 1, 1,
                    nullptr, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess
                );
                mpNormalBuffer[i]->setName("PPMNormal" + std::to_string(i));
            }
        }
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
    var["CB"]["gDisplayMode"] = static_cast<uint>(mDisplayMode);

    // Light sampler
    if (mpEmissiveLightSampler)
        mpEmissiveLightSampler->setShaderData(var["Light"]["gEmissiveSampler"]);

    // Output buffers
    var["gPhotonBuffer"] = mpPhotonBuffer;
    var["gPhotonCounter"] = mpPhotonCounter;
    var["gCellCount"] = mpCellCount;

    // Emit photons
    mpScene->raytrace(pRenderContext, mTracePhotonProgram.pProgram.get(), mTracePhotonProgram.pVars, uint3(dispatchDim, dispatchDim, 1));

    // UAV barrier
    pRenderContext->uavBarrier(mpPhotonBuffer.get());
    pRenderContext->uavBarrier(mpPhotonCounter.get());
    pRenderContext->uavBarrier(mpCellCount.get());

    // Read back photon count for overflow detection
    pRenderContext->copyResource(mpPhotonCounterStaging.get(), mpPhotonCounter.get());
    const uint32_t* pData = reinterpret_cast<const uint32_t*>(mpPhotonCounterStaging->map(Buffer::MapType::Read));
    if (pData)
    {
        mLastPhotonCount = *pData;
        mpPhotonCounterStaging->unmap();
    }
}

void ProgressivePhotonMapping::sortPhotonsPass(RenderContext* pRenderContext)
{
    FALCOR_PROFILE(pRenderContext, "PPM Sort Photons");

    if (mLastPhotonCount == 0)
        return;

    // Step 1: In-place prefix sum on mpCellCount
    // After this, mpCellCount[i] = sum of original counts[0..i-1] (exclusive scan)
    // mpCellCount is a raw buffer, compatible with PrefixSum
    mpPrefixSum->execute(pRenderContext, mpCellCount, kHashGridSize);

    // Step 2: Clear cell counter (used as per-cell atomic counter in scatter pass)
    pRenderContext->clearUAV(mpCellCounter->getUAV().get(), uint4(0));

    // Step 3: Scatter photons into sorted buffer
    if (!mpScatterPhotonsPass)
    {
        Program::Desc desc;
        desc.addShaderLibrary(kShaderScatterPhotons).csEntry("main").setShaderModel(kShaderModel);

        DefineList defines;
        mpScatterPhotonsPass = ComputePass::create(mpDevice, desc, defines, true);
    }

    auto var = mpScatterPhotonsPass->getRootVar();
    var["CB"]["gPhotonCount"] = mLastPhotonCount;
    var["gPhotonBuffer"] = mpPhotonBuffer;
    var["gCellOffset"] = mpCellCount;  // mpCellCount now contains offsets after prefix sum
    var["gSortedPhotonBuffer"] = mpSortedPhotonBuffer;
    var["gCellCounter"] = mpCellCounter;

    // Dispatch: one thread per photon (execute takes thread count, auto-divides by group size)
    mpScatterPhotonsPass->execute(pRenderContext, uint3(mLastPhotonCount, 1, 1));

    // UAV barrier
    pRenderContext->uavBarrier(mpSortedPhotonBuffer.get());
    pRenderContext->uavBarrier(mpCellCounter.get());
}

void ProgressivePhotonMapping::collectPhotonsPass(RenderContext* pRenderContext, const RenderData& renderData)
{
    if (mUseReSTIR)
    {
        // ========================================
        // ReSTIR Collect Path
        // ========================================
        FALCOR_PROFILE(pRenderContext, "PPM ReSTIR Collect");

        // Initialize ReSTIR compute pass
        if (!mpReSTIRCollectPass)
        {
            Program::Desc desc;
            desc.addShaderModules(mpScene->getShaderModules());
            desc.addShaderLibrary(kShaderReSTIRCollect).csEntry("main").setShaderModel(kShaderModel);
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
            if (mpEmissiveLightSampler)
                defines.add(mpEmissiveLightSampler->getDefines());
            defines.add(getValidResourceDefines(kInputChannels, renderData));
            defines.add(getValidResourceDefines(kOutputChannels, renderData));

            mpReSTIRCollectPass = ComputePass::create(mpDevice, desc, defines, true);
        }
        FALCOR_ASSERT(mpReSTIRCollectPass);

        // Runtime defines
        mpReSTIRCollectPass->getProgram()->addDefine("PHOTON_BUFFER_SIZE", std::to_string(mMaxPhotonBufferSize));
        mpReSTIRCollectPass->getProgram()->addDefine("ROUGHNESS_THRESHOLD", std::to_string(mSpecularRoughnessThreshold));
        mpReSTIRCollectPass->getProgram()->addDefine("MAX_CAMERA_BOUNCES", std::to_string(mMaxCameraBounces));
        if (mpEmissiveLightSampler)
            mpReSTIRCollectPass->getProgram()->addDefines(mpEmissiveLightSampler->getDefines());
        mpReSTIRCollectPass->getProgram()->addDefines(getValidResourceDefines(kInputChannels, renderData));
        mpReSTIRCollectPass->getProgram()->addDefines(getValidResourceDefines(kOutputChannels, renderData));

        // Set variables
        auto var = mpReSTIRCollectPass->getRootVar();
        mpScene->setRaytracingShaderData(pRenderContext, var);
        mpSampleGenerator->setShaderData(var);

        // Constant buffer
        float currentIteration = static_cast<float>(mFrameCount + 1);
        float globalRadius = mInitialRadius * std::pow(currentIteration, -(1.0f - mAlpha) * 0.5f);

        var["CB"]["gFrameCount"] = mFrameCount;
        var["CB"]["gFrameDim"] = mScreenRes;
        var["CB"]["gInitialRadius"] = mInitialRadius;
        var["CB"]["gNumPhotonsPerFrame"] = mNumPhotonsPerFrame;
        var["CB"]["gAlpha"] = mAlpha;
        var["CB"]["gCellSize"] = mInitialRadius;
        var["CB"]["gCurrentRadius"] = globalRadius;
        var["CB"]["gDisplayMode"] = static_cast<uint>(mDisplayMode);
        var["CB"]["gTemporalMaxM"] = mTemporalMaxM;
        var["CB"]["gSpatialNeighbors"] = mSpatialNeighbors;
        var["CB"]["gSpatialRadius"] = mSpatialRadius;

        // Emissive light sampler
        if (mpEmissiveLightSampler)
            mpEmissiveLightSampler->setShaderData(var["EmissiveLight"]["gEmissiveSampler"]);

        // Input
        var["gVBuffer"] = renderData[kInputVBuffer]->asTexture();
        auto pViewW = renderData[kInputViewDir];
        if (pViewW) var["gViewW"] = pViewW->asTexture();

        var["gSortedPhotonBuffer"] = mpSortedPhotonBuffer;
        var["gPhotonCounter"] = mpPhotonCounter;
        var["gCellOffset"] = mpCellCount;
        var["gCellCounter"] = mpCellCounter;

        // Reservoir double buffering: current frame writes to [mReservoirFrameIndex], reads from [1 - mReservoirFrameIndex]
        uint currIdx = mReservoirFrameIndex;
        uint prevIdx = 1 - mReservoirFrameIndex;

        var["gReservoirBuffer"] = mpReservoirBuffer[currIdx];
        var["gPrevReservoirBuffer"] = mpReservoirBuffer[prevIdx];
        var["gPositionBuffer"] = mpPositionBuffer[currIdx];
        var["gNormalBuffer"] = mpNormalBuffer[currIdx];
        var["gPrevPositionBuffer"] = mpPositionBuffer[prevIdx];
        var["gPrevNormalBuffer"] = mpNormalBuffer[prevIdx];

        // Per-pixel statistics
        var["gPixelStats"] = mpPixelStatsBuffer;

        // Output
        var["gOutputColor"] = renderData.getTexture("color");

        // Execute
        const uint2 targetDim = renderData.getDefaultTextureDims();
        FALCOR_ASSERT(targetDim.x > 0 && targetDim.y > 0);
        mpReSTIRCollectPass->execute(pRenderContext, uint3(targetDim, 1));

        // Swap reservoir frame index for next frame
        mReservoirFrameIndex = 1 - mReservoirFrameIndex;
    }
    else
    {
        // ========================================
        // Traditional PPM Collect Path
        // ========================================
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
        var["CB"]["gDisplayMode"] = static_cast<uint>(mDisplayMode);

        // Probabilistic PPM: compute global radius schedule r_k = r_0 * k^{-(1-alpha)/2}
        float currentIteration = static_cast<float>(mFrameCount + 1);
        float globalRadius = mInitialRadius * std::pow(currentIteration, -(1.0f - mAlpha) * 0.5f);
        var["CB"]["gCellSize"] = mInitialRadius;
        var["CB"]["gCurrentRadius"] = globalRadius;

        // Emissive light sampler
        if (mpEmissiveLightSampler)
            mpEmissiveLightSampler->setShaderData(var["EmissiveLight"]["gEmissiveSampler"]);

        // Input
        var["gVBuffer"] = renderData[kInputVBuffer]->asTexture();
        auto pViewW = renderData[kInputViewDir];
        if (pViewW) var["gViewW"] = pViewW->asTexture();

        var["gSortedPhotonBuffer"] = mpSortedPhotonBuffer;
        var["gPhotonCounter"] = mpPhotonCounter;
        var["gCellOffset"] = mpCellCount;
        var["gCellCounter"] = mpCellCounter;

        // Input/Output: per-pixel statistics
        var["gPixelStats"] = mpPixelStatsBuffer;

        // Output
        var["gOutputColor"] = renderData.getTexture("color");

        // Execute
        const uint2 targetDim = renderData.getDefaultTextureDims();
        FALCOR_ASSERT(targetDim.x > 0 && targetDim.y > 0);
        mpCollectPhotonsPass->execute(pRenderContext, uint3(targetDim, 1));
    }
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
