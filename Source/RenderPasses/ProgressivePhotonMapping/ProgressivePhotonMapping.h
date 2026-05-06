#pragma once
#include "Falcor.h"
#include "RenderGraph/RenderPass.h"
#include "Utils/Sampling/SampleGenerator.h"
#include "Rendering/Lights/EmissiveLightSampler.h"

using namespace Falcor;

/**
 * Progressive Photon Mapping (PPM) render pass.
 * Implements "Progressive Photon Mapping: A Probabilistic Approach" (Knaus & Zwicker, TOG 2011)
 *
 * Algorithm:
 * 1. Camera Pass: Trace paths from camera to first diffuse surface (supports random glossy sampling)
 * 2. Photon Tracing Pass: Emit photons from light sources, store at diffuse surfaces
 * 3. Photon Collection Pass: Estimate radiance via photon density estimation
 * 4. Progressive radius reduction: r_{i+1}^2 = r_i^2 * (i + alpha) / (i + 1)
 * 5. Running average of per-frame radiance estimates
 *
 * Key advantage over standard SPPM: Each frame independently samples camera paths,
 * enabling correct handling of glossy surfaces, DOF, and motion blur.
 */
class ProgressivePhotonMapping : public RenderPass
{
public:
    FALCOR_PLUGIN_CLASS(ProgressivePhotonMapping, "ProgressivePhotonMapping", "Progressive Photon Mapping renderer.");

    static ref<ProgressivePhotonMapping> create(ref<Device> pDevice, const Properties& props)
    {
        return make_ref<ProgressivePhotonMapping>(pDevice, props);
    }

    ProgressivePhotonMapping(ref<Device> pDevice, const Properties& props);

    virtual Properties getProperties() const override;
    virtual RenderPassReflection reflect(const CompileData& compileData) override;
    virtual void execute(RenderContext* pRenderContext, const RenderData& renderData) override;
    virtual void renderUI(Gui::Widgets& widget) override;
    virtual void setScene(RenderContext* pRenderContext, const ref<Scene>& pScene) override;
    virtual bool onMouseEvent(const MouseEvent& mouseEvent) override { return false; }
    virtual bool onKeyEvent(const KeyboardEvent& keyEvent) override { return false; }

private:
    void parseProperties(const Properties& props);

    // Prepare light sampler
    void prepareLighting(RenderContext* pRenderContext);

    // Prepare GPU resources
    void prepareResources(RenderContext* pRenderContext, const RenderData& renderData);

    // Camera Pass: Trace camera paths, record hit points
    void traceCameraPass(RenderContext* pRenderContext, const RenderData& renderData);

    // Photon Tracing Pass: Emit photons from light sources
    void tracePhotonsPass(RenderContext* pRenderContext, const RenderData& renderData);

    // Photon Collection Pass: Collect photons for density estimation
    void collectPhotonsPass(RenderContext* pRenderContext, const RenderData& renderData);

    // Get material-related defines
    DefineList getMaterialDefines();

    //
    // Scene and infrastructure
    //
    ref<Scene> mpScene;
    ref<SampleGenerator> mpSampleGenerator;
    std::unique_ptr<EmissiveLightSampler> mpEmissiveLightSampler;

    //
    // PPM parameters
    //
    uint mFrameCount = 0;
    uint2 mScreenRes = uint2(0);
    bool mOptionsChanged = false;
    bool mResetAccumulation = true;

    // Camera path parameters
    uint mMaxCameraBounces = 10;            // Max camera path bounces (before reaching diffuse surface)
    float mSpecularRoughnessThreshold = 0.25f; // Surfaces below this roughness are treated as specular

    // Photon parameters
    uint mPhotonMaxBounces = 10;            // Max photon bounces (increase to capture more indirect lighting)
    uint mNumPhotonsPerFrame = 2000000;     // Photons emitted per frame (2 million)
    uint mMaxPhotonBufferSize = 4000000;    // Photon buffer capacity (4 million, larger due to bounced photons)

    // PPM core parameters
    float mInitialRadius = 0.1f;            // Initial search radius
    float mAlpha = 0.7f;                    // Radius reduction parameter alpha (standard SPPM recommends 0.7)
    bool mUseProbabilisticPPM = true;       // true = Probabilistic PPM (Knaus 2011), false = Standard SPPM (Hachisuka 2009)

    // Light source parameters
    bool mHasLights = false;
    bool mHasAnalyticLights = false;
    bool mMixedLights = false;
    float mPhotonAnalyticRatio = 0.5f;      // Ratio of analytic lights when using mixed lights

    //
    // GPU resources
    //
    ref<Buffer> mpPixelStatsBuffer;         // Per-pixel persistent statistics (PPMPixelStats)
    ref<Buffer> mpPhotonBuffer;             // Photon buffer
    ref<Buffer> mpPhotonCounter;            // Photon counter
    ref<Buffer> mpHashGrid;                 // Hash grid for spatial photon lookup

    //
    // Render programs
    //
    struct RayTraceProgramHelper
    {
        ref<RtProgram> pProgram;
        ref<RtBindingTable> pBindingTable;
        ref<RtProgramVars> pVars;

        static RayTraceProgramHelper create()
        {
            RayTraceProgramHelper r;
            r.pProgram = nullptr;
            r.pBindingTable = nullptr;
            r.pVars = nullptr;
            return r;
        }

        void initProgramVars(ref<Device> pDevice, ref<Scene> pScene, ref<SampleGenerator> pSampleGenerator);
    };

    RayTraceProgramHelper mTraceCameraProgram;
    RayTraceProgramHelper mTracePhotonProgram;
    ref<ComputePass> mpCollectPhotonsPass;
};
