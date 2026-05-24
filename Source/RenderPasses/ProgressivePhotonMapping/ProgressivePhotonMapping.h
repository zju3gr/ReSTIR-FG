#pragma once
#include "Falcor.h"
#include "RenderGraph/RenderPass.h"
#include "Utils/Sampling/SampleGenerator.h"
#include "Rendering/Lights/EmissiveLightSampler.h"
#include "Utils/Algorithm/PrefixSum.h"

using namespace Falcor;

/**
 * Progressive Photon Mapping (PPM) render pass.
 * Supports two modes:
 *
 * Mode 1: "Memoryless" Probabilistic PPM (Knaus & Zwicker, TOG 2011)
 *   - Global radius schedule: r_k = r_0 * k^{-(1-alpha)/2}
 *   - Each frame is an independent density estimate: L_k = Phi / (pi * r_k^2)
 *   - Final result: L = (1/N) * sum L_k (simple running average)
 *   - NO per-pixel radius, NO per-pixel photon count, NO tau recursion
 *   - Supports random camera paths each frame (glossy, DOF, motion blur)
 *
 * Mode 2: Standard SPPM (Hachisuka et al. 2009)
 *   - Per-pixel radius reduction: Nnew = N + alpha*M, r_new^2 = r_old^2 * Nnew/(N+M)
 *   - Tau accumulation: tau = (tau + Phi) * (r_new^2 / r_old^2)
 *   - Requires per-pixel local photon statistics
 *
 * Algorithm:
 * 1. Camera Pass: Trace paths from camera to first diffuse surface
 * 2. Photon Tracing Pass: Emit photons from light sources, store at diffuse surfaces
 * 3. Photon Collection Pass: Estimate radiance via photon density estimation
 */
/** Display mode for PPM output. */
enum class PPMDisplayMode : uint32_t
{
    Full = 0,           // Direct + Indirect (default)
    DirectOnly = 1,     // Direct lighting only
    IndirectOnly = 2,   // Indirect lighting only (photon density estimation)
    PhotonDirectTest = 3, // Sanity test: first-bounce photon map direct only (no NEE, depth==0 photons)
};

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

    // Photon Sort Pass: Sort photons into contiguous memory by hash cell
    void sortPhotonsPass(RenderContext* pRenderContext);

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
    uint mPhotonMaxBounces = 5;             // Max photon bounces
    uint mNumPhotonsPerFrame = 500000;      // Photons emitted per frame (500K, balance quality/performance)
    uint mMaxPhotonBufferSize = 5000000;    // Photon buffer capacity (5M)
    uint mLastPhotonCount = 0;              // Last frame's actual photon count (for UI display)

    // PPM core parameters
    float mInitialRadius = 0.05f;           // Initial search radius (smaller = less photons per pixel = faster)
    float mAlpha = 0.7f;                    // Radius reduction parameter alpha (standard SPPM recommends 0.7)
    bool mUseProbabilisticPPM = true;       // true = Probabilistic PPM (Knaus 2011), false = Standard SPPM (Hachisuka 2009)
    PPMDisplayMode mDisplayMode = PPMDisplayMode::IndirectOnly; // Display mode: start with Indirect Only for testing

    // ReSTIR parameters
    bool mUseReSTIR = false;                // 是否启用 ReSTIR 优化光子收集
    uint mTemporalMaxM = 20;                // 时间重采样最大 M 值（限制历史复用量，防止 temporal lag）
    uint mSpatialNeighbors = 3;             // 空间重采样邻居数量
    float mSpatialRadius = 10.0f;           // 空间重采样搜索半径（像素）

    // Light source parameters
    bool mHasLights = false;
    bool mHasAnalyticLights = false;
    bool mMixedLights = false;
    float mPhotonAnalyticRatio = 0.5f;      // Ratio of analytic lights when using mixed lights

    //
    // GPU resources
    //
    ref<Buffer> mpPixelStatsBuffer;         // Per-pixel persistent statistics (PPMPixelStats)
    ref<Buffer> mpPhotonBuffer;             // Unsorted photon buffer (trace pass 写入)
    ref<Buffer> mpPhotonCounter;            // Photon counter
    ref<Buffer> mpPhotonCounterStaging;     // Staging buffer for reading back photon count
    ref<Buffer> mpCellCount;                // 每个 hash cell 的光子计数 → in-place prefix sum 后变为 offset table
    ref<Buffer> mpCellCounter;              // Scatter pass 中的 per-cell atomic counter
    ref<Buffer> mpSortedPhotonBuffer;       // Sorted photon buffer (按 cell 连续存储)
    std::unique_ptr<PrefixSum> mpPrefixSum; // GPU parallel prefix sum utility

    // ReSTIR resources
    ref<Buffer> mpReservoirBuffer[2];       // 双缓冲 Reservoir (当前帧/上一帧)
    ref<Texture> mpPositionBuffer[2];       // 双缓冲 position (当前帧/上一帧)
    ref<Texture> mpNormalBuffer[2];         // 双缓冲 normal (当前帧/上一帧)
    uint mReservoirFrameIndex = 0;          // 当前帧的 buffer index (0 or 1)

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
    ref<ComputePass> mpScatterPhotonsPass;
    ref<ComputePass> mpCollectPhotonsPass;
    ref<ComputePass> mpReSTIRCollectPass;   // ReSTIR 版本的 Collect Pass
};
