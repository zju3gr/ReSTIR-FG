add_plugin(ProgressivePhotonMapping)

target_sources(ProgressivePhotonMapping PRIVATE
    ProgressivePhotonMapping.cpp
    ProgressivePhotonMapping.h
    PPMTracePhotons.rt.slang
    PPMCollectPhotons.cs.slang
    PPMParams.slang
)

target_copy_shaders(ProgressivePhotonMapping RenderPasses/ProgressivePhotonMapping)

target_source_group(ProgressivePhotonMapping "RenderPasses")
