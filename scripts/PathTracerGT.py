from pathlib import WindowsPath, PosixPath
from falcor import *

def render_graph_PathTracerGT():
    g = RenderGraph('PathTracerGT')

    # VBuffer: 使用 Stratified 采样模式，提高 primary ray 质量
    g.create_pass('VBufferRT', 'VBufferRT', {
        'outputSize': 'Default',
        'samplePattern': 'Stratified',
        'sampleCount': 16,
        'useAlphaTest': True,
        'adjustShadingNormals': True,
        'forceCullMode': False,
        'cull': 'Back',
        'useTraceRayInline': False,
        'useDOF': True
    })

    # PathTracer: 每像素 1 spp，通过 AccumulatePass 累积实现高 spp
    # maxSurfaceBounces 设为较高值以捕获复杂间接光照（包括 caustics 等）
    # 未单独设置的 diffuse/specular/transmission bounces 会自动继承 maxSurfaceBounces
    g.create_pass('PathTracer', 'PathTracer', {
        'samplesPerPixel': 1,
        'maxSurfaceBounces': 10,
        'useRussianRoulette': True
    })

    # AccumulatePass: 启用累积，Single 精度，无帧数上限（手动停止）
    # overflowMode 设为 'Stop'，达到 maxFrameCount 后停止累积
    # maxFrameCount = 0 表示无限累积
    g.create_pass('AccumulatePass', 'AccumulatePass', {
        'enabled': True,
        'outputSize': 'Default',
        'autoReset': True,
        'precisionMode': 'Single',
        'maxFrameCount': 0,
        'overflowMode': 'Stop'
    })

    # ToneMapper: 线性映射，关闭自动曝光，方便精确对比
    g.create_pass('ToneMapper', 'ToneMapper', {
        'outputSize': 'Default',
        'useSceneMetadata': True,
        'exposureCompensation': 0.0,
        'autoExposure': False,
        'operator': 'Linear',
        'clamp': True,
        'fNumber': 1.0,
        'shutter': 1.0,
        'exposureMode': 'AperturePriority'
    })

    # 连接 render graph
    g.add_edge('VBufferRT.vbuffer', 'PathTracer.vbuffer')
    g.add_edge('VBufferRT.viewW', 'PathTracer.viewW')
    g.add_edge('VBufferRT.mvec', 'PathTracer.mvec')
    g.add_edge('PathTracer.color', 'AccumulatePass.input')
    g.add_edge('AccumulatePass.output', 'ToneMapper.src')

    # 输出: ToneMapper 的结果用于显示，AccumulatePass 的原始 HDR 输出用于精确对比
    g.mark_output('ToneMapper.dst')
    g.mark_output('AccumulatePass.output')
    g.mark_output('PathTracer.directLighting')
    g.mark_output('PathTracer.indirectLighting')

    return g

PathTracerGT = render_graph_PathTracerGT()
try: m.addGraph(PathTracerGT)
except NameError: None
