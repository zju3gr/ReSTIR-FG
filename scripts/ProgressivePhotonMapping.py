# -*- coding: utf-8 -*-
from falcor import *

def render_graph_ProgressivePhotonMapping():
    g = RenderGraph("ProgressivePhotonMapping")
    
    VBufferRT = createPass("VBufferRT", {"samplePattern": "Center", "sampleCount": 1})
    g.addPass(VBufferRT, "VBufferRT")
    
    PPM = createPass("ProgressivePhotonMapping")
    g.addPass(PPM, "PPM")
    
    ToneMapper = createPass("ToneMapper", {"autoExposure": False})
    g.addPass(ToneMapper, "ToneMapper")
    
    g.addEdge("VBufferRT.vbuffer", "PPM.vbuffer")
    g.addEdge("VBufferRT.viewW", "PPM.viewW")
    g.addEdge("PPM.color", "ToneMapper.src")
    
    g.markOutput("ToneMapper.dst")
    
    return g

ProgressivePhotonMapping = render_graph_ProgressivePhotonMapping()
try: m.addGraph(ProgressivePhotonMapping)
except NameError: None
