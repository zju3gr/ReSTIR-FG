/// UpdateTileGuide.cs.slang
/// Compute shader: 从 Reservoir winner 的 FGData 更新 Tile Guide 直方图
/// 使用两个 entry point:
///   - accumulateTileGuide: 每个像素将其 reservoir winner 方向写入一个 per-pixel bin index + weight buffer
///   - blendTileGuide: 每个 tile 汇总所有像素贡献，并与历史混合

#include "Scene/SceneDefines.slangh"
#include "Utils/Math/MathConstants.slangh"

import Utils.Math.MathHelpers;
import Utils.Math.PackedFormats;
import Utils.Math.FormatConversion;
import Utils.Color.ColorHelpers;

import Reservoir;
import FinalGatherData;
import SurfaceDataFG;
import TileGuide;

// Reservoir buffer
#if USE_REDUCED_RESERVOIR_FORMAT
    Texture2D<uint2> gReservoir;
#else
    Texture2D<uint4> gReservoir;
#endif

// FG sample data
StructuredBuffer<FGData> gFGSampleData;

// Surface data
StructuredBuffer<PackedSurfaceDataFG> gSurfaceData;
Texture2D<float4> gView;

// Per-pixel 中间数据: 存储 (binIndex, weight) 打包为 uint2
// x = binIndex (0xFFFFFFFF 表示无效), y = asuint(weight)
RWStructuredBuffer<uint2> gPixelGuideBinWeight;

// Tile Guide buffers
StructuredBuffer<float> gTileGuidePrev;     // 上一帧的 tile guide（只读）
RWStructuredBuffer<float> gTileGuideOut;    // 输出的 tile guide

cbuffer TileGuideCB
{
    uint2 gFrameDim;        // 屏幕分辨率
    uint2 gTileDim;         // Tile 网格维度
    uint gTileSize;         // Tile 大小（像素）
    float gTemporalAlpha;   // 时间混合系数 alpha (新帧权重)
    float gAttenuationRadius; // 衰减半径
    uint gFrameCount;       // 帧计数
};

/// Pass 1: 每个像素计算其 reservoir winner 的方向 bin 和权重
[numthreads(16, 16, 1)]
void accumulateTileGuide(uint2 pixel : SV_DispatchThreadID)
{
    if (any(pixel >= gFrameDim))
        return;

    uint pixelIdx = pixel.x + pixel.y * gFrameDim.x;

    // 默认无效
    gPixelGuideBinWeight[pixelIdx] = uint2(0xFFFFFFFF, 0);

    // 读取 Reservoir
    Reservoir r = Reservoir(gReservoir[pixel]);
    if (r.M == 0 || r.weightSum <= 0.0)
        return;

    // 读取 FGData
    FGData fgData = gFGSampleData[pixelIdx];
    if (!fgData.valid)
        return;

    // 读取 Surface 获取 posW
    float3 viewDir = gView[pixel].xyz;
    SurfaceFG surface = SurfaceFG(gSurfaceData[pixelIdx], viewDir);
    if (!surface.valid)
        return;

    // 计算 Gather 方向
    float3 gatherDir = fgData.posW - surface.posW;
    float dist = length(gatherDir);
    if (dist < 1e-6)
        return;
    gatherDir /= dist;

    // 计算置信度权重
    float confidence = r.targetFunc * r.weightSum;
    if (confidence <= 0.0 || isinf(confidence) || isnan(confidence))
        return;
    confidence = min(confidence, 100.0);

    // 计算 bin 索引并存储
    uint binIdx = directionToBinIndex(gatherDir);
    gPixelGuideBinWeight[pixelIdx] = uint2(binIdx, asuint(confidence));
}

/// Pass 2: 每个 tile 汇总其所有像素的贡献，并与历史混合
[numthreads(256, 1, 1)]
void blendTileGuide(uint tileIndex : SV_DispatchThreadID)
{
    uint totalTiles = gTileDim.x * gTileDim.y;
    if (tileIndex >= totalTiles)
        return;

    // 计算 tile 的像素范围
    uint tileX = tileIndex % gTileDim.x;
    uint tileY = tileIndex / gTileDim.x;
    uint2 tileStart = uint2(tileX, tileY) * gTileSize;
    uint2 tileEnd = min(tileStart + gTileSize, gFrameDim);

    // 累积当前帧的 bin 权重
    float newBins[kTileBinCount];
    [unroll]
    for (uint i = 0; i < kTileBinCount; i++)
        newBins[i] = 0.0;

    float newTotalWeight = 0.0;

    for (uint py = tileStart.y; py < tileEnd.y; py++)
    {
        for (uint px = tileStart.x; px < tileEnd.x; px++)
        {
            uint pixelIdx = px + py * gFrameDim.x;
            uint2 binWeight = gPixelGuideBinWeight[pixelIdx];
            uint binIdx = binWeight.x;
            float weight = asfloat(binWeight.y);

            if (binIdx < kTileBinCount && weight > 0.0)
            {
                newBins[binIdx] += weight;
                newTotalWeight += weight;
            }
        }
    }

    // 加载上一帧的 guide
    uint baseOffset = tileIndex * (kTileBinCount + 1);
    float prevTotalWeight = gTileGuidePrev[baseOffset + kTileBinCount];

    // 指数移动平均混合
    float alpha = (prevTotalWeight > 1e-8) ? gTemporalAlpha : 1.0;

    float outTotalWeight = 0.0;
    [unroll]
    for (uint j = 0; j < kTileBinCount; j++)
    {
        float prevBin = max(0.0, gTileGuidePrev[baseOffset + j]);
        float newNorm = (newTotalWeight > 1e-8) ? (newBins[j] / newTotalWeight) : 0.0;
        float prevNorm = (prevTotalWeight > 1e-8) ? (prevBin / prevTotalWeight) : 0.0;
        float blended = alpha * newNorm + (1.0 - alpha) * prevNorm;
        gTileGuideOut[baseOffset + j] = blended;
        outTotalWeight += blended;
    }
    gTileGuideOut[baseOffset + kTileBinCount] = outTotalWeight;
}
