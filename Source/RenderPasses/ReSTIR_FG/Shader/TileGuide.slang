/// TileGuide.slang
/// Tile-based directional guide using octahedral histogram.
/// 每个 Tile 存储一个 4×4 octahedral 方向直方图，用于引导 Gather Ray 采样。

#include "Utils/Math/MathConstants.slangh"
import Utils.Math.MathHelpers;

static const uint kTileBinRes = 4;                         // 4×4 bins
static const uint kTileBinCount = kTileBinRes * kTileBinRes; // 16 bins

/// 将 3D 方向映射到 octahedral UV [0,1]^2
float2 directionToOctUV(float3 dir)
{
    float3 n = dir / (abs(dir.x) + abs(dir.y) + abs(dir.z));
    // 下半球折叠
    if (n.z < 0.0)
    {
        float2 wrapped;
        wrapped.x = (1.0 - abs(n.y)) * (n.x >= 0.0 ? 1.0 : -1.0);
        wrapped.y = (1.0 - abs(n.x)) * (n.y >= 0.0 ? 1.0 : -1.0);
        n.xy = wrapped;
    }
    // 映射到 [0,1]
    return n.xy * 0.5 + 0.5;
}

/// 将 octahedral UV [0,1]^2 映射回 3D 方向
float3 octUVToDirection(float2 uv)
{
    float2 f = uv * 2.0 - 1.0;
    float3 n = float3(f.x, f.y, 1.0 - abs(f.x) - abs(f.y));
    if (n.z < 0.0)
    {
        float2 wrapped;
        wrapped.x = (1.0 - abs(n.y)) * (n.x >= 0.0 ? 1.0 : -1.0);
        wrapped.y = (1.0 - abs(n.x)) * (n.y >= 0.0 ? 1.0 : -1.0);
        n.xy = wrapped;
    }
    return normalize(n);
}

/// 将方向映射到 bin 索引
uint directionToBinIndex(float3 dir)
{
    float2 uv = directionToOctUV(dir);
    uint2 bin = uint2(clamp(uv * float(kTileBinRes), float2(0), float2(kTileBinRes - 1)));
    return bin.x + bin.y * kTileBinRes;
}

/// 将 bin 索引映射到 bin 中心方向
float3 binIndexToDirection(uint binIdx)
{
    uint bx = binIdx % kTileBinRes;
    uint by = binIdx / kTileBinRes;
    float2 uv = (float2(bx, by) + 0.5) / float(kTileBinRes);
    return octUVToDirection(uv);
}

/// 计算 octahedral mapping 在给定方向处的 Jacobian |dω/dudv|
/// 即 UV 空间微元 du·dv 对应的立体角微元 dω 的大小。
/// 对于标准 octahedral mapping: |dω/dudv| = 4 / ||n_unnorm||3
/// 其中 n_unnorm 是 octahedral 参数化中未归一化的方向向量。
/// 数值验证: ∫|J|dudv over [0,1]2 = 4π（球面总立体角），确认公式正确。
float octahedralJacobian(float3 dir)
{
    // 将归一化方向 dir 投影回 L1 球面上的未归一化坐标 n_unnorm
    // n_unnorm = dir / L1_norm, 其中 L1_norm = |dir.x| + |dir.y| + |dir.z|
    float L1 = abs(dir.x) + abs(dir.y) + abs(dir.z);
    float3 n_unnorm = dir / L1;
    // ||n_unnorm||? = 1 / L1
    // Jacobian: |dω/dudv| = 4 / ||n_unnorm||?3 = 4 * L13
    return 4.0 * L1 * L1 * L1;
}

/// Tile Guide 数据结构
/// 每个 tile 存储 16 个 float bin + 1 个 totalWeight
/// 内存布局: bins[0..15], totalWeight => 17 floats = 68 bytes
struct TileGuideData
{
    float bins[kTileBinCount]; // 4×4 octahedral 方向直方图
    float totalWeight;         // 所有 bin 权重之和

    /// 空初始化
    __init()
    {
        [unroll]
        for (uint i = 0; i < kTileBinCount; i++)
            bins[i] = 0.0;
        totalWeight = 0.0;
    }

    /// 判断 guide 是否有效（有足够的信息）
    property bool valid
    {
        get { return totalWeight > 1e-8; }
    }

    /// 评估给定方向的 PDF（关于立体角 dω）
    /// 返回值: 归一化的 PDF（在球面上积分为 1）
    float evalPdf(float3 dir)
    {
        if (!valid)
            return 0.0;

        uint binIdx = directionToBinIndex(dir);
        // UV 空间的 PDF: p_uv = (bin权重 / 总权重) * binCount
        float pdfUV = (bins[binIdx] / totalWeight) * float(kTileBinCount);
        // 转换到立体角空间: p_ω = p_uv / |dω/dudv|
        // 其中 |dω/dudv| 是 octahedral mapping 的 Jacobian
        float J = octahedralJacobian(dir);
        return pdfUV / max(J, 1e-8);
    }

    /// 从直方图中 importance sample 一个方向
    /// rnd1: 用于选择 bin 的随机数 [0,1)
    /// rnd2: 用于在 bin 内部抖动的 2D 随机数 [0,1)^2
    /// outPdf: 输出采样的 PDF
    /// 返回值: 采样的方向
    float3 sampleDirection(float rnd1, float2 rnd2, out float outPdf)
    {
        if (!valid)
        {
            outPdf = 0.0;
            return float3(0, 0, 1);
        }

        // CDF 查找：按权重选择 bin
        float target = rnd1 * totalWeight;
        float cumulative = 0.0;
        uint selectedBin = kTileBinCount - 1; // 默认最后一个

        [unroll]
        for (uint i = 0; i < kTileBinCount; i++)
        {
            cumulative += bins[i];
            if (cumulative > target && i <= selectedBin)
            {
                selectedBin = i;
                break;
            }
        }

        // 在选中的 bin 内部做均匀抖动
        uint bx = selectedBin % kTileBinRes;
        uint by = selectedBin / kTileBinRes;
        float2 uv = (float2(bx, by) + rnd2) / float(kTileBinRes);
        uv = clamp(uv, float2(0.001), float2(0.999));

        float3 dir = octUVToDirection(uv);

        // UV 空间的 PDF: p_uv = (bin权重 / 总权重) * binCount
        float pdfUV = (bins[selectedBin] / totalWeight) * float(kTileBinCount);
        // 转换到立体角空间: p_ω = p_uv / |dω/dudv|
        float J = octahedralJacobian(dir);
        outPdf = pdfUV / max(J, 1e-8);

        return dir;
    }
};

/// 从 StructuredBuffer 加载 TileGuideData
/// buffer: 存储所有 tile 数据的 buffer（每个 tile 17 个 float）
/// tileIndex: tile 的线性索引
TileGuideData loadTileGuide(StructuredBuffer<float> buffer, uint tileIndex)
{
    TileGuideData guide;
    uint baseOffset = tileIndex * (kTileBinCount + 1);
    [unroll]
    for (uint i = 0; i < kTileBinCount; i++)
        guide.bins[i] = buffer[baseOffset + i];
    guide.totalWeight = buffer[baseOffset + kTileBinCount];
    return guide;
}

/// 将 TileGuideData 存储到 RWStructuredBuffer
void storeTileGuide(RWStructuredBuffer<float> buffer, uint tileIndex, TileGuideData guide)
{
    uint baseOffset = tileIndex * (kTileBinCount + 1);
    [unroll]
    for (uint i = 0; i < kTileBinCount; i++)
        buffer[baseOffset + i] = guide.bins[i];
    buffer[baseOffset + kTileBinCount] = guide.totalWeight;
}
