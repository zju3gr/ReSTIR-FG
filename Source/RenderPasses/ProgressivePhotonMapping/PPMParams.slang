// PPM shared parameter definitions
// Supports two modes:
//   1. Probabilistic PPM (Knaus & Zwicker, TOG 2011): per-frame independent estimation + running average
//   2. Standard SPPM (Hachisuka et al. 2009): tau accumulation with per-pixel radius reduction
#pragma once

#include "Utils/Math/MathConstants.slangh"

// Hash grid parameters
// 使用 Sorted Photon Array 方案：
//   1. Count Pass: 统计每个 cell 中的光子数量
//   2. Prefix Sum: 计算每个 cell 的起始偏移
//   3. Scatter Pass: 将光子按 cell 排序写入连续 buffer
//   4. Collect Pass: 使用 offset + count 直接连续访问光子数据
static const uint HASH_GRID_SIZE = 2000000;  // Hash grid cell 数量（2M，减少 hash 冲突）
static const uint HASH_GRID_INVALID = 0xFFFFFFFF;
static const uint MAX_PHOTONS_PER_CELL = 2048; // 每个 cell 最多存储的光子数量（需要足够大以避免丢弃光子导致归一化偏差）

// PPM Pixel Statistics (per-pixel persistent state)
// Two modes:
//   Probabilistic PPM (Knaus 2011): Only uses tau (radianceSum) and emissionSum.
//     Radius is a GLOBAL schedule computed from iteration count, no per-pixel state needed.
//   Standard SPPM (Hachisuka 2009): Uses all fields (radius2, N, tau, emissionSum).
// Layout: 48 bytes total (16-byte aligned)
struct PPMPixelStats
{
    float radius2;          // SPPM only: current search radius squared (r^2)
    float N;                // SPPM only: effective photon count
    float _pad0;
    float _pad1;
    float3 tau;             // SPPM: accumulated flux (tau); Probabilistic: radianceSum (running sum of L_k)
    float _pad2;
    float3 emissionSum;     // Running sum of per-frame emission/direct contributions
    float _pad3;

    __init()
    {
        radius2 = 0;
        N = 0;
        _pad0 = 0;
        _pad1 = 0;
        tau = float3(0);
        _pad2 = 0;
        emissionSum = float3(0);
        _pad3 = 0;
    }
};

// Photon data structure (用于 unsorted photon buffer，trace pass 写入)
struct PPMPhoton
{
    float3 posW;            // World-space position
    uint cellIndex;         // Hash grid cell index (用于排序)
    float3 flux;            // Photon flux (power)
    uint _pad1;
    float3 dirW;            // Incident direction (pointing towards light)
    uint _pad2;

    __init()
    {
        posW = float3(0);
        cellIndex = HASH_GRID_INVALID;
        flux = float3(0);
        _pad1 = 0;
        dirW = float3(0);
        _pad2 = 0;
    }
};

// Sorted Photon data structure (用于 sorted photon buffer，collect pass 读取)
// 更紧凑的布局，去掉 cellIndex 以节省带宽
struct PPMPhotonSorted
{
    float3 posW;            // World-space position
    float _pad0;
    float3 flux;            // Photon flux (power)
    float _pad1;
    float3 dirW;            // Incident direction (pointing towards light)
    float _pad2;

    __init()
    {
        posW = float3(0);
        _pad0 = 0;
        flux = float3(0);
        _pad1 = 0;
        dirW = float3(0);
        _pad2 = 0;
    }
};

// ========================================
// ReSTIR Reservoir 数据结构
// 用于 ReSTIR 光子收集：从大量光子中通过 RIS 选出 1 个高质量样本
// ========================================
struct PhotonReservoir
{
    float3 photonPosW;      // 被选中光子的世界坐标
    float wSum;             // RIS 权重累加和
    float3 photonFlux;      // 被选中光子的 flux
    uint M;                 // 已见过的候选光子数量
    float3 photonDirW;      // 被选中光子的入射方向
    float targetPdf;        // 被选中光子的 target function 值
    float3 F;               // 被选中光子的最终贡献 (throughput * BRDF * flux)
    float _pad0;

    __init()
    {
        photonPosW = float3(0);
        wSum = 0;
        photonFlux = float3(0);
        M = 0;
        photonDirW = float3(0);
        targetPdf = 0;
        F = float3(0);
        _pad0 = 0;
    }

    // 流式 RIS 更新：以概率 w/wSum 替换当前样本
    [mutating] bool update(float3 posW, float3 flux, float3 dirW, float w, float targetP, float3 contribution, float rand)
    {
        wSum += w;
        M++;
        if (rand * wSum < w)
        {
            photonPosW = posW;
            photonFlux = flux;
            photonDirW = dirW;
            targetPdf = targetP;
            F = contribution;
            return true;
        }
        return false;
    }

    // 合并另一个 Reservoir（用于时空重采样）
    [mutating] bool merge(PhotonReservoir other, float targetPdfAtThis, float rand)
    {
        // 另一个 reservoir 的 RIS 权重 = targetPdfAtThis * other.W * other.M
        // 其中 other.W = other.wSum / (other.targetPdf * other.M)
        float otherW = 0;
        if (other.targetPdf > 0 && other.M > 0)
            otherW = targetPdfAtThis * (other.wSum / other.targetPdf);

        wSum += otherW;
        M += other.M;
        if (wSum > 0 && rand * wSum < otherW)
        {
            photonPosW = other.photonPosW;
            photonFlux = other.photonFlux;
            photonDirW = other.photonDirW;
            targetPdf = targetPdfAtThis;
            // F 需要在合并后重新计算
            return true;
        }
        return false;
    }

    // 计算最终的 RIS 权重 W = wSum / (targetPdf * M)
    float getInvPdf()
    {
        if (targetPdf > 0 && M > 0)
            return wSum / (targetPdf * float(M));
        return 0;
    }
};

// Hash grid helper functions
uint hashGridCellIndex(int3 cell)
{
    // Simple spatial hash function
    uint h = uint(cell.x * 73856093u ^ cell.y * 19349663u ^ cell.z * 83492791u);
    return h % HASH_GRID_SIZE;
}

int3 worldToCell(float3 posW, float cellSize)
{
    return int3(floor(posW / cellSize));
}
