/**
 * PPM Photon Scatter Shader (Compute)
 *
 * 将 unsorted photon buffer 中的光子按 hash cell 排序写入 sorted photon buffer。
 * 前置条件：
 *   - gCellOffset[] 已经通过 prefix sum 计算完成（exclusive scan of cell counts）
 *   - gPhotonCounter[0] 包含实际光子数量
 *
 * 每个线程处理一个光子：
 *   1. 读取光子的 cellIndex
 *   2. 使用 InterlockedAdd 在 gCellCounter[cellIndex] 上获取 cell 内偏移
 *   3. 写入 sorted buffer 的 gCellOffset[cellIndex] + localOffset 位置
 */

#include "Utils/Math/MathConstants.slangh"
import PPMParams;

cbuffer CB
{
    uint gPhotonCount;  // 实际光子数量
}

// Input: unsorted photon buffer (trace pass 写入)
StructuredBuffer<PPMPhoton> gPhotonBuffer;

// Input: cell offset (prefix sum 结果，exclusive scan)
ByteAddressBuffer gCellOffset;

// Output: sorted photon buffer (连续内存，按 cell 排列)
RWStructuredBuffer<PPMPhotonSorted> gSortedPhotonBuffer;

// Temp: per-cell atomic counter (用于获取 cell 内写入位置)
RWBuffer<uint> gCellCounter;

[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint photonIdx = dispatchThreadId.x;
    if (photonIdx >= gPhotonCount)
        return;

    PPMPhoton photon = gPhotonBuffer[photonIdx];

    // 跳过无效光子
    if (photon.cellIndex == HASH_GRID_INVALID)
        return;

    uint cellIdx = photon.cellIndex;

    // 获取 cell 内的写入偏移
    uint localOffset;
    InterlockedAdd(gCellCounter[cellIdx], 1u, localOffset);

    // 限制每个 cell 的光子数量，防止极端情况
    if (localOffset >= MAX_PHOTONS_PER_CELL)
        return;

    // 计算在 sorted buffer 中的全局写入位置
    uint globalOffset = gCellOffset.Load(cellIdx * 4) + localOffset;

    // 写入 sorted photon
    PPMPhotonSorted sortedPhoton;
    sortedPhoton.posW = photon.posW;
    sortedPhoton._pad0 = 0;
    sortedPhoton.flux = photon.flux;
    sortedPhoton._pad1 = 0;
    sortedPhoton.dirW = photon.dirW;
    sortedPhoton._pad2 = 0;

    gSortedPhotonBuffer[globalOffset] = sortedPhoton;
}
