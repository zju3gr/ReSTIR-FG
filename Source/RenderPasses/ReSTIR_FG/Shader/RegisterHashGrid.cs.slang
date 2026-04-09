// RegisterHashGrid.cs.slang
// 将每个像素的 surface 位置注册到 World Space Hash Grid 中
// 每个像素计算其所属的 cell，并原子地增加 cell 计数器

#include "Scene/SceneDefines.slangh"

import WorldSpaceHashGrid;
import SurfaceDataFG;

StructuredBuffer<PackedSurfaceDataFG> gSurface;
Texture2D<float4> gView;

RWStructuredBuffer<HashAppendData> gAppendBuffer;
RWByteAddressBuffer gCheckSumBuffer;
RWByteAddressBuffer gCellCounters;

cbuffer HashGridCB
{
    uint2 gFrameDim;
    uint gHashTableSize;    // hash table 大小（cell 数量）
    float gFov;
    float3 gCameraPos;
    float gMinCellSize;
    float3 gSceneBBMin;
    float _pad0;            // 对齐填充
};

[numthreads(16, 16, 1)]
void main(uint2 dTid : SV_DispatchThreadID)
{
    if (any(dTid >= gFrameDim))
        return;

    uint linearIdx = dTid.x + gFrameDim.x * dTid.y;

    // 初始化为无效
    HashAppendData data = {};
    data.isValid = 0;
    data.pixelIndex = linearIdx;
    data.cellIdx = 0;
    data.inCellIdx = 0;

    // 获取 surface 数据
    float3 viewDir = gView[dTid].xyz;
    PackedSurfaceDataFG packedSurface = gSurface[linearIdx];
    SurfaceFG surface = SurfaceFG(packedSurface, viewDir);

    if (surface.valid)
    {
        float3 pos = surface.posW;
        float3 norm = surface.normal;

        if (any(norm != 0))
        {
            float cellSize = CalculateCellSize(pos, gCameraPos, gFov, float2(gFrameDim), gMinCellSize);
            int cellIdx = FindOrInsertCell(pos, norm, cellSize, gSceneBBMin, gHashTableSize, gCheckSumBuffer);

            if (cellIdx != -1)
            {
                uint inCellIdx;
                gCellCounters.InterlockedAdd(uint(cellIdx) * 4, 1, inCellIdx);
                data.isValid = 1;
                data.cellIdx = uint(cellIdx);
                data.inCellIdx = inCellIdx;
            }
        }
    }

    gAppendBuffer[linearIdx] = data;
}
