// BuildHashGrid.cs.slang
// 根据 PrefixSum 结果，将像素索引写入 cell storage
// 每个像素根据其 HashAppendData 中的 cellIdx 和 inCellIdx，
// 将自己的线性索引写入 cellStorage 的对应位置

#include "Scene/SceneDefines.slangh"

import WorldSpaceHashGrid;

StructuredBuffer<HashAppendData> gAppendBuffer;
ByteAddressBuffer gIndexBuffer;     // PrefixSum 后的 cell 起始偏移
RWStructuredBuffer<uint> gCellStorage;  // 最终的 cell 存储

cbuffer BuildCB
{
    uint2 gFrameDim;
};

[numthreads(16, 16, 1)]
void main(uint2 dTid : SV_DispatchThreadID)
{
    if (any(dTid >= gFrameDim))
        return;

    uint linearIdx = dTid.x + gFrameDim.x * dTid.y;
    HashAppendData data = gAppendBuffer[linearIdx];

    if (data.isValid == 0)
        return;

    // 从 PrefixSum 结果中获取 cell 的起始偏移
    uint baseIdx = gIndexBuffer.Load(data.cellIdx * 4);
    // 将像素索引写入 cell storage
    gCellStorage[baseIdx + data.inCellIdx] = data.pixelIndex;
}
