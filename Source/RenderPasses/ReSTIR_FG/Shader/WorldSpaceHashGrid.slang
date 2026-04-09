// World Space Hash Grid 数据结构
// 移植自 World Space ReSTIR GI，用于 spatial resampling 中的邻居查找
// 将像素按 world space 位置 + 法线方向注册到 hash grid 中

import Utils.Math.HashUtils;

// Hash Grid 注册数据：每个像素在 hash grid 中的位置信息
struct HashAppendData
{
    uint isValid;       // 是否有效
    uint pixelIndex;    // 像素的线性索引
    uint cellIdx;       // 所属 cell 的索引
    uint inCellIdx;     // 在 cell 内的偏移
};

// PCG32 哈希函数
uint pcg32(uint input)
{
    uint state = input * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// 将法线量化为 8 个方向桶（每个轴正/负）
// 确保法线相似的像素被分配到同一个 cell
uint BinaryNorm(float3 norm)
{
    uint a = norm.x > 0.f ? 1 : 0;
    uint b = norm.y > 0.f ? 1 : 0;
    uint c = norm.z > 0.f ? 1 : 0;
    return a * 100 + b * 10 + c;
}

// 基于相机距离自适应计算 cell 大小
// 远处的物体使用更大的 cell，近处使用更小的 cell
float CalculateCellSize(float3 pos, float3 cameraPos, float fov, float2 frameDim, float minCellSize)
{
    float cellSizeStep = length(pos - cameraPos) * tan(120 * fov * max(1.0 / frameDim.y, frameDim.y / float((frameDim.x * frameDim.x))));
    int logStep = int(floor(log2(cellSizeStep / minCellSize)));
    return minCellSize * max(0.12f, exp2(float(logStep)));
}

// 在 hash table 中查找或插入一个 cell
// 使用开放寻址法处理冲突，checksum 用于验证
// 返回 cell 索引，-1 表示失败
int FindOrInsertCell(float3 pos, float3 norm, float cellSize, float3 sceneBBMin, uint hashTableSize, RWByteAddressBuffer checkSumBuffer)
{
    uint3 p = uint3(floor((pos - sceneBBMin) / cellSize));
    uint normprint = BinaryNorm(norm);

    uint cellIndex = pcg32(normprint + pcg32(asuint(cellSize) + pcg32(p.z + pcg32(p.y + pcg32(p.x))))) % hashTableSize;
    uint checkSum = max(jenkinsHash(normprint + jenkinsHash(asuint(cellSize) + jenkinsHash(p.z + jenkinsHash(p.y + jenkinsHash(p.x))))), 1);

    // 开放寻址：最多尝试 32 个槽位
    for (uint i = 0; i < 32; i++)
    {
        uint idx = cellIndex * 32 + i;
        uint checkSumPre;

        checkSumBuffer.InterlockedCompareExchange(idx * 4, 0, checkSum, checkSumPre);
        if (checkSumPre == 0 || checkSumPre == checkSum)
            return int(idx);
    }

    return -1;
}

// 在 hash table 中查找一个 cell（只读版本）
// 返回 cell 索引，-1 表示未找到
int FindCell(float3 pos, float3 norm, float cellSize, float3 sceneBBMin, uint hashTableSize, ByteAddressBuffer checkSumBuffer)
{
    uint3 p = uint3(floor((pos - sceneBBMin) / cellSize));
    uint normprint = BinaryNorm(norm);

    uint cellIndex = pcg32(normprint + pcg32(asuint(cellSize) + pcg32(p.z + pcg32(p.y + pcg32(p.x))))) % hashTableSize;
    uint checkSum = max(jenkinsHash(normprint + jenkinsHash(asuint(cellSize) + jenkinsHash(p.z + jenkinsHash(p.y + jenkinsHash(p.x))))), 1);

    for (uint i = 0; i < 32; i++)
    {
        uint idx = cellIndex * 32 + i;

        if (checkSumBuffer.Load(idx * 4) == checkSum)
            return int(idx);
    }

    return -1;
}
