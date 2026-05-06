// PPM shared parameter definitions
// Supports two modes:
//   1. Probabilistic PPM (Knaus & Zwicker, TOG 2011): per-frame independent estimation + running average
//   2. Standard SPPM (Hachisuka et al. 2009): tau accumulation with per-pixel radius reduction
#pragma once

#include "Utils/Math/MathConstants.slangh"

// Hash grid parameters
static const uint HASH_GRID_SIZE = 2000000;  // Number of hash grid cells (2M to reduce hash collisions)
static const uint HASH_GRID_INVALID = 0xFFFFFFFF;

// PPM Pixel Statistics (per-pixel persistent state)
// Stores all fields needed by both Probabilistic PPM and Standard SPPM modes.
// Layout: 64 bytes total (16-byte aligned)
struct PPMPixelStats
{
    float radius2;          // Current search radius squared (r^2), reduced each iteration
    float N;                // Probabilistic: iteration count; SPPM: effective photon count
    float _pad0;
    float _pad1;
    float3 tau;             // SPPM: accumulated flux (tau); Probabilistic: radianceSum (running sum)
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

// Photon data structure
struct PPMPhoton
{
    float3 posW;            // World-space position
    uint _pad0;
    float3 flux;            // Photon flux (power)
    uint _pad1;
    float3 dirW;            // Incident direction (pointing towards light)
    uint nextInCell;        // Next photon index in the same hash cell (linked list)

    __init()
    {
        posW = float3(0);
        _pad0 = 0;
        flux = float3(0);
        _pad1 = 0;
        dirW = float3(0);
        nextInCell = HASH_GRID_INVALID;
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
