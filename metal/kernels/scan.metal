/*
 * metal/kernels/scan.metal
 *
 * Metal port of gpu/gpu_scan.cuf — integer inclusive prefix scan.
 * Scope: int32 only (NPRE=4; dp variant not needed for the AMR sort path).
 *
 * Two kernels replace block_scan + uniform_add from gpu_scan.cuf:
 *
 *   scan_block_kernel   — phase 1: within-block inclusive scan; writes block
 *                         total to partial_sums[bid].  Mirrors block_scan<<<>>>.
 *   scan_fixup_kernel   — phase 3: adds partial_sums[bid-1] to all elements of
 *                         block bid (bid > 0).  Mirrors uniform_add<<<>>>.
 *
 * The caller (mtl_prefix_scan in metal_bridge.mm) orchestrates three dispatches:
 *   1. scan_block_kernel over prefix_sum (ceil(n/256) threadgroups)
 *   2. scan_block_kernel over partial_sums (1 threadgroup, if n > 256)
 *   3. scan_fixup_kernel over prefix_sum  (ceil(n/256) threadgroups, if n > 256)
 *
 * Thread layout: 256 threads / threadgroup = 8 SIMD groups × 32 lanes.
 * Apple Silicon guarantees simd_size == 32; hardcoded as SCAN_SIMD_W below.
 *
 * Offset convention: kernels operate on data[offset .. offset+n-1] (0-based),
 * matching the gpu_scan.cuf `offset` parameter (converted from Fortran 1-based
 * head_idx by subtracting 1 in the bridge).
 */

#include <metal_stdlib>
using namespace metal;

constant int SCAN_TG_SIZE   = 256;
constant int SCAN_SIMD_W    = 32;
constant int SCAN_NUM_SIMD  = 8;    /* SCAN_TG_SIZE / SCAN_SIMD_W */

/* =========================================================================
 * scan_block_kernel — phase 1 (and phase 2 applied to partial_sums).
 *
 * Each threadgroup of 256 threads performs an inclusive prefix scan of its
 * 256 elements using a two-level SIMD approach:
 *   1. SIMD-level inclusive scan via simd_prefix_inclusive_sum.
 *   2. The last lane of each SIMD group writes its sum to threadgroup sums[].
 *   3. The first SIMD group scans sums[] (8 values embedded in lanes 0..7).
 *   4. Each thread with simd_id > 0 adds sums[simd_id - 1].
 *
 * Result is written back to data[offset + gid].
 * The last thread in the threadgroup writes its value (= block total) to
 * partial_sums[bid].
 * ========================================================================= */
kernel void scan_block_kernel(
    device int*       data          [[buffer(0)]],
    device int*       partial_sums  [[buffer(1)]],
    constant int&     offset        [[buffer(2)]],
    constant int&     n             [[buffer(3)]],
    uint  tid   [[thread_position_in_threadgroup]],
    uint  bid   [[threadgroup_position_in_grid]],
    uint  gid   [[thread_position_in_grid]])
{
    uint simd_id   = tid / (uint)SCAN_SIMD_W;
    uint simd_lane = tid % (uint)SCAN_SIMD_W;

    threadgroup int sums[SCAN_NUM_SIMD];

    /* Load — out-of-range threads contribute 0 */
    int scan = 0;
    if ((int)gid < n) scan = data[offset + (int)gid];

    /* SIMD-level inclusive prefix scan */
    scan = simd_prefix_inclusive_sum(scan);

    /* Last lane in each SIMD group saves its (inclusive) group sum */
    if (simd_lane == (uint)(SCAN_SIMD_W - 1)) sums[simd_id] = scan;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    /* First SIMD group scans the 8 group sums.
     * Lanes 0..7 carry sums[0..7]; lanes 8..31 contribute 0. */
    if (simd_id == 0) {
        int w = (simd_lane < (uint)SCAN_NUM_SIMD) ? sums[simd_lane] : 0;
        w = simd_prefix_inclusive_sum(w);
        if (simd_lane < (uint)SCAN_NUM_SIMD) sums[simd_lane] = w;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    /* Add the sum of all preceding SIMD groups */
    if (simd_id > 0) scan += sums[simd_id - 1];

    /* Write inclusive prefix sum back */
    if ((int)gid < n) data[offset + (int)gid] = scan;

    /* Block total: written by the last thread in the threadgroup.
     * If the block is partial (last block, some gid >= n), tid == SCAN_TG_SIZE-1
     * may be out of range; scan already incorporates only valid (non-zero) data,
     * so partial_sums[bid] = sum of valid elements in this block. */
    if (tid == (uint)(SCAN_TG_SIZE - 1)) partial_sums[bid] = scan;
}

/* =========================================================================
 * scan_fixup_kernel — phase 3.
 *
 * For each threadgroup bid > 0, adds partial_sums[bid - 1] (the inclusive
 * sum of all preceding blocks after the phase-2 scan of partial_sums) to
 * every in-range element in this block.  Block 0 returns immediately.
 *
 * Mirrors uniform_add<<<>>> in gpu_scan.cuf.
 * ========================================================================= */
kernel void scan_fixup_kernel(
    device int*           data          [[buffer(0)]],
    device const int*     partial_sums  [[buffer(1)]],
    constant int&         offset        [[buffer(2)]],
    constant int&         n             [[buffer(3)]],
    uint  tid   [[thread_position_in_threadgroup]],
    uint  bid   [[threadgroup_position_in_grid]],
    uint  gid   [[thread_position_in_grid]])
{
    /* Block 0 is already correct from phase 1 */
    if (bid == 0) return;

    threadgroup int buf;

    /* Thread 0 broadcasts the preceding block's cumulative sum */
    if (tid == 0) buf = partial_sums[bid - 1];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if ((int)gid < n) data[offset + (int)gid] += buf;
}
