/*
 * metal/kernels/refine.metal
 *
 * Metal port of gpu/gpu_refine.cuf — AMR neighbour-array construction.
 * Scope: PoC (levelmin==levelmax, NDIM=3, NSUBGRID=1).
 *
 * Hash table layout (set by insert_hash_kernel in hash.metal):
 *   hash_key : device long[hash_size]   (64-bit Hilbert key; 0 == empty)
 *   hash_val : device int[hash_size]    (1-based oct index; 0 == not found)
 * Both are read non-atomically here because hash.metal's insert dispatch
 * has already completed (waitUntilCompleted in metal_bridge.mm).
 *
 * Kernel:
 *   build_nbor_kernel — replaces the 27-launch update_nbor_array loop;
 *                       each thread handles all 27 neighbours of one subgrid.
 */

#include <metal_stdlib>
#include "../metal_types.h"
using namespace metal;

constant int NSUBGRIDP2_RF  = 3;    /* nsubgrid + 2 (nsubgrid == 1) */
constant int SUBGRIDSIZE_RF = 27;   /* NSUBGRIDP2^NDIM, NDIM=3      */

/* =========================================================================
 * fnv64 / hash_bucket / hash_get — read-only hash helpers.
 * These mirror the insert-side versions in hash.metal but operate on plain
 * (non-atomic) device pointers, which is safe because the insert dispatch
 * completed before this kernel is enqueued.
 * ========================================================================= */
inline long fnv64_r(long key_signed)
{
    ulong key = as_type<ulong>(key_signed);
    ulong h   = 14695981039346656037UL;
    for (int j = 0; j < 8; j++) {
        ulong b = (key >> (8 * j)) & 0xFFUL;
        h ^= b;
        h *= 1099511628211UL;
    }
    return as_type<long>(h);
}

inline int hash_bucket_r(long key, int hash_size)
{
    long ib = fnv64_r(key) % long(hash_size);
    if (ib < 0) ib += long(hash_size);
    return int(ib) + 1;
}

inline int hash_get(device const long* hash_key, device const int* hash_val,
                    int hash_size, long key)
{
    int b = hash_bucket_r(key, hash_size);
    for (;;) {
        long cur = hash_key[b - 1];
        if (cur == key) return hash_val[b - 1];
        if (cur == 0L)  return 0;
        b = (b % hash_size) + 1;
    }
}

/* =========================================================================
 * build_nbor_kernel — replaces the 27-launch update_nbor_array loop from
 *                     r_set_grid_device in gpu_manager.cuf.
 *
 * Each thread handles one subgrid (= one oct for nsubgrid=1) and computes
 * all 27 neighbour indices.  The inner loop body mirrors update_nbor_array
 * for nsubgrid=1 (gpu_refine.cuf lines 1908-1939):
 *
 *   ind = 1 + i_off + j_off*nsubgridp2 + k_off*nsubgridp2^2   (1..27)
 *   ckey_n[d] = ckey[d] + off - 1    (nsubgrid=1)
 *   if periodic: wrap when ckey_n < box_min or ckey_n >= box_max
 *   key = key_off + ix + iy*nx + iz*nx*nx
 *   nbor[(subgrid_idx-1)*27 + (ind-1)] = hash_get(key)
 *
 * Thread layout: 128 threads/threadgroup.
 * ========================================================================= */
kernel void build_nbor_kernel(
    device const oct_t   *grid         [[buffer(0)]],
    device int           *nbor         [[buffer(1)]],
    device const long    *hash_key     [[buffer(2)]],   /* non-atomic long */
    device const int     *hash_val     [[buffer(3)]],   /* non-atomic int  */
    constant int         &hash_size    [[buffer(4)]],
    constant int         &ckey_max_l   [[buffer(5)]],
    constant long        &key_off_l    [[buffer(6)]],
    constant int         *box_ckey_min [[buffer(7)]],   /* [3] */
    constant int         *box_ckey_max [[buffer(8)]],   /* [3] */
    constant int         *periodic     [[buffer(9)]],   /* [3]: non-zero = true */
    constant int         &head_idx     [[buffer(10)]],
    constant int         &num_subgrids [[buffer(11)]],
    uint thread_idx [[thread_position_in_grid]])
{
    if (int(thread_idx) >= num_subgrids) return;
    int subgrid_idx = head_idx + int(thread_idx);   /* 1-based */

    long nx = long(ckey_max_l);

    for (int k_off = 0; k_off < NSUBGRIDP2_RF; k_off++) {
        for (int j_off = 0; j_off < NSUBGRIDP2_RF; j_off++) {
            for (int i_off = 0; i_off < NSUBGRIDP2_RF; i_off++) {
                int ind = 1 + i_off
                            + NSUBGRIDP2_RF * j_off
                            + NSUBGRIDP2_RF * NSUBGRIDP2_RF * k_off;   /* 1..27 */

                int ckey_n[3];
                ckey_n[0] = grid[subgrid_idx - 1].ckey[0] + i_off - 1;
                ckey_n[1] = grid[subgrid_idx - 1].ckey[1] + j_off - 1;
                ckey_n[2] = grid[subgrid_idx - 1].ckey[2] + k_off - 1;

                for (int d = 0; d < 3; d++) {
                    if (periodic[d]) {
                        if (ckey_n[d] <  box_ckey_min[d]) ckey_n[d] = box_ckey_max[d] - 1;
                        if (ckey_n[d] >= box_ckey_max[d]) ckey_n[d] = box_ckey_min[d];
                    }
                }

                long ix  = long(ckey_n[0]);
                long iy  = long(ckey_n[1]);
                long iz  = long(ckey_n[2]);
                long key = key_off_l + ix + iy * nx + iz * nx * nx;

                int nbor_idx = hash_get(hash_key, hash_val, hash_size, key);
                nbor[(subgrid_idx - 1) * SUBGRIDSIZE_RF + (ind - 1)] = nbor_idx;
            }
        }
    }
}
