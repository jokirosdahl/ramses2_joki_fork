#include <metal_stdlib>
#include "../metal_types.h"
#include "metal_utils.h"
using namespace metal;

/* ---- Hilbert state tables (NDIM=3), copied from refine.metal ---------- */
constant long next_digits_p[96] = {
    0, 1, 3, 2, 7, 6, 4, 5,
    0, 7, 1, 6, 3, 4, 2, 5,
    0, 3, 7, 4, 1, 2, 6, 5,
    2, 3, 1, 0, 5, 4, 6, 7,
    4, 3, 5, 2, 7, 0, 6, 1,
    6, 5, 1, 2, 7, 4, 0, 3,
    4, 7, 3, 0, 5, 6, 2, 1,
    6, 7, 5, 4, 1, 0, 2, 3,
    2, 5, 3, 4, 1, 6, 0, 7,
    2, 1, 5, 6, 3, 0, 4, 7,
    4, 5, 7, 6, 3, 2, 0, 1,
    6, 1, 7, 0, 5, 2, 4, 3
};

constant int next_state_p[96] = {
     1,  2,  3,  2,  4,  5,  3,  5,
     2,  6,  0,  7,  8,  8,  0,  7,
     0,  9, 10,  9,  1,  1, 11, 11,
     6,  0,  6, 11,  9,  0,  9,  8,
    11, 11,  0,  7,  5,  9,  0,  7,
     4,  4,  8,  8,  0,  6, 10,  6,
     5,  7,  5,  3,  1,  1, 11, 11,
     6,  1,  6, 10,  9,  4,  9, 10,
    10,  3,  1,  1, 10,  3,  5,  9,
     4,  4,  8,  8,  2,  7,  2,  3,
     7,  2, 11,  2,  7,  5,  8,  5,
    10,  3,  2,  6, 10,  3,  4,  4
};

inline long hilbert_key_p(int ckey[3], int level)
{
    ulong hkey   = 0;
    int   cstate = 0;
    for (int ibit = level - 1; ibit >= 0; ibit--) {
        hkey = (hkey << 4) >> 1;
        int sdigit = 0;
        if ((ckey[0] >> ibit) & 1) sdigit += 4;
        if ((ckey[1] >> ibit) & 1) sdigit += 2;
        if ((ckey[2] >> ibit) & 1) sdigit += 1;
        int ind = cstate * 8 + sdigit;
        cstate  = next_state_p[ind];
        hkey   += (ulong)next_digits_p[ind];
    }
    return as_type<long>(hkey);
}

/* =========================================================================
 * read-only hash helpers
 * ========================================================================= */
inline ulong fnv64_p(long key_signed)
{
    ulong key = as_type<ulong>(key_signed);
    ulong h   = 14695981039346656037UL;
    for (int j = 0; j < 8; j++) {
        ulong b = (key >> (8 * j)) & 0xFFUL;
        h ^= b;
        h *= 1099511628211UL;
    }
    return h;
}

inline int hash_bucket_p(long key, int hash_size)
{
    return int(fnv64_p(key) % ulong(hash_size)) + 1;
}

inline int hash_get_p(device const long* hash_key, device const int* hash_val,
                      int hash_size, long key)
{
    int b = hash_bucket_p(key, hash_size);
    for (;;) {
        long cur = hash_key[b - 1];
        if (cur == key) return hash_val[b - 1];
        if (cur == 0L)  return 0;
        b = (b % hash_size) + 1;
    }
}

/* =========================================================================
 * gather_cic_force_part equivalent
 * ========================================================================= */
inline void gather_cic_force_part(
    device const float* f_d,
    device const long* hash_key_d,
    device const int* hash_val_d,
    int hash_size_d,
    device const int* ckey_max_d,
    device const long* key_off_d,
    device const int* box_ckey_min_d,
    device const int* box_ckey_max_d,
    device const int* periodic_d,
    int ngridmax,
    float x_in[3],
    int cell_level,
    int grid_level,
    thread float ff[3],
    thread bool &ok_level)
{
    ok_level = true;
    ff[0] = 0.0f; ff[1] = 0.0f; ff[2] = 0.0f;
    if (grid_level < 1) {
        ok_level = false;
        return;
    }

    int il[3] = {0, 0, 0};
    int ir[3] = {0, 0, 0};
    float dr[3] = {0.0f, 0.0f, 0.0f};
    float dl[3] = {1.0f, 1.0f, 1.0f};
    int box_min[3] = {0, 0, 0};
    int box_max[3] = {0, 0, 0};

    for (int idim = 0; idim < 3; idim++) {
        dr[idim] = x_in[idim] + 0.5f;
        ir[idim] = (int)floor(dr[idim]);
        dr[idim] = dr[idim] - (float)ir[idim];
        dl[idim] = 1.0f - dr[idim];
        il[idim] = ir[idim] - 1;
        box_min[idim] = box_ckey_min_d[idim + (cell_level - 1) * 3];
        box_max[idim] = box_ckey_max_d[idim + (cell_level - 1) * 3];
        if (periodic_d[idim]) {
            if (il[idim] <  box_min[idim]) il[idim] = box_max[idim] - 1;
            if (ir[idim] >= box_max[idim]) ir[idim] = box_min[idim];
        }
    }

    for (int bz = 0; bz <= 1; bz++) {
        for (int by = 0; by <= 1; by++) {
            for (int bx = 0; bx <= 1; bx++) {
                int target_ckey[3] = {0, 0, 0};
                float weight = 1.0f;

                target_ckey[0] = (bx == 1) ? ir[0] : il[0];
                weight *= (bx == 1) ? dr[0] : dl[0];
                target_ckey[1] = (by == 1) ? ir[1] : il[1];
                weight *= (by == 1) ? dr[1] : dl[1];
                target_ckey[2] = (bz == 1) ? ir[2] : il[2];
                weight *= (bz == 1) ? dr[2] : dl[2];

                bool in_domain = true;
                for (int idim = 0; idim < 3; idim++) {
                    if (!periodic_d[idim]) {
                        if (target_ckey[idim] < box_min[idim])  in_domain = false;
                        if (target_ckey[idim] >= box_max[idim]) in_domain = false;
                    }
                }
                if (!in_domain) {
                    ok_level = false;
                    continue;
                }

                int father_ckey[3] = {target_ckey[0] / 2, target_ckey[1] / 2, target_ckey[2] / 2};
                int ii[3] = {target_ckey[0] - 2 * father_ckey[0],
                             target_ckey[1] - 2 * father_ckey[1],
                             target_ckey[2] - 2 * father_ckey[2]};
                int icell = ii[0] + ii[1] * 2 + ii[2] * 4;

                long ix8 = (long)father_ckey[0];
                long iy8 = (long)father_ckey[1];
                long iz8 = (long)father_ckey[2];
                long nx  = (long)ckey_max_d[grid_level - 1];
                long offset = key_off_d[grid_level - 1];
                long key = offset + ix8 + iy8*nx + iz8*nx*nx;

                int igrid = hash_get_p(hash_key_d, hash_val_d, hash_size_d, key);

                if (igrid == 0 || igrid > ngridmax) {
                    ok_level = false;
                    continue;
                }

                for (int idim = 0; idim < 3; idim++) {
                    // f_d is 1-based twotondim, ndim, igrid in Fortran.
                    // Flatten: f_d[(igrid - 1) * 24 + idim * 8 + icell]
                    ff[idim] += f_d[(igrid - 1) * 24 + idim * 8 + icell] * weight;
                }
            }
        }
    }

    if (!ok_level) {
        ff[0] = 0.0f; ff[1] = 0.0f; ff[2] = 0.0f;
    }
}

/* =========================================================================
 * Kernel 1: kick_drift_part_kernel
 * ========================================================================= */
kernel void kick_drift_part_kernel(
    device float* xp_d                    [[buffer(0)]],
    device float* vp_d                    [[buffer(1)]],
    device int* levelp_d                   [[buffer(2)]],
    device const float* f_d               [[buffer(3)]],
    device const long* hash_key_d         [[buffer(4)]],
    device const int* hash_val_d          [[buffer(5)]],
    device const int* ckey_max_d          [[buffer(6)]],
    device const long* key_off_d          [[buffer(7)]],
    device const int* box_ckey_min_d      [[buffer(8)]],
    device const int* box_ckey_max_d      [[buffer(9)]],
    device const float* box_size_d        [[buffer(10)]],
    device const int* periodic_d           [[buffer(11)]],
    device const float* dtnew_d           [[buffer(12)]],
    device const float* dtold_d           [[buffer(13)]],
    constant int &hash_size_d             [[buffer(14)]],
    constant int &ngridmax                [[buffer(15)]],
    constant float &skip1                 [[buffer(16)]],
    constant float &skip2                 [[buffer(17)]],
    constant float &skip3                 [[buffer(18)]],
    constant float &dx_loc                [[buffer(19)]],
    constant int &action_part             [[buffer(20)]],
    constant int &ilevel                  [[buffer(21)]],
    constant int &head_idx                [[buffer(22)]],
    constant int &num_parts               [[buffer(23)]],
    constant long &leading                [[buffer(24)]],
    uint tid                              [[thread_position_in_grid]])
{
    if ((int)tid >= num_parts) return;
    int ipart = (head_idx - 1) + (int)tid; // 0-based index conversion

    float skip[3] = {skip1, skip2, skip3};
    float x[3];
    for (int idim = 0; idim < 3; idim++) {
        x[idim] = (xp_d[ipart + idim * leading] + skip[idim]) / dx_loc;
    }

    float ff[3] = {0.0f, 0.0f, 0.0f};
    bool ok_level = true;

    gather_cic_force_part(f_d, hash_key_d, hash_val_d, hash_size_d,
                          ckey_max_d, key_off_d, box_ckey_min_d, box_ckey_max_d, periodic_d,
                          ngridmax, x, ilevel + 1, ilevel, ff, ok_level);

    if (!ok_level) {
        float x_coarse[3] = {x[0] * 0.5f, x[1] * 0.5f, x[2] * 0.5f};
        gather_cic_force_part(f_d, hash_key_d, hash_val_d, hash_size_d,
                              ckey_max_d, key_off_d, box_ckey_min_d, box_ckey_max_d, periodic_d,
                              ngridmax, x_coarse, ilevel, ilevel - 1, ff, ok_level);
    }

    if (action_part == 2) {
        float dt = dtnew_d[ilevel - 1];
        for (int idim = 0; idim < 3; idim++) {
            vp_d[ipart + idim * leading] += ff[idim] * 0.5f * dt;
            float xnew = xp_d[ipart + idim * leading] + vp_d[ipart + idim * leading] * dt;
            if (periodic_d[idim]) {
                if (xnew < 0.0f)              xnew += box_size_d[idim];
                if (xnew >= box_size_d[idim]) xnew -= box_size_d[idim];
            }
            xp_d[ipart + idim * leading] = xnew;
        }
    } else if (action_part == 1) {
        int lp = levelp_d[ipart];
        float dteff = (lp >= ilevel) ? dtnew_d[lp - 1] : dtold_d[lp - 1];
        levelp_d[ipart] = ilevel;

        for (int idim = 0; idim < 3; idim++) {
            vp_d[ipart + idim * leading] += ff[idim] * 0.5f * dteff;
        }
    }
}

/* =========================================================================
 * Kernel 2: newdt_part_kernel
 * ========================================================================= */
kernel void newdt_part_kernel(
    device const float* vp_d              [[buffer(0)]],
    device const float* mp_d              [[buffer(1)]],
    device atomic_uint* vmax_d            [[buffer(2)]],
    device atomic_uint* ekin_d            [[buffer(3)]],
    constant int &head_idx                [[buffer(4)]],
    constant int &num_parts               [[buffer(5)]],
    constant long &leading                [[buffer(6)]],
    uint tid                              [[thread_position_in_grid]],
    uint ltid                             [[thread_position_in_threadgroup]])
{
    threadgroup float tg_vmax[256];
    threadgroup float tg_ekin[256];

    int ipart = (head_idx - 1) + (int)tid;
    float vmax_local = 0.0f;
    float ekin_local = 0.0f;

    if ((int)tid < num_parts) {
        float mi = mp_d[ipart];
        for (int idim = 0; idim < 3; idim++) {
            float v = vp_d[ipart + idim * leading];
            float av = abs(v);
            if (av > vmax_local) vmax_local = av;
            ekin_local += 0.5f * mi * v * v;
        }
    }

    tg_vmax[ltid] = vmax_local;
    tg_ekin[ltid] = ekin_local;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduction within threadgroup
    for (uint s = 128; s > 0; s /= 2) {
        if (ltid < s) {
            tg_vmax[ltid] = max(tg_vmax[ltid], tg_vmax[ltid + s]);
            tg_ekin[ltid] += tg_ekin[ltid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (ltid == 0) {
        atomic_max_float(vmax_d, tg_vmax[0]);
        atomic_add_float(ekin_d, tg_ekin[0]);
    }
}

/* =========================================================================
 * Kernel 3: bucket_part_kernel
 * ========================================================================= */
kernel void bucket_part_kernel(
    device const float* xp                [[buffer(0)]],
    device int* bucket_part_d             [[buffer(1)]],
    device const oct_t* grid              [[buffer(2)]],
    device const long* hash_key_d         [[buffer(3)]],
    device const int* hash_val_d          [[buffer(4)]],
    device const int* ckey_max_d          [[buffer(5)]],
    device const long* key_off_d          [[buffer(6)]],
    constant int &hash_size_d             [[buffer(7)]],
    constant float &skip1                 [[buffer(8)]],
    constant float &skip2                 [[buffer(9)]],
    constant float &skip3                 [[buffer(10)]],
    constant float &dx_loc                [[buffer(11)]],
    constant int &ilevel                  [[buffer(12)]],
    constant int &head_idx                [[buffer(13)]],
    constant int &num_parts               [[buffer(14)]],
    constant long &leading                [[buffer(15)]],
    uint tid                              [[thread_position_in_grid]])
{
    if ((int)tid >= num_parts) return;
    int ipart = (head_idx - 1) + (int)tid;

    float skip[3] = {skip1, skip2, skip3};
    float inv_dx_loc = 1.0f / dx_loc;
    float inv_2dx_loc = 0.5f * inv_dx_loc;

    int ix[3];
    for (int idim = 0; idim < 3; idim++) {
        float xs = xp[ipart + idim * leading] + skip[idim];
        ix[idim] = (int)floor(xs * inv_2dx_loc);
    }

    long ix8 = (long)ix[0];
    long iy8 = (long)ix[1];
    long iz8 = (long)ix[2];
    long nx  = (long)ckey_max_d[ilevel - 1];
    long offset = key_off_d[ilevel - 1];
    long key = offset + ix8 + iy8*nx + iz8*nx*nx;

    int igrid = hash_get_p(hash_key_d, hash_val_d, hash_size_d, key);

    if (igrid == 0) {
        bucket_part_d[ipart] = 0;
        return;
    }

    int ii[3];
    for (int idim = 0; idim < 3; idim++) {
        float x = (xp[ipart + idim * leading] + skip[idim]) * inv_dx_loc;
        ii[idim] = (int)(x - 2.0f * (float)ix[idim]);
    }
    int icell = ii[0] + ii[1] * 2 + ii[2] * 4;

    // grid uses 1-based indices in Fortran; convert igrid to 0-based
    if (!grid[igrid - 1].refined[icell]) {
        bucket_part_d[ipart] = 0;
    } else {
        bucket_part_d[ipart] = 1;
    }
}

/* =========================================================================
 * Kernel 4: init_prefix_sum_part_hilbert_fine
 * ========================================================================= */
kernel void init_prefix_sum_part_hilbert_fine(
    device const int* bucket_part_d       [[buffer(0)]],
    device const int* sortp_d             [[buffer(1)]],
    device int* prefix_sum_d              [[buffer(2)]],
    constant int &head_idx                [[buffer(3)]],
    constant int &num_parts               [[buffer(4)]],
    uint tid                              [[thread_position_in_grid]])
{
    if ((int)tid >= num_parts) return;
    int idx = (head_idx - 1) + (int)tid;
    int ipart = sortp_d[idx] - 1; // Convert 1-based sortp to 0-based index
    prefix_sum_d[idx] = bucket_part_d[ipart];
}

/* =========================================================================
 * Kernel 5: write_swap_global_hilbert_partition
 * ========================================================================= */
kernel void write_swap_global_hilbert_partition(
    device int* swap_global_d             [[buffer(0)]],
    device const int* sortp_d             [[buffer(1)]],
    device const int* prefix_sum_d        [[buffer(2)]],
    constant int &head_idx                [[buffer(3)]],
    constant int &num_parts               [[buffer(4)]],
    constant int &n_coarse                [[buffer(5)]],
    uint tid                              [[thread_position_in_grid]])
{
    if ((int)tid >= num_parts) return;
    int idx = (head_idx - 1) + (int)tid;

    int ipart = sortp_d[idx];
    int fine_rank = prefix_sum_d[idx];
    int previous_sum = (idx > (head_idx - 1)) ? prefix_sum_d[idx - 1] : 0;
    int bit = fine_rank - previous_sum;

    int seen = (int)tid + 1;
    int coarse_rank = seen - fine_rank;
    if (bit == 0) {
        swap_global_d[(head_idx - 1) + coarse_rank - 1] = ipart;
    } else {
        swap_global_d[(head_idx - 1) + n_coarse + fine_rank - 1] = ipart;
    }
}

/* =========================================================================
 * Kernel 6: write_sortp_part
 * ========================================================================= */
kernel void write_sortp_part(
    device int* sortp_d                   [[buffer(0)]],
    device const int* swap_global_d       [[buffer(1)]],
    constant int &head_idx                [[buffer(2)]],
    constant int &num_parts               [[buffer(3)]],
    uint tid                              [[thread_position_in_grid]])
{
    if ((int)tid >= num_parts) return;
    int idx = (head_idx - 1) + (int)tid;
    sortp_d[idx] = swap_global_d[idx];
}

/* =========================================================================
 * Radix Sorting Helpers
 * ========================================================================= */
kernel void compute_hkey_part_kernel(
    device const float* xp                [[buffer(0)]],
    device long* hkey_part_d              [[buffer(1)]],
    device const int* box_ckey_min_d      [[buffer(2)]],
    device const int* box_ckey_max_d      [[buffer(3)]],
    device const float* skip              [[buffer(4)]],
    constant float &dx_inv                [[buffer(5)]],
    constant int &head_idx                [[buffer(6)]],
    constant int &num_parts               [[buffer(7)]],
    constant long &leading                [[buffer(8)]],
    constant int &level                   [[buffer(9)]],
    constant float &shift                 [[buffer(10)]],
    uint tid                              [[thread_position_in_grid]])
{
    if ((int)tid >= num_parts) return;
    int ipart = (head_idx - 1) + (int)tid;

    int box_min[3] = {0, 0, 0};
    int box_max[3] = {0, 0, 0};
    for (int idim = 0; idim < 3; idim++) {
        box_min[idim] = box_ckey_min_d[idim + level * 3];
        box_max[idim] = box_ckey_max_d[idim + level * 3];
    }

    int ix[3];
    for (int idim = 0; idim < 3; idim++) {
        float xs = (xp[ipart + idim * leading] + skip[idim]) * dx_inv + shift;
        ix[idim] = (int)floor(xs);
        if (ix[idim] <  box_min[idim]) ix[idim] = box_max[idim] - 1;
        if (ix[idim] >= box_max[idim]) ix[idim] = box_min[idim];
    }

    hkey_part_d[ipart] = hilbert_key_p(ix, level);
}

kernel void init_prefix_sum_part_bit(
    device const long* hkey_part_d        [[buffer(0)]],
    device const int* sortp_d             [[buffer(1)]],
    device int* prefix_sum_d              [[buffer(2)]],
    constant int &head_idx                [[buffer(3)]],
    constant int &num_parts               [[buffer(4)]],
    constant int &ibit                    [[buffer(5)]],
    uint tid                              [[thread_position_in_grid]])
{
    if ((int)tid >= num_parts) return;
    int idx = (head_idx - 1) + (int)tid;

    int old_idx = sortp_d[idx] - 1;
    prefix_sum_d[idx] = (int)((hkey_part_d[old_idx] >> ibit) & 1L);
}

/* =========================================================================
 * Gather / Scatter Helper Kernels
 * ========================================================================= */
kernel void sort_gather_part_real_col(
    device float* buf                     [[buffer(0)]],
    device const float* src               [[buffer(1)]],
    device const int* swap_global_d       [[buffer(2)]],
    constant int &head_idx                [[buffer(3)]],
    constant int &num_parts               [[buffer(4)]],
    constant long &leading                [[buffer(5)]],
    constant int &idim                    [[buffer(6)]],
    uint tid                              [[thread_position_in_grid]])
{
    if ((int)tid >= num_parts) return;
    int idx = (head_idx - 1) + (int)tid;
    int old_idx = swap_global_d[idx] - 1;
    long off = (long)(idim - 1) * leading;
    buf[idx] = src[old_idx + off];
}

kernel void sort_scatter_part_real_col(
    device float* dst                     [[buffer(0)]],
    device const float* buf               [[buffer(1)]],
    constant int &head_idx                [[buffer(2)]],
    constant int &num_parts               [[buffer(3)]],
    constant long &leading                [[buffer(4)]],
    constant int &idim                    [[buffer(5)]],
    uint tid                              [[thread_position_in_grid]])
{
    if ((int)tid >= num_parts) return;
    int idx = (head_idx - 1) + (int)tid;
    long off = (long)(idim - 1) * leading;
    dst[idx + off] = buf[idx];
}

kernel void sort_gather_part_real_1d(
    device float* buf                     [[buffer(0)]],
    device const float* src               [[buffer(1)]],
    device const int* swap_global_d       [[buffer(2)]],
    constant int &head_idx                [[buffer(3)]],
    constant int &num_parts               [[buffer(4)]],
    uint tid                              [[thread_position_in_grid]])
{
    if ((int)tid >= num_parts) return;
    int idx = (head_idx - 1) + (int)tid;
    int old_idx = swap_global_d[idx] - 1;
    buf[idx] = src[old_idx];
}

kernel void sort_scatter_part_real_1d(
    device float* dst                     [[buffer(0)]],
    device const float* buf               [[buffer(1)]],
    constant int &head_idx                [[buffer(2)]],
    constant int &num_parts               [[buffer(3)]],
    uint tid                              [[thread_position_in_grid]])
{
    if ((int)tid >= num_parts) return;
    int idx = (head_idx - 1) + (int)tid;
    dst[idx] = buf[idx];
}

kernel void sort_gather_part_i8_1d(
    device long* buf                      [[buffer(0)]],
    device const long* src                [[buffer(1)]],
    device const int* swap_global_d       [[buffer(2)]],
    constant int &head_idx                [[buffer(3)]],
    constant int &num_parts               [[buffer(4)]],
    uint tid                              [[thread_position_in_grid]])
{
    if ((int)tid >= num_parts) return;
    int idx = (head_idx - 1) + (int)tid;
    int old_idx = swap_global_d[idx] - 1;
    buf[idx] = src[old_idx];
}

kernel void sort_scatter_part_i8_1d(
    device long* dst                      [[buffer(0)]],
    device const long* buf                [[buffer(1)]],
    constant int &head_idx                [[buffer(2)]],
    constant int &num_parts               [[buffer(3)]],
    uint tid                              [[thread_position_in_grid]])
{
    if ((int)tid >= num_parts) return;
    int idx = (head_idx - 1) + (int)tid;
    dst[idx] = buf[idx];
}

kernel void sort_gather_part_i4_1d(
    device int* buf                       [[buffer(0)]],
    device const int* src                 [[buffer(1)]],
    device const int* swap_global_d       [[buffer(2)]],
    constant int &head_idx                [[buffer(3)]],
    constant int &num_parts               [[buffer(4)]],
    uint tid                              [[thread_position_in_grid]])
{
    if ((int)tid >= num_parts) return;
    int idx = (head_idx - 1) + (int)tid;
    int old_idx = swap_global_d[idx] - 1;
    buf[idx] = src[old_idx];
}

kernel void sort_scatter_part_i4_1d(
    device int* dst                       [[buffer(0)]],
    device const int* buf                 [[buffer(1)]],
    constant int &head_idx                [[buffer(2)]],
    constant int &num_parts               [[buffer(3)]],
    uint tid                              [[thread_position_in_grid]])
{
    if ((int)tid >= num_parts) return;
    int idx = (head_idx - 1) + (int)tid;
    dst[idx] = buf[idx];
}

/* =========================================================================
 * Kernel 7: cic_part_medium_kernel
 * ========================================================================= */
kernel void cic_part_medium_kernel(
    device const int* sortp_d             [[buffer(0)]],
    device const long* hash_key_d         [[buffer(1)]],
    device const int* hash_val_d          [[buffer(2)]],
    device const int* ckey_max_d          [[buffer(3)]],
    device const long* key_off_d          [[buffer(4)]],
    device const int* box_ckey_min_d      [[buffer(5)]],
    device const int* box_ckey_max_d      [[buffer(6)]],
    device const float* xp_d              [[buffer(7)]],
    device const float* mp_d              [[buffer(8)]],
    device float* rho_d                   [[buffer(9)]],
    device float* nref_d                  [[buffer(10)]],
    constant int &hash_size_d             [[buffer(11)]],
    constant float &skip1                 [[buffer(12)]],
    constant float &skip2                 [[buffer(13)]],
    constant float &skip3                 [[buffer(14)]],
    constant float &dx_loc                [[buffer(15)]],
    constant float &vol_loc               [[buffer(16)]],
    constant float &mass_sph              [[buffer(17)]],
    constant int &star                    [[buffer(18)]],
    constant float &m_refine_at_level     [[buffer(19)]],
    constant float &mass_cut_refine       [[buffer(20)]],
    constant int &ilevel                  [[buffer(21)]],
    constant long &leading                [[buffer(22)]],
    constant int &head_idx                [[buffer(23)]],
    constant int &num_parts               [[buffer(24)]],
    uint tid                              [[thread_position_in_grid]],
    uint lane                             [[thread_index_in_simdgroup]])
{
    bool valid_lane = (tid < (uint)num_parts);

    float skip[3] = {skip1, skip2, skip3};
    long nx          = (long)ckey_max_d[ilevel - 1];
    long hkey_offset = key_off_d[ilevel - 1];

    int box_min[3] = {0, 0, 0};
    int box_max[3] = {0, 0, 0};
    for (int idim = 0; idim < 3; idim++) {
        box_min[idim] = box_ckey_min_d[idim + ilevel * 3];
        box_max[idim] = box_ckey_max_d[idim + ilevel * 3];
    }

    int combined = 0;
    int icell_src_lane = 0;
    int igrid_src_lane = 0;
    float mp_i = 0.0f;
    float frac[3] = {0.0f, 0.0f, 0.0f};
    int src_full_ckey[3] = {0, 0, 0};

    if (valid_lane) {
        int ipart = sortp_d[(head_idx - 1) + (int)tid] - 1;
        mp_i = mp_d[ipart];

        for (int idim = 0; idim < 3; idim++) {
            float xs = (xp_d[ipart + idim * leading] + skip[idim]) / dx_loc + 0.5f;
            src_full_ckey[idim] = (int)floor(xs);
            frac[idim] = xs - (float)src_full_ckey[idim];
            if (src_full_ckey[idim] <  box_min[idim]) src_full_ckey[idim] = box_max[idim] - 1;
            if (src_full_ckey[idim] >= box_max[idim]) src_full_ckey[idim] = box_min[idim];
        }

        int father_ckey[3] = {src_full_ckey[0] / 2, src_full_ckey[1] / 2, src_full_ckey[2] / 2};
        int ii[3] = {src_full_ckey[0] - 2 * father_ckey[0],
                     src_full_ckey[1] - 2 * father_ckey[1],
                     src_full_ckey[2] - 2 * father_ckey[2]};

        icell_src_lane = ii[0] + ii[1] * 2 + ii[2] * 4;

        long ix8 = (long)father_ckey[0];
        long iy8 = (long)father_ckey[1];
        long iz8 = (long)father_ckey[2];
        long key = hkey_offset + ix8 + iy8*nx + iz8*nx*nx;
        igrid_src_lane = hash_get_p(hash_key_d, hash_val_d, hash_size_d, key);

        if (igrid_src_lane == 0) {
            valid_lane = false;
            mp_i = 0.0f;
            frac[0] = 0.0f; frac[1] = 0.0f; frac[2] = 0.0f;
            src_full_ckey[0] = 0; src_full_ckey[1] = 0; src_full_ckey[2] = 0;
        } else {
            combined = (igrid_src_lane << 5) | icell_src_lane;
        }
    }

    float wx[2] = {1.0f - frac[0], frac[0]};
    float wy[2] = {1.0f - frac[1], frac[1]};
    float wz[2] = {1.0f - frac[2], frac[2]};

    int prev_combined = simd_shuffle_up(combined, 1);
    int next_combined = simd_shuffle_down(combined, 1);
    bool is_head = (lane == 0)  || (combined != prev_combined);
    bool is_tail = (lane == 31) || (combined != next_combined);

    for (int k = 1; k <= 8; k++) {
        int bx = (k - 1) & 1;
        int by = ((k - 1) >> 1) & 1;
        int bz = ((k - 1) >> 2) & 1;

        float my_rho = 0.0f;
        float my_nref = 0.0f;

        if (valid_lane) {
            float w = wx[bx] * wy[by] * wz[bz];
            if (w > 0.0f) {
                my_rho = mp_i * w / vol_loc;
                if (m_refine_at_level >= 0.0f) {
                    if (!star) {
                        if (mass_cut_refine > 0.0f) {
                            if (mp_i < mass_cut_refine) {
                                my_nref = w;
                            }
                        } else {
                            my_nref = w;
                        }
                    } else {
                        my_nref = mp_i * w / mass_sph;
                    }
                }
            }
        }

        int dst_igrid_lane = 0;
        int dst_icell_lane = 0;

        if (is_tail && valid_lane) {
            int target_ckey[3];
            target_ckey[0] = src_full_ckey[0] - 1 + bx;
            target_ckey[1] = src_full_ckey[1] - 1 + by;
            target_ckey[2] = src_full_ckey[2] - 1 + bz;

            for (int idim = 0; idim < 3; idim++) {
                if (target_ckey[idim] <  box_min[idim]) target_ckey[idim] = box_max[idim] - 1;
                if (target_ckey[idim] >= box_max[idim]) target_ckey[idim] = box_min[idim];
            }

            int father_ckey[3] = {target_ckey[0] / 2, target_ckey[1] / 2, target_ckey[2] / 2};
            int ii[3] = {target_ckey[0] - 2 * father_ckey[0],
                         target_ckey[1] - 2 * father_ckey[1],
                         target_ckey[2] - 2 * father_ckey[2]};

            dst_icell_lane = ii[0] + ii[1] * 2 + ii[2] * 4;

            long ix8 = (long)father_ckey[0];
            long iy8 = (long)father_ckey[1];
            long iz8 = (long)father_ckey[2];
            long key = hkey_offset + ix8 + iy8*nx + iz8*nx*nx;
            dst_igrid_lane = hash_get_p(hash_key_d, hash_val_d, hash_size_d, key);
        }

        int head_acc = is_head ? 1 : 0;
        for (int scan_iter = 0; scan_iter <= 4; scan_iter++) {
            int scan_offset = 1 << scan_iter;
            float other_rho  = simd_shuffle_up(my_rho,  scan_offset);
            float other_nref = simd_shuffle_up(my_nref, scan_offset);
            int other_head   = simd_shuffle_up(head_acc, scan_offset);
            if (lane >= (uint)scan_offset) {
                if (head_acc == 0) {
                    my_rho  += other_rho;
                    my_nref += other_nref;
                }
                head_acc = head_acc | other_head;
            }
        }

        if (is_tail && valid_lane && dst_igrid_lane != 0) {
            int dst_idx = (dst_igrid_lane - 1) * 8 + dst_icell_lane;
            if (my_rho != 0.0f) {
                atomic_add_float((device atomic_uint*)&rho_d[dst_idx], my_rho);
            }
            if (m_refine_at_level >= 0.0f && my_nref != 0.0f) {
                atomic_add_float((device atomic_uint*)&nref_d[dst_idx], my_nref);
            }
        }
    }
}

/* =========================================================================
 * multipole_q_part_kernel — accumulate monopole q[0] and dipole q[1..3]
 * from all particles [head_idx-1 .. head_idx-1+num_parts-1].
 * Mirrors multipole_q_kernel (gpu_part.cuf).
 * multipole_q_d: float[4] = { sum(mp), sum(mp*xp[0]), sum(mp*xp[1]), sum(mp*xp[2]) }
 * Uses 256-thread threadgroups (8 SIMD groups of 32).
 * ========================================================================= */
kernel void multipole_q_part_kernel(
    device const float* xp_d           [[buffer(0)]],
    device const float* mp_d           [[buffer(1)]],
    device atomic_uint* multipole_q_d  [[buffer(2)]],
    constant int &head_idx             [[buffer(3)]],
    constant int &num_parts            [[buffer(4)]],
    constant long &leading             [[buffer(5)]],
    uint tid                           [[thread_position_in_grid]],
    uint tid_local                     [[thread_position_in_threadgroup]])
{
    float q[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    if ((int)tid < num_parts) {
        int ipart = (head_idx - 1) + (int)tid;
        float mp_i = mp_d[ipart];
        q[0] = mp_i;
        q[1] = mp_i * xp_d[ipart + 0 * leading];
        q[2] = mp_i * xp_d[ipart + 1 * leading];
        q[3] = mp_i * xp_d[ipart + 2 * leading];
    }

    threadgroup float tg_sums[8];
    for (int c = 0; c < 4; c++) {
        float reduced = tg_reduce_sum_f(q[c], tid_local, tg_sums);
        if (tid_local == 0) {
            atomic_add_float(&multipole_q_d[c], reduced);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
