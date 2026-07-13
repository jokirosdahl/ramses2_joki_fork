/*
 * metal/kernels/flag.metal
 *
 * Metal port of gpu/gpu_flag.cuf and hydro_flag_kernel from gpu/gpu_hydro.cuf.
 * Scope: HYDRO=1, GRAV=0, MHD=0, NDIM=3, NPRE=4 (float32), nsubgrid=1.
 *
 * Kernels (thread layout):
 *   reset_flag1_kernel      2D TG(128): 8 cells × 16 octs
 *   reset_flag2_kernel      2D TG(128): 8 cells × 16 octs
 *   init_flag_kernel        1D TG(128): 1 thread per fine oct
 *   count_flag1_kernel      1D TG(256): 1 thread per cell, SIMD reduce
 *   hydro_flag_kernel       2D TG(128): 8 cells × 16 octs
 *   count_neighbors_kernel  2D TG(128): 8 cells × 16 octs
 *   flag_count_kernel       2D TG(128): 8 cells × 16 octs
 *   enforce_rules_kernel    1D TG(128): 1 thread per oct
 *
 * Memory conventions (0-based throughout):
 *   flag1[cell_0 + 8 * oct_abs_0]            — column-major (8, ntotal)
 *   uold[cell_0 + 8*ivar_0 + 8*NVAR*oct_0]   — column-major (8, NVAR, ntotal)
 *   nbor[sg_0 * 27 + ind_0]                  — Metal row-major built by build_nbor_kernel
 */

#ifndef NVAR
#define NVAR 5
#endif

#include <metal_stdlib>
#include "../metal_types.h"
using namespace metal;

/* =========================================================================
 * Constants
 * ========================================================================= */
constant int SUBGRIDSIZE_FL  = 27;   /* (nsubgrid+2)^3, nsubgrid=1   */
constant int NSUBGRIDP2_FL   = 3;    /* nsubgrid + 2                 */
constant int NSUBGRIDP2SQ_FL = 9;    /* NSUBGRIDP2^2                 */
constant int FLAG_TG_OCTS    = 16;   /* octs per 2D threadgroup      */

/* =========================================================================
 * nbor access: row-major [sg_0][ind_0] = nb[(sg_1-1)*27 + (ind_1-1)]
 * Matches build_nbor_kernel and hydro.metal convention.
 * sg_1 and ind_1 are 1-based; returns 1-based oct index (0 = absent).
 * ========================================================================= */
inline int nbor_fl(device const int *nb, int sg_1, int ind_1) {
    return nb[(sg_1 - 1) * SUBGRIDSIZE_FL + (ind_1 - 1)];
}

/* =========================================================================
 * Neighbour lookup tables: hhh_c[cell_0][idir_0] and iii_c[cell_0][idir_0].
 * Ported from gpu_flag.cuf (Fortran 1-based → 0-based).
 * idir_0 = 0,1 → ±x;  2,3 → ±y;  4,5 → ±z.
 * hhh_c[c][d]: neighbour cell index (0-based) in oct for direction d.
 * iii_c[c][d]: grid offset in that direction (-1, 0, or +1).
 * ========================================================================= */
constant int hhh_c[8][6] = {
    {1, 1, 2, 2, 4, 4},
    {0, 0, 3, 3, 5, 5},
    {3, 3, 0, 0, 6, 6},
    {2, 2, 1, 1, 7, 7},
    {5, 5, 6, 6, 0, 0},
    {4, 4, 7, 7, 1, 1},
    {7, 7, 4, 4, 2, 2},
    {6, 6, 5, 5, 3, 3},
};
constant int iii_c[8][6] = {
    {-1,  0, -1,  0, -1,  0},
    { 0,  1, -1,  0, -1,  0},
    {-1,  0,  0,  1, -1,  0},
    { 0,  1,  0,  1, -1,  0},
    {-1,  0, -1,  0,  0,  1},
    { 0,  1, -1,  0,  0,  1},
    {-1,  0,  0,  1,  0,  1},
    { 0,  1,  0,  1,  0,  1},
};

/* =========================================================================
 * Minimal conserved/primitive structs for hydro_flag (NPRE=4, no MHD).
 * Redeclared here because .metal files compile independently.
 * ========================================================================= */
struct fl_conserved_t {
    float density, momentum_x, momentum_y, momentum_z, energy;
};
struct fl_primitive_t {
    float density, velocity_x, velocity_y, velocity_z, pressure;
};

inline float fl_compute_pressure(fl_conserved_t c, float gamma,
                                  float smallr, float smallc2) {
    float d  = max(c.density, smallr);
    float ke = 0.5f * (c.momentum_x * c.momentum_x +
                       c.momentum_y * c.momentum_y +
                       c.momentum_z * c.momentum_z) / d;
    return max((gamma - 1.0f) * (c.energy - ke), smallc2 * d);
}

inline fl_primitive_t fl_c2p(fl_conserved_t c, float gamma,
                               float smallr, float smallc2) {
    fl_primitive_t p;
    p.density    = max(c.density, smallr);
    p.velocity_x = c.momentum_x / p.density;
    p.velocity_y = c.momentum_y / p.density;
    p.velocity_z = c.momentum_z / p.density;
    p.pressure   = fl_compute_pressure(c, gamma, smallr, smallc2);
    return p;
}

/* Gradient-based refinement criterion — density and pressure. */
inline bool fl_hydro_crit(fl_primitive_t l, fl_primitive_t m, fl_primitive_t r,
                           float eg_d, float fl_d, float eg_p, float fl_p) {
    bool ok = false;
    if (eg_d > 0.0f) {
        float el = abs((m.density - l.density) / (m.density + l.density + fl_d));
        float er = abs((r.density - m.density) / (r.density + m.density + fl_d));
        ok = ok || (2.0f * max(el, er) > eg_d);
    }
    if (eg_p > 0.0f) {
        float el = abs((m.pressure - l.pressure) / (m.pressure + l.pressure + fl_p));
        float er = abs((r.pressure - m.pressure) / (r.pressure + m.pressure + fl_p));
        ok = ok || (2.0f * max(el, er) > eg_p);
    }
    return ok;
}

/* Helper: load conserved state from uold buffer (column-major layout). */
inline fl_conserved_t fl_load(device const float *uold, int cell_0, int oct_0) {
    int base = cell_0 + 8 * (NVAR) * oct_0;
    fl_conserved_t c;
    c.density    = uold[base];
    c.momentum_x = uold[base + 8];
    c.momentum_y = uold[base + 16];
    c.momentum_z = uold[base + 24];
    c.energy     = uold[base + 32];
    return c;
}

/* =========================================================================
 * reset_flag1_kernel — zero flag1 for octs [head_idx .. head_idx+num_octs-1].
 * 2D threadgroup: 128 threads = 8 cells (low bits) × 16 octs (high bits).
 * ========================================================================= */
kernel void reset_flag1_kernel(
    device int        *flag1    [[buffer(0)]],
    constant int      &head_idx [[buffer(1)]],
    constant int      &num_octs [[buffer(2)]],
    uint tid [[thread_position_in_threadgroup]],
    uint bid [[threadgroup_position_in_grid]])
{
    uint cell_0  = tid % 8u;
    uint oct_off = bid * (uint)FLAG_TG_OCTS + tid / 8u;
    if (int(oct_off) >= num_octs) return;
    flag1[cell_0 + 8u * uint(head_idx - 1 + int(oct_off))] = 0;
}

/* =========================================================================
 * reset_flag2_kernel — same layout as reset_flag1_kernel.
 * ========================================================================= */
kernel void reset_flag2_kernel(
    device int        *flag2    [[buffer(0)]],
    constant int      &head_idx [[buffer(1)]],
    constant int      &num_octs [[buffer(2)]],
    uint tid [[thread_position_in_threadgroup]],
    uint bid [[threadgroup_position_in_grid]])
{
    uint cell_0  = tid % 8u;
    uint oct_off = bid * (uint)FLAG_TG_OCTS + tid / 8u;
    if (int(oct_off) >= num_octs) return;
    flag2[cell_0 + 8u * uint(head_idx - 1 + int(oct_off))] = 0;
}

/* =========================================================================
 * init_flag_kernel — for each fine oct (ilevel+1), flag the parent cell in
 * flag1 if any child cell is refined or already flagged.
 * 1D: 1 thread per fine oct; head_idx/num_octs refer to ilevel+1.
 * father[oct_abs_0] = 1-based parent oct (populated by update_father_kernel).
 * ========================================================================= */
kernel void init_flag_kernel(
    device int          *flag1    [[buffer(0)]],
    device const oct_t  *grid     [[buffer(1)]],
    device const int    *father   [[buffer(2)]],
    constant int        &head_idx [[buffer(3)]],
    constant int        &num_octs [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (int(gid) >= num_octs) return;
    int oct_abs_0 = (head_idx - 1) + int(gid);
    int father_1  = father[oct_abs_0];
    if (father_1 <= 0) return;
    int father_0 = father_1 - 1;

    /* Child-cell position inside parent oct from ckey parity */
    int ci = grid[oct_abs_0].ckey[0] - 2 * grid[father_0].ckey[0];
    int cj = grid[oct_abs_0].ckey[1] - 2 * grid[father_0].ckey[1];
    int ck = grid[oct_abs_0].ckey[2] - 2 * grid[father_0].ckey[2];
    int cell_0 = ci + 2 * cj + 4 * ck;   /* 0-based, 0..7 */

    bool ok = false;
    for (int ind = 0; ind < 8; ind++) {
        ok = ok || (bool)grid[oct_abs_0].refined[ind];
        ok = ok || (flag1[ind + 8 * oct_abs_0] == 1);
    }
    if (ok) flag1[cell_0 + 8 * father_0] = 1;
}

/* =========================================================================
 * count_flag1_kernel — atomic-reduce sum of all flag1 cells at ilevel.
 * 1D: 1024 threads/TG, 32 SIMD groups × 32 lanes.
 * Two-level reduction: simd_sum per group → tg_sums[32],
 * then SIMD group 0 reads tg_sums[lane] and simd_sum → one atomic per TG.
 * 4× fewer threadgroups than the old 256-thread version → 4× fewer atomics.
 * result buffer must be zeroed before dispatch.
 * ========================================================================= */
kernel void count_flag1_kernel(
    device const int   *flag1    [[buffer(0)]],
    device atomic_int  *result   [[buffer(1)]],
    constant int       &head_idx [[buffer(2)]],
    constant int       &num_octs [[buffer(3)]],
    uint gid    [[thread_position_in_grid]],
    uint lane   [[thread_index_in_simdgroup]],
    uint sg_idx [[simdgroup_index_in_threadgroup]])
{
    threadgroup int tg_sums[32];   /* 1024 / 32 = 32 SIMD groups */

    uint oct_off = gid / 8u;
    uint cell_0  = gid % 8u;
    int val = 0;
    if (int(oct_off) < num_octs) {
        int oct_abs_0 = (head_idx - 1) + int(oct_off);
        val = flag1[cell_0 + 8u * uint(oct_abs_0)];
    }

    /* Level 1: SIMD-group reduction */
    int sv = simd_sum(val);
    if (lane == 0) tg_sums[sg_idx] = sv;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    /* Level 2: SIMD group 0 reads all 32 partial sums and reduces */
    if (sg_idx == 0) {
        int blk_total = simd_sum(tg_sums[lane]);
        if (lane == 0)
            atomic_fetch_add_explicit(result, blk_total, memory_order_relaxed);
    }
}

/* =========================================================================
 * hydro_flag_kernel — gradient-based density/pressure refinement criterion.
 * 2D threadgroup: 8 cells × 16 octs.  No MHD, no GRAV.
 * For nsubgrid=1: true oct = nbor centre slot (ind=14).
 * ========================================================================= */
kernel void hydro_flag_kernel(
    device int          *flag1      [[buffer(0)]],
    device const int    *nbor       [[buffer(1)]],
    device const float  *uold       [[buffer(2)]],
    constant int        &head_idx   [[buffer(3)]],
    constant int        &num_octs   [[buffer(4)]],
    constant float      &gamma      [[buffer(5)]],
    constant float      &smallr     [[buffer(6)]],
    constant float      &smallc2    [[buffer(7)]],
    constant float      &err_grad_d [[buffer(8)]],
    constant float      &err_grad_p [[buffer(9)]],
    constant float      &floor_d    [[buffer(10)]],
    constant float      &floor_p    [[buffer(11)]],
    uint tid [[thread_position_in_threadgroup]],
    uint bid [[threadgroup_position_in_grid]])
{
    uint cell_0  = tid % 8u;
    uint oct_off = bid * (uint)FLAG_TG_OCTS + tid / 8u;
    if (int(oct_off) >= num_octs) return;

    /* sg_1 = 1-based subgrid index; for nsubgrid=1 this is the input oct */
    int sg_1      = head_idx + int(oct_off);   /* 1-based */
    int oct_abs_0 = nbor_fl(nbor, sg_1, 14) - 1;  /* true oct, 0-based */
    if (oct_abs_0 < 0) return;

    fl_conserved_t cm = fl_load(uold, int(cell_0), oct_abs_0);
    fl_primitive_t pm = fl_c2p(cm, gamma, smallr, smallc2);
    bool ok = false;

    for (int idim = 0; idim < 3; idim++) {
        int idir_l = 2 * idim;
        int idir_r = 2 * idim + 1;

        /* Grid offsets for left/right in this dimension */
        int in_l = 0, jn_l = 0, kn_l = 0;
        int in_r = 0, jn_r = 0, kn_r = 0;
        if (idim == 0) {
            in_l = iii_c[cell_0][idir_l];
            in_r = iii_c[cell_0][idir_r];
        } else if (idim == 1) {
            jn_l = iii_c[cell_0][idir_l];
            jn_r = iii_c[cell_0][idir_r];
        } else {
            kn_l = iii_c[cell_0][idir_l];
            kn_r = iii_c[cell_0][idir_r];
        }

        /* nsubgrid=1 → i=j=k=1 (center of 3×3×3 subgrid) */
        int ind_l = 1 + (1 + in_l) + NSUBGRIDP2_FL * (1 + jn_l) + NSUBGRIDP2SQ_FL * (1 + kn_l);
        int ind_r = 1 + (1 + in_r) + NSUBGRIDP2_FL * (1 + jn_r) + NSUBGRIDP2SQ_FL * (1 + kn_r);

        int src_l_1 = nbor_fl(nbor, sg_1, ind_l);
        int src_r_1 = nbor_fl(nbor, sg_1, ind_r);

        int ic_l = hhh_c[cell_0][idir_l];
        int ic_r = hhh_c[cell_0][idir_r];

        fl_conserved_t cl = (src_l_1 > 0) ? fl_load(uold, ic_l, src_l_1 - 1) : cm;
        fl_conserved_t cr = (src_r_1 > 0) ? fl_load(uold, ic_r, src_r_1 - 1) : cm;

        fl_primitive_t pl = fl_c2p(cl, gamma, smallr, smallc2);
        fl_primitive_t pr = fl_c2p(cr, gamma, smallr, smallc2);

        ok = ok || fl_hydro_crit(pl, pm, pr, err_grad_d, floor_d, err_grad_p, floor_p);
    }

    if (ok) flag1[cell_0 + 8u * uint(oct_abs_0)] = 1;
}

/* =========================================================================
 * count_neighbors_kernel — for each cell, count face-adjacent neighbours
 * that have flag1 == 1; write count to flag2[cell, true_oct].
 * 2D threadgroup: 8 cells × 16 octs.
 * Mirrors count_neighbors<<<dim3(N,1,1),dim3(8,16,1)>>> in gpu_flag.cuf.
 * ========================================================================= */
kernel void count_neighbors_kernel(
    device int        *flag2    [[buffer(0)]],
    device const int  *flag1    [[buffer(1)]],
    device const int  *nbor     [[buffer(2)]],
    constant int      &head_idx [[buffer(3)]],
    constant int      &num_octs [[buffer(4)]],
    uint tid [[thread_position_in_threadgroup]],
    uint bid [[threadgroup_position_in_grid]])
{
    uint cell_0  = tid % 8u;
    uint oct_off = bid * (uint)FLAG_TG_OCTS + tid / 8u;
    if (int(oct_off) >= num_octs) return;

    int sg_1      = head_idx + int(oct_off);
    int oct_abs_0 = nbor_fl(nbor, sg_1, 14) - 1;
    if (oct_abs_0 < 0) return;

    int count = 0;
    for (int idir = 0; idir < 6; idir++) {
        int ic    = hhh_c[cell_0][idir];
        int in = 0, jn = 0, kn = 0;
        if      (idir < 2) in = iii_c[cell_0][idir];
        else if (idir < 4) jn = iii_c[cell_0][idir];
        else               kn = iii_c[cell_0][idir];

        int ind   = 1 + (1 + in) + NSUBGRIDP2_FL * (1 + jn) + NSUBGRIDP2SQ_FL * (1 + kn);
        int src_1 = nbor_fl(nbor, sg_1, ind);
        if (src_1 > 0) count += flag1[ic + 8 * (src_1 - 1)];
    }
    flag2[cell_0 + 8 * oct_abs_0] = count;
}

/* =========================================================================
 * flag_count_kernel — if flag1==1 clear flag2; if flag2>=num_nbors set flag1=1.
 * Operates directly on octs by head_idx index (no nbor lookup needed).
 * 2D threadgroup: 8 cells × 16 octs.
 * Mirrors flag_count_kernel in gpu_flag.cuf.
 * ========================================================================= */
kernel void flag_count_kernel(
    device int        *flag1     [[buffer(0)]],
    device int        *flag2     [[buffer(1)]],
    constant int      &head_idx  [[buffer(2)]],
    constant int      &num_octs  [[buffer(3)]],
    constant int      &num_nbors [[buffer(4)]],
    uint tid [[thread_position_in_threadgroup]],
    uint bid [[threadgroup_position_in_grid]])
{
    uint cell_0  = tid % 8u;
    uint oct_off = bid * (uint)FLAG_TG_OCTS + tid / 8u;
    if (int(oct_off) >= num_octs) return;
    int oct_abs_0 = (head_idx - 1) + int(oct_off);
    int idx = int(cell_0) + 8 * oct_abs_0;

    int f1 = flag1[idx];
    int f2 = flag2[idx];
    if (f1 == 1) f2 = 0;
    if (f2 >= num_nbors) f1 = 1;
    flag1[idx] = f1;
    flag2[idx] = f2;
}

/* =========================================================================
 * enforce_rules_kernel — clear flag1 for an oct if any of its 27 subgrid
 * neighbour slots is missing (0) or in the ghost cache (> ngridmax).
 * 1D: 1 thread per oct; head_idx/num_octs at the current level.
 * Mirrors enforce_rules<<<128 threads>>> in gpu_flag.cuf.
 * ========================================================================= */
kernel void enforce_rules_kernel(
    device int        *flag1    [[buffer(0)]],
    device const int  *nbor     [[buffer(1)]],
    constant int      &head_idx [[buffer(2)]],
    constant int      &num_octs [[buffer(3)]],
    constant int      &ngridmax [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (int(gid) >= num_octs) return;

    int sg_1      = head_idx + int(gid);           /* 1-based subgrid */
    int oct_abs_0 = nbor_fl(nbor, sg_1, 14) - 1;  /* true oct, 0-based */
    if (oct_abs_0 < 0) return;

    bool ok = false;
    for (int kn = 0; kn <= 2; kn++) {
        for (int jn = 0; jn <= 2; jn++) {
            for (int in = 0; in <= 2; in++) {
                /* ind_0 in [0..26]; same as ind_1 - 1 */
                int ind_0  = in + NSUBGRIDP2_FL * jn + NSUBGRIDP2SQ_FL * kn;
                int nb_val = nbor[(sg_1 - 1) * SUBGRIDSIZE_FL + ind_0];
                ok = ok || (nb_val == 0) || (nb_val > ngridmax);
            }
        }
    }
    if (ok) {
        for (int ic = 0; ic < 8; ic++) flag1[ic + 8 * oct_abs_0] = 0;
    }
}

/* =========================================================================
 * poisson_flag_kernel — flag cells based on Jeans length or density threshold.
 * 2D threadgroup: 128 threads = 8 cells (low bits) × 16 octs (high bits).
 * ========================================================================= */
kernel void poisson_flag_kernel(
    device int          *flag1        [[buffer(0)]],
    device const float  *nref         [[buffer(1)]],
    device const float  *uold         [[buffer(2)]],
    device const float  *bold         [[buffer(3)]],
    constant int        &head_idx     [[buffer(4)]],
    constant int        &num_octs     [[buffer(5)]],
    constant float      &gamma        [[buffer(6)]],
    constant float      &smallr       [[buffer(7)]],
    constant float      &smallc2      [[buffer(8)]],
    constant float      &mass_sph     [[buffer(9)]],
    constant float      &m_refine     [[buffer(10)]],
    constant float      &jeans_refine [[buffer(11)]],
    constant float      &factG        [[buffer(12)]],
    constant float      &dx_loc       [[buffer(13)]],
    uint tid [[thread_position_in_threadgroup]],
    uint bid [[threadgroup_position_in_grid]])
{
    uint cell_0  = tid % 8u;
    uint oct_offset = bid * (uint)FLAG_TG_OCTS + tid / 8u;
    if (int(oct_offset) >= num_octs) return;
    int oct_0  = (head_idx - 1) + int(oct_offset);

#ifndef GRAV
    float vol_loc = dx_loc * dx_loc * dx_loc;
    float d_scale = mass_sph / vol_loc;
#endif

    bool ok = false;

#ifdef GRAV
    /* Refine if number of particles exceeds threshold */
    if (m_refine >= 0.0f) {
        ok = ok || (nref[cell_0 + 8 * oct_0] >= m_refine);
    }
#else
    /* Refine if gas mass exceeds threshold */
    if (mass_sph > 0.0f && m_refine >= 0.0f) {
        ok = ok || (uold[cell_0 + 8 * 0 + 8 * (NVAR) * oct_0] >= m_refine * d_scale);
    }
#endif

#ifdef HYDRO
    if (jeans_refine >= 0.0f) {
        /* Convert to primitive variables */
        int base = (int)cell_0 + 8 * (NVAR) * oct_0;
        float density    = max(uold[base], smallr);
        float momentum_x = uold[base + 8];
        float momentum_y = uold[base + 16];
        float momentum_z = uold[base + 24];
        float energy     = uold[base + 32];
        float emag       = 0.0f;
#ifdef MHD
        int b_base = (int)cell_0 + 8 * 6 * oct_0;
        for (int idim = 0; idim < 3; idim++) {
            float b_val = bold[b_base + 8 * idim] + bold[b_base + 8 * (idim + 3)];
            emag += 0.125f * b_val * b_val;
        }
#endif
        float pressure   = energy - 0.5f * (momentum_x*momentum_x + momentum_y*momentum_y + momentum_z*momentum_z) / density - emag;
        pressure   = max((gamma - 1.0f) * pressure, smallc2 * density);

        /* Compute Jeans length */
        float c_iso = sqrt(pressure / density);
        float t_ff = sqrt(3.1415926535f / density / factG);
        float jeans_length = c_iso * t_ff;

        /* Refine if Jeans length is not resolved enough */
        ok = ok || (jeans_refine * dx_loc >= jeans_length);
    }
#endif

    /* If satisfied set flag1 to 1 */
    if (ok) {
        flag1[cell_0 + 8 * oct_0] = 1;
    }
}
