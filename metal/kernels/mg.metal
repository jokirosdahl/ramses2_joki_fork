/*
 * metal/kernels/mg.metal
 *
 * Metal Shading Language port of gpu/gpu_mg.cuf mg_device module.
 * 21 kernels for the multigrid Poisson gravity solver.
 *
 * All arrays use float32 throughout (accepted precision loss for PoC).
 * "fine" variants bind AMR buffers; "mg" variants bind MG buffers.
 *
 * Buffer layout (Fortran column-major flattened):
 *   phi(cell,oct)         → phi[(oct-1)*8+(cell-1)]
 *   f(cell,idim,oct)      → f[(oct-1)*3*8+(idim-1)*8+(cell-1)]
 *   nbor(ind,sg)          → nbor[(sg-1)*27+(ind-1)]
 *   father(oct)           → father[oct-1]
 */

#include <metal_stdlib>
#include <metal_atomic>
using namespace metal;

#include "../metal_types.h"
#include "metal_utils.h"

/* ---------------------------------------------------------------------------
 * Constant lookup tables — verbatim from gpu_mg.cuf module parameters.
 * Access: hhh_mg[face-1][cell-1]  ↔  Fortran hhh(cell_idx, face)
 * (C row-major with first index = face-1 maps the same flat bytes as
 *  Fortran column-major with first index = cell, second = face.)
 * --------------------------------------------------------------------------*/

constant int hhh_mg[6][8] = {
    {2, 1, 4, 3, 6, 5, 8, 7},
    {2, 1, 4, 3, 6, 5, 8, 7},
    {3, 4, 1, 2, 7, 8, 5, 6},
    {3, 4, 1, 2, 7, 8, 5, 6},
    {5, 6, 7, 8, 1, 2, 3, 4},
    {5, 6, 7, 8, 1, 2, 3, 4}
};

constant int iii_mg[6][8] = {
    {-1, 0,-1, 0,-1, 0,-1, 0},
    { 0, 1, 0, 1, 0, 1, 0, 1},
    {-1,-1, 0, 0,-1,-1, 0, 0},
    { 0, 0, 1, 1, 0, 0, 1, 1},
    {-1,-1,-1,-1, 0, 0, 0, 0},
    { 0, 0, 0, 0, 1, 1, 1, 1}
};

constant float bbb_mg[8] = {
    1.0f/64.0f, 3.0f/64.0f, 3.0f/64.0f, 9.0f/64.0f,
    3.0f/64.0f, 9.0f/64.0f, 9.0f/64.0f, 27.0f/64.0f
};

/* ccc_mg[cell_idx-1][ind_average-1]  ↔  Fortran ccc(ind_average, cell_idx) */
constant int ccc_mg[8][8] = {
    { 1,  2,  4,  5, 10, 11, 13, 14},
    { 3,  2,  6,  5, 12, 11, 15, 14},
    { 7,  8,  4,  5, 16, 17, 13, 14},
    { 9,  8,  6,  5, 18, 17, 15, 14},
    {19, 20, 22, 23, 10, 11, 13, 14},
    {21, 20, 24, 23, 12, 11, 15, 14},
    {25, 26, 22, 23, 16, 17, 13, 14},
    {27, 26, 24, 23, 18, 17, 15, 14}
};

/* gg?_mg[idim-1][cell-1] ↔ Fortran gg?(cell,idim) */
constant int gg1_mg[3][8] = {
    {1,0,1,0,1,0,1,0},
    {3,3,0,0,3,3,0,0},
    {5,5,5,5,0,0,0,0}
};
constant int gg2_mg[3][8] = {
    {0,2,0,2,0,2,0,2},
    {0,0,4,4,0,0,4,4},
    {0,0,0,0,6,6,6,6}
};
constant int gg3_mg[3][8] = {
    {1,1,1,1,1,1,1,1},
    {3,3,3,3,3,3,3,3},
    {5,5,5,5,5,5,5,5}
};
constant int gg4_mg[3][8] = {
    {2,2,2,2,2,2,2,2},
    {4,4,4,4,4,4,4,4},
    {6,6,6,6,6,6,6,6}
};

/* hh?_mg[idim-1][cell-1] ↔ Fortran hh?(cell,idim) */
constant int hh1_mg[3][8] = {
    {2,1,4,3,6,5,8,7},
    {3,4,1,2,7,8,5,6},
    {5,6,7,8,1,2,3,4}
};
constant int hh2_mg[3][8] = {
    {2,1,4,3,6,5,8,7},
    {3,4,1,2,7,8,5,6},
    {5,6,7,8,1,2,3,4}
};
constant int hh3_mg[3][8] = {
    {1,2,3,4,5,6,7,8},
    {1,2,3,4,5,6,7,8},
    {1,2,3,4,5,6,7,8}
};
constant int hh4_mg[3][8] = {
    {1,2,3,4,5,6,7,8},
    {1,2,3,4,5,6,7,8},
    {1,2,3,4,5,6,7,8}
};

constant int ired_mg[4]   = {1, 4, 6, 7};
constant int iblack_mg[4] = {2, 3, 5, 8};

/* ---------------------------------------------------------------------------
 * Array access helpers
 * --------------------------------------------------------------------------*/

static inline float phi_get_mg(device const float *p, int oct_1, int cell_1) {
    return p[(oct_1-1)*8 + (cell_1-1)];
}
static inline void phi_set_mg(device float *p, int oct_1, int cell_1, float v) {
    p[(oct_1-1)*8 + (cell_1-1)] = v;
}
/* f(cell,idim,oct) layout: stride = 8 per idim, 3*8=24 per oct */
static inline float f_get_mg(device const float *f, int oct_1, int idim_1, int cell_1) {
    return f[(oct_1-1)*24 + (idim_1-1)*8 + (cell_1-1)];
}
static inline void f_set_mg(device float *f, int oct_1, int idim_1, int cell_1, float v) {
    f[(oct_1-1)*24 + (idim_1-1)*8 + (cell_1-1)] = v;
}
static inline float rho_get_mg(device const float *r, int oct_1, int cell_1) {
    return r[(oct_1-1)*8 + (cell_1-1)];
}

/* FNV-1a hash + linear probing — see metal_utils.h for hash_get / fnv64. */

/* ---------------------------------------------------------------------------
 * floor_div2 helper (mirrors Fortran: for n<0: (n-1)/2)
 * --------------------------------------------------------------------------*/
static inline int floor_div2_mg(int n) {
    return (n >= 0) ? n/2 : (n-1)/2;
}

/* ---------------------------------------------------------------------------
 * nbor_father_cells: find 27 neighboring father cells (AMR nbor).
 * Specialised for NSUBGRID=1: nsubgridtondim=1, nsubgridp2=3, nsubgridp2sq=9.
 * father_idx = father[oct_idx-1] (caller reads the appropriate father array).
 * --------------------------------------------------------------------------*/
static inline void nbor_father_cells_impl(
    device const oct_t *grid,
    device const int   *nbor,
    int father_idx,
    int oct_idx,
    thread int igrid_nbor[27],
    thread int icell_nbor[27])
{
    /* For NSUBGRID=1: subgrid_idx == father_idx, i_subgrid=j_subgrid=k_subgrid=1 */
    int subgrid_idx = father_idx;

    int ck0 = grid[oct_idx-1].ckey[0];
    int ck1 = grid[oct_idx-1].ckey[1];
    int ck2 = grid[oct_idx-1].ckey[2];
    int i = ck0 - 2*(ck0/2);
    int j = ck1 - 2*(ck1/2);
    int k = ck2 - 2*(ck2/2);

    int inbor = 0;
    for (int kk = -1; kk <= 1; kk++) {
        for (int jj = -1; jj <= 1; jj++) {
            for (int ii = -1; ii <= 1; ii++) {
                int in = i + ii;  /* i_subgrid=1 → 2*(i_subgrid-1)=0 */
                int jn = j + jj;
                int kn = k + kk;
                int io = floor_div2_mg(in);
                int jo = floor_div2_mg(jn);
                int ko = floor_div2_mg(kn);
                int ic = in - 2*io;
                int jc = jn - 2*jo;
                int kc = kn - 2*ko;
                /* ind_nbor = 1 + (io+1) + (jo+1)*3 + (ko+1)*9 */
                int ind_nbor = 1 + (io+1) + (jo+1)*3 + (ko+1)*9;
                igrid_nbor[inbor] = nbor[(subgrid_idx-1)*27 + (ind_nbor-1)];
                icell_nbor[inbor] = 1 + ic + jc*2 + kc*4;
                inbor++;
            }
        }
    }
}

static inline void nbor_father_cells_amr(
    device const oct_t *grid,
    device const int   *father,
    device const int   *nbor,
    int oct_idx,
    thread int igrid_nbor[27],
    thread int icell_nbor[27])
{
    nbor_father_cells_impl(grid, nbor, father[oct_idx-1], oct_idx, igrid_nbor, icell_nbor);
}

static inline void nbor_father_cells_mg_fn(
    device const oct_t *grid,
    device const int   *father,
    device const int   *nbor,
    int oct_idx,
    int mg_idx,
    thread int igrid_nbor[27],
    thread int icell_nbor[27])
{
    nbor_father_cells_impl(grid, nbor, father[mg_idx-1], oct_idx, igrid_nbor, icell_nbor);
}

/* ===========================================================================
 * Kernel 1: save_phi_old — copy phi → phi_old.
 * Thread layout: 2D {8, 16}.
 * ========================================================================= */
kernel void save_phi_old_kernel(
    device float       *phi_old  [[buffer(0)]],
    device const float *phi      [[buffer(1)]],
    constant int       &head_idx [[buffer(2)]],
    constant int       &num_octs [[buffer(3)]],
    uint2 tptg [[threads_per_threadgroup]],
    uint2 tgpi [[threadgroup_position_in_grid]],
    uint2 tpg  [[thread_position_in_threadgroup]])
{
    int oct_off = (int)(tgpi.x * tptg.y + tpg.y);
    if (oct_off >= num_octs) return;
    int oct = head_idx + oct_off;
    int c   = (int)tpg.x + 1;
    phi_set_mg(phi_old, oct, c, phi_get_mg(phi, oct, c));
}

/* ===========================================================================
 * Kernel 2: reset_phi_kernel_mg — zero phi and f[1..3] for AMR or MG grid.
 * Thread layout: 2D {8, 16}.
 * ========================================================================= */
kernel void reset_phi_kernel_mg(
    device float *phi     [[buffer(0)]],
    device float *f       [[buffer(1)]],
    constant int &head_idx [[buffer(2)]],
    constant int &num_octs [[buffer(3)]],
    uint2 tptg [[threads_per_threadgroup]],
    uint2 tgpi [[threadgroup_position_in_grid]],
    uint2 tpg  [[thread_position_in_threadgroup]])
{
    int oct_off = (int)(tgpi.x * tptg.y + tpg.y);
    if (oct_off >= num_octs) return;
    int oct = head_idx + oct_off;
    int c   = (int)tpg.x + 1;
    phi_set_mg(phi, oct, c, 0.0f);
    f_set_mg(f, oct, 1, c, 0.0f);
    f_set_mg(f, oct, 2, c, 0.0f);
    f_set_mg(f, oct, 3, c, 0.0f);
}

/* ===========================================================================
 * Kernel 3: make_initial_phi_kernel — CIC interpolation + time extrapolation.
 * Thread layout: 2D {8, 16}.
 * Uses AMR father and nbor.
 * ========================================================================= */
kernel void make_initial_phi_kernel(
    device const oct_t *grid      [[buffer(0)]],
    device const int   *father    [[buffer(1)]],
    device const int   *nbor      [[buffer(2)]],
    device float       *phi       [[buffer(3)]],
    device const float *phi_old   [[buffer(4)]],
    constant float     &tfrac     [[buffer(5)]],
    constant int       &head_idx  [[buffer(6)]],
    constant int       &num_octs  [[buffer(7)]],
    constant int       &ngridmax  [[buffer(8)]],
    uint2 tptg [[threads_per_threadgroup]],
    uint2 tgpi [[threadgroup_position_in_grid]],
    uint2 tpg  [[thread_position_in_threadgroup]])
{
    int oct_off = (int)(tgpi.x * tptg.y + tpg.y);
    if (oct_off >= num_octs) return;
    int oct_idx  = head_idx + oct_off;
    int cell_idx = (int)tpg.x + 1;

    thread int igrid_nbor[27], icell_nbor[27];
    nbor_father_cells_amr(grid, father, nbor, oct_idx, igrid_nbor, icell_nbor);

    int igrid_cen = igrid_nbor[13];  /* index 14 in 1-based = index 13 here */
    int ind_cen   = icell_nbor[13];

    float acc = 0.0f;
    for (int ind_average = 0; ind_average < 8; ind_average++) {
        int ind_father = ccc_mg[cell_idx-1][ind_average] - 1;  /* 0-based */
        float coeff    = bbb_mg[ind_average];
        int igrid_nbr  = igrid_nbor[ind_father];
        int ind_nbr    = icell_nbor[ind_father];
        if (igrid_nbr == 0 || igrid_nbr > ngridmax) {
            igrid_nbr = igrid_cen;
            ind_nbr   = ind_cen;
        }
        float ph  = phi_get_mg(phi,     igrid_nbr, ind_nbr);
        float pho = phi_get_mg(phi_old, igrid_nbr, ind_nbr);
        acc += coeff * (ph + (ph - pho) * tfrac);
    }
    phi_set_mg(phi, oct_idx, cell_idx, acc);
}

/* ===========================================================================
 * Kernel 4: reset_mask_kernel — set f(cell,3,oct) to mask_val.
 * Thread layout: 2D {8, 16}.
 * ========================================================================= */
kernel void reset_mask_kernel_mg(
    device float  *f        [[buffer(0)]],
    constant int  &head_idx [[buffer(1)]],
    constant int  &num_octs [[buffer(2)]],
    constant float &mask_val [[buffer(3)]],
    uint2 tptg [[threads_per_threadgroup]],
    uint2 tgpi [[threadgroup_position_in_grid]],
    uint2 tpg  [[thread_position_in_threadgroup]])
{
    int oct_off = (int)(tgpi.x * tptg.y + tpg.y);
    if (oct_off >= num_octs) return;
    int oct = head_idx + oct_off;
    int c   = (int)tpg.x + 1;
    f_set_mg(f, oct, 3, c, mask_val);
}

/* ===========================================================================
 * Kernel 5: reset_rhs_kernel — compute RHS: f2 = fourpi*(rho-offset),
 *           apply Dirichlet BC correction near mask boundary.
 * Thread layout: 2D {8, 16}.
 * ========================================================================= */
kernel void reset_rhs_kernel_mg(
    device const float *phi       [[buffer(0)]],
    device const float *rho       [[buffer(1)]],
    device float       *f         [[buffer(2)]],
    device const int   *nbor      [[buffer(3)]],
    constant int       &head_idx  [[buffer(4)]],
    constant int       &num_octs  [[buffer(5)]],
    constant int       &ngridmax  [[buffer(6)]],
    constant float     &fourpi    [[buffer(7)]],
    constant float     &offset    [[buffer(8)]],
    constant float     &oneoverdx2 [[buffer(9)]],
    uint2 tptg [[threads_per_threadgroup]],
    uint2 tgpi [[threadgroup_position_in_grid]],
    uint2 tpg  [[thread_position_in_threadgroup]])
{
    int oct_off = (int)(tgpi.x * tptg.y + tpg.y);
    if (oct_off >= num_octs) return;
    int oct_idx  = head_idx + oct_off;
    int cell_idx = (int)tpg.x + 1;

    /* For NSUBGRID=1: subgrid_idx == oct_idx, i=j=k=1 → ind_nbor = 1+1+3+9 = 14 */
    int subgrid_idx = oct_idx;
    int i = 1, j = 1, k = 1;
    int ind_nbor = 1 + i + 3*j + 9*k;  /* = 14 for NSUBGRID=1 */
    int oct_true = nbor[(subgrid_idx-1)*27 + (ind_nbor-1)];

    f_set_mg(f, oct_true, 2, cell_idx,
             fourpi * (rho_get_mg(rho, oct_true, cell_idx) - offset));

    for (int idim = 1; idim <= 3; idim++) {
        for (int inbor = 1; inbor <= 2; inbor++) {
            int face = 2*(idim-1) + inbor - 1;  /* 0-based into iii/hhh */
            int in = 0, jn = 0, kn = 0;
            if      (idim == 1) in = iii_mg[face][cell_idx-1];
            else if (idim == 2) jn = iii_mg[face][cell_idx-1];
            else                kn = iii_mg[face][cell_idx-1];
            int ind_n = 1 + (i+in) + 3*(j+jn) + 9*(k+kn);
            int oct_nbr_idx = nbor[(subgrid_idx-1)*27 + (ind_n-1)];
            int cell_nbr_idx = hhh_mg[face][cell_idx-1];

            float phi_nbr;
            float dis_nbr;
            if (oct_nbr_idx > 0) {
                phi_nbr = phi_get_mg(phi, oct_nbr_idx, cell_nbr_idx);
                if (oct_nbr_idx <= ngridmax) {
                    dis_nbr = f_get_mg(f, oct_nbr_idx, 3, cell_nbr_idx);
                } else {
                    dis_nbr = -1.0f;
                }
            } else {
                phi_nbr = 0.0f;
                dis_nbr = -1.0f;
            }

            float phi_c = phi_get_mg(phi, oct_true, cell_idx);
            float dis_c = f_get_mg(f, oct_true, 3, cell_idx);
            if (dis_nbr <= 0.0f && dis_c > 0.0f) {
                float w   = dis_nbr / (dis_nbr - dis_c);
                float phi_b = (1.0f-w)*phi_nbr + w*phi_c;
                f_set_mg(f, oct_true, 2, cell_idx,
                         f_get_mg(f,oct_true,2,cell_idx) - 2.0f*oneoverdx2*phi_b);
            }
        }
    }
}

/* ===========================================================================
 * Kernel 6: update_father_array_kernel — find parent oct in MG hash table.
 * Thread layout: 1D 128.
 * Uses MG hash (hash_key_mg, hash_val_mg) and ckey_max/key_off from device.
 * "fine" variant: reads grid[oct_idx] for ckey/lev.
 * "mg"   variant: reads grid_mg[oct_idx] for ckey/lev.  (same kernel, different buffer binding)
 * ========================================================================= */
kernel void update_father_array_kernel(
    device int         *father_mg    [[buffer(0)]],
    device const oct_t *grid_src     [[buffer(1)]],  /* AMR or MG grid */
    device const long  *hash_key_mg  [[buffer(2)]],
    device const int   *hash_val_mg  [[buffer(3)]],
    device const int   *ckey_max_dev [[buffer(4)]],
    device const long  *key_off_dev  [[buffer(5)]],
    constant int       &hash_size_mg [[buffer(6)]],
    constant int       &head_idx     [[buffer(7)]],
    constant int       &head_father  [[buffer(8)]],
    constant int       &num_octs     [[buffer(9)]],
    uint tid [[thread_position_in_grid]])
{
    int oct_off = (int)tid;
    if (oct_off >= num_octs) return;
    int mg_idx  = head_father + oct_off;
    int oct_idx = head_idx    + oct_off;

    int ilevel = grid_src[oct_idx-1].lev;
    int ck0    = grid_src[oct_idx-1].ckey[0];
    int ck1    = grid_src[oct_idx-1].ckey[1];
    int ck2    = grid_src[oct_idx-1].ckey[2];

    /* Father oct Cartesian key (one level coarser) */
    int fl = ilevel - 1;
    long nx     = (long)ckey_max_dev[fl - 1];
    long offset = key_off_dev[fl - 1];
    long fck0   = (long)(ck0/2);
    long fck1   = (long)(ck1/2);
    long fck2   = (long)(ck2/2);
    long key    = offset + fck0 + fck1*nx + fck2*nx*nx;

    father_mg[mg_idx-1] = hash_get(hash_key_mg, hash_val_mg, hash_size_mg, key);
}

/* ===========================================================================
 * Kernel 7: init_prefix_sum_mg_kernel — mark first oct per parent group.
 * Thread layout: 1D 128.
 * "fine" variant binds s_grid; "mg" variant binds s_grid_mg. (same kernel)
 * ========================================================================= */
kernel void init_prefix_sum_mg_kernel(
    device const oct_t *grid       [[buffer(0)]],
    device int         *prefix_sum [[buffer(1)]],
    constant int       &head_idx   [[buffer(2)]],
    constant int       &num_octs   [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    int oct_off = (int)tid;
    if (oct_off >= num_octs) return;
    int oct_idx = head_idx + oct_off;
    int val;
    if (oct_idx == head_idx) {
        val = 1;
    } else {
        val = (grid[oct_idx-1].hkey / 8L != grid[oct_idx-2].hkey / 8L) ? 1 : 0;
    }
    prefix_sum[oct_idx-1] = val;
}

/* ===========================================================================
 * Kernel 8: compute_father_swap_kernel — build swap table from prefix sums.
 * Thread layout: 1D 128.
 * ========================================================================= */
kernel void compute_father_swap_kernel(
    device int         *swap_local  [[buffer(0)]],
    device const int   *prefix_sum  [[buffer(1)]],
    constant int       &head_idx    [[buffer(2)]],
    constant int       &num_octs    [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    int oct_off = (int)tid;
    if (oct_off >= num_octs) return;
    int oct_idx = head_idx + oct_off;

    int prev_sum = (oct_idx > head_idx) ? prefix_sum[oct_idx-2] : 0;
    int bit      = prefix_sum[oct_idx-1] - prev_sum;
    if (bit == 1) {
        int father_idx     = prev_sum + 1;
        swap_local[father_idx-1] = oct_idx;
    }
}

/* ===========================================================================
 * Kernel 9: make_father_octs_kernel — build new MG parent octs.
 * Thread layout: 1D 128.
 * "fine" variant: reads ckey/lev from s_grid; "mg": from s_grid_mg. (same kernel)
 * ========================================================================= */
kernel void make_father_octs_kernel(
    device const oct_t *grid_src    [[buffer(0)]],
    device oct_t       *grid_mg     [[buffer(1)]],
    device float       *phi_mg      [[buffer(2)]],
    device float       *f_mg        [[buffer(3)]],
    device const int   *swap_local  [[buffer(4)]],
    constant int       &head_mg     [[buffer(5)]],
    constant int       &new_noct    [[buffer(6)]],
    uint tid [[thread_position_in_grid]])
{
    int father_off = (int)tid;
    if (father_off >= new_noct) return;

    int oct_idx    = swap_local[father_off];  /* 1-based original oct */
    int father_idx = head_mg + father_off;     /* 1-based MG target   */

    int ilevel = grid_src[oct_idx-1].lev;
    int ck0    = grid_src[oct_idx-1].ckey[0];
    int ck1    = grid_src[oct_idx-1].ckey[1];
    int ck2    = grid_src[oct_idx-1].ckey[2];

    grid_mg[father_idx-1].lev      = ilevel - 1;
    grid_mg[father_idx-1].ckey[0]  = ck0/2;
    grid_mg[father_idx-1].ckey[1]  = ck1/2;
    grid_mg[father_idx-1].ckey[2]  = ck2/2;
    for (int c = 0; c < 8; c++) grid_mg[father_idx-1].refined[c] = 0;
    grid_mg[father_idx-1].hkey     = grid_src[oct_idx-1].hkey / 8L;

    for (int c = 1; c <= 8; c++) {
        phi_set_mg(phi_mg, father_idx, c, 0.0f);
        f_set_mg(f_mg, father_idx, 1, c, 0.0f);
        f_set_mg(f_mg, father_idx, 2, c, 0.0f);
        f_set_mg(f_mg, father_idx, 3, c, 0.0f);
    }
}

/* ===========================================================================
 * Kernel 10: restrict_mask_kernel — average fine mask up to coarse level.
 * Thread layout: 1D 128.
 * "fine": reads s_grid + s_f → s_f_mg.
 * "mg":   reads s_grid_mg + s_f_mg → s_f_mg.
 * ========================================================================= */
kernel void restrict_mask_kernel_mg(
    device const oct_t *grid        [[buffer(0)]],
    device const int   *father_mg   [[buffer(1)]],
    device const float *f_fine      [[buffer(2)]],
    device float       *f_mg        [[buffer(3)]],
    constant int       &head_idx    [[buffer(4)]],
    constant int       &head_father [[buffer(5)]],
    constant int       &num_octs    [[buffer(6)]],
    uint tid [[thread_position_in_grid]])
{
    int oct_off = (int)tid;
    if (oct_off >= num_octs) return;
    int mg_idx  = head_father + oct_off;
    int oct_idx = head_idx    + oct_off;

    int father_idx = father_mg[mg_idx-1];
    int ck0 = grid[oct_idx-1].ckey[0];
    int ck1 = grid[oct_idx-1].ckey[1];
    int ck2 = grid[oct_idx-1].ckey[2];
    int pi  = ck0 - 2*(ck0/2);
    int pj  = ck1 - 2*(ck1/2);
    int pk  = ck2 - 2*(ck2/2);
    int cell_idx = 1 + pi + 2*pj + 4*pk;

    float sum_mask = 0.0f;
    for (int ind = 1; ind <= 8; ind++) {
        sum_mask += (1.0f + f_get_mg(f_fine, oct_idx, 3, ind)) * 0.5f / 8.0f;
    }
    f_set_mg(f_mg, father_idx, 3, cell_idx, sum_mask);
}

/* ===========================================================================
 * Kernel 11: volume_to_mask_kernel — convert accumulated count to signed mask.
 * Thread layout: 1D 256 (8 cells per oct, 32 octs per TG).
 * Modifies f(cell,3,oct) in-place; returns max into scalar buffer.
 * ========================================================================= */
kernel void volume_to_mask_kernel(
    device float       *f        [[buffer(0)]],
    device float       *mask_max [[buffer(1)]],   /* pre-zeroed float[1] */
    constant int       &head_idx [[buffer(2)]],
    constant int       &num_octs [[buffer(3)]],
    uint tid  [[thread_position_in_grid]],
    uint ltid [[thread_position_in_threadgroup]])
{
    threadgroup float tg_sums[8];

    int oct_local = (int)(tid / 8u);
    int cell_idx  = (int)(tid % 8u) + 1;

    float mask_loc = -1.0f;
    if (oct_local < num_octs) {
        int oct_idx = head_idx + oct_local;
        float v = 2.0f * f_get_mg(f, oct_idx, 3, cell_idx) - 1.0f;
        f_set_mg(f, oct_idx, 3, cell_idx, v);
        mask_loc = v;
    }

    mask_loc = tg_reduce_max_f(mask_loc, ltid, tg_sums);
    if (ltid == 0u)
        atomic_max_float((device atomic_uint*)mask_max, mask_loc);
}

/* ===========================================================================
 * Shared inner loop for cmp_residual and gauss_seidel.
 * For NSUBGRID=1: subgrid_idx == oct_idx, i=j=k=1, ind_nbor=14 → oct_true from nbor.
 * --------------------------------------------------------------------------*/
static inline void mg_get_subgrid_true(device const int *nbor, int oct_idx,
                                        thread int &subgrid_idx, thread int &oct_true,
                                        thread int &i, thread int &j, thread int &k) {
    subgrid_idx = oct_idx;
    i = 1; j = 1; k = 1;
    int ind_nbor = 1 + i + 3*j + 9*k;
    oct_true = nbor[(subgrid_idx-1)*27 + (ind_nbor-1)];
}

/* ===========================================================================
 * Kernel 12: cmp_residual_kernel — compute Laplacian residual.
 * Thread layout: 2D {8, 16}.
 * ========================================================================= */
kernel void cmp_residual_kernel(
    device const float *phi       [[buffer(0)]],
    device float       *f         [[buffer(1)]],
    device const int   *nbor      [[buffer(2)]],
    constant int       &head_idx  [[buffer(3)]],
    constant int       &num_octs  [[buffer(4)]],
    constant int       &ngridmax  [[buffer(5)]],
    constant float     &fourpi    [[buffer(6)]],
    constant float     &offset    [[buffer(7)]],
    constant float     &oneoverdx2 [[buffer(8)]],
    uint2 tptg [[threads_per_threadgroup]],
    uint2 tgpi [[threadgroup_position_in_grid]],
    uint2 tpg  [[thread_position_in_threadgroup]])
{
    int oct_off  = (int)(tgpi.x * tptg.y + tpg.y);
    if (oct_off >= num_octs) return;
    int oct_idx  = head_idx + oct_off;
    int cell_idx = (int)tpg.x + 1;

    int sg, oct_true, i, j, k;
    mg_get_subgrid_true(nbor, oct_idx, sg, oct_true, i, j, k);

    float phi_c = phi_get_mg(phi, oct_true, cell_idx);
    float dis_c = f_get_mg(f, oct_true, 3, cell_idx);

    if (dis_c > 0.0f) {
        float nb_sum = 0.0f;
        for (int idim = 1; idim <= 3; idim++) {
            for (int inbor = 1; inbor <= 2; inbor++) {
                int face = 2*(idim-1) + inbor - 1;
                int in = 0, jn = 0, kn = 0;
                if      (idim==1) in = iii_mg[face][cell_idx-1];
                else if (idim==2) jn = iii_mg[face][cell_idx-1];
                else              kn = iii_mg[face][cell_idx-1];
                int ind_n = 1 + (i+in) + 3*(j+jn) + 9*(k+kn);
                int oct_nbr_idx  = nbor[(sg-1)*27+(ind_n-1)];
                int cell_nbr_idx = hhh_mg[face][cell_idx-1];

                float phi_nbr, dis_nbr;
                if (oct_nbr_idx > 0 && oct_nbr_idx <= ngridmax) {
                    phi_nbr = phi_get_mg(phi, oct_nbr_idx, cell_nbr_idx);
                    dis_nbr = f_get_mg(f, oct_nbr_idx, 3, cell_nbr_idx);
                } else {
                    phi_nbr = 0.0f; dis_nbr = -1.0f;
                }
                if (dis_nbr <= 0.0f)
                    nb_sum += phi_c * dis_nbr / dis_c;
                else
                    nb_sum += phi_nbr;
            }
        }
        f_set_mg(f, oct_true, 1, cell_idx,
                 -oneoverdx2*(nb_sum - 6.0f*phi_c) + f_get_mg(f,oct_true,2,cell_idx));
    } else {
        f_set_mg(f, oct_true, 1, cell_idx, 0.0f);
    }
}

/* ===========================================================================
 * Kernel 13: gauss_seidel_kernel — red-black Gauss-Seidel smoother.
 * Thread layout: 2D {4, 32}.
 * ========================================================================= */
kernel void gauss_seidel_kernel(
    device float       *phi       [[buffer(0)]],
    device const float *f         [[buffer(1)]],
    device const int   *nbor      [[buffer(2)]],
    constant int       &head_idx  [[buffer(3)]],
    constant int       &num_octs  [[buffer(4)]],
    constant int       &ngridmax  [[buffer(5)]],
    constant float     &dx2       [[buffer(6)]],
    constant int       &safe_i    [[buffer(7)]],
    constant int       &redstep_i [[buffer(8)]],
    uint2 tptg [[threads_per_threadgroup]],
    uint2 tgpi [[threadgroup_position_in_grid]],
    uint2 tpg  [[thread_position_in_threadgroup]])
{
    int oct_off = (int)(tgpi.x * tptg.y + tpg.y);
    if (oct_off >= num_octs) return;
    int oct_idx = head_idx + oct_off;

    bool safe    = (safe_i    != 0);
    bool redstep = (redstep_i != 0);
    int cell_idx = redstep ? ired_mg[tpg.x] : iblack_mg[tpg.x];

    int sg, oct_true, i, j, k;
    mg_get_subgrid_true(nbor, oct_idx, sg, oct_true, i, j, k);

    float dis_c = f_get_mg(f, oct_true, 3, cell_idx);

    if (dis_c > 0.0f && (!safe || dis_c >= 1.0f)) {
        float nb_sum = 0.0f;
        float weight = 0.0f;
        for (int idim = 1; idim <= 3; idim++) {
            for (int inbor = 1; inbor <= 2; inbor++) {
                int face = 2*(idim-1)+inbor-1;
                int in = 0, jn = 0, kn = 0;
                if      (idim==1) in = iii_mg[face][cell_idx-1];
                else if (idim==2) jn = iii_mg[face][cell_idx-1];
                else              kn = iii_mg[face][cell_idx-1];
                int ind_n = 1+(i+in)+3*(j+jn)+9*(k+kn);
                int oct_nbr_idx  = nbor[(sg-1)*27+(ind_n-1)];
                int cell_nbr_idx = hhh_mg[face][cell_idx-1];

                float phi_nbr, dis_nbr;
                if (oct_nbr_idx > 0 && oct_nbr_idx <= ngridmax) {
                    phi_nbr = phi_get_mg(phi, oct_nbr_idx, cell_nbr_idx);
                    dis_nbr = f_get_mg(f, oct_nbr_idx, 3, cell_nbr_idx);
                } else {
                    phi_nbr = 0.0f; dis_nbr = -1.0f;
                }
                if (dis_nbr <= 0.0f)
                    weight += dis_nbr / dis_c;
                else
                    nb_sum += phi_nbr;
            }
        }
        phi_set_mg(phi, oct_true, cell_idx,
                   (nb_sum - dx2*f_get_mg(f,oct_true,2,cell_idx)) / (6.0f - weight));
    }
}

/* ===========================================================================
 * Kernel 14: reset_phi_val_kernel — set phi(cell,oct) to a scalar value.
 * Thread layout: 2D {8, 16}.
 * ========================================================================= */
kernel void reset_phi_val_kernel(
    device float  *phi      [[buffer(0)]],
    constant int  &head_idx [[buffer(1)]],
    constant int  &num_octs [[buffer(2)]],
    constant float &phi_val  [[buffer(3)]],
    uint2 tptg [[threads_per_threadgroup]],
    uint2 tgpi [[threadgroup_position_in_grid]],
    uint2 tpg  [[thread_position_in_threadgroup]])
{
    int oct_off = (int)(tgpi.x * tptg.y + tpg.y);
    if (oct_off >= num_octs) return;
    int oct = head_idx + oct_off;
    int c   = (int)tpg.x + 1;
    phi_set_mg(phi, oct, c, phi_val);
}

/* ===========================================================================
 * Kernel 15: restrict_residual_kernel — restrict f1 from fine to coarse.
 * Thread layout: 1D 128.
 * ========================================================================= */
kernel void restrict_residual_kernel(
    device const oct_t *grid        [[buffer(0)]],
    device const int   *father_mg   [[buffer(1)]],
    device const float *f_fine      [[buffer(2)]],
    device float       *f_mg        [[buffer(3)]],
    constant int       &head_idx    [[buffer(4)]],
    constant int       &head_father [[buffer(5)]],
    constant int       &num_octs    [[buffer(6)]],
    uint tid [[thread_position_in_grid]])
{
    int oct_off = (int)tid;
    if (oct_off >= num_octs) return;
    int mg_idx  = head_father + oct_off;
    int oct_idx = head_idx    + oct_off;

    int father_idx = father_mg[mg_idx-1];
    int ck0 = grid[oct_idx-1].ckey[0];
    int ck1 = grid[oct_idx-1].ckey[1];
    int ck2 = grid[oct_idx-1].ckey[2];
    int pi  = ck0 - 2*(ck0/2);
    int pj  = ck1 - 2*(ck1/2);
    int pk  = ck2 - 2*(ck2/2);
    int cell_idx = 1 + pi + 2*pj + 4*pk;

    float dis_father = f_get_mg(f_mg, father_idx, 3, cell_idx);
    float local_sum  = 0.0f;
    if (dis_father > 0.0f) {
        for (int ind = 1; ind <= 8; ind++) {
            if (f_get_mg(f_fine, oct_idx, 3, ind) > 0.0f)
                local_sum += f_get_mg(f_fine, oct_idx, 1, ind);
        }
    }
    f_set_mg(f_mg, father_idx, 2, cell_idx, local_sum / 8.0f);
}

/* ===========================================================================
 * Kernel 16: interpolate_correct_kernel — CIC interpolation of coarse
 *             correction and add to fine grid.
 * Thread layout: 2D {8, 16}.
 * Shared memory: igrid_nbor[27*16], icell_nbor[27*16].
 * ========================================================================= */
kernel void interpolate_correct_kernel(
    device const oct_t *grid        [[buffer(0)]],
    device const int   *father_mg   [[buffer(1)]],
    device const int   *nbor        [[buffer(2)]],
    device float       *phi         [[buffer(3)]],
    device const float *phi_mg      [[buffer(4)]],
    device const float *f           [[buffer(5)]],
    constant int       &head_idx    [[buffer(6)]],
    constant int       &head_father [[buffer(7)]],
    constant int       &num_octs    [[buffer(8)]],
    uint2 tptg [[threads_per_threadgroup]],
    uint2 tgpi [[threadgroup_position_in_grid]],
    uint2 tpg  [[thread_position_in_threadgroup]],
    threadgroup int    *igrid_nbor  [[threadgroup(0)]],
    threadgroup int    *icell_nbor  [[threadgroup(1)]])
{
    int oct_off  = (int)(tgpi.x * tptg.y + tpg.y);
    int mg_idx   = head_father + oct_off;
    int cell_idx = (int)tpg.x + 1;

    /* Thread x==0 loads nbor data for this oct into threadgroup memory */
    if (tpg.x == 0u && oct_off < num_octs) {
        int oct_idx = head_idx + oct_off;
        thread int tg_ig[27], tg_ic[27];
        nbor_father_cells_mg_fn(grid, father_mg, nbor, oct_idx, mg_idx, tg_ig, tg_ic);
        for (int n = 0; n < 27; n++) {
            igrid_nbor[tpg.y * 27 + n] = tg_ig[n];
            icell_nbor[tpg.y * 27 + n] = tg_ic[n];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (oct_off >= num_octs) return;
    int oct_idx = head_idx + oct_off;

    float corr = 0.0f;
    if (f_get_mg(f, oct_idx, 3, cell_idx) > 0.0f) {
        for (int ind_average = 0; ind_average < 8; ind_average++) {
            int ind_father = ccc_mg[cell_idx-1][ind_average] - 1;  /* 0-based */
            float coeff    = bbb_mg[ind_average];
            int igrid_nbr  = igrid_nbor[tpg.y * 27 + ind_father];
            int ind_nbr    = icell_nbor[tpg.y * 27 + ind_father];
            if (igrid_nbr > 0)
                corr += coeff * phi_get_mg(phi_mg, igrid_nbr, ind_nbr);
        }
    }
    phi_set_mg(phi, oct_idx, cell_idx, phi_get_mg(phi, oct_idx, cell_idx) + corr);
}

/* ===========================================================================
 * Kernel 17: residual_norm_kernel — sum squares of residual over inner cells.
 * Thread layout: 1D 256.
 * ========================================================================= */
kernel void residual_norm_kernel(
    device const float *f        [[buffer(0)]],
    device float       *norm_tot [[buffer(1)]],
    constant int       &head_idx [[buffer(2)]],
    constant int       &num_octs [[buffer(3)]],
    uint tid  [[thread_position_in_grid]],
    uint ltid [[thread_position_in_threadgroup]])
{
    threadgroup float tg_sums[8];

    int oct_local = (int)(tid / 8u);
    int cell_idx  = (int)(tid % 8u) + 1;

    float norm_loc = 0.0f;
    if (oct_local < num_octs) {
        int oct_idx = head_idx + oct_local;
        float r = f_get_mg(f, oct_idx, 1, cell_idx);
        float d = f_get_mg(f, oct_idx, 3, cell_idx);
        if (d > 0.0f) norm_loc = r * r;
    }

    norm_loc = tg_reduce_sum_f(norm_loc, ltid, tg_sums);
    if (ltid == 0u)
        atomic_add_float((device atomic_uint*)norm_tot, norm_loc);
}

/* ===========================================================================
 * Kernel 18: cmp_epot_kernel — potential energy (sum of |f|^2 at leaves).
 * Thread layout: 1D 256.
 * ========================================================================= */
kernel void cmp_epot_kernel(
    device const oct_t *grid     [[buffer(0)]],
    device const float *f        [[buffer(1)]],
    device float       *epot_tot [[buffer(2)]],
    constant int       &head_idx [[buffer(3)]],
    constant int       &num_octs [[buffer(4)]],
    uint tid  [[thread_position_in_grid]],
    uint ltid [[thread_position_in_threadgroup]])
{
    threadgroup float tg_sums[8];

    int oct_local = (int)(tid / 8u);
    int cell_idx  = (int)(tid % 8u) + 1;

    float epot_loc = 0.0f;
    if (oct_local < num_octs) {
        int oct_idx = head_idx + oct_local;
        if (grid[oct_idx-1].refined[cell_idx-1] == 0) {
            float f1 = f_get_mg(f, oct_idx, 1, cell_idx);
            float f2 = f_get_mg(f, oct_idx, 2, cell_idx);
            float f3 = f_get_mg(f, oct_idx, 3, cell_idx);
            epot_loc = f1*f1 + f2*f2 + f3*f3;
        }
    }

    epot_loc = tg_reduce_sum_f(epot_loc, ltid, tg_sums);
    if (ltid == 0u)
        atomic_add_float((device atomic_uint*)epot_tot, epot_loc);
}

/* ===========================================================================
 * Kernel 19: cmp_rhomax_kernel — maximum rho over all cells.
 * Thread layout: 1D 256.
 * ========================================================================= */
kernel void cmp_rhomax_kernel(
    device const float *rho       [[buffer(0)]],
    device float       *rhomax_tot [[buffer(1)]],
    constant int       &head_idx  [[buffer(2)]],
    constant int       &num_octs  [[buffer(3)]],
    uint tid  [[thread_position_in_grid]],
    uint ltid [[thread_position_in_threadgroup]])
{
    threadgroup float tg_sums[8];

    int oct_local = (int)(tid / 8u);
    int cell_idx  = (int)(tid % 8u) + 1;

    float rmax = 0.0f;
    if (oct_local < num_octs) {
        int oct_idx = head_idx + oct_local;
        rmax = rho_get_mg(rho, oct_idx, cell_idx);
    }

    rmax = tg_reduce_max_f(rmax, ltid, tg_sums);
    if (ltid == 0u)
        atomic_max_float((device atomic_uint*)rhomax_tot, rmax);
}

/* ===========================================================================
 * Kernel 20: gradient_phi_kernel — high-order finite-difference gradient.
 * Thread layout: 2D {8, 16}.
 * ========================================================================= */
kernel void gradient_phi_kernel(
    device const float *phi      [[buffer(0)]],
    device float       *f        [[buffer(1)]],
    device const int   *nbor     [[buffer(2)]],
    constant int       &head_idx [[buffer(3)]],
    constant int       &num_octs [[buffer(4)]],
    constant float     &dx       [[buffer(5)]],
    uint2 tptg [[threads_per_threadgroup]],
    uint2 tgpi [[threadgroup_position_in_grid]],
    uint2 tpg  [[thread_position_in_threadgroup]])
{
    int oct_off  = (int)(tgpi.x * tptg.y + tpg.y);
    if (oct_off >= num_octs) return;
    int oct_idx  = head_idx + oct_off;
    int cell_idx = (int)tpg.x + 1;

    int sg = oct_idx, i = 1, j = 1, k = 1;
    int ind_nbor_c = 1 + i + 3*j + 9*k;
    int oct_true = nbor[(sg-1)*27 + (ind_nbor_c-1)];

    /* Build nbor_idx[0..6]: 0=central, 1..6=±x,±y,±z */
    thread int nbor_idx[7];
    nbor_idx[0] = oct_true;
    for (int idim = 1; idim <= 3; idim++) {
        for (int inbor = 1; inbor <= 2; inbor++) {
            int ind_nbor_l = 2*(idim-1) + inbor;
            int in=0, jn=0, kn=0;
            if      (idim==1) in = -1+2*(inbor-1);
            else if (idim==2) jn = -1+2*(inbor-1);
            else              kn = -1+2*(inbor-1);
            int ind_n = 1+(i+in)+3*(j+jn)+9*(k+kn);
            nbor_idx[ind_nbor_l] = nbor[(sg-1)*27+(ind_n-1)];
        }
    }

    float a = 0.5f * (4.0f/3.0f) / dx;
    float b = 0.25f * (1.0f/3.0f) / dx;

    for (int idim = 1; idim <= 3; idim++) {
        int id0 = idim - 1;
        int oct1 = nbor_idx[gg1_mg[id0][cell_idx-1]];
        int oct2 = nbor_idx[gg2_mg[id0][cell_idx-1]];
        int oct3 = nbor_idx[gg3_mg[id0][cell_idx-1]];
        int oct4 = nbor_idx[gg4_mg[id0][cell_idx-1]];
        int c1   = hh1_mg[id0][cell_idx-1];
        int c2   = hh2_mg[id0][cell_idx-1];
        int c3   = hh3_mg[id0][cell_idx-1];
        int c4   = hh4_mg[id0][cell_idx-1];
        float phi1 = phi_get_mg(phi, oct1, c1);
        float phi2 = phi_get_mg(phi, oct2, c2);
        float phi3 = phi_get_mg(phi, oct3, c3);
        float phi4 = phi_get_mg(phi, oct4, c4);
        f_set_mg(f, oct_true, idim, cell_idx, a*(phi1-phi2) - b*(phi3-phi4));
    }
}

/* ===========================================================================
 * Kernel 21: update_nbor_array_mg_kernel — build MG nbor table from hash.
 * Thread layout: 1D 128 — one thread per subgrid (= one per MG oct for NSUBGRID=1).
 * ========================================================================= */
kernel void update_nbor_array_mg_kernel(
    device int         *nbor_mg      [[buffer(0)]],
    device const oct_t *grid_mg      [[buffer(1)]],
    device const long  *hash_key_mg  [[buffer(2)]],
    device const int   *hash_val_mg  [[buffer(3)]],
    device const int   *ckey_max_dev [[buffer(4)]],
    device const long  *key_off_dev  [[buffer(5)]],
    device const int   *box_ckey_min [[buffer(6)]],
    device const int   *box_ckey_max [[buffer(7)]],
    device const int   *periodic_dev [[buffer(8)]],
    constant int       &hash_size_mg  [[buffer(9)]],
    constant int       &head_idx      [[buffer(10)]],
    constant int       &num_subgrids  [[buffer(11)]],
    uint tid [[thread_position_in_grid]])
{
    int sg_off = (int)tid;
    if (sg_off >= num_subgrids) return;
    int subgrid_idx = head_idx + sg_off;
    int oct_idx     = subgrid_idx;   /* NSUBGRID=1 */

    int ilevel = grid_mg[oct_idx-1].lev;
    long nx     = (long)ckey_max_dev[ilevel-1];
    long offset = key_off_dev[ilevel-1];

    for (int input_ind = 1; input_ind <= 27; input_ind++) {
        int ind = input_ind - 1;
        int kk  = ind / 9;
        int jj  = (ind - kk*9) / 3;
        int ii  = ind - kk*9 - jj*3;

        /* For NSUBGRID=1: subgrid corner = (ckey/1)*1 = ckey */
        int ck0 = grid_mg[oct_idx-1].ckey[0] + ii - 1;
        int ck1 = grid_mg[oct_idx-1].ckey[1] + jj - 1;
        int ck2 = grid_mg[oct_idx-1].ckey[2] + kk - 1;

        /* Periodic wrapping */
        if (periodic_dev[0]) {
            if (ck0 <  box_ckey_min[3*(ilevel-1)+0]) ck0 = box_ckey_max[3*(ilevel-1)+0]-1;
            if (ck0 >= box_ckey_max[3*(ilevel-1)+0]) ck0 = box_ckey_min[3*(ilevel-1)+0];
        }
        if (periodic_dev[1]) {
            if (ck1 <  box_ckey_min[3*(ilevel-1)+1]) ck1 = box_ckey_max[3*(ilevel-1)+1]-1;
            if (ck1 >= box_ckey_max[3*(ilevel-1)+1]) ck1 = box_ckey_min[3*(ilevel-1)+1];
        }
        if (periodic_dev[2]) {
            if (ck2 <  box_ckey_min[3*(ilevel-1)+2]) ck2 = box_ckey_max[3*(ilevel-1)+2]-1;
            if (ck2 >= box_ckey_max[3*(ilevel-1)+2]) ck2 = box_ckey_min[3*(ilevel-1)+2];
        }

        long key = offset + (long)ck0 + (long)ck1*nx + (long)ck2*nx*nx;
        nbor_mg[(subgrid_idx-1)*27 + (input_ind-1)] =
            hash_get(hash_key_mg, hash_val_mg, hash_size_mg, key);
    }
}
