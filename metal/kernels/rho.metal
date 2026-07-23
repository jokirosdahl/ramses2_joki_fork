/*
 * metal/kernels/rho.metal
 *
 * Metal Shading Language port of gpu/gpu_rho.cuf rho_device module.
 * Kernels: reset_rho, multipole_leaf, multipole_upload, multipole_tot,
 *          deposit_rho.
 * multipole_cic is NOT ported (Metal path uses deposit_rho for nsubgrid=1).
 *
 * All arrays use float32 (single precision) — accepted precision loss for PoC.
 * Buffer layout mirrors gpu_rho.cuf (Fortran column-major flattened):
 *   rho(cell,oct)         → rho[(oct-1)*8+(cell-1)]
 *   unew(cell,ivar,oct)   → unew[(oct-1)*NVAR*8+(ivar-1)*8+(cell-1)]
 *   nbor(ind,sg)          → nbor[(sg-1)*27+(ind-1)]
 */

#include <metal_stdlib>
#include <metal_atomic>
using namespace metal;

#include "../metal_types.h"
#include "metal_utils.h"

#ifndef NVAR
#define NVAR 5
#endif

constant int NSUBGRIDP2_R = 3;   /* NSUBGRID+2 for NSUBGRID=1 */
constant int NSUBGRIDP2SQ_R = 9; /* NSUBGRIDP2^2               */

/* ---------------------------------------------------------------------------
 * Array access helpers (column-major flat indexing)
 * --------------------------------------------------------------------------*/

static inline void rho_set(device float *r, int oct_1, int cell_1, float v) {
    r[(oct_1-1)*8 + (cell_1-1)] = v;
}
static inline void rho_atomic_add(device float *r, int oct_1, int cell_1, float v) {
    atomic_add_float((device atomic_uint*)&r[(oct_1-1)*8+(cell_1-1)], v);
}

static inline float u_get_r(device const float *u, int oct_1, int ivar_1, int cell_1) {
    return u[(oct_1-1)*NVAR*8 + (ivar_1-1)*8 + (cell_1-1)];
}
static inline void u_set_r(device float *u, int oct_1, int ivar_1, int cell_1, float v) {
    u[(oct_1-1)*NVAR*8 + (ivar_1-1)*8 + (cell_1-1)] = v;
}

/* CIC weight array: dl,dr are 1-vectors per dimension */
static inline void cic_weight_r(float dl0, float dr0, float dl1, float dr1, float dl2, float dr2,
                                  thread float vol[8]) {
    vol[0]=dl0*dl1*dl2; vol[1]=dr0*dl1*dl2; vol[2]=dl0*dr1*dl2; vol[3]=dr0*dr1*dl2;
    vol[4]=dl0*dl1*dr2; vol[5]=dr0*dl1*dr2; vol[6]=dl0*dr1*dr2; vol[7]=dr0*dr1*dr2;
}

static inline void cic_index_r(int il0, int ir0, int il1, int ir1, int il2, int ir2,
                                 thread int key[3][8]) {
    key[0][0]=il0; key[1][0]=il1; key[2][0]=il2;
    key[0][1]=ir0; key[1][1]=il1; key[2][1]=il2;
    key[0][2]=il0; key[1][2]=ir1; key[2][2]=il2;
    key[0][3]=ir0; key[1][3]=ir1; key[2][3]=il2;
    key[0][4]=il0; key[1][4]=il1; key[2][4]=ir2;
    key[0][5]=ir0; key[1][5]=il1; key[2][5]=ir2;
    key[0][6]=il0; key[1][6]=ir1; key[2][6]=ir2;
    key[0][7]=ir0; key[1][7]=ir1; key[2][7]=ir2;
}

/* ===========================================================================
 * reset_rho_kernel — zero rho and nref for all cells in [head_idx, head_idx+num_octs).
 * Thread layout: 2D {8, 16} — x=cell_idx(1..8), y=oct offset within TG.
 * ========================================================================= */
kernel void reset_rho_kernel(
    device float         *rho     [[buffer(0)]],
    device float         *nref    [[buffer(1)]],
    constant int         &head_idx [[buffer(2)]],
    constant int         &num_octs [[buffer(3)]],
    uint2 tptg [[threads_per_threadgroup]],
    uint2 tgpi [[threadgroup_position_in_grid]],
    uint2 tpg  [[thread_position_in_threadgroup]])
{
    int oct_offset = (int)(tgpi.x * tptg.y + tpg.y);
    if (oct_offset >= num_octs) return;
    int oct_idx  = head_idx + oct_offset;
    int cell_idx = (int)tpg.x + 1;   /* 1-based */

    rho_set(rho,  oct_idx, cell_idx, 0.0f);
    rho_set(nref, oct_idx, cell_idx, 0.0f);
}

/* ===========================================================================
 * multipole_leaf_kernel — compute monopole and 3 dipoles for leaf cells.
 * Thread layout: 2D {8, 16}.
 * ========================================================================= */
kernel void multipole_leaf_kernel(
    device float         *unew    [[buffer(0)]],
    device const float   *uold    [[buffer(1)]],
    device const oct_t   *grid    [[buffer(2)]],
    constant int         &head_idx [[buffer(3)]],
    constant int         &num_octs [[buffer(4)]],
    constant float       &dx_loc   [[buffer(5)]],
    uint2 tptg [[threads_per_threadgroup]],
    uint2 tgpi [[threadgroup_position_in_grid]],
    uint2 tpg  [[thread_position_in_threadgroup]])
{
    int oct_offset = (int)(tgpi.x * tptg.y + tpg.y);
    if (oct_offset >= num_octs) return;
    int oct_idx  = head_idx + oct_offset;
    int cell_idx = (int)tpg.x + 1;

    int ind = cell_idx - 1;
    int k = ind / 4; ind -= k*4;
    int j = ind / 2;
    int i = ind - j*2;

    bool ok_leaf = (grid[oct_idx-1].refined[cell_idx-1] == 0);

    float x0 = (2.0f*grid[oct_idx-1].ckey[0] + i + 0.5f) * dx_loc;
    float x1 = (2.0f*grid[oct_idx-1].ckey[1] + j + 0.5f) * dx_loc;
    float x2 = (2.0f*grid[oct_idx-1].ckey[2] + k + 0.5f) * dx_loc;

    float vol_loc = dx_loc * dx_loc * dx_loc;
    float monopole = u_get_r(uold, oct_idx, 1, cell_idx) * vol_loc;

    if (ok_leaf) {
        u_set_r(unew, oct_idx, 1, cell_idx, monopole);
        u_set_r(unew, oct_idx, 2, cell_idx, monopole * x0);
        u_set_r(unew, oct_idx, 3, cell_idx, monopole * x1);
        u_set_r(unew, oct_idx, 4, cell_idx, monopole * x2);
    } else {
        u_set_r(unew, oct_idx, 1, cell_idx, 0.0f);
        u_set_r(unew, oct_idx, 2, cell_idx, 0.0f);
        u_set_r(unew, oct_idx, 3, cell_idx, 0.0f);
        u_set_r(unew, oct_idx, 4, cell_idx, 0.0f);
    }
}

/* ===========================================================================
 * multipole_upload_kernel — sum 8 children into parent cell (NVAR=4 components).
 * Thread layout: 1D 256 — one thread per oct.
 * ========================================================================= */
kernel void multipole_upload_kernel(
    device const oct_t   *grid    [[buffer(0)]],
    device const int     *father  [[buffer(1)]],
    device float         *unew    [[buffer(2)]],
    constant int         &head_idx [[buffer(3)]],
    constant int         &num_octs [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    int oct_offset = (int)tid;
    if (oct_offset >= num_octs) return;
    int oct_idx    = head_idx + oct_offset;
    int father_idx = father[oct_idx - 1];

    int ck0 = grid[oct_idx-1].ckey[0];
    int ck1 = grid[oct_idx-1].ckey[1];
    int ck2 = grid[oct_idx-1].ckey[2];
    int pi = ck0 - 2*(ck0/2);
    int pj = ck1 - 2*(ck1/2);
    int pk = ck2 - 2*(ck2/2);
    int cell_idx = 1 + pi + 2*pj + 4*pk;

    /* Reset parent cell */
    u_set_r(unew, father_idx, 1, cell_idx, 0.0f);
    u_set_r(unew, father_idx, 2, cell_idx, 0.0f);
    u_set_r(unew, father_idx, 3, cell_idx, 0.0f);
    u_set_r(unew, father_idx, 4, cell_idx, 0.0f);

    /* Sum 8 children into parent */
    for (int ind = 1; ind <= 8; ind++) {
        u_set_r(unew, father_idx, 1, cell_idx,
                u_get_r(unew,father_idx,1,cell_idx) + u_get_r(unew,oct_idx,1,ind));
        u_set_r(unew, father_idx, 2, cell_idx,
                u_get_r(unew,father_idx,2,cell_idx) + u_get_r(unew,oct_idx,2,ind));
        u_set_r(unew, father_idx, 3, cell_idx,
                u_get_r(unew,father_idx,3,cell_idx) + u_get_r(unew,oct_idx,3,ind));
        u_set_r(unew, father_idx, 4, cell_idx,
                u_get_r(unew,father_idx,4,cell_idx) + u_get_r(unew,oct_idx,4,ind));
    }
}

/* ===========================================================================
 * multipole_tot_kernel — global reduction: sum all cell multipoles into 4 floats.
 * Thread layout: 1D 256 — 32 octs per TG (256/8 cells each).
 * Output buffer: device float[4] pre-zeroed by bridge; atomic-added into.
 * ========================================================================= */
kernel void multipole_tot_kernel(
    device const float    *unew      [[buffer(0)]],
    device atomic_float   *multipole [[buffer(1)]],
    constant int          &head_idx  [[buffer(2)]],
    constant int          &num_octs  [[buffer(3)]],
    uint tid  [[thread_position_in_grid]],
    uint ltid [[thread_position_in_threadgroup]])
{
    threadgroup float tg_sums[8];

    int oct_local = (int)(tid / 8u);
    int cell_idx  = (int)(tid % 8u) + 1;

    float m0 = 0.0f, m1 = 0.0f, m2 = 0.0f, m3 = 0.0f;
    if (oct_local < num_octs) {
        int oct_idx = head_idx + oct_local;
        m0 = u_get_r(unew, oct_idx, 1, cell_idx);
        m1 = u_get_r(unew, oct_idx, 2, cell_idx);
        m2 = u_get_r(unew, oct_idx, 3, cell_idx);
        m3 = u_get_r(unew, oct_idx, 4, cell_idx);
    }

    m0 = tg_reduce_sum_f(m0, ltid, tg_sums);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    m1 = tg_reduce_sum_f(m1, ltid, tg_sums);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    m2 = tg_reduce_sum_f(m2, ltid, tg_sums);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    m3 = tg_reduce_sum_f(m3, ltid, tg_sums);

    if (ltid == 0u) {
        atomic_fetch_add_explicit(&multipole[0], m0, memory_order_relaxed);
        atomic_fetch_add_explicit(&multipole[1], m1, memory_order_relaxed);
        atomic_fetch_add_explicit(&multipole[2], m2, memory_order_relaxed);
        atomic_fetch_add_explicit(&multipole[3], m3, memory_order_relaxed);
    }
}

/* ===========================================================================
 * deposit_rho_kernel — CIC deposit of gas mass onto rho grid.
 * Thread layout: 2D {8, 16}.
 * Mirrors deposit_rho in gpu_rho.cuf.
 * ========================================================================= */
kernel void deposit_rho_kernel(
    device const oct_t   *grid            [[buffer(0)]],
    device const float   *uold            [[buffer(1)]],
    device const float   *unew            [[buffer(2)]],
    device float         *rho             [[buffer(3)]],
    device float         *nref            [[buffer(4)]],
    device const int     *nbor            [[buffer(5)]],
    constant int         &head_idx        [[buffer(6)]],
    constant int         &num_octs        [[buffer(7)]],
    constant float       &dx              [[buffer(8)]],
    constant float       &vol_loc         [[buffer(9)]],
    constant float       &m_refine        [[buffer(10)]],
    constant float       &mass_sph        [[buffer(11)]],
    constant float       &var_cut_refine  [[buffer(12)]],
    constant int         &ivar_refine     [[buffer(13)]],
    constant int         &ngridmax        [[buffer(14)]],
    uint2 tptg [[threads_per_threadgroup]],
    uint2 tgpi [[threadgroup_position_in_grid]],
    uint2 tpg  [[thread_position_in_threadgroup]])
{
    int oct_offset = (int)(tgpi.x * tptg.y + tpg.y);
    if (oct_offset >= num_octs) return;
    int oct_idx  = head_idx + oct_offset;
    int cell_idx = (int)tpg.x + 1;

    float monopole = u_get_r(unew, oct_idx, 1, cell_idx);
    if (monopole == 0.0f) return;

    if (grid[oct_idx-1].refined[cell_idx-1] == 0) {
        /* Leaf cell: deposit onto self */
        float deposit = monopole / vol_loc;
        rho_atomic_add(rho, oct_idx, cell_idx, deposit);

        if (m_refine >= 0.0f) {
            deposit = monopole / mass_sph;
            if (ivar_refine > 0) {
                float mask = u_get_r(uold, oct_idx, ivar_refine, cell_idx)
                           / u_get_r(uold, oct_idx, 1, cell_idx);
                if (mask > var_cut_refine)
                    rho_atomic_add(nref, oct_idx, cell_idx, deposit);
            } else {
                rho_atomic_add(nref, oct_idx, cell_idx, deposit);
            }
        }
    } else {
        /* Split cell: CIC to up to 8 neighbours */
        float x_com0 = u_get_r(unew, oct_idx, 2, cell_idx) / monopole;
        float x_com1 = u_get_r(unew, oct_idx, 3, cell_idx) / monopole;
        float x_com2 = u_get_r(unew, oct_idx, 4, cell_idx) / monopole;

        float dd0 = x_com0/dx + 0.5f; int id0 = (int)dd0; dd0 -= id0; float dg0 = 1.0f-dd0; int ig0 = id0-1;
        float dd1 = x_com1/dx + 0.5f; int id1 = (int)dd1; dd1 -= id1; float dg1 = 1.0f-dd1; int ig1 = id1-1;
        float dd2 = x_com2/dx + 0.5f; int id2 = (int)dd2; dd2 -= id2; float dg2 = 1.0f-dd2; int ig2 = id2-1;

        thread float vol_cic[8];
        thread int key_cic[3][8];
        cic_weight_r(dg0,dd0,dg1,dd1,dg2,dd2, vol_cic);
        cic_index_r(ig0,id0,ig1,id1,ig2,id2, key_cic);

        for (int ind = 0; ind < 8; ind++) {
            if (vol_cic[ind] == 0.0f) continue;

            int sum_d0 = key_cic[0][ind] - 2*grid[oct_idx-1].ckey[0];
            int sum_d1 = key_cic[1][ind] - 2*grid[oct_idx-1].ckey[1];
            int sum_d2 = key_cic[2][ind] - 2*grid[oct_idx-1].ckey[2];

            int in_vec0 = (sum_d0 < 0) ? -1 : sum_d0/2;
            int in_vec1 = (sum_d1 < 0) ? -1 : sum_d1/2;
            int in_vec2 = (sum_d2 < 0) ? -1 : sum_d2/2;

            int tgt_cell_l0 = sum_d0 - 2*in_vec0;
            int tgt_cell_l1 = sum_d1 - 2*in_vec1;
            int tgt_cell_l2 = sum_d2 - 2*in_vec2;

            int ind_nbor = 1 + (1+in_vec0) + NSUBGRIDP2_R*(1+in_vec1) + NSUBGRIDP2SQ_R*(1+in_vec2);
            int tgt_oct_idx = nbor_get(nbor, oct_idx, ind_nbor);
            if (tgt_oct_idx > ngridmax) continue;

            int tgt_cell_idx = 1 + tgt_cell_l0 + 2*tgt_cell_l1 + 4*tgt_cell_l2;

            float deposit = monopole * vol_cic[ind] / vol_loc;
            rho_atomic_add(rho, tgt_oct_idx, tgt_cell_idx, deposit);

            if (m_refine >= 0.0f) {
                deposit = monopole * vol_cic[ind] / mass_sph;
                if (ivar_refine > 0) {
                    float mask = u_get_r(uold, tgt_oct_idx, ivar_refine, tgt_cell_idx)
                               / u_get_r(uold, tgt_oct_idx, 1, tgt_cell_idx);
                    if (mask > var_cut_refine)
                        rho_atomic_add(nref, tgt_oct_idx, tgt_cell_idx, deposit);
                } else {
                    rho_atomic_add(nref, tgt_oct_idx, tgt_cell_idx, deposit);
                }
            }
        }
    }
}
