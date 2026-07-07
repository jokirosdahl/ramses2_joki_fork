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
inline ulong fnv64_r(long key_signed)
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

inline int hash_bucket_r(long key, int hash_size)
{
    return int(fnv64_r(key) % ulong(hash_size)) + 1;
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

/* =========================================================================
 * update_father_kernel — populate father[oct_abs_0] with the 1-based parent
 * oct index by hashing the parent's Cartesian key.
 * Mirrors update_father_array<<<128>>> in gpu_refine.cuf.
 * Per-level Hilbert parameters are passed directly (not via s_ckey_max_dev)
 * so this kernel can be called once per level.
 * ========================================================================= */
kernel void update_father_kernel(
    device const oct_t  *grid     [[buffer(0)]],
    device int          *father   [[buffer(1)]],
    device const long   *hash_key [[buffer(2)]],
    device const int    *hash_val [[buffer(3)]],
    constant int        &hash_size  [[buffer(4)]],
    constant int        &ckey_max_l [[buffer(5)]],
    constant long       &key_off_l  [[buffer(6)]],
    constant int        &head_idx   [[buffer(7)]],
    constant int        &num_octs   [[buffer(8)]],
    uint gid [[thread_position_in_grid]])
{
    if (int(gid) >= num_octs) return;
    int oct_abs_0 = (head_idx - 1) + int(gid);

    /* Parent ckey: divide current ckey by 2 (right-shift) */
    long ix = long(grid[oct_abs_0].ckey[0] / 2);
    long iy = long(grid[oct_abs_0].ckey[1] / 2);
    long iz = long(grid[oct_abs_0].ckey[2] / 2);
    long nx = long(ckey_max_l);

    long parent_key = key_off_l + ix + iy * nx + iz * nx * nx;
    father[oct_abs_0] = hash_get(hash_key, hash_val, hash_size, parent_key);
}

/* =========================================================================
 * AMR refinement kernels — port of gpu/gpu_refine.cuf
 * Scope: HYDRO=1, GRAV=0, MHD=0, NDIM=3, NSUBGRID=1, no CUB.
 *
 * Memory layout (0-based):
 *   uold    : float[8*(NVAR)*ntotal]  cell_0 + 8*ivar_0 + 8*(NVAR)*oct_abs_0
 *   flag1/2 : int[8*ntotal]           cell_0 + 8*oct_abs_0
 *   nbor    : int[27*ntotal]          (subgrid_abs_0)*27 + (ind_0)   row-major
 *   grid    : oct_t[ntotal]           0-based oct index
 *   father  : int[ntotal]             1-based parent oct index per oct
 *   swap_*  : int[ntotal]             1-based oct index
 *   prefix  : int[ntotal]
 *
 * Level params uploaded by mtl_upload_level_params (0-based in device buffers):
 *   ckey_max_dev[lev-1]            = ckey_max(lev)
 *   key_off_dev[lev-1]             = key_off(lev)   (long)
 *   box_ckey_min_dev[3*(lev-1)+d]  = box_ckey_min(d+1, lev)
 *   box_ckey_max_dev[3*(lev-1)+d]  = box_ckey_max(d+1, lev)
 *   periodic_dev[d]                = periodic(d+1)
 * ========================================================================= */

#ifndef NVAR
#define NVAR 5
#endif

/* ---- Hilbert state tables (NDIM=3), from gpu/gpu_hilbert.cuf ---------- */
/* left_shift=4, right_shift=-1 → ISHFT(hkey,4) then ISHFT(hkey,-1)       */
/* Net effect per loop iteration: hkey = (hkey << 4) >> 1  =  hkey << 3   */
constant long next_digits_rf[96] = {
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

constant int next_state_rf[96] = {
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

/* hilbert_key_rf — direct port of gpu_hilbert.cuf::hilbert_key (device fn).
 * Uses ulong internally (ISHFT is logical; C >> on signed is implementation-
 * defined, so we keep ulong throughout). Max level=21: 3*21=63 bits < 64. */
inline long hilbert_key_rf(int ckey[3], int level)
{
    ulong hkey   = 0;
    int   cstate = 0;
    for (int ibit = level - 1; ibit >= 0; ibit--) {
        hkey = (hkey << 4) >> 1;               /* ISHFT(hkey,4) then ISHFT(hkey,-1) */
        int sdigit = 0;
        if ((ckey[0] >> ibit) & 1) sdigit += 4; /* idim=1: add_digit=2^(3-1)=4 */
        if ((ckey[1] >> ibit) & 1) sdigit += 2; /* idim=2: add_digit=2^(3-2)=2 */
        if ((ckey[2] >> ibit) & 1) sdigit += 1; /* idim=3: add_digit=2^(3-3)=1 */
        int ind = cstate * 8 + sdigit;
        cstate  = next_state_rf[ind];
        hkey   += (ulong)next_digits_rf[ind];
    }
    return as_type<long>(hkey);
}

/* ---- Read-write hash helpers (device atomic_int* for val) ------------- */
/* hash_bucket_r is already defined above in the read-only section.        */

/* hash_set_r: insert or overwrite key→val using 32-bit CAS on hash_val.  */
inline void hash_set_r(device long* hash_key, device atomic_int* hash_val,
                       int hash_size, long key, int val)
{
    int b = hash_bucket_r(key, hash_size);
    for (;;) {
        int expected = 0;
        if (atomic_compare_exchange_weak_explicit(
                &hash_val[b - 1], &expected, val,
                memory_order_relaxed, memory_order_relaxed)) {
            hash_key[b - 1] = key;
            return;
        }
        if (expected == val) { hash_key[b - 1] = key; return; } /* same key re-insert */
        if (expected != 0)    b = (b % hash_size) + 1;          /* occupied → probe   */
        /* expected == 0: spurious weak-CAS failure → retry same slot */
    }
}

/* hash_free_r: set val=0 for key, leaving key in table (matches CUDA).
 * No atomics needed: each thread owns a unique key → unique bucket.
 * The prior kernel dispatch (a full barrier on Metal's serial queue) has
 * completed before this runs, so no concurrent insertion is possible.      */
inline void hash_free_r(device long* hash_key, device int* hash_val,
                        int hash_size, long key)
{
    int b = hash_bucket_r(key, hash_size);
    for (;;) {
        long cur = hash_key[b - 1];
        if (cur == key) { hash_val[b - 1] = 0; return; }
        if (cur == 0L) return;
        b = (b % hash_size) + 1;
    }
}

/* hash_update_r: find key, overwrite val (key must already be in table).
 * No atomics needed: same unique-key-per-thread guarantee as hash_free_r.  */
inline void hash_update_r(device long* hash_key, device int* hash_val,
                          int hash_size, long key, int val)
{
    int b = hash_bucket_r(key, hash_size);
    for (;;) {
        long cur = hash_key[b - 1];
        if (cur == key) { hash_val[b - 1] = val; return; }
        if (cur == 0L) return;
        b = (b % hash_size) + 1;
    }
}

/* =========================================================================
 * refine_kernel — create new child octs for flagged, unrefined cells.
 * Mirrors refine_kernel + make_new_oct (gpu_refine.cuf).
 * HYDRO=1, GRAV=0, MHD=0 — straight injection only.
 *
 * 1D thread layout: gid = oct_offset * 8 + cell_0 (< 8 * num_octs).
 * Each thread independently creates ONE child oct if flag1[cell_0, oct] is
 * set and grid[oct].refined[cell_0] is false.  atomicAdd on ifree_dev gives
 * each creating thread a unique 1-based child index.
 * ========================================================================= */
kernel void refine_kernel(
    device oct_t       *grid     [[buffer(0)]],
    device int         *flag1    [[buffer(1)]],
    device float       *uold     [[buffer(2)]],
    device atomic_int  *ifree_dev [[buffer(3)]],
    constant int       &head_idx [[buffer(4)]],
    constant int       &num_octs [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if (int(gid) >= num_octs * 8) return;
    int oct_offset = int(gid) / 8;
    int cell_0     = int(gid) % 8;    /* 0-based cell index */
    int oct_abs_0  = (head_idx - 1) + oct_offset;

    bool flagged = (flag1[cell_0 + 8 * oct_abs_0] == 1);
    bool refined = (grid[oct_abs_0].refined[cell_0] != 0);
    if (!flagged || refined) return;

    /* make_new_oct inline (HYDRO=1 only) */
    int i = cell_0 & 1;
    int j = (cell_0 >> 1) & 1;
    int k = (cell_0 >> 2) & 1;

    int ilevel  = grid[oct_abs_0].lev + 1;
    int cckey[3];
    cckey[0] = 2 * grid[oct_abs_0].ckey[0] + i;
    cckey[1] = 2 * grid[oct_abs_0].ckey[1] + j;
    cckey[2] = 2 * grid[oct_abs_0].ckey[2] + k;

    /* Claim the next free slot (1-based). */
    int child_1based = atomic_fetch_add_explicit(ifree_dev, 1, memory_order_relaxed);
    int child_abs_0  = child_1based - 1;

    /* Initialise child oct. */
    grid[child_abs_0].lev    = ilevel;
    grid[child_abs_0].ckey[0] = cckey[0];
    grid[child_abs_0].ckey[1] = cckey[1];
    grid[child_abs_0].ckey[2] = cckey[2];
    for (int c = 0; c < 8; c++) grid[child_abs_0].refined[c] = 0;
    for (int c = 0; c < 8; c++) flag1[c + 8 * child_abs_0] = 0;
    grid[child_abs_0].hkey = hilbert_key_rf(cckey, ilevel - 1);

    /* Straight injection: copy parent cell into all 8 child cells. */
    for (int ivar_0 = 0; ivar_0 < (NVAR); ivar_0++) {
        float val = uold[cell_0 + 8 * ivar_0 + 8 * (NVAR) * oct_abs_0];
        for (int c = 0; c < 8; c++)
            uold[c + 8 * ivar_0 + 8 * (NVAR) * child_abs_0] = val;
    }

    /* Mark parent cell as refined. */
    grid[oct_abs_0].refined[cell_0] = 1;
}

/* =========================================================================
 * derefine_kernel — remove child octs whose parent cells are no longer flagged.
 * Mirrors derefine_kernel in gpu_refine.cuf. 1 thread/oct.
 * ========================================================================= */
kernel void derefine_kernel(
    device oct_t       *grid        [[buffer(0)]],
    device const int   *flag1       [[buffer(1)]],
    device long        *hash_key    [[buffer(2)]],
    device int         *hash_val    [[buffer(3)]],
    device const int   *ckey_max_dev [[buffer(4)]],
    device const long  *key_off_dev  [[buffer(5)]],
    constant int       &hash_size   [[buffer(6)]],
    constant int       &head_idx    [[buffer(7)]],
    constant int       &num_octs    [[buffer(8)]],
    uint gid [[thread_position_in_grid]])
{
    if (int(gid) >= num_octs) return;
    int oct_abs_0 = (head_idx - 1) + int(gid);
    int ilevel    = grid[oct_abs_0].lev;
    if (ilevel <= 0) return;

    /* Compute oct's own Hilbert key and hash it. */
    long nx  = long(ckey_max_dev[ilevel - 1]);
    long ix  = long(grid[oct_abs_0].ckey[0]);
    long iy  = long(grid[oct_abs_0].ckey[1]);
    long iz  = long(grid[oct_abs_0].ckey[2]);
    long key = key_off_dev[ilevel - 1] + ix + iy * nx + iz * nx * nx;

    /* Compute parent level Hilbert key. */
    int parent_lev = ilevel - 1;
    if (parent_lev <= 0) return;
    long pnx = long(ckey_max_dev[parent_lev - 1]);
    long pix = ix / 2L;
    long piy = iy / 2L;
    long piz = iz / 2L;
    long parent_key = key_off_dev[parent_lev - 1] + pix + piy * pnx + piz * pnx * pnx;

    int parent_1based = hash_get(hash_key, hash_val, hash_size, parent_key);
    if (parent_1based <= 0) return;
    int parent_abs_0 = parent_1based - 1;

    int pi = int(ix - 2L * pix);
    int pj = int(iy - 2L * piy);
    int pk = int(iz - 2L * piz);
    int cell_0 = pi + 2 * pj + 4 * pk;

    bool flagged = (flag1[cell_0 + 8 * parent_abs_0] == 1);
    bool refined = (grid[parent_abs_0].refined[cell_0] != 0);

    if (!flagged && refined) {
        hash_free_r(hash_key, hash_val, hash_size, key);
        grid[oct_abs_0].lev = 0;
        grid[parent_abs_0].refined[cell_0] = 0;
    }
}

/* =========================================================================
 * free_hash_kernel — free hash entries for a range of octs (cache wipe).
 * Mirrors free_hash_kernel in gpu_refine.cuf. 1 thread/oct.
 * ========================================================================= */
kernel void free_hash_kernel(
    device const oct_t *grid        [[buffer(0)]],
    device long        *hash_key    [[buffer(1)]],
    device int         *hash_val    [[buffer(2)]],
    device const int   *ckey_max_dev [[buffer(3)]],
    device const long  *key_off_dev  [[buffer(4)]],
    constant int       &hash_size   [[buffer(5)]],
    constant int       &head_idx    [[buffer(6)]],
    constant int       &num_octs    [[buffer(7)]],
    uint gid [[thread_position_in_grid]])
{
    if (int(gid) >= num_octs) return;
    int oct_abs_0 = (head_idx - 1) + int(gid);
    int ilevel    = grid[oct_abs_0].lev;
    if (ilevel <= 0) return;

    long nx  = long(ckey_max_dev[ilevel - 1]);
    long ix  = long(grid[oct_abs_0].ckey[0]);
    long iy  = long(grid[oct_abs_0].ckey[1]);
    long iz  = long(grid[oct_abs_0].ckey[2]);
    long key = key_off_dev[ilevel - 1] + ix + iy * nx + iz * nx * nx;
    hash_free_r(hash_key, hash_val, hash_size, key);
}

/* =========================================================================
 * update_hash_kernel — update val for each oct's key after sort rearrangement.
 * Mirrors update_hash_kernel in gpu_refine.cuf. 1 thread/oct.
 * ========================================================================= */
kernel void update_hash_kernel(
    device const oct_t *grid        [[buffer(0)]],
    device long        *hash_key    [[buffer(1)]],
    device int         *hash_val    [[buffer(2)]],
    device const int   *ckey_max_dev [[buffer(3)]],
    device const long  *key_off_dev  [[buffer(4)]],
    constant int       &hash_size   [[buffer(5)]],
    constant int       &head_idx    [[buffer(6)]],
    constant int       &num_octs    [[buffer(7)]],
    uint gid [[thread_position_in_grid]])
{
    if (int(gid) >= num_octs) return;
    int oct_abs_0 = (head_idx - 1) + int(gid);
    int ilevel    = grid[oct_abs_0].lev;
    if (ilevel <= 0) return;

    long nx  = long(ckey_max_dev[ilevel - 1]);
    long ix  = long(grid[oct_abs_0].ckey[0]);
    long iy  = long(grid[oct_abs_0].ckey[1]);
    long iz  = long(grid[oct_abs_0].ckey[2]);
    long key = key_off_dev[ilevel - 1] + ix + iy * nx + iz * nx * nx;
    int  val = (head_idx - 1) + int(gid) + 1;   /* 1-based oct index */
    hash_update_r(hash_key, hash_val, hash_size, key, val);
}

/* =========================================================================
 * insert_hash_all_kernel — insert all octs in a range, reading level from grid.
 * Used for newly created octs (which may span multiple levels).
 * Mirrors insert_hash_kernel in gpu_refine.cuf (the multi-level version).
 * ========================================================================= */
kernel void insert_hash_all_kernel(
    device const oct_t *grid        [[buffer(0)]],
    device long        *hash_key    [[buffer(1)]],
    device atomic_int  *hash_val    [[buffer(2)]],
    device const int   *ckey_max_dev [[buffer(3)]],
    device const long  *key_off_dev  [[buffer(4)]],
    constant int       &hash_size   [[buffer(5)]],
    constant int       &head_idx    [[buffer(6)]],
    constant int       &num_octs    [[buffer(7)]],
    uint gid [[thread_position_in_grid]])
{
    if (int(gid) >= num_octs) return;
    int oct_abs_0 = (head_idx - 1) + int(gid);
    int ilevel    = grid[oct_abs_0].lev;
    if (ilevel <= 0) return;

    long nx  = long(ckey_max_dev[ilevel - 1]);
    long ix  = long(grid[oct_abs_0].ckey[0]);
    long iy  = long(grid[oct_abs_0].ckey[1]);
    long iz  = long(grid[oct_abs_0].ckey[2]);
    long key = key_off_dev[ilevel - 1] + ix + iy * nx + iz * nx * nx;
    int  val = oct_abs_0 + 1;   /* 1-based */
    hash_set_r(hash_key, hash_val, hash_size, key, val);
}

/* =========================================================================
 * init_global_swap_table_kernel — swap_global[oct] = oct (identity).
 * Mirrors init_global_swap_table_kernel in gpu_refine.cuf. 1 thread/oct.
 * ========================================================================= */
kernel void init_global_swap_table_kernel(
    device int   *swap_global [[buffer(0)]],
    constant int &head_idx    [[buffer(1)]],
    constant int &num_octs    [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (int(gid) >= num_octs) return;
    int oct_abs_0 = (head_idx - 1) + int(gid);
    swap_global[oct_abs_0] = oct_abs_0 + 1;   /* 1-based */
}

/* =========================================================================
 * init_prefix_sum_level_kernel
 * prefix_sum[oct] = 0 if grid[swap_global[oct]-1].lev == ilevel, else 1.
 * Mirrors init_prefix_sum_level in gpu_refine.cuf. 1 thread/oct.
 * ========================================================================= */
kernel void init_prefix_sum_level_kernel(
    device const oct_t *grid        [[buffer(0)]],
    device const int   *swap_global [[buffer(1)]],
    device int         *prefix_sum  [[buffer(2)]],
    constant int       &head_idx    [[buffer(3)]],
    constant int       &num_octs    [[buffer(4)]],
    constant int       &ilevel      [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if (int(gid) >= num_octs) return;
    int oct_abs_0     = (head_idx - 1) + int(gid);
    int old_1based    = swap_global[oct_abs_0];
    int old_abs_0     = old_1based - 1;
    prefix_sum[oct_abs_0] = (grid[old_abs_0].lev == ilevel) ? 0 : 1;
}

/* =========================================================================
 * init_prefix_sum_bit_kernel
 * prefix_sum[oct] = bit `ibit` of grid[swap_global[oct]-1].hkey.
 * Mirrors init_prefix_sum_bit in gpu_refine.cuf. 1 thread/oct.
 * ========================================================================= */
kernel void init_prefix_sum_bit_kernel(
    device const oct_t *grid        [[buffer(0)]],
    device const int   *swap_global [[buffer(1)]],
    device int         *prefix_sum  [[buffer(2)]],
    constant int       &head_idx    [[buffer(3)]],
    constant int       &num_octs    [[buffer(4)]],
    constant int       &ibit        [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if (int(gid) >= num_octs) return;
    int oct_abs_0  = (head_idx - 1) + int(gid);
    int old_1based = swap_global[oct_abs_0];
    int old_abs_0  = old_1based - 1;
    ulong hkey     = as_type<ulong>(grid[old_abs_0].hkey);
    prefix_sum[oct_abs_0] = int((hkey >> ibit) & 1UL);
}

/* =========================================================================
 * compute_local_swap_table_kernel — LSD scatter step.
 * Mirrors compute_local_swap_table_impl in gpu_refine.cuf. 1 thread/oct.
 * ========================================================================= */
kernel void compute_local_swap_table_kernel(
    device int       *swap_local  [[buffer(0)]],
    device const int *swap_global [[buffer(1)]],
    device const int *prefix_sum  [[buffer(2)]],
    constant int     &head_idx    [[buffer(3)]],
    constant int     &num_octs    [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (int(gid) >= num_octs) return;
    int oct_abs_0 = (head_idx - 1) + int(gid);
    int tail_abs  = (head_idx - 1) + num_octs - 1;

    int nones  = prefix_sum[tail_abs];
    int nzeros = num_octs - nones;

    int prev_sum = (oct_abs_0 > (head_idx - 1)) ? prefix_sum[oct_abs_0 - 1] : 0;
    int bit      = prefix_sum[oct_abs_0] - prev_sum;
    int old_val  = swap_global[oct_abs_0];   /* 1-based oct index */

    int sorted_abs_0;
    if (bit == 0)
        sorted_abs_0 = oct_abs_0 - prev_sum;
    else
        sorted_abs_0 = (head_idx - 1) + nzeros + prev_sum;

    swap_local[sorted_abs_0] = old_val;
}

/* =========================================================================
 * update_global_swap_table_kernel — swap_global[oct] = swap_local[oct].
 * Mirrors update_global_swap_table_impl. 1 thread/oct.
 * ========================================================================= */
kernel void update_global_swap_table_kernel(
    device int       *swap_global [[buffer(0)]],
    device const int *swap_local  [[buffer(1)]],
    constant int     &head_idx    [[buffer(2)]],
    constant int     &num_octs    [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (int(gid) >= num_octs) return;
    int oct_abs_0 = (head_idx - 1) + int(gid);
    swap_global[oct_abs_0] = swap_local[oct_abs_0];
}

/* =========================================================================
 * sort_gather_grid_kernel — pack grid properties into flag2 scratch.
 * flag2 slots (0-based, per oct):
 *   [0] = lev, [1]=ckey[0], [2]=ckey[1], [3]=ckey[2], [4]=refined_mask
 * Mirrors sort_gather_grid_impl. 1 thread/oct.
 * ========================================================================= */
kernel void sort_gather_grid_kernel(
    device int       *flag2       [[buffer(0)]],
    device const oct_t *grid      [[buffer(1)]],
    device const int *swap_global [[buffer(2)]],
    constant int     &head_idx    [[buffer(3)]],
    constant int     &num_octs    [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (int(gid) >= num_octs) return;
    int oct_abs_0     = (head_idx - 1) + int(gid);
    int old_1based    = swap_global[oct_abs_0];
    int old_abs_0     = old_1based - 1;

    flag2[0 + 8 * oct_abs_0] = grid[old_abs_0].lev;
    flag2[1 + 8 * oct_abs_0] = grid[old_abs_0].ckey[0];
    flag2[2 + 8 * oct_abs_0] = grid[old_abs_0].ckey[1];
    flag2[3 + 8 * oct_abs_0] = grid[old_abs_0].ckey[2];
    int mask = 0;
    for (int c = 0; c < 8; c++)
        if (grid[old_abs_0].refined[c]) mask |= (1 << c);
    flag2[4 + 8 * oct_abs_0] = mask;
}

/* =========================================================================
 * sort_scatter_grid_kernel — unpack flag2 scratch back to grid, recompute hkey.
 * Mirrors sort_scatter_grid_impl. 1 thread/oct.
 * ========================================================================= */
kernel void sort_scatter_grid_kernel(
    device oct_t     *grid    [[buffer(0)]],
    device const int *flag2   [[buffer(1)]],
    constant int     &head_idx [[buffer(2)]],
    constant int     &num_octs [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (int(gid) >= num_octs) return;
    int oct_abs_0 = (head_idx - 1) + int(gid);

    int ilevel    = flag2[0 + 8 * oct_abs_0];
    int ckey[3];
    ckey[0] = flag2[1 + 8 * oct_abs_0];
    ckey[1] = flag2[2 + 8 * oct_abs_0];
    ckey[2] = flag2[3 + 8 * oct_abs_0];
    int mask      = flag2[4 + 8 * oct_abs_0];

    grid[oct_abs_0].lev    = ilevel;
    grid[oct_abs_0].ckey[0] = ckey[0];
    grid[oct_abs_0].ckey[1] = ckey[1];
    grid[oct_abs_0].ckey[2] = ckey[2];
    for (int c = 0; c < 8; c++)
        grid[oct_abs_0].refined[c] = (mask >> c) & 1;
    grid[oct_abs_0].hkey = hilbert_key_rf(ckey, ilevel - 1);
}

/* =========================================================================
 * sort_gather_flag_kernel — flag2[:,oct] = flag1[:,swap_global[oct]-1].
 * Mirrors sort_gather_flag_impl. 1 thread/oct.
 * ========================================================================= */
kernel void sort_gather_flag_kernel(
    device int       *flag2       [[buffer(0)]],
    device const int *flag1       [[buffer(1)]],
    device const int *swap_global [[buffer(2)]],
    constant int     &head_idx    [[buffer(3)]],
    constant int     &num_octs    [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (int(gid) >= num_octs) return;
    int oct_abs_0  = (head_idx - 1) + int(gid);
    int old_abs_0  = swap_global[oct_abs_0] - 1;
    for (int c = 0; c < 8; c++)
        flag2[c + 8 * oct_abs_0] = flag1[c + 8 * old_abs_0];
}

/* =========================================================================
 * sort_scatter_flag_kernel — flag1[:,oct] = flag2[:,oct].
 * Mirrors sort_scatter_flag_impl. 1 thread/oct.
 * ========================================================================= */
kernel void sort_scatter_flag_kernel(
    device int       *flag1   [[buffer(0)]],
    device const int *flag2   [[buffer(1)]],
    constant int     &head_idx [[buffer(2)]],
    constant int     &num_octs [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (int(gid) >= num_octs) return;
    int oct_abs_0 = (head_idx - 1) + int(gid);
    for (int c = 0; c < 8; c++)
        flag1[c + 8 * oct_abs_0] = flag2[c + 8 * oct_abs_0];
}

/* =========================================================================
 * sort_gather_hydro_kernel — unew[cell,ivar,oct] = uold[cell,ivar,swap_global[oct]-1].
 * Mirrors sort_gather_hydro_impl (HYDRO=1, no MHD).
 * 1D gid = oct_offset * 8 + cell_0; inner loop over NVAR variables.
 * ========================================================================= */
kernel void sort_gather_hydro_kernel(
    device float     *unew        [[buffer(0)]],
    device const float *uold      [[buffer(1)]],
    device const int *swap_global [[buffer(2)]],
    constant int     &head_idx    [[buffer(3)]],
    constant int     &num_octs    [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (int(gid) >= num_octs * 8) return;
    int oct_offset = int(gid) / 8;
    int cell_0     = int(gid) % 8;
    int oct_abs_0  = (head_idx - 1) + oct_offset;
    int old_abs_0  = swap_global[oct_abs_0] - 1;

    for (int ivar_0 = 0; ivar_0 < (NVAR); ivar_0++)
        unew[cell_0 + 8 * ivar_0 + 8 * (NVAR) * oct_abs_0] =
            uold[cell_0 + 8 * ivar_0 + 8 * (NVAR) * old_abs_0];
}

/* =========================================================================
 * update_nbor_prefix_kernel — compute nbor[input_ind, subgrid] and set
 * prefix_sum[subgrid] = 1 if missing, 0 if found.
 * Mirrors update_nbor_array(..., prefix_sum=prefix_sum) in gpu_refine.cuf.
 * NSUBGRID=1 → subgrid_idx == oct_idx; nsubgridp2=3, nsubgridp2sq=9.
 * 1 thread/subgrid.
 * ========================================================================= */
kernel void update_nbor_prefix_kernel(
    device int         *nbor          [[buffer(0)]],
    device const oct_t *grid          [[buffer(1)]],
    device const long  *hash_key      [[buffer(2)]],
    device const int   *hash_val      [[buffer(3)]],
    device const int   *ckey_max_dev  [[buffer(4)]],
    device const long  *key_off_dev   [[buffer(5)]],
    device const int   *box_ckey_min  [[buffer(6)]],
    device const int   *box_ckey_max  [[buffer(7)]],
    device const int   *periodic      [[buffer(8)]],
    device int         *prefix_sum    [[buffer(9)]],
    constant int       &hash_size     [[buffer(10)]],
    constant int       &head_idx      [[buffer(11)]],
    constant int       &num_subgrids  [[buffer(12)]],
    constant int       &input_ind     [[buffer(13)]],
    uint gid [[thread_position_in_grid]])
{
    if (int(gid) >= num_subgrids) return;
    int subgrid_abs_0 = (head_idx - 1) + int(gid);
    int oct_abs_0     = subgrid_abs_0;   /* nsubgrid=1 */

    /* Decode 3D offset from input_ind (1-based, 1..27). */
    int ind0 = input_ind - 1;
    int k_off = ind0 / 9;
    int j_off = (ind0 - k_off * 9) / 3;
    int i_off = ind0 - k_off * 9 - j_off * 3;

    int ilevel = grid[oct_abs_0].lev;
    /* nsubgrid=1 → ckey/nsubgrid = ckey; offset = (ckey/1)*1 + i_off - 1 */
    int ckey_n[3];
    ckey_n[0] = grid[oct_abs_0].ckey[0] + i_off - 1;
    ckey_n[1] = grid[oct_abs_0].ckey[1] + j_off - 1;
    ckey_n[2] = grid[oct_abs_0].ckey[2] + k_off - 1;

    /* Periodic boundary conditions (level-specific box bounds). */
    for (int d = 0; d < 3; d++) {
        if (periodic[d]) {
            int bmin = box_ckey_min[3 * (ilevel - 1) + d];
            int bmax = box_ckey_max[3 * (ilevel - 1) + d];
            if (ckey_n[d] <  bmin) ckey_n[d] = bmax - 1;
            if (ckey_n[d] >= bmax) ckey_n[d] = bmin;
        }
    }

    long nx  = long(ckey_max_dev[ilevel - 1]);
    long key = key_off_dev[ilevel - 1]
               + long(ckey_n[0])
               + long(ckey_n[1]) * nx
               + long(ckey_n[2]) * nx * nx;

    int nbor_val = hash_get(hash_key, hash_val, hash_size, key);
    nbor[subgrid_abs_0 * 27 + (input_ind - 1)] = nbor_val;
    prefix_sum[subgrid_abs_0] = (nbor_val == 0) ? 1 : 0;
}

/* =========================================================================
 * compute_cache_swap_table_kernel — build swap_local for cache oct creation.
 * If prefix_sum bit is 1 (nbor missing), write subgrid_idx (1-based) to
 * swap_local[prev_sum] (0-based slot).
 * Mirrors compute_cache_swap_table_impl in gpu_refine.cuf. 1 thread/subgrid.
 * ========================================================================= */
kernel void compute_cache_swap_table_kernel(
    device int       *swap_local   [[buffer(0)]],
    device const int *prefix_sum   [[buffer(1)]],
    constant int     &head_idx     [[buffer(2)]],
    constant int     &num_subgrids [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (int(gid) >= num_subgrids) return;
    int subgrid_abs_0 = (head_idx - 1) + int(gid);
    int prev_sum = (subgrid_abs_0 > (head_idx - 1)) ? prefix_sum[subgrid_abs_0 - 1] : 0;
    int bit      = prefix_sum[subgrid_abs_0] - prev_sum;
    if (bit == 1)
        swap_local[prev_sum] = subgrid_abs_0 + 1;   /* 1-based subgrid idx */
}

/* =========================================================================
 * make_cache_octs_kernel — create ghost octs in the cache region.
 * Mirrors make_cache_octs_impl (HYDRO=1, no GRAV, no MHD).
 * NSUBGRID=1 → oct_abs_0 = subgrid_abs_0.
 * 1 thread per new cache oct (gid < new_noct).
 * ========================================================================= */
kernel void make_cache_octs_kernel(
    device oct_t       *grid          [[buffer(0)]],
    device int         *flag1         [[buffer(1)]],
    device float       *uold          [[buffer(2)]],
    device const int   *swap_local    [[buffer(3)]],
    device int         *father        [[buffer(4)]],
    device int         *nbor          [[buffer(5)]],
    device const long  *hash_key      [[buffer(6)]],
    device const int   *hash_val      [[buffer(7)]],
    device const int   *ckey_max_dev  [[buffer(8)]],
    device const long  *key_off_dev   [[buffer(9)]],
    device const int   *box_ckey_min  [[buffer(10)]],
    device const int   *box_ckey_max  [[buffer(11)]],
    device const int   *periodic      [[buffer(12)]],
    constant int       &hash_size     [[buffer(13)]],
    constant int       &ngridmax      [[buffer(14)]],
    constant int       &ifree_cache   [[buffer(15)]],   /* 1-based offset (host-read) */
    constant int       &new_noct      [[buffer(16)]],
    constant int       &input_ind     [[buffer(17)]],
    device float       *f_grav        [[buffer(18)]],
    device float       *phi           [[buffer(19)]],
    device float       *phi_old       [[buffer(20)]],
    uint gid [[thread_position_in_grid]])
{
    if (int(gid) >= new_noct) return;

    /* swap_local[gid] holds the 1-based subgrid idx (set by compute_cache_swap_table). */
    int subgrid_1based = swap_local[int(gid)];
    int subgrid_abs_0  = subgrid_1based - 1;
    int oct_abs_0      = subgrid_abs_0;   /* nsubgrid=1 */

    /* Cache grid slot: 1-based = ngridmax + ifree_cache + gid */
    int cache_1based = ngridmax + ifree_cache + int(gid);
    int cache_abs_0  = cache_1based - 1;

    /* Update nbor for this subgrid (mirrors make_cache_octs_impl line 2192). */
    nbor[subgrid_abs_0 * 27 + (input_ind - 1)] = cache_1based;

    /* Decode 3D offset for the neighbour direction. */
    int ind0  = input_ind - 1;
    int k_off = ind0 / 9;
    int j_off = (ind0 - k_off * 9) / 3;
    int i_off = ind0 - k_off * 9 - j_off * 3;

    int ilevel = grid[oct_abs_0].lev;
    int ckey[3];
    ckey[0] = grid[oct_abs_0].ckey[0] + i_off - 1;
    ckey[1] = grid[oct_abs_0].ckey[1] + j_off - 1;
    ckey[2] = grid[oct_abs_0].ckey[2] + k_off - 1;

    for (int d = 0; d < 3; d++) {
        if (periodic[d]) {
            int bmin = box_ckey_min[3 * (ilevel - 1) + d];
            int bmax = box_ckey_max[3 * (ilevel - 1) + d];
            if (ckey[d] <  bmin) ckey[d] = bmax - 1;
            if (ckey[d] >= bmax) ckey[d] = bmin;
        }
    }

    /* Set cache oct grid properties. */
    grid[cache_abs_0].lev    = ilevel;
    grid[cache_abs_0].ckey[0] = ckey[0];
    grid[cache_abs_0].ckey[1] = ckey[1];
    grid[cache_abs_0].ckey[2] = ckey[2];
    for (int c = 0; c < 8; c++) grid[cache_abs_0].refined[c] = 0;
    for (int c = 0; c < 8; c++) flag1[c + 8 * cache_abs_0] = 0;
    grid[cache_abs_0].hkey = hilbert_key_rf(ckey, ilevel - 1);

    /* Compute parent hash key. */
    int parent_lev = ilevel - 1;
    long pnx = long(ckey_max_dev[parent_lev - 1]);
    long pix = long(ckey[0]) / 2L;
    long piy = long(ckey[1]) / 2L;
    long piz = long(ckey[2]) / 2L;
    long parent_key = key_off_dev[parent_lev - 1] + pix + piy * pnx + piz * pnx * pnx;

    int parent_1based = hash_get(hash_key, hash_val, hash_size, parent_key);
    father[cache_abs_0] = parent_1based;

    /* Straight injection: copy parent cell data into all 8 cache cells. */
    if (parent_1based > 0) {
        int parent_abs_0 = parent_1based - 1;
        int pi = int(long(ckey[0]) - 2L * pix);
        int pj = int(long(ckey[1]) - 2L * piy);
        int pk = int(long(ckey[2]) - 2L * piz);
        int cell_0 = pi + 2 * pj + 4 * pk;
        for (int ivar_0 = 0; ivar_0 < (NVAR); ivar_0++) {
            float val = uold[cell_0 + 8 * ivar_0 + 8 * (NVAR) * parent_abs_0];
            for (int c = 0; c < 8; c++)
                uold[c + 8 * ivar_0 + 8 * (NVAR) * cache_abs_0] = val;
        }

        if (f_grav && phi && phi_old) {
            int cell_idx_1based = cell_0 + 1;
            for (int c = 1; c <= 8; c++) {
                f_grav[(cache_1based - 1) * 24 + 0 * 8 + (c - 1)] = f_grav[(parent_1based - 1) * 24 + 0 * 8 + (cell_idx_1based - 1)];
                f_grav[(cache_1based - 1) * 24 + 1 * 8 + (c - 1)] = f_grav[(parent_1based - 1) * 24 + 1 * 8 + (cell_idx_1based - 1)];
                f_grav[(cache_1based - 1) * 24 + 2 * 8 + (c - 1)] = f_grav[(parent_1based - 1) * 24 + 2 * 8 + (cell_idx_1based - 1)];

                phi[(cache_1based - 1) * 8 + (c - 1)] = phi[(parent_1based - 1) * 8 + (cell_idx_1based - 1)];
                phi_old[(cache_1based - 1) * 8 + (c - 1)] = phi_old[(parent_1based - 1) * 8 + (cell_idx_1based - 1)];
            }
        }
    }
}

/* =========================================================================
 * insert_hash_cache_kernel — insert new cache octs into hash table.
 * Mirrors insert_hash_cache_impl in gpu_refine.cuf. 1 thread per cache oct.
 * ========================================================================= */
kernel void insert_hash_cache_kernel(
    device const oct_t *grid        [[buffer(0)]],
    device long        *hash_key    [[buffer(1)]],
    device atomic_int  *hash_val    [[buffer(2)]],
    device const int   *ckey_max_dev [[buffer(3)]],
    device const long  *key_off_dev  [[buffer(4)]],
    constant int       &hash_size   [[buffer(5)]],
    constant int       &ngridmax    [[buffer(6)]],
    constant int       &ifree_cache [[buffer(7)]],
    constant int       &new_noct    [[buffer(8)]],
    uint gid [[thread_position_in_grid]])
{
    if (int(gid) >= new_noct) return;
    int cache_1based = ngridmax + ifree_cache + int(gid);
    int cache_abs_0  = cache_1based - 1;
    int ilevel       = grid[cache_abs_0].lev;
    if (ilevel <= 0) return;

    long nx  = long(ckey_max_dev[ilevel - 1]);
    long ix  = long(grid[cache_abs_0].ckey[0]);
    long iy  = long(grid[cache_abs_0].ckey[1]);
    long iz  = long(grid[cache_abs_0].ckey[2]);
    long key = key_off_dev[ilevel - 1] + ix + iy * nx + iz * nx * nx;
    hash_set_r(hash_key, hash_val, hash_size, key, cache_1based);
}

/* =========================================================================
 * Gravity/Poisson variable sorting kernels (for refinement/reordering)
 * ========================================================================= */
kernel void sort_gather_force_kernel(
    device float       *scratch      [[buffer(0)]], // s_nref
    device const float *f_grav       [[buffer(1)]], // s_f_grav
    device const int   *swap_global  [[buffer(2)]], // s_swap_global
    constant int       &idim         [[buffer(3)]], // 1-based dimension
    constant int       &head_idx     [[buffer(4)]], // 1-based start
    constant int       &num_octs     [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if (int(gid) >= num_octs) return;
    int oct_idx = head_idx + int(gid);
    int old_oct_idx = swap_global[oct_idx - 1];

    for (int c = 0; c < 8; c++) {
        scratch[(oct_idx - 1) * 8 + c] = f_grav[(old_oct_idx - 1) * 24 + (idim - 1) * 8 + c];
    }
}

kernel void sort_scatter_force_kernel(
    device float       *f_grav       [[buffer(0)]], // s_f_grav
    device const float *scratch      [[buffer(1)]], // s_nref
    constant int       &idim         [[buffer(2)]], // 1-based dimension
    constant int       &head_idx     [[buffer(3)]], // 1-based start
    constant int       &num_octs     [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (int(gid) >= num_octs) return;
    int oct_idx = head_idx + int(gid);

    for (int c = 0; c < 8; c++) {
        f_grav[(oct_idx - 1) * 24 + (idim - 1) * 8 + c] = scratch[(oct_idx - 1) * 8 + c];
    }
}

kernel void sort_gather_phi_kernel(
    device float       *scratch      [[buffer(0)]], // s_nref
    device const float *phi          [[buffer(1)]], // s_phi or s_phi_old
    device const int   *swap_global  [[buffer(2)]], // s_swap_global
    constant int       &head_idx     [[buffer(3)]], // 1-based start
    constant int       &num_octs     [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (int(gid) >= num_octs) return;
    int oct_idx = head_idx + int(gid);
    int old_oct_idx = swap_global[oct_idx - 1];

    for (int c = 0; c < 8; c++) {
        scratch[(oct_idx - 1) * 8 + c] = phi[(old_oct_idx - 1) * 8 + c];
    }
}

kernel void sort_scatter_phi_kernel(
    device float       *phi          [[buffer(0)]], // s_phi or s_phi_old
    device const float *scratch      [[buffer(1)]], // s_nref
    constant int       &head_idx     [[buffer(2)]], // 1-based start
    constant int       &num_octs     [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (int(gid) >= num_octs) return;
    int oct_idx = head_idx + int(gid);

    for (int c = 0; c < 8; c++) {
        phi[(oct_idx - 1) * 8 + c] = scratch[(oct_idx - 1) * 8 + c];
    }
}
