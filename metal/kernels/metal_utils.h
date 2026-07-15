/*
 * metal/kernels/metal_utils.h
 *
 * Shared inline helpers for Metal kernel files.
 * All functions are static inline to avoid multiple-definition linker errors
 * when included by multiple .metal translation units.
 */
#pragma once
#include <metal_stdlib>
#include <metal_atomic>
using namespace metal;

/* ---------------------------------------------------------------------------
 * Atomic helpers
 * --------------------------------------------------------------------------*/

/* CAS loop for float atomic add. */
static inline void atomic_add_float(device atomic_uint *a, float val) {
    uint expected = atomic_load_explicit(a, memory_order_relaxed);
    uint desired;
    do {
        desired = as_type<uint>(as_type<float>(expected) + val);
    } while (!atomic_compare_exchange_weak_explicit(
             a, &expected, desired, memory_order_relaxed, memory_order_relaxed));
}

/* Float min via uint bit-cast — valid for positive IEEE-754 floats only. */
static inline void atomic_min_float_bits(device atomic_uint *a, float val) {
    atomic_fetch_min_explicit(a, as_type<uint>(val), memory_order_relaxed);
}

/* CAS loop for float atomic max. */
static inline void atomic_max_float(device atomic_uint *a, float val) {
    uint expected = atomic_load_explicit(a, memory_order_relaxed);
    uint desired;
    do {
        desired = as_type<uint>(max(as_type<float>(expected), val));
    } while (!atomic_compare_exchange_weak_explicit(
             a, &expected, desired, memory_order_relaxed, memory_order_relaxed));
}

/* ---------------------------------------------------------------------------
 * Layout helpers
 * --------------------------------------------------------------------------*/

/* Decomposes 1D index into (i,j,k) with i varying fastest (Fortran column-major). */
static inline void index_1Dto3D(int idx, int sx, int sy,
                                 thread int &i, thread int &j, thread int &k) {
    i = idx % sx;
    j = (idx / sx) % sy;
    k = idx / (sx * sy);
}

/* nbor(ind_1, sg_1) — Fortran column-major: ind_1 varies fastest.
 * Metal flat index: (sg_1-1)*27 + (ind_1-1). */
static inline int nbor_get(device const int *nb, int sg_1, int ind_1) {
    return nb[(sg_1-1)*27 + (ind_1-1)];
}

/* ---------------------------------------------------------------------------
 * Block reductions — for 256-thread threadgroups (8 SIMD groups of 32 lanes).
 * Caller must declare threadgroup float tg_sums[8] in the kernel.
 * Return value is only valid in thread 0.
 * --------------------------------------------------------------------------*/

/* 256-thread (8 SIMD groups) variants — caller declares threadgroup float tg_sums[8].
 * NOTE: no trailing barrier. If called multiple times with the same tg_sums array,
 * the caller must insert threadgroup_barrier(mem_flags::mem_threadgroup) between calls. */
static inline float tg_reduce_sum_f(float val, uint tid,
                                     threadgroup float tg_sums[8]) {
    uint lane   = tid % 32u;
    uint sg_idx = tid / 32u;
    val = simd_sum(val);
    if (lane == 0u) tg_sums[sg_idx] = val;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg_idx == 0u) {
        val = (lane < 8u) ? tg_sums[lane] : 0.0f;
        val = simd_sum(val);
    }
    return val;
}

/* NOTE: same contract as tg_reduce_sum_f — caller must barrier between repeated calls. */
static inline float tg_reduce_max_f(float val, uint tid,
                                     threadgroup float tg_sums[8]) {
    uint lane   = tid % 32u;
    uint sg_idx = tid / 32u;
    val = simd_max(val);
    if (lane == 0u) tg_sums[sg_idx] = val;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg_idx == 0u) {
        val = (lane < 8u) ? tg_sums[lane] : -HUGE_VALF;
        val = simd_max(val);
    }
    return val;
}

/* 1024-thread (32 SIMD groups) variants — caller declares threadgroup float tg_sums[32]. */
static inline float tg_reduce_sum_f_1024(float val, uint tid,
                                          threadgroup float tg_sums[32]) {
    uint lane   = tid % 32u;
    uint sg_idx = tid / 32u;
    val = simd_sum(val);
    if (lane == 0u) tg_sums[sg_idx] = val;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg_idx == 0u) {
        val = tg_sums[lane];   /* exactly 32 SIMD groups → all 32 lanes used */
        val = simd_sum(val);
    }
    return val;
}

/* ---------------------------------------------------------------------------
 * FNV-1a 64-bit hash + linear-probing hash table lookup.
 * Matches the convention used by refine.metal's hash_get / insert_hash_all.
 * --------------------------------------------------------------------------*/

static inline ulong fnv64(long key_signed) {
    ulong key = as_type<ulong>(key_signed);
    ulong h   = 14695981039346656037UL;
    for (int j = 0; j < 8; j++) {
        ulong b = (key >> (8 * j)) & 0xFFUL;
        h ^= b;
        h *= 1099511628211UL;
    }
    return h;
}

/* 1-based bucket index — mirrors hash_bucket_r in refine.metal. */
static inline int hash_bucket(long key, int hash_size) {
    return int(fnv64(key) % ulong(hash_size)) + 1;
}

/* Linear-probing lookup — mirrors hash_get in refine.metal.
 * Returns 0 if key is absent. */
static inline int hash_get(device const long *hash_key,
                            device const int  *hash_val,
                            int hash_size, long key) {
    int b = hash_bucket(key, hash_size);
    for (;;) {
        long cur = hash_key[b - 1];
        if (cur == key) return hash_val[b - 1];
        if (cur == 0L)  return 0;
        b = (b % hash_size) + 1;
    }
}
