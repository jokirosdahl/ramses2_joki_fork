/*
 * metal/kernels/hash.metal
 *
 * Metal port of gpu/gpu_hash.cuf — insert_hash_kernel.
 *
 * Apple GPU has no 64-bit CAS.  The design mirrors the reference branch:
 *   hash_val : device atomic_int[hash_size]   (1-based oct index; 0 == empty)
 *   hash_key : device long[hash_size]          (64-bit Hilbert key; non-atomic)
 *
 * Insertion: CAS on hash_val to claim a bucket (0 → oct_idx), then write
 * hash_key non-atomically.  This is safe because oct keys are unique — no two
 * threads contest the same final slot.
 *
 * Read path: hash_get() is inline in the read-only kernel headers (refine.metal
 * reads hash_key / hash_val non-atomically after the insert dispatch completes).
 */

#include <metal_stdlib>
#include <metal_atomic>
#include "../metal_types.h"
using namespace metal;

/* =========================================================================
 * fnv64 — FNV-1a 64-bit hash, mirrors gpu_hash.cuf::fnv64.
 * Operates byte-by-byte to match the Fortran ichar() loop exactly.
 * Returns the signed bit-pattern so the bucket modulo reproduces the
 * signed Fortran MOD + fixup.
 * ========================================================================= */
inline long fnv64(long key_signed)
{
    ulong key = as_type<ulong>(key_signed);
    ulong h   = 14695981039346656037UL;   /* FNV-1a offset basis */
    for (int j = 0; j < 8; j++) {
        ulong b = (key >> (8 * j)) & 0xFFUL;
        h ^= b;
        h *= 1099511628211UL;             /* FNV-1a prime */
    }
    return as_type<long>(h);
}

/* =========================================================================
 * hash_bucket — 1-based bucket index, matching Fortran:
 *   ibucket = MOD(fnv64(key), hash_size); if(<0) += hash_size; ibucket+1
 * ========================================================================= */
inline int hash_bucket(long key, int hash_size)
{
    long ib = fnv64(key) % long(hash_size);
    if (ib < 0) ib += long(hash_size);
    return int(ib) + 1;
}

/* =========================================================================
 * insert_hash_kernel — mirrors insert_hash_kernel in gpu_refine.cuf.
 *
 * Each thread inserts one oct's Hilbert key into the hash table using a
 * 32-bit CAS on hash_val (the claim slot).  After claiming a bucket the
 * thread writes hash_key non-atomically; because oct keys are unique, no
 * other thread will target this slot.
 *
 * Thread layout: 128 threads/threadgroup — mirrors CUDA num_threads=128.
 * ========================================================================= */
kernel void insert_hash_kernel(
    device const oct_t   *grid       [[buffer(0)]],
    device long          *hash_key   [[buffer(1)]],   /* non-atomic long  */
    device atomic_int    *hash_val   [[buffer(2)]],   /* 32-bit CAS claim */
    constant int         &hash_size  [[buffer(3)]],
    constant int         &ckey_max_l [[buffer(4)]],
    constant long        &key_off_l  [[buffer(5)]],
    constant int         &head_idx   [[buffer(6)]],
    constant int         &num_octs   [[buffer(7)]],
    uint gid [[thread_position_in_grid]])
{
    if (int(gid) >= num_octs) return;
    int oct_idx = head_idx + int(gid);   /* 1-based */

    long nx  = long(ckey_max_l);
    long ix  = long(grid[oct_idx - 1].ckey[0]);
    long iy  = long(grid[oct_idx - 1].ckey[1]);
    long iz  = long(grid[oct_idx - 1].ckey[2]);
    long key = key_off_l + ix + iy * nx + iz * nx * nx;

    int b = hash_bucket(key, hash_size);
    for (;;) {
        int expected = 0;
        if (atomic_compare_exchange_weak_explicit(
                &hash_val[b - 1], &expected, oct_idx,
                memory_order_relaxed, memory_order_relaxed)) {
            hash_key[b - 1] = key;   /* non-atomic write after slot claimed */
            return;
        }
        if (expected != 0) b = (b % hash_size) + 1;   /* occupied → probe */
        /* expected == 0: spurious weak-CAS failure → retry same slot       */
    }
}
