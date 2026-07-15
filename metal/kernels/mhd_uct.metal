#include <metal_stdlib>
#include <metal_atomic>
using namespace metal;

#include "../metal_types.h"

#ifndef NVAR
#define NVAR 5
#endif

#if NSUBGRID != 1 && NSUBGRID != 2
#error NSUBGRID must be 1 or 2
#endif

#define MR_TWOTONDIM 8
#define MR_NSUBGRID NSUBGRID
#define MR_NSUBGRIDP2 (MR_NSUBGRID + 2)
#define MR_SUBGRIDSIZE (MR_NSUBGRIDP2 * MR_NSUBGRIDP2 * MR_NSUBGRIDP2)
#define MR_M (2 * MR_NSUBGRID)
#define MR_STENCIL (MR_M + 4)
#define MR_TRACE (MR_M + 2)
#define MR_LOCAL_CELLS (MR_STENCIL * MR_STENCIL * MR_STENCIL)
#define MR_REFINED_CELLS (MR_TRACE * MR_TRACE * MR_TRACE)
#define MR_BF_COMPONENT ((MR_STENCIL - 1) * MR_STENCIL * MR_STENCIL)
#define MR_FACE_CELLS ((MR_M + 1) * MR_M * MR_M)
#define MR_ALL_FACES (3 * MR_FACE_CELLS)
#define MR_SHELL_CELLS (4 * MR_M * (MR_M + 1))
#define MR_ALL_SHELLS (3 * MR_SHELL_CELLS)
#define MR_EDGE_CELLS (MR_M * (MR_M + 1) * (MR_M + 1))
#define MR_ALL_EDGES (3 * MR_EDGE_CELLS)
#define MR_INTERIOR_CELLS (MR_M * MR_M * MR_M)
#define MR_LOCAL_BASE 0
#define MR_BF_BASE (5 * MR_LOCAL_CELLS)
#define MR_FACE_BASE (MR_BF_BASE + 3 * MR_BF_COMPONENT)
#define MR_FACE_BUFFER (8 * MR_FACE_CELLS)
#define MR_FLUX_FLOATS (5 * MR_ALL_FACES)
#define MR_RECORD_FLOATS (6 * MR_ALL_FACES)
#define MR_SHELL_FREE (6 * MR_FACE_BUFFER - MR_FLUX_FLOATS - MR_RECORD_FLOATS)
#define MR_SHELL_RECORD_FLOATS (6 * MR_ALL_SHELLS)
#define MR_SHELL_SPILL (MR_SHELL_RECORD_FLOATS - MR_SHELL_FREE)
#define MR_EMF_BASE MR_SHELL_SPILL
#define MR_SMEM_FLOATS (MR_FACE_BASE + 6 * MR_FACE_BUFFER)
#define MR_SET_Y 16

struct mr_primitive_t {
    float density;
    float velocity_x;
    float velocity_y;
    float velocity_z;
    float pressure;
    float Bx;
    float By;
    float Bz;
};

struct mr_conserved_t {
    float density;
    float momentum_x;
    float momentum_y;
    float momentum_z;
    float energy;
    float Bx;
    float By;
    float Bz;
};

struct mr_uct_record_t {
    float aL;
    float dL;
    float dR;
    float vt1;
    float vt2;
    float Bn;
};

struct mr_trace_t {
    mr_primitive_t cell;
    mr_primitive_t sx;
    mr_primitive_t sy;
    mr_primitive_t sz;
    float AL;
    float AR;
    float BL;
    float BR;
    float CL;
    float CR;
};

inline int mr_u_flat(int oct_idx, int ivar, int cell_idx) {
    return (oct_idx - 1) * NVAR * MR_TWOTONDIM + (ivar - 1) * MR_TWOTONDIM + cell_idx - 1;
}
inline float mr_u_get(device const float *u, int oct_idx, int ivar, int cell_idx) {
    return u[mr_u_flat(oct_idx, ivar, cell_idx)];
}

inline void mr_u_set(device float *u, int oct_idx, int ivar, int cell_idx, float value) {
    u[mr_u_flat(oct_idx, ivar, cell_idx)] = value;
}

inline int mr_b_flat(int oct_idx, int ivar, int cell_idx) {
    return (oct_idx - 1) * 6 * MR_TWOTONDIM + (ivar - 1) * MR_TWOTONDIM + cell_idx - 1;
}

inline float mr_b_get(device const float *b, int oct_idx, int ivar, int cell_idx) {
    return b[mr_b_flat(oct_idx, ivar, cell_idx)];
}

inline void mr_b_set(device float *b, int oct_idx, int ivar, int cell_idx, float value) {
    b[mr_b_flat(oct_idx, ivar, cell_idx)] = value;
}

inline int mr_nbor_get(device const int *nbor, int subgrid_idx, int ind_nbor) {
    return nbor[(subgrid_idx - 1) * MR_SUBGRIDSIZE + ind_nbor - 1];
}

inline void mr_index3(int index, int nx, int ny, thread int &i, thread int &j, thread int &k) {
    i = index % nx;
    j = (index / nx) % ny;
    k = index / (nx * ny);
}

inline int mr_local_index(int i, int j, int k) {
    return i + MR_STENCIL * (j + MR_STENCIL * k);
}

inline float mr_local_get(threadgroup const float *s, int field, int i, int j, int k) {
    return s[MR_LOCAL_BASE + field * MR_LOCAL_CELLS + mr_local_index(i, j, k)];
}

inline void mr_local_set(threadgroup float *s, int field, int i, int j, int k, float value) {
    s[MR_LOCAL_BASE + field * MR_LOCAL_CELLS + mr_local_index(i, j, k)] = value;
}

inline int mr_refined_index(int i, int j, int k) {
    return (i - 1) + MR_TRACE * ((j - 1) + MR_TRACE * (k - 1));
}

inline bool mr_refined_get(threadgroup const bool *refined, int i, int j, int k) {
    return refined[mr_refined_index(i, j, k)];
}

inline void mr_refined_set(threadgroup bool *refined, int i, int j, int k, bool value) {
    refined[mr_refined_index(i, j, k)] = value;
}

inline int mr_bx_index(int i, int j, int k) {
    return i - 1 + (MR_STENCIL - 1) * (j + MR_STENCIL * k);
}

inline int mr_by_index(int i, int j, int k) {
    return i + MR_STENCIL * (j - 1 + (MR_STENCIL - 1) * k);
}

inline int mr_bz_index(int i, int j, int k) {
    return i + MR_STENCIL * (j + MR_STENCIL * (k - 1));
}

inline float mr_bf_get(threadgroup const float *s, int component, int i, int j, int k) {
    int index = component == 0 ? mr_bx_index(i, j, k) : component == 1 ? mr_by_index(i, j, k) : mr_bz_index(i, j, k);
    return s[MR_BF_BASE + component * MR_BF_COMPONENT + index];
}

inline void mr_bf_set(threadgroup float *s, int component, int i, int j, int k, float value) {
    int index = component == 0 ? mr_bx_index(i, j, k) : component == 1 ? mr_by_index(i, j, k) : mr_bz_index(i, j, k);
    s[MR_BF_BASE + component * MR_BF_COMPONENT + index] = value;
}

inline int mr_face_index(int orientation, int i, int j, int k) {
    if (orientation == 0) return i + (MR_M + 1) * (j + MR_M * k);
    if (orientation == 1) return i + MR_M * (j + (MR_M + 1) * k);
    return i + MR_M * (j + MR_M * k);
}

inline float mr_face_get(threadgroup const float *s, int buffer, int field, int index) {
    return s[MR_FACE_BASE + buffer * MR_FACE_BUFFER + field * MR_FACE_CELLS + index];
}

inline void mr_face_set(threadgroup float *s, int buffer, int field, int index, float value) {
    s[MR_FACE_BASE + buffer * MR_FACE_BUFFER + field * MR_FACE_CELLS + index] = value;
}

inline mr_primitive_t mr_face_load(threadgroup const float *s, int buffer, int index) {
    mr_primitive_t q;
    q.density = mr_face_get(s, buffer, 0, index);
    q.velocity_x = mr_face_get(s, buffer, 1, index);
    q.velocity_y = mr_face_get(s, buffer, 2, index);
    q.velocity_z = mr_face_get(s, buffer, 3, index);
    q.pressure = mr_face_get(s, buffer, 4, index);
    q.Bx = mr_face_get(s, buffer, 5, index);
    q.By = mr_face_get(s, buffer, 6, index);
    q.Bz = mr_face_get(s, buffer, 7, index);
    return q;
}

inline void mr_face_store(threadgroup float *s, int buffer, int index, mr_primitive_t q) {
    mr_face_set(s, buffer, 0, index, q.density);
    mr_face_set(s, buffer, 1, index, q.velocity_x);
    mr_face_set(s, buffer, 2, index, q.velocity_y);
    mr_face_set(s, buffer, 3, index, q.velocity_z);
    mr_face_set(s, buffer, 4, index, q.pressure);
    mr_face_set(s, buffer, 5, index, q.Bx);
    mr_face_set(s, buffer, 6, index, q.By);
    mr_face_set(s, buffer, 7, index, q.Bz);
}

inline float mr_flux_get(threadgroup const float *s, int field, int face) {
    return s[MR_FACE_BASE + field * MR_ALL_FACES + face];
}

inline void mr_flux_set(threadgroup float *s, int field, int face, float value) {
    s[MR_FACE_BASE + field * MR_ALL_FACES + face] = value;
}

inline float mr_record_value(mr_uct_record_t r, int field) {
    if (field == 0) return r.aL;
    if (field == 1) return r.dL;
    if (field == 2) return r.dR;
    if (field == 3) return r.vt1;
    if (field == 4) return r.vt2;
    return r.Bn;
}

inline void mr_record_set_value(thread mr_uct_record_t &r, int field, float value) {
    if (field == 0) r.aL = value;
    else if (field == 1) r.dL = value;
    else if (field == 2) r.dR = value;
    else if (field == 3) r.vt1 = value;
    else if (field == 4) r.vt2 = value;
    else r.Bn = value;
}

inline void mr_interior_record_store(threadgroup float *s, int face, mr_uct_record_t r) {
    for (int field = 0; field < 6; ++field) s[MR_FACE_BASE + MR_FLUX_FLOATS + field * MR_ALL_FACES + face] = mr_record_value(r, field);
}

inline mr_uct_record_t mr_interior_record_load(threadgroup const float *s, int face) {
    mr_uct_record_t r;
    for (int field = 0; field < 6; ++field) mr_record_set_value(r, field, s[MR_FACE_BASE + MR_FLUX_FLOATS + field * MR_ALL_FACES + face]);
    return r;
}

inline void mr_shell_record_store(threadgroup float *s, int shell, mr_uct_record_t r) {
    for (int field = 0; field < 6; ++field) {
        int index = field * MR_ALL_SHELLS + shell;
        if (index < MR_SHELL_FREE) s[MR_FACE_BASE + MR_FLUX_FLOATS + MR_RECORD_FLOATS + index] = mr_record_value(r, field);
        else s[index - MR_SHELL_FREE] = mr_record_value(r, field);
    }
}

inline mr_uct_record_t mr_shell_record_load(threadgroup const float *s, int shell) {
    mr_uct_record_t r;
    for (int field = 0; field < 6; ++field) {
        int index = field * MR_ALL_SHELLS + shell;
        float value = index < MR_SHELL_FREE ? s[MR_FACE_BASE + MR_FLUX_FLOATS + MR_RECORD_FLOATS + index] : s[index - MR_SHELL_FREE];
        mr_record_set_value(r, field, value);
    }
    return r;
}

inline int mr_emfz_index(int i, int j, int k) {
    return i + (MR_M + 1) * (j + (MR_M + 1) * k);
}

inline int mr_emfy_index(int i, int j, int k) {
    return MR_EDGE_CELLS + i + (MR_M + 1) * (j + MR_M * k);
}

inline int mr_emfx_index(int i, int j, int k) {
    return 2 * MR_EDGE_CELLS + i + MR_M * (j + (MR_M + 1) * k);
}

inline float mr_emf_get(threadgroup const float *s, int orientation, int i, int j, int k) {
    int index = orientation == 0 ? mr_emfz_index(i, j, k) : orientation == 1 ? mr_emfy_index(i, j, k) : mr_emfx_index(i, j, k);
    return s[MR_EMF_BASE + index];
}

inline void mr_emf_set(threadgroup float *s, int orientation, int i, int j, int k, float value) {
    int index = orientation == 0 ? mr_emfz_index(i, j, k) : orientation == 1 ? mr_emfy_index(i, j, k) : mr_emfx_index(i, j, k);
    s[MR_EMF_BASE + index] = value;
}

inline float mr_magnitude_squared(float x, float y, float z) {
    return x * x + (y * y + z * z);
}

inline float mr_bsquared(float x, float y, float z) {
    return x * x + (y * y + z * z);
}

inline float mr_emag(float x, float y, float z) {
    return 0.5f * mr_bsquared(x, y, z);
}

inline float mr_compute_pressure(mr_conserved_t u, float gamma, float smallr, float smallc2) {
    float density = max(u.density, smallr);
    float eint = u.energy - 0.5f * mr_magnitude_squared(u.momentum_x, u.momentum_y, u.momentum_z) / density;
    eint -= mr_emag(u.Bx, u.By, u.Bz);
    return max((gamma - 1.0f) * eint, density * smallc2 / gamma);
}

inline float mr_compute_energy(mr_primitive_t q, float gamma) {
    return q.pressure / (gamma - 1.0f) + 0.5f * q.density * mr_magnitude_squared(q.velocity_x, q.velocity_y, q.velocity_z) + mr_emag(q.Bx, q.By, q.Bz);
}

inline mr_primitive_t mr_conserved_to_primitive(mr_conserved_t u, float gamma, float smallr, float smallc2) {
    mr_primitive_t q;
    q.density = max(u.density, smallr);
    q.velocity_x = u.momentum_x / q.density;
    q.velocity_y = u.momentum_y / q.density;
    q.velocity_z = u.momentum_z / q.density;
    q.pressure = mr_compute_pressure(u, gamma, smallr, smallc2);
    q.Bx = u.Bx;
    q.By = u.By;
    q.Bz = u.Bz;
    return q;
}

inline mr_conserved_t mr_primitive_to_conserved(mr_primitive_t q, float gamma) {
    mr_conserved_t u;
    u.density = q.density;
    u.momentum_x = q.density * q.velocity_x;
    u.momentum_y = q.density * q.velocity_y;
    u.momentum_z = q.density * q.velocity_z;
    u.energy = mr_compute_energy(q, gamma);
    u.Bx = q.Bx;
    u.By = q.By;
    u.Bz = q.Bz;
    return u;
}

inline float mr_slope_moncen(float left, float middle, float right, int slope) {
    float dl = middle - left;
    float dr = right - middle;
    float dc = 0.5f * (dl + dr);
    float factor = float(slope);
    if (dl * dr <= 0.0f) return 0.0f;
    if (dl > 0.0f) return min(factor * min(dl, dr), dc);
    return max(factor * max(dl, dr), dc);
}

inline float mr_face_b_slope(float bm, float b0, float bp, int slope_mag) {
    if (slope_mag == 0) return 0.0f;
    float factor = float(min(slope_mag, 2));
    float dl = factor * (b0 - bm);
    float dr = factor * (bp - b0);
    float dc = 0.5f * (dl + dr) / factor;
    float limit = dl * dr <= 0.0f ? 0.0f : min(abs(dl), abs(dr));
    return copysign(min(limit, abs(dc)), dc);
}

inline float mr_hll(float sl, float sr, float fl, float fr, float ul, float ur) {
    if (sl >= 0.0f) return fl;
    if (sr <= 0.0f) return fr;
    return (sr * fl - sl * fr + sl * sr * (ur - ul)) / (sr - sl);
}

inline mr_primitive_t mr_rotate_face(mr_primitive_t q, int orientation) {
    if (orientation == 0) return q;
    mr_primitive_t r = q;
    if (orientation == 1) {
        r.velocity_x = q.velocity_y;
        r.velocity_y = q.velocity_x;
        r.velocity_z = q.velocity_z;
        r.Bx = q.By;
        r.By = q.Bx;
        r.Bz = q.Bz;
    } else {
        r.velocity_x = q.velocity_z;
        r.velocity_y = q.velocity_x;
        r.velocity_z = q.velocity_y;
        r.Bx = q.Bz;
        r.By = q.Bx;
        r.Bz = q.By;
    }
    return r;
}

inline mr_conserved_t mr_unrotate_flux(mr_conserved_t f, int orientation) {
    if (orientation == 0) return f;
    mr_conserved_t r = f;
    if (orientation == 1) {
        r.momentum_x = f.momentum_y;
        r.momentum_y = f.momentum_x;
        r.momentum_z = f.momentum_z;
    } else {
        r.momentum_x = f.momentum_y;
        r.momentum_y = f.momentum_z;
        r.momentum_z = f.momentum_x;
    }
    return r;
}

inline void mr_set_uct_record(thread mr_uct_record_t &record, float ul, float ur, float vl, float vr, float wl, float wr, float A, float SL, float SR, float ustar, float SAL, float SAR) {
    float ap = max(SR, 0.0f);
    float am = -min(SL, 0.0f);
    float asum = ap + am;
    record.vt1 = asum > 0.0f ? (ap * vl + am * vr) / asum : 0.5f * (vl + vr);
    record.vt2 = asum > 0.0f ? (ap * wl + am * wr) / asum : 0.5f * (wl + wr);
    float eps = max(1.0e-9f, 16.0f * FLT_EPSILON);
    float den = abs(SAR) + abs(SAL);
    float nustar = abs(SAR - SAL) > eps * abs(SR - SL) && den > 0.0f ? (SAR + SAL) / den : 0.0f;
    den = abs(SAL) + abs(SL);
    float nuL = den > 0.0f ? (SAL + SL) / den : 0.0f;
    den = abs(SAR) + abs(SR);
    float nuR = den > 0.0f ? (SAR + SR) / den : 0.0f;
    den = SAL + SL - 2.0f * ustar;
    float scale = max(abs(SAL), max(abs(SL), abs(ustar)));
    float tchiL = abs(den) > eps * scale ? (ul - ustar) * (SL - ustar) / den : 0.5f * (ul - ustar);
    den = SAR + SR - 2.0f * ustar;
    scale = max(abs(SAR), max(abs(SR), abs(ustar)));
    float tchiR = abs(den) > eps * scale ? (ur - ustar) * (SR - ustar) / den : 0.5f * (ur - ustar);
    record.aL = 0.5f * (1.0f + nustar);
    record.dL = 0.5f * (nuL - nustar) * tchiL + 0.5f * (abs(SAL) - nustar * SAL);
    record.dR = 0.5f * (nuR - nustar) * tchiR + 0.5f * (abs(SAR) - nustar * SAR);
    record.Bn = A;
}

inline mr_uct_record_t mr_llf_record(mr_primitive_t left, mr_primitive_t right, float gamma, float smallr, float smallc2) {
    float smallp = smallc2 / gamma;
    left.density = max(left.density, smallr);
    right.density = max(right.density, smallr);
    left.pressure = max(left.pressure, smallp * left.density);
    right.pressure = max(right.pressure, smallp * right.density);
    float A = 0.5f * (left.Bx + right.Bx);
    float b2l = A * A + left.By * left.By + left.Bz * left.Bz;
    float c2l = gamma * left.pressure / left.density;
    float h2l = 0.5f * (b2l / left.density + c2l);
    float cfl = sqrt(h2l + sqrt(max(h2l * h2l - c2l * A * A / left.density, 0.0f)));
    float b2r = A * A + right.By * right.By + right.Bz * right.Bz;
    float c2r = gamma * right.pressure / right.density;
    float h2r = 0.5f * (b2r / right.density + c2r);
    float cfr = sqrt(h2r + sqrt(max(h2r * h2r - c2r * A * A / right.density, 0.0f)));
    float lmax = max(abs(left.velocity_x) + cfl, abs(right.velocity_x) + cfr);
    mr_uct_record_t record;
    record.aL = 0.5f;
    record.dL = 0.5f * lmax;
    record.dR = 0.5f * lmax;
    record.vt1 = 0.5f * (left.velocity_y + right.velocity_y);
    record.vt2 = 0.5f * (left.velocity_z + right.velocity_z);
    record.Bn = A;
    return record;
}

inline mr_uct_record_t mr_hlld_record(mr_primitive_t left, mr_primitive_t right, float gamma, float smallr, float smallc2) {
    float smallp = smallc2 / gamma;
    left.density = max(left.density, smallr);
    right.density = max(right.density, smallr);
    left.pressure = max(left.pressure, smallp * left.density);
    right.pressure = max(right.pressure, smallp * right.density);
    float A = 0.5f * (left.Bx + right.Bx);
    float rl = left.density;
    float rr = right.density;
    float ul = left.velocity_x;
    float ur = right.velocity_x;
    float b2l = A * A + left.By * left.By + left.Bz * left.Bz;
    float b2r = A * A + right.By * right.By + right.Bz * right.Bz;
    float ptotl = left.pressure + 0.5f * b2l;
    float ptotr = right.pressure + 0.5f * b2r;
    float c2l = gamma * left.pressure / rl;
    float h2l = 0.5f * (b2l / rl + c2l);
    float cfl = sqrt(h2l + sqrt(max(h2l * h2l - c2l * A * A / rl, 0.0f)));
    float c2r = gamma * right.pressure / rr;
    float h2r = 0.5f * (b2r / rr + c2r);
    float cfr = sqrt(h2r + sqrt(max(h2r * h2r - c2r * A * A / rr, 0.0f)));
    float SL = min(ul, ur) - max(cfl, cfr);
    float SR = max(ul, ur) + max(cfl, cfr);
    float rcl = rl * (ul - SL);
    float rcr = rr * (SR - ur);
    float ustar = (rcr * ur + rcl * ul + ptotl - ptotr) / (rcr + rcl);
    float rstarl = max(rl * (SL - ul) / (SL - ustar), smallr);
    float rstarr = max(rr * (SR - ur) / (SR - ustar), smallr);
    float SAL = ustar - abs(A) / sqrt(rstarl);
    float SAR = ustar + abs(A) / sqrt(rstarr);
    mr_uct_record_t record;
    mr_set_uct_record(record, ul, ur, left.velocity_y, right.velocity_y, left.velocity_z, right.velocity_z, A, SL, SR, ustar, SAL, SAR);
    return record;
}

inline mr_conserved_t mr_hll_mhd_flux(mr_primitive_t left, mr_primitive_t right, float gamma, float smallr, float smallc2) {
    float smallp = smallc2 / gamma;
    float A = 0.5f * (left.Bx + right.Bx);
    left.Bx = A;
    right.Bx = A;
    left.density = max(left.density, smallr);
    right.density = max(right.density, smallr);
    left.pressure = max(left.pressure, smallp * left.density);
    right.pressure = max(right.pressure, smallp * right.density);
    mr_conserved_t ul = mr_primitive_to_conserved(left, gamma);
    mr_conserved_t ur = mr_primitive_to_conserved(right, gamma);
    float ptl = left.pressure + mr_emag(left.Bx, left.By, left.Bz);
    float ptr = right.pressure + mr_emag(right.Bx, right.By, right.Bz);
    float vbl = left.Bx * left.velocity_x + left.By * left.velocity_y + left.Bz * left.velocity_z;
    float vbr = right.Bx * right.velocity_x + right.By * right.velocity_y + right.Bz * right.velocity_z;
    mr_conserved_t fl;
    fl.density = left.density * left.velocity_x;
    fl.momentum_x = left.density * left.velocity_x * left.velocity_x + ptl - A * A;
    fl.momentum_y = left.density * left.velocity_x * left.velocity_y - A * left.By;
    fl.momentum_z = left.density * left.velocity_x * left.velocity_z - A * left.Bz;
    fl.energy = (ul.energy + ptl) * left.velocity_x - A * vbl;
    fl.Bx = 0.0f;
    fl.By = left.By * left.velocity_x - A * left.velocity_y;
    fl.Bz = left.Bz * left.velocity_x - A * left.velocity_z;
    mr_conserved_t fr;
    fr.density = right.density * right.velocity_x;
    fr.momentum_x = right.density * right.velocity_x * right.velocity_x + ptr - A * A;
    fr.momentum_y = right.density * right.velocity_x * right.velocity_y - A * right.By;
    fr.momentum_z = right.density * right.velocity_x * right.velocity_z - A * right.Bz;
    fr.energy = (ur.energy + ptr) * right.velocity_x - A * vbr;
    fr.Bx = 0.0f;
    fr.By = right.By * right.velocity_x - A * right.velocity_y;
    fr.Bz = right.Bz * right.velocity_x - A * right.velocity_z;
    float b2l = mr_bsquared(A, left.By, left.Bz);
    float c2l = gamma * left.pressure / left.density;
    float d2l = 0.5f * (b2l / left.density + c2l);
    float cfl = sqrt(d2l + sqrt(max(d2l * d2l - c2l * A * A / left.density, 0.0f)));
    float b2r = mr_bsquared(A, right.By, right.Bz);
    float c2r = gamma * right.pressure / right.density;
    float d2r = 0.5f * (b2r / right.density + c2r);
    float cfr = sqrt(d2r + sqrt(max(d2r * d2r - c2r * A * A / right.density, 0.0f)));
    float speed = max(abs(left.velocity_x) + cfl, abs(right.velocity_x) + cfr);
    float sl = -speed;
    float sr = speed;
    mr_conserved_t flux;
    flux.density = mr_hll(sl, sr, fl.density, fr.density, ul.density, ur.density);
    flux.momentum_x = mr_hll(sl, sr, fl.momentum_x, fr.momentum_x, ul.momentum_x, ur.momentum_x);
    flux.momentum_y = mr_hll(sl, sr, fl.momentum_y, fr.momentum_y, ul.momentum_y, ur.momentum_y);
    flux.momentum_z = mr_hll(sl, sr, fl.momentum_z, fr.momentum_z, ul.momentum_z, ur.momentum_z);
    flux.energy = mr_hll(sl, sr, fl.energy, fr.energy, ul.energy, ur.energy);
    flux.Bx = mr_hll(sl, sr, fl.Bx, fr.Bx, ul.Bx, ur.Bx);
    flux.By = mr_hll(sl, sr, fl.By, fr.By, ul.By, ur.By);
    flux.Bz = mr_hll(sl, sr, fl.Bz, fr.Bz, ul.Bz, ur.Bz);
    return flux;
}

inline mr_conserved_t mr_hlld_mhd_flux(mr_primitive_t left, mr_primitive_t right, float gamma, float smallr, float smallc2, thread mr_uct_record_t &uct) {
    float smallp = smallc2 / gamma;
    float entho = 1.0f / (gamma - 1.0f);
    left.density = max(left.density, smallr);
    right.density = max(right.density, smallr);
    left.pressure = max(left.pressure, smallp * left.density);
    right.pressure = max(right.pressure, smallp * right.density);
    float A = 0.5f * (left.Bx + right.Bx);
    float sgnm = copysign(1.0f, A);
    float rl = left.density;
    float ul = left.velocity_x;
    float vl = left.velocity_y;
    float wl = left.velocity_z;
    float Pl = left.pressure;
    float Bl = left.By;
    float Cl = left.Bz;
    float ekinl = 0.5f * (ul * ul + vl * vl + wl * wl) * rl;
    float emagl = 0.5f * (A * A + Bl * Bl + Cl * Cl);
    float etotl = Pl * entho + ekinl + emagl;
    float Ptotl = Pl + emagl;
    float vdotBl = ul * A + vl * Bl + wl * Cl;
    float rr = right.density;
    float ur = right.velocity_x;
    float vr = right.velocity_y;
    float wr = right.velocity_z;
    float Pr = right.pressure;
    float Br = right.By;
    float Cr = right.Bz;
    float ekinr = 0.5f * (ur * ur + vr * vr + wr * wr) * rr;
    float emagr = 0.5f * (A * A + Br * Br + Cr * Cr);
    float etotr = Pr * entho + ekinr + emagr;
    float Ptotr = Pr + emagr;
    float vdotBr = ur * A + vr * Br + wr * Cr;
    float b2l = A * A + Bl * Bl + Cl * Cl;
    float c2l = gamma * Pl / rl;
    float d2l = 0.5f * (b2l / rl + c2l);
    float cfastl = sqrt(d2l + sqrt(max(d2l * d2l - c2l * A * A / rl, 0.0f)));
    float b2r = A * A + Br * Br + Cr * Cr;
    float c2r = gamma * Pr / rr;
    float d2r = 0.5f * (b2r / rr + c2r);
    float cfastr = sqrt(d2r + sqrt(max(d2r * d2r - c2r * A * A / rr, 0.0f)));
    float SL = min(ul, ur) - max(cfastl, cfastr);
    float SR = max(ul, ur) + max(cfastl, cfastr);
    float rcl = rl * (ul - SL);
    float rcr = rr * (SR - ur);
    float ustar = (rcr * ur + rcl * ul + Ptotl - Ptotr) / (rcr + rcl);
    float Ptotstar = (rcr * Ptotl + rcl * Ptotr + rcl * rcr * (ul - ur)) / (rcr + rcl);
    float rstarl = max(rl * (SL - ul) / (SL - ustar), smallr);
    float estar = rl * (SL - ul) * (SL - ustar) - A * A;
    float el = rl * (SL - ul) * (SL - ul) - A * A;
    float vstarl;
    float Bstarl;
    float wstarl;
    float Cstarl;
    if (abs(estar) < 1.0e-4f * A * A) {
        vstarl = vl;
        Bstarl = Bl;
        wstarl = wl;
        Cstarl = Cl;
    } else {
        vstarl = vl - A * Bl * (ustar - ul) / estar;
        Bstarl = Bl * el / estar;
        wstarl = wl - A * Cl * (ustar - ul) / estar;
        Cstarl = Cl * el / estar;
    }
    float vdotBstarl = ustar * A + vstarl * Bstarl + wstarl * Cstarl;
    float etotstarl = ((SL - ul) * etotl - Ptotl * ul + Ptotstar * ustar + A * (vdotBl - vdotBstarl)) / (SL - ustar);
    float sqrrstarl = sqrt(rstarl);
    float SAL = ustar - abs(A) / sqrrstarl;
    float rstarr = max(rr * (SR - ur) / (SR - ustar), smallr);
    estar = rr * (SR - ur) * (SR - ustar) - A * A;
    float er = rr * (SR - ur) * (SR - ur) - A * A;
    float vstarr;
    float Bstarr;
    float wstarr;
    float Cstarr;
    if (abs(estar) < 1.0e-4f * A * A) {
        vstarr = vr;
        Bstarr = Br;
        wstarr = wr;
        Cstarr = Cr;
    } else {
        vstarr = vr - A * Br * (ustar - ur) / estar;
        Bstarr = Br * er / estar;
        wstarr = wr - A * Cr * (ustar - ur) / estar;
        Cstarr = Cr * er / estar;
    }
    float vdotBstarr = ustar * A + vstarr * Bstarr + wstarr * Cstarr;
    float etotstarr = ((SR - ur) * etotr - Ptotr * ur + Ptotstar * ustar + A * (vdotBr - vdotBstarr)) / (SR - ustar);
    float sqrrstarr = sqrt(rstarr);
    float SAR = ustar + abs(A) / sqrrstarr;
    mr_set_uct_record(uct, ul, ur, vl, vr, wl, wr, A, SL, SR, ustar, SAL, SAR);
    float denom = sqrrstarl + sqrrstarr;
    float vstarstar = (sqrrstarl * vstarl + sqrrstarr * vstarr + sgnm * (Bstarr - Bstarl)) / denom;
    float wstarstar = (sqrrstarl * wstarl + sqrrstarr * wstarr + sgnm * (Cstarr - Cstarl)) / denom;
    float Bstarstar = (sqrrstarl * Bstarr + sqrrstarr * Bstarl + sgnm * sqrrstarl * sqrrstarr * (vstarr - vstarl)) / denom;
    float Cstarstar = (sqrrstarl * Cstarr + sqrrstarr * Cstarl + sgnm * sqrrstarl * sqrrstarr * (wstarr - wstarl)) / denom;
    float vdotBstarstar = ustar * A + vstarstar * Bstarstar + wstarstar * Cstarstar;
    float etotstarstarl = etotstarl - sgnm * sqrrstarl * (vdotBstarl - vdotBstarstar);
    float etotstarstarr = etotstarr + sgnm * sqrrstarr * (vdotBstarr - vdotBstarstar);
    float ro;
    float uo;
    float vo;
    float wo;
    float Bo;
    float Co;
    float Ptoto;
    float etoto;
    float vdotBo;
    if (SL > 0.0f) {
        ro = rl; uo = ul; vo = vl; wo = wl; Bo = Bl; Co = Cl; Ptoto = Ptotl; etoto = etotl; vdotBo = vdotBl;
    } else if (SAL > 0.0f) {
        ro = rstarl; uo = ustar; vo = vstarl; wo = wstarl; Bo = Bstarl; Co = Cstarl; Ptoto = Ptotstar; etoto = etotstarl; vdotBo = vdotBstarl;
    } else if (ustar > 0.0f) {
        ro = rstarl; uo = ustar; vo = vstarstar; wo = wstarstar; Bo = Bstarstar; Co = Cstarstar; Ptoto = Ptotstar; etoto = etotstarstarl; vdotBo = vdotBstarstar;
    } else if (SAR > 0.0f) {
        ro = rstarr; uo = ustar; vo = vstarstar; wo = wstarstar; Bo = Bstarstar; Co = Cstarstar; Ptoto = Ptotstar; etoto = etotstarstarr; vdotBo = vdotBstarstar;
    } else if (SR > 0.0f) {
        ro = rstarr; uo = ustar; vo = vstarr; wo = wstarr; Bo = Bstarr; Co = Cstarr; Ptoto = Ptotstar; etoto = etotstarr; vdotBo = vdotBstarr;
    } else {
        ro = rr; uo = ur; vo = vr; wo = wr; Bo = Br; Co = Cr; Ptoto = Ptotr; etoto = etotr; vdotBo = vdotBr;
    }
    mr_conserved_t flux;
    flux.density = ro * uo;
    flux.momentum_x = ro * uo * uo + Ptoto - A * A;
    flux.momentum_y = ro * uo * vo - A * Bo;
    flux.momentum_z = ro * uo * wo - A * Co;
    flux.energy = (etoto + Ptoto) * uo - A * vdotBo;
    flux.Bx = 0.0f;
    flux.By = Bo * uo - A * vo;
    flux.Bz = Co * uo - A * wo;
    return flux;
}

inline mr_primitive_t mr_cell_load(threadgroup const float *s, int i, int j, int k) {
    mr_primitive_t q;
    q.density = mr_local_get(s, 0, i, j, k);
    q.velocity_x = mr_local_get(s, 1, i, j, k);
    q.velocity_y = mr_local_get(s, 2, i, j, k);
    q.velocity_z = mr_local_get(s, 3, i, j, k);
    q.pressure = mr_local_get(s, 4, i, j, k);
    q.Bx = 0.5f * (mr_bf_get(s, 0, i, j, k) + mr_bf_get(s, 0, i + 1, j, k));
    q.By = 0.5f * (mr_bf_get(s, 1, i, j, k) + mr_bf_get(s, 1, i, j + 1, k));
    q.Bz = 0.5f * (mr_bf_get(s, 2, i, j, k) + mr_bf_get(s, 2, i, j, k + 1));
    return q;
}

inline void mr_compute_slopes(threadgroup const float *s, int i, int j, int k, int slope, thread mr_primitive_t &sx, thread mr_primitive_t &sy, thread mr_primitive_t &sz) {
    mr_primitive_t q = mr_cell_load(s, i, j, k);
    sx.density = 0.5f * mr_slope_moncen(mr_local_get(s, 0, i - 1, j, k), q.density, mr_local_get(s, 0, i + 1, j, k), slope);
    sx.velocity_x = 0.5f * mr_slope_moncen(mr_local_get(s, 1, i - 1, j, k), q.velocity_x, mr_local_get(s, 1, i + 1, j, k), slope);
    sx.velocity_y = 0.5f * mr_slope_moncen(mr_local_get(s, 2, i - 1, j, k), q.velocity_y, mr_local_get(s, 2, i + 1, j, k), slope);
    sx.velocity_z = 0.5f * mr_slope_moncen(mr_local_get(s, 3, i - 1, j, k), q.velocity_z, mr_local_get(s, 3, i + 1, j, k), slope);
    sx.pressure = 0.5f * mr_slope_moncen(mr_local_get(s, 4, i - 1, j, k), q.pressure, mr_local_get(s, 4, i + 1, j, k), slope);
    float bym = 0.5f * (mr_bf_get(s, 1, i - 1, j, k) + mr_bf_get(s, 1, i - 1, j + 1, k));
    float byp = 0.5f * (mr_bf_get(s, 1, i + 1, j, k) + mr_bf_get(s, 1, i + 1, j + 1, k));
    float bzm = 0.5f * (mr_bf_get(s, 2, i - 1, j, k) + mr_bf_get(s, 2, i - 1, j, k + 1));
    float bzp = 0.5f * (mr_bf_get(s, 2, i + 1, j, k) + mr_bf_get(s, 2, i + 1, j, k + 1));
    sx.Bx = 0.0f;
    sx.By = 0.5f * mr_slope_moncen(bym, q.By, byp, slope);
    sx.Bz = 0.5f * mr_slope_moncen(bzm, q.Bz, bzp, slope);
    sy.density = 0.5f * mr_slope_moncen(mr_local_get(s, 0, i, j - 1, k), q.density, mr_local_get(s, 0, i, j + 1, k), slope);
    sy.velocity_x = 0.5f * mr_slope_moncen(mr_local_get(s, 1, i, j - 1, k), q.velocity_x, mr_local_get(s, 1, i, j + 1, k), slope);
    sy.velocity_y = 0.5f * mr_slope_moncen(mr_local_get(s, 2, i, j - 1, k), q.velocity_y, mr_local_get(s, 2, i, j + 1, k), slope);
    sy.velocity_z = 0.5f * mr_slope_moncen(mr_local_get(s, 3, i, j - 1, k), q.velocity_z, mr_local_get(s, 3, i, j + 1, k), slope);
    sy.pressure = 0.5f * mr_slope_moncen(mr_local_get(s, 4, i, j - 1, k), q.pressure, mr_local_get(s, 4, i, j + 1, k), slope);
    float bxm = 0.5f * (mr_bf_get(s, 0, i, j - 1, k) + mr_bf_get(s, 0, i + 1, j - 1, k));
    float bxp = 0.5f * (mr_bf_get(s, 0, i, j + 1, k) + mr_bf_get(s, 0, i + 1, j + 1, k));
    bzm = 0.5f * (mr_bf_get(s, 2, i, j - 1, k) + mr_bf_get(s, 2, i, j - 1, k + 1));
    bzp = 0.5f * (mr_bf_get(s, 2, i, j + 1, k) + mr_bf_get(s, 2, i, j + 1, k + 1));
    sy.Bx = 0.5f * mr_slope_moncen(bxm, q.Bx, bxp, slope);
    sy.By = 0.0f;
    sy.Bz = 0.5f * mr_slope_moncen(bzm, q.Bz, bzp, slope);
    sz.density = 0.5f * mr_slope_moncen(mr_local_get(s, 0, i, j, k - 1), q.density, mr_local_get(s, 0, i, j, k + 1), slope);
    sz.velocity_x = 0.5f * mr_slope_moncen(mr_local_get(s, 1, i, j, k - 1), q.velocity_x, mr_local_get(s, 1, i, j, k + 1), slope);
    sz.velocity_y = 0.5f * mr_slope_moncen(mr_local_get(s, 2, i, j, k - 1), q.velocity_y, mr_local_get(s, 2, i, j, k + 1), slope);
    sz.velocity_z = 0.5f * mr_slope_moncen(mr_local_get(s, 3, i, j, k - 1), q.velocity_z, mr_local_get(s, 3, i, j, k + 1), slope);
    sz.pressure = 0.5f * mr_slope_moncen(mr_local_get(s, 4, i, j, k - 1), q.pressure, mr_local_get(s, 4, i, j, k + 1), slope);
    bxm = 0.5f * (mr_bf_get(s, 0, i, j, k - 1) + mr_bf_get(s, 0, i + 1, j, k - 1));
    bxp = 0.5f * (mr_bf_get(s, 0, i, j, k + 1) + mr_bf_get(s, 0, i + 1, j, k + 1));
    bym = 0.5f * (mr_bf_get(s, 1, i, j, k - 1) + mr_bf_get(s, 1, i, j + 1, k - 1));
    byp = 0.5f * (mr_bf_get(s, 1, i, j, k + 1) + mr_bf_get(s, 1, i, j + 1, k + 1));
    sz.Bx = 0.5f * mr_slope_moncen(bxm, q.Bx, bxp, slope);
    sz.By = 0.5f * mr_slope_moncen(bym, q.By, byp, slope);
    sz.Bz = 0.0f;
}

inline float mr_edge_x(threadgroup const float *s, int i, int j, int k) {
    float v = 0.25f * (mr_local_get(s, 2, i, j - 1, k - 1) + mr_local_get(s, 2, i, j - 1, k) + mr_local_get(s, 2, i, j, k - 1) + mr_local_get(s, 2, i, j, k));
    float w = 0.25f * (mr_local_get(s, 3, i, j - 1, k - 1) + mr_local_get(s, 3, i, j - 1, k) + mr_local_get(s, 3, i, j, k - 1) + mr_local_get(s, 3, i, j, k));
    float B = 0.5f * (mr_bf_get(s, 1, i, j, k - 1) + mr_bf_get(s, 1, i, j, k));
    float C = 0.5f * (mr_bf_get(s, 2, i, j - 1, k) + mr_bf_get(s, 2, i, j, k));
    return v * C - w * B;
}

inline float mr_edge_y(threadgroup const float *s, int i, int j, int k) {
    float u = 0.25f * (mr_local_get(s, 1, i - 1, j, k - 1) + mr_local_get(s, 1, i - 1, j, k) + mr_local_get(s, 1, i, j, k - 1) + mr_local_get(s, 1, i, j, k));
    float w = 0.25f * (mr_local_get(s, 3, i - 1, j, k - 1) + mr_local_get(s, 3, i - 1, j, k) + mr_local_get(s, 3, i, j, k - 1) + mr_local_get(s, 3, i, j, k));
    float A = 0.5f * (mr_bf_get(s, 0, i, j, k - 1) + mr_bf_get(s, 0, i, j, k));
    float C = 0.5f * (mr_bf_get(s, 2, i - 1, j, k) + mr_bf_get(s, 2, i, j, k));
    return w * A - u * C;
}

inline float mr_edge_z(threadgroup const float *s, int i, int j, int k) {
    float u = 0.25f * (mr_local_get(s, 1, i - 1, j - 1, k) + mr_local_get(s, 1, i - 1, j, k) + mr_local_get(s, 1, i, j - 1, k) + mr_local_get(s, 1, i, j, k));
    float v = 0.25f * (mr_local_get(s, 2, i - 1, j - 1, k) + mr_local_get(s, 2, i - 1, j, k) + mr_local_get(s, 2, i, j - 1, k) + mr_local_get(s, 2, i, j, k));
    float A = 0.5f * (mr_bf_get(s, 0, i, j - 1, k) + mr_bf_get(s, 0, i, j, k));
    float B = 0.5f * (mr_bf_get(s, 1, i - 1, j, k) + mr_bf_get(s, 1, i, j, k));
    return u * B - v * A;
}

inline mr_trace_t mr_predict_cell(threadgroup const float *s, int i, int j, int k, float gamma, float dtdx, int slope, int slope_mag, int induction) {
    mr_trace_t t;
    t.AL = mr_bf_get(s, 0, i, j, k);
    t.AR = mr_bf_get(s, 0, i + 1, j, k);
    t.BL = mr_bf_get(s, 1, i, j, k);
    t.BR = mr_bf_get(s, 1, i, j + 1, k);
    t.CL = mr_bf_get(s, 2, i, j, k);
    t.CR = mr_bf_get(s, 2, i, j, k + 1);
    float A = 0.5f * (t.AL + t.AR);
    float B = 0.5f * (t.BL + t.BR);
    float C = 0.5f * (t.CL + t.CR);
    float ELL = mr_edge_x(s, i, j, k);
    float ELR = mr_edge_x(s, i, j, k + 1);
    float ERL = mr_edge_x(s, i, j + 1, k);
    float ERR = mr_edge_x(s, i, j + 1, k + 1);
    float FLL = mr_edge_y(s, i, j, k);
    float FLR = mr_edge_y(s, i, j, k + 1);
    float FRL = mr_edge_y(s, i + 1, j, k);
    float FRR = mr_edge_y(s, i + 1, j, k + 1);
    float GLL = mr_edge_z(s, i, j, k);
    float GLR = mr_edge_z(s, i, j + 1, k);
    float GRL = mr_edge_z(s, i + 1, j, k);
    float GRR = mr_edge_z(s, i + 1, j + 1, k);
    t.AL += ((GLR - GLL) - (FLR - FLL)) * dtdx * 0.5f;
    t.AR += ((GRR - GRL) - (FRR - FRL)) * dtdx * 0.5f;
    t.BL += (-(GRL - GLL) + (ELR - ELL)) * dtdx * 0.5f;
    t.BR += (-(GRR - GLR) + (ERR - ERL)) * dtdx * 0.5f;
    t.CL += ((FRL - FLL) - (ERL - ELL)) * dtdx * 0.5f;
    t.CR += ((FRR - FLR) - (ERR - ELR)) * dtdx * 0.5f;
    t.cell = mr_cell_load(s, i, j, k);
    mr_compute_slopes(s, i, j, k, slope, t.sx, t.sy, t.sz);
    float divv = t.sx.velocity_x + t.sy.velocity_y + t.sz.velocity_z;
    mr_primitive_t source;
    source.density = -t.cell.velocity_x * t.sx.density - t.cell.velocity_y * t.sy.density - t.cell.velocity_z * t.sz.density - divv * t.cell.density;
    source.velocity_x = -t.cell.velocity_x * t.sx.velocity_x - t.cell.velocity_y * t.sy.velocity_x - t.cell.velocity_z * t.sz.velocity_x - t.sx.pressure / t.cell.density;
    source.velocity_y = -t.cell.velocity_x * t.sx.velocity_y - t.cell.velocity_y * t.sy.velocity_y - t.cell.velocity_z * t.sz.velocity_y - t.sy.pressure / t.cell.density;
    source.velocity_z = -t.cell.velocity_x * t.sx.velocity_z - t.cell.velocity_y * t.sy.velocity_z - t.cell.velocity_z * t.sz.velocity_z - t.sz.pressure / t.cell.density;
    source.pressure = -t.cell.velocity_x * t.sx.pressure - t.cell.velocity_y * t.sy.pressure - t.cell.velocity_z * t.sz.pressure - divv * gamma * t.cell.pressure;
    source.velocity_x += (-(B * t.sx.By + C * t.sx.Bz) + B * t.sy.Bx + C * t.sz.Bx) / t.cell.density;
    source.velocity_y += (A * t.sx.By - (A * t.sy.Bx + C * t.sy.Bz) + C * t.sz.By) / t.cell.density;
    source.velocity_z += (A * t.sx.Bz + B * t.sy.Bz - (A * t.sz.Bx + B * t.sz.By)) / t.cell.density;
    if (induction != 0) {
        source.velocity_x = 0.0f;
        source.velocity_y = 0.0f;
        source.velocity_z = 0.0f;
    }
    t.cell.density += dtdx * source.density;
    t.cell.velocity_x += dtdx * source.velocity_x;
    t.cell.velocity_y += dtdx * source.velocity_y;
    t.cell.velocity_z += dtdx * source.velocity_z;
    t.cell.pressure += dtdx * source.pressure;
    t.cell.Bx = 0.5f * (t.AL + t.AR);
    t.cell.By = 0.5f * (t.BL + t.BR);
    t.cell.Bz = 0.5f * (t.CL + t.CR);
    return t;
}

inline mr_primitive_t mr_traced_face(threadgroup const float *s, mr_trace_t t, int orientation, int side, int i, int j, int k, float smallr, float smallc2) {
    mr_primitive_t q = t.cell;
    mr_primitive_t ds = orientation == 0 ? t.sx : orientation == 1 ? t.sy : t.sz;
    float sign = side < 0 ? -1.0f : 1.0f;
    q.density += sign * ds.density;
    q.velocity_x += sign * ds.velocity_x;
    q.velocity_y += sign * ds.velocity_y;
    q.velocity_z += sign * ds.velocity_z;
    q.pressure += sign * ds.pressure;
    if (orientation == 0) {
        q.Bx = side < 0 ? t.AL : t.AR;
        q.By = t.cell.By + sign * t.sx.By;
        q.Bz = t.cell.Bz + sign * t.sx.Bz;
    } else if (orientation == 1) {
        q.Bx = t.cell.Bx + sign * t.sy.Bx;
        q.By = side < 0 ? t.BL : t.BR;
        q.Bz = t.cell.Bz + sign * t.sy.Bz;
    } else {
        q.Bx = t.cell.Bx + sign * t.sz.Bx;
        q.By = t.cell.By + sign * t.sz.By;
        q.Bz = side < 0 ? t.CL : t.CR;
    }
    if (q.density < smallr) q.density = mr_local_get(s, 0, i, j, k);
    if (q.pressure < smallr * smallc2) q.pressure = mr_local_get(s, 4, i, j, k);
    return q;
}

inline int mr_shell_index(int orientation, int normal, int t1, int t2) {
    int perimeter;
    if (t2 == -1) perimeter = t1;
    else if (t2 == MR_M) perimeter = MR_M + t1;
    else if (t1 == -1) perimeter = 2 * MR_M + t2;
    else perimeter = 3 * MR_M + t2;
    return orientation * MR_SHELL_CELLS + normal + (MR_M + 1) * perimeter;
}

inline void mr_decode_shell(int shell, thread int &orientation, thread int &normal, thread int &t1, thread int &t2) {
    orientation = shell / MR_SHELL_CELLS;
    int local = shell - orientation * MR_SHELL_CELLS;
    normal = local % (MR_M + 1);
    int perimeter = local / (MR_M + 1);
    if (perimeter < MR_M) {
        t1 = perimeter;
        t2 = -1;
    } else if (perimeter < 2 * MR_M) {
        t1 = perimeter - MR_M;
        t2 = MR_M;
    } else if (perimeter < 3 * MR_M) {
        t1 = -1;
        t2 = perimeter - 2 * MR_M;
    } else {
        t1 = MR_M;
        t2 = perimeter - 3 * MR_M;
    }
}

inline mr_uct_record_t mr_get_record(threadgroup const float *s, int orientation, int normal, int t1, int t2) {
    bool interior = t1 >= 0 && t1 < MR_M && t2 >= 0 && t2 < MR_M;
    if (interior) {
        int i = orientation == 0 ? normal : t1;
        int j = orientation == 1 ? normal : orientation == 0 ? t1 : t2;
        int k = orientation == 2 ? normal : t2;
        int face = orientation * MR_FACE_CELLS + mr_face_index(orientation, i, j, k);
        return mr_interior_record_load(s, face);
    }
    return mr_shell_record_load(s, mr_shell_index(orientation, normal, t1, t2));
}

inline float mr_uct_edge(mr_uct_record_t f1lo, mr_uct_record_t f1hi, mr_uct_record_t f2lo, mr_uct_record_t f2hi, int slot1, int slot2) {
    float a1L = 0.5f * (f1lo.aL + f1hi.aL);
    float a1R = 1.0f - a1L;
    float d1L = 0.5f * (f1lo.dL + f1hi.dL);
    float d1R = 0.5f * (f1lo.dR + f1hi.dR);
    float a2L = 0.5f * (f2lo.aL + f2hi.aL);
    float a2R = 1.0f - a2L;
    float d2L = 0.5f * (f2lo.dL + f2hi.dL);
    float d2R = 0.5f * (f2lo.dR + f2hi.dR);
    float v1L = slot2 == 1 ? f2lo.vt1 : f2lo.vt2;
    float v1R = slot2 == 1 ? f2hi.vt1 : f2hi.vt2;
    float v2L = slot1 == 1 ? f1lo.vt1 : f1lo.vt2;
    float v2R = slot1 == 1 ? f1hi.vt1 : f1hi.vt2;
    return a1L * v1L * f2lo.Bn + a1R * v1R * f2hi.Bn - a2L * v2L * f1lo.Bn - a2R * v2R * f1hi.Bn + d1L * f2lo.Bn - d1R * f2hi.Bn - d2L * f1lo.Bn + d2R * f1hi.Bn;
}

inline mr_uct_record_t mr_shell_solve(threadgroup const float *s, int shell, float gamma, float smallr, float smallc2, float dtdx, int slope, int slope_mag, int induction, float switch_dmin, float switch_pmin) {
    int orientation;
    int normal;
    int t1;
    int t2;
    mr_decode_shell(shell, orientation, normal, t1, t2);
    int il;
    int jl;
    int kl;
    int ir;
    int jr;
    int kr;
    if (orientation == 0) {
        il = normal + 1; jl = t1 + 2; kl = t2 + 2;
        ir = normal + 2; jr = jl; kr = kl;
    } else if (orientation == 1) {
        il = t1 + 2; jl = normal + 1; kl = t2 + 2;
        ir = il; jr = normal + 2; kr = kl;
    } else {
        il = t1 + 2; jl = t2 + 2; kl = normal + 1;
        ir = il; jr = jl; kr = normal + 2;
    }
    mr_trace_t tl = mr_predict_cell(s, il, jl, kl, gamma, dtdx, slope, slope_mag, induction);
    mr_trace_t tr = mr_predict_cell(s, ir, jr, kr, gamma, dtdx, slope, slope_mag, induction);
    mr_primitive_t left = mr_rotate_face(mr_traced_face(s, tl, orientation, 1, il, jl, kl, smallr, smallc2), orientation);
    mr_primitive_t right = mr_rotate_face(mr_traced_face(s, tr, orientation, -1, ir, jr, kr, smallr, smallc2), orientation);
    bool use_llf = (switch_dmin > 0.0f && min(left.density, right.density) < switch_dmin) || (switch_pmin > 0.0f && min(left.pressure, right.pressure) < switch_pmin);
    return use_llf ? mr_llf_record(left, right, gamma, smallr, smallc2) : mr_hlld_record(left, right, gamma, smallr, smallc2);
}

inline void mr_atomic_add(device atomic_uint *value, float addend) {
    uint expected = atomic_load_explicit(value, memory_order_relaxed);
    uint desired;
    do {
        desired = as_type<uint>(as_type<float>(expected) + addend);
    } while (!atomic_compare_exchange_weak_explicit(value, &expected, desired, memory_order_relaxed, memory_order_relaxed));
}

inline void mr_atomic_min(device atomic_uint *value, float candidate) {
    atomic_fetch_min_explicit(value, as_type<uint>(candidate), memory_order_relaxed);
}

struct mr_cf_cell_t {
    bool coarse;
    int oct_idx;
    int cell_idx;
};

inline mr_cf_cell_t mr_get_cf_cell(device const oct_t *grid, device const int *father, device const int *nbor, int subgrid_idx, int ngridmax, int i, int j, int k) {
    mr_cf_cell_t cell;
    cell.coarse = false;
    cell.oct_idx = 0;
    cell.cell_idx = 1;
    int stencil_idx = 1 + i + MR_NSUBGRIDP2 * j + MR_NSUBGRIDP2 * MR_NSUBGRIDP2 * k;
    int nbor_idx = mr_nbor_get(nbor, subgrid_idx, stencil_idx);
    if (nbor_idx > ngridmax) {
        int father_idx = father[nbor_idx - 1];
        int ci = grid[nbor_idx - 1].ckey[0] - 2 * grid[father_idx - 1].ckey[0];
        int cj = grid[nbor_idx - 1].ckey[1] - 2 * grid[father_idx - 1].ckey[1];
        int ck = grid[nbor_idx - 1].ckey[2] - 2 * grid[father_idx - 1].ckey[2];
        cell.coarse = true;
        cell.oct_idx = father_idx;
        cell.cell_idx = 1 + ci + 2 * cj + 4 * ck;
    }
    return cell;
}

inline void mr_u_atomic_add(device float *u, int oct_idx, int ivar, int cell_idx, float addend) {
    mr_atomic_add((device atomic_uint *)&u[mr_u_flat(oct_idx, ivar, cell_idx)], addend);
}

inline void mr_cf_b_add(device float *b, mr_cf_cell_t cell, int ivar, float addend) {
    if (cell.coarse) mr_atomic_add((device atomic_uint *)&b[mr_b_flat(cell.oct_idx, ivar, cell.cell_idx)], addend);
}

inline float mr_cf_weight(mr_cf_cell_t c1, mr_cf_cell_t c2, mr_cf_cell_t c3) {
    return c1.coarse && c2.coarse && c3.coarse ? 1.0f : 0.5f;
}

inline void mr_coarse_hydro_update(threadgroup const float *s, device const oct_t *grid, device const int *father, device const int *nbor, device float *unew, int subgrid_idx, int ngridmax, int tid, float dtdx) {
    int interface_size = MR_NSUBGRID * MR_NSUBGRID;
    if (tid >= 6 * interface_size) return;
    int boundary = tid / interface_size;
    int work = tid - boundary * interface_size;
    int first = work % MR_NSUBGRID;
    int second = work / MR_NSUBGRID;
    int orientation = boundary / 2;
    int side = boundary % 2;
    int normal = side == 0 ? 0 : MR_M;
    int i_sg = orientation == 0 ? (side == 0 ? 0 : MR_NSUBGRID + 1) : first + 1;
    int j_sg = orientation == 1 ? (side == 0 ? 0 : MR_NSUBGRID + 1) : orientation == 0 ? first + 1 : second + 1;
    int k_sg = orientation == 2 ? (side == 0 ? 0 : MR_NSUBGRID + 1) : second + 1;
    float flux[5] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float scale = (side == 0 ? -0.125f : 0.125f) * dtdx;
    for (int a = 0; a < 2; ++a) {
        for (int b = 0; b < 2; ++b) {
            int t1 = 2 * first + a;
            int t2 = 2 * second + b;
            int i = orientation == 0 ? normal : t1;
            int j = orientation == 1 ? normal : orientation == 0 ? t1 : t2;
            int k = orientation == 2 ? normal : t2;
            int face = orientation * MR_FACE_CELLS + mr_face_index(orientation, i, j, k);
            for (int field = 0; field < 5; ++field) flux[field] += mr_flux_get(s, field, face) * scale;
        }
    }
    mr_cf_cell_t cell = mr_get_cf_cell(grid, father, nbor, subgrid_idx, ngridmax, i_sg, j_sg, k_sg);
    if (cell.coarse) for (int field = 0; field < 5; ++field) mr_u_atomic_add(unew, cell.oct_idx, field + 1, cell.cell_idx, flux[field]);
}

inline void mr_coarse_ct_update(threadgroup const float *s, device const oct_t *grid, device const int *father, device const int *nbor, device float *bnew, int subgrid_idx, int ngridmax, float dtdx) {
    for (int k_sg = 1; k_sg <= MR_NSUBGRID; ++k_sg) {
        int k = 2 * k_sg - 2;
        for (int j_sg = 1; j_sg <= MR_NSUBGRID; ++j_sg) {
            int j = 2 * j_sg - 2;
            for (int i_sg = 1; i_sg <= MR_NSUBGRID; ++i_sg) {
                int i = 2 * i_sg - 2;
                mr_cf_cell_t c1 = mr_get_cf_cell(grid, father, nbor, subgrid_idx, ngridmax, i_sg, j_sg - 1, k_sg);
                mr_cf_cell_t c2 = mr_get_cf_cell(grid, father, nbor, subgrid_idx, ngridmax, i_sg - 1, j_sg - 1, k_sg);
                mr_cf_cell_t c3 = mr_get_cf_cell(grid, father, nbor, subgrid_idx, ngridmax, i_sg - 1, j_sg, k_sg);
                if (c1.coarse || c3.coarse) {
                    float dflux = (mr_emf_get(s, 0, i, j, k) + mr_emf_get(s, 0, i, j, k + 1)) * 0.25f * mr_cf_weight(c1, c2, c3) * dtdx;
                    mr_cf_b_add(bnew, c1, 5, dflux); mr_cf_b_add(bnew, c1, 1, dflux);
                    mr_cf_b_add(bnew, c2, 4, dflux); mr_cf_b_add(bnew, c2, 5, -dflux);
                    mr_cf_b_add(bnew, c3, 2, -dflux); mr_cf_b_add(bnew, c3, 4, -dflux);
                }
                c1 = mr_get_cf_cell(grid, father, nbor, subgrid_idx, ngridmax, i_sg - 1, j_sg, k_sg);
                c2 = mr_get_cf_cell(grid, father, nbor, subgrid_idx, ngridmax, i_sg - 1, j_sg + 1, k_sg);
                c3 = mr_get_cf_cell(grid, father, nbor, subgrid_idx, ngridmax, i_sg, j_sg + 1, k_sg);
                if (c1.coarse || c3.coarse) {
                    float dflux = (mr_emf_get(s, 0, i, j + 2, k) + mr_emf_get(s, 0, i, j + 2, k + 1)) * 0.25f * mr_cf_weight(c1, c2, c3) * dtdx;
                    mr_cf_b_add(bnew, c1, 4, dflux); mr_cf_b_add(bnew, c1, 5, -dflux);
                    mr_cf_b_add(bnew, c2, 2, -dflux); mr_cf_b_add(bnew, c2, 4, -dflux);
                    mr_cf_b_add(bnew, c3, 1, -dflux); mr_cf_b_add(bnew, c3, 2, dflux);
                }
                c1 = mr_get_cf_cell(grid, father, nbor, subgrid_idx, ngridmax, i_sg, j_sg + 1, k_sg);
                c2 = mr_get_cf_cell(grid, father, nbor, subgrid_idx, ngridmax, i_sg + 1, j_sg + 1, k_sg);
                c3 = mr_get_cf_cell(grid, father, nbor, subgrid_idx, ngridmax, i_sg + 1, j_sg, k_sg);
                if (c1.coarse || c3.coarse) {
                    float dflux = (mr_emf_get(s, 0, i + 2, j + 2, k) + mr_emf_get(s, 0, i + 2, j + 2, k + 1)) * 0.25f * mr_cf_weight(c1, c2, c3) * dtdx;
                    mr_cf_b_add(bnew, c1, 2, -dflux); mr_cf_b_add(bnew, c1, 4, -dflux);
                    mr_cf_b_add(bnew, c2, 1, -dflux); mr_cf_b_add(bnew, c2, 2, dflux);
                    mr_cf_b_add(bnew, c3, 5, dflux); mr_cf_b_add(bnew, c3, 1, dflux);
                }
                c1 = mr_get_cf_cell(grid, father, nbor, subgrid_idx, ngridmax, i_sg + 1, j_sg, k_sg);
                c2 = mr_get_cf_cell(grid, father, nbor, subgrid_idx, ngridmax, i_sg + 1, j_sg - 1, k_sg);
                c3 = mr_get_cf_cell(grid, father, nbor, subgrid_idx, ngridmax, i_sg, j_sg - 1, k_sg);
                if (c1.coarse || c3.coarse) {
                    float dflux = (mr_emf_get(s, 0, i + 2, j, k) + mr_emf_get(s, 0, i + 2, j, k + 1)) * 0.25f * mr_cf_weight(c1, c2, c3) * dtdx;
                    mr_cf_b_add(bnew, c1, 1, -dflux); mr_cf_b_add(bnew, c1, 2, dflux);
                    mr_cf_b_add(bnew, c2, 5, dflux); mr_cf_b_add(bnew, c2, 1, dflux);
                    mr_cf_b_add(bnew, c3, 4, dflux); mr_cf_b_add(bnew, c3, 5, -dflux);
                }
                c1 = mr_get_cf_cell(grid, father, nbor, subgrid_idx, ngridmax, i_sg, j_sg, k_sg - 1);
                c2 = mr_get_cf_cell(grid, father, nbor, subgrid_idx, ngridmax, i_sg, j_sg - 1, k_sg - 1);
                c3 = mr_get_cf_cell(grid, father, nbor, subgrid_idx, ngridmax, i_sg, j_sg - 1, k_sg);
                if (c1.coarse || c3.coarse) {
                    float dflux = (mr_emf_get(s, 2, i, j, k) + mr_emf_get(s, 2, i + 1, j, k)) * 0.25f * mr_cf_weight(c1, c2, c3) * dtdx;
                    mr_cf_b_add(bnew, c1, 6, dflux); mr_cf_b_add(bnew, c1, 2, dflux);
                    mr_cf_b_add(bnew, c2, 5, dflux); mr_cf_b_add(bnew, c2, 6, -dflux);
                    mr_cf_b_add(bnew, c3, 3, -dflux); mr_cf_b_add(bnew, c3, 5, -dflux);
                }
                c1 = mr_get_cf_cell(grid, father, nbor, subgrid_idx, ngridmax, i_sg, j_sg - 1, k_sg);
                c2 = mr_get_cf_cell(grid, father, nbor, subgrid_idx, ngridmax, i_sg, j_sg - 1, k_sg + 1);
                c3 = mr_get_cf_cell(grid, father, nbor, subgrid_idx, ngridmax, i_sg, j_sg, k_sg + 1);
                if (c1.coarse || c3.coarse) {
                    float dflux = (mr_emf_get(s, 2, i, j, k + 2) + mr_emf_get(s, 2, i + 1, j, k + 2)) * 0.25f * mr_cf_weight(c1, c2, c3) * dtdx;
                    mr_cf_b_add(bnew, c1, 5, dflux); mr_cf_b_add(bnew, c1, 6, -dflux);
                    mr_cf_b_add(bnew, c2, 3, -dflux); mr_cf_b_add(bnew, c2, 5, -dflux);
                    mr_cf_b_add(bnew, c3, 2, -dflux); mr_cf_b_add(bnew, c3, 3, dflux);
                }
                c1 = mr_get_cf_cell(grid, father, nbor, subgrid_idx, ngridmax, i_sg, j_sg, k_sg + 1);
                c2 = mr_get_cf_cell(grid, father, nbor, subgrid_idx, ngridmax, i_sg, j_sg + 1, k_sg + 1);
                c3 = mr_get_cf_cell(grid, father, nbor, subgrid_idx, ngridmax, i_sg, j_sg + 1, k_sg);
                if (c1.coarse || c3.coarse) {
                    float dflux = (mr_emf_get(s, 2, i, j + 2, k + 2) + mr_emf_get(s, 2, i + 1, j + 2, k + 2)) * 0.25f * mr_cf_weight(c1, c2, c3) * dtdx;
                    mr_cf_b_add(bnew, c1, 3, -dflux); mr_cf_b_add(bnew, c1, 5, -dflux);
                    mr_cf_b_add(bnew, c2, 2, -dflux); mr_cf_b_add(bnew, c2, 3, dflux);
                    mr_cf_b_add(bnew, c3, 6, dflux); mr_cf_b_add(bnew, c3, 2, dflux);
                }
                c1 = mr_get_cf_cell(grid, father, nbor, subgrid_idx, ngridmax, i_sg, j_sg + 1, k_sg);
                c2 = mr_get_cf_cell(grid, father, nbor, subgrid_idx, ngridmax, i_sg, j_sg + 1, k_sg - 1);
                c3 = mr_get_cf_cell(grid, father, nbor, subgrid_idx, ngridmax, i_sg, j_sg, k_sg - 1);
                if (c1.coarse || c3.coarse) {
                    float dflux = (mr_emf_get(s, 2, i, j + 2, k) + mr_emf_get(s, 2, i + 1, j + 2, k)) * 0.25f * mr_cf_weight(c1, c2, c3) * dtdx;
                    mr_cf_b_add(bnew, c1, 2, -dflux); mr_cf_b_add(bnew, c1, 3, dflux);
                    mr_cf_b_add(bnew, c2, 6, dflux); mr_cf_b_add(bnew, c2, 2, dflux);
                    mr_cf_b_add(bnew, c3, 5, dflux); mr_cf_b_add(bnew, c3, 6, -dflux);
                }
                c1 = mr_get_cf_cell(grid, father, nbor, subgrid_idx, ngridmax, i_sg, j_sg, k_sg - 1);
                c2 = mr_get_cf_cell(grid, father, nbor, subgrid_idx, ngridmax, i_sg - 1, j_sg, k_sg - 1);
                c3 = mr_get_cf_cell(grid, father, nbor, subgrid_idx, ngridmax, i_sg - 1, j_sg, k_sg);
                if (c1.coarse || c3.coarse) {
                    float dflux = (mr_emf_get(s, 1, i, j, k) + mr_emf_get(s, 1, i, j + 1, k)) * 0.25f * mr_cf_weight(c1, c2, c3) * dtdx;
                    mr_cf_b_add(bnew, c1, 6, -dflux); mr_cf_b_add(bnew, c1, 1, -dflux);
                    mr_cf_b_add(bnew, c2, 4, -dflux); mr_cf_b_add(bnew, c2, 6, dflux);
                    mr_cf_b_add(bnew, c3, 3, dflux); mr_cf_b_add(bnew, c3, 4, dflux);
                }
                c1 = mr_get_cf_cell(grid, father, nbor, subgrid_idx, ngridmax, i_sg - 1, j_sg, k_sg);
                c2 = mr_get_cf_cell(grid, father, nbor, subgrid_idx, ngridmax, i_sg - 1, j_sg, k_sg + 1);
                c3 = mr_get_cf_cell(grid, father, nbor, subgrid_idx, ngridmax, i_sg, j_sg, k_sg + 1);
                if (c1.coarse || c3.coarse) {
                    float dflux = (mr_emf_get(s, 1, i, j, k + 2) + mr_emf_get(s, 1, i, j + 1, k + 2)) * 0.25f * mr_cf_weight(c1, c2, c3) * dtdx;
                    mr_cf_b_add(bnew, c1, 4, -dflux); mr_cf_b_add(bnew, c1, 6, dflux);
                    mr_cf_b_add(bnew, c2, 3, dflux); mr_cf_b_add(bnew, c2, 4, dflux);
                    mr_cf_b_add(bnew, c3, 1, dflux); mr_cf_b_add(bnew, c3, 3, -dflux);
                }
                c1 = mr_get_cf_cell(grid, father, nbor, subgrid_idx, ngridmax, i_sg, j_sg, k_sg + 1);
                c2 = mr_get_cf_cell(grid, father, nbor, subgrid_idx, ngridmax, i_sg + 1, j_sg, k_sg + 1);
                c3 = mr_get_cf_cell(grid, father, nbor, subgrid_idx, ngridmax, i_sg + 1, j_sg, k_sg);
                if (c1.coarse || c3.coarse) {
                    float dflux = (mr_emf_get(s, 1, i + 2, j, k + 2) + mr_emf_get(s, 1, i + 2, j + 1, k + 2)) * 0.25f * mr_cf_weight(c1, c2, c3) * dtdx;
                    mr_cf_b_add(bnew, c1, 3, dflux); mr_cf_b_add(bnew, c1, 4, dflux);
                    mr_cf_b_add(bnew, c2, 1, dflux); mr_cf_b_add(bnew, c2, 3, -dflux);
                    mr_cf_b_add(bnew, c3, 6, -dflux); mr_cf_b_add(bnew, c3, 1, -dflux);
                }
                c1 = mr_get_cf_cell(grid, father, nbor, subgrid_idx, ngridmax, i_sg + 1, j_sg, k_sg);
                c2 = mr_get_cf_cell(grid, father, nbor, subgrid_idx, ngridmax, i_sg + 1, j_sg, k_sg - 1);
                c3 = mr_get_cf_cell(grid, father, nbor, subgrid_idx, ngridmax, i_sg, j_sg, k_sg - 1);
                if (c1.coarse || c3.coarse) {
                    float dflux = (mr_emf_get(s, 1, i + 2, j, k) + mr_emf_get(s, 1, i + 2, j + 1, k)) * 0.25f * mr_cf_weight(c1, c2, c3) * dtdx;
                    mr_cf_b_add(bnew, c1, 1, dflux); mr_cf_b_add(bnew, c1, 3, -dflux);
                    mr_cf_b_add(bnew, c2, 6, -dflux); mr_cf_b_add(bnew, c2, 1, -dflux);
                    mr_cf_b_add(bnew, c3, 4, -dflux); mr_cf_b_add(bnew, c3, 6, dflux);
                }
            }
        }
    }
}

kernel void mhd_set_uold_kernel(
    device float *uold [[buffer(0)]],
    device const float *unew [[buffer(1)]],
    device float *bold [[buffer(2)]],
    device const float *bnew [[buffer(3)]],
    constant int &head_idx [[buffer(4)]],
    constant int &num_octs [[buffer(5)]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 bid [[threadgroup_position_in_grid]])
{
    int offset = int(bid.x) * MR_SET_Y + int(tid.y);
    if (offset >= num_octs) return;
    int oct_idx = head_idx + offset;
    int cell_idx = int(tid.x) + 1;
    for (int ivar = 1; ivar <= NVAR; ++ivar) mr_u_set(uold, oct_idx, ivar, cell_idx, mr_u_get(unew, oct_idx, ivar, cell_idx));
    for (int ivar = 1; ivar <= 6; ++ivar) mr_b_set(bold, oct_idx, ivar, cell_idx, mr_b_get(bnew, oct_idx, ivar, cell_idx));
}

kernel void mhd_set_unew_kernel(
    device const float *uold [[buffer(0)]],
    device float *unew [[buffer(1)]],
    device const float *bold [[buffer(2)]],
    device float *bnew [[buffer(3)]],
    constant int &head_idx [[buffer(4)]],
    constant int &num_octs [[buffer(5)]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 bid [[threadgroup_position_in_grid]])
{
    int offset = int(bid.x) * MR_SET_Y + int(tid.y);
    if (offset >= num_octs) return;
    int oct_idx = head_idx + offset;
    int cell_idx = int(tid.x) + 1;
    for (int ivar = 1; ivar <= NVAR; ++ivar) mr_u_set(unew, oct_idx, ivar, cell_idx, mr_u_get(uold, oct_idx, ivar, cell_idx));
    for (int ivar = 1; ivar <= 6; ++ivar) mr_b_set(bnew, oct_idx, ivar, cell_idx, mr_b_get(bold, oct_idx, ivar, cell_idx));
}

kernel void mhd_cmpdt_kernel(
    device const oct_t *grid [[buffer(0)]],
    device const float *uold [[buffer(1)]],
    device const float *bold [[buffer(2)]],
    device atomic_uint *data_out [[buffer(3)]],
    constant int &head_idx [[buffer(4)]],
    constant int &num_octs [[buffer(5)]],
    constant float &dx [[buffer(6)]],
    constant float &gamma [[buffer(7)]],
    constant float &smallr [[buffer(8)]],
    constant float &smallc2 [[buffer(9)]],
    constant float &courant_factor [[buffer(10)]],
    constant int &induction [[buffer(11)]],
    constant float *constant_gravity [[buffer(12)]],
    uint tid [[thread_position_in_threadgroup]],
    uint bid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_index [[simdgroup_index_in_threadgroup]],
    uint simd_width [[threads_per_simdgroup]],
    uint threads_per_group [[threads_per_threadgroup]])
{
    uint global = bid * threads_per_group + tid;
    int oct_offset = int(global / MR_TWOTONDIM);
    float mass = 0.0f;
    float ekin = 0.0f;
    float eint = 0.0f;
    float emag = 0.0f;
    float dtloc = HUGE_VALF;
    if (oct_offset < num_octs) {
        int oct_idx = head_idx + oct_offset;
        int cell_idx = int(global % MR_TWOTONDIM) + 1;
        if (grid[oct_idx - 1].refined[cell_idx - 1] == 0) {
            mr_conserved_t u;
            u.density = mr_u_get(uold, oct_idx, 1, cell_idx);
            u.momentum_x = mr_u_get(uold, oct_idx, 2, cell_idx);
            u.momentum_y = mr_u_get(uold, oct_idx, 3, cell_idx);
            u.momentum_z = mr_u_get(uold, oct_idx, 4, cell_idx);
            u.energy = mr_u_get(uold, oct_idx, 5, cell_idx);
            u.Bx = 0.5f * (mr_b_get(bold, oct_idx, 1, cell_idx) + mr_b_get(bold, oct_idx, 4, cell_idx));
            u.By = 0.5f * (mr_b_get(bold, oct_idx, 2, cell_idx) + mr_b_get(bold, oct_idx, 5, cell_idx));
            u.Bz = 0.5f * (mr_b_get(bold, oct_idx, 3, cell_idx) + mr_b_get(bold, oct_idx, 6, cell_idx));
            mr_primitive_t q = mr_conserved_to_primitive(u, gamma, smallr, smallc2);
            float volume = dx * dx * dx;
            float pressure = max(q.pressure, q.density * smallc2 / gamma);
            float b2 = mr_bsquared(q.Bx, q.By, q.Bz);
            mass = u.density * volume;
            ekin = u.energy * volume;
            eint = pressure / (gamma - 1.0f) * volume;
            emag = 0.5f * b2 * volume;
            float rinv = 1.0f / q.density;
            float a2 = gamma * pressure * rinv;
            float ctot;
            if (induction != 0) {
                float smallc = sqrt(smallc2);
                ctot = abs(q.velocity_x) + abs(q.velocity_y) + abs(q.velocity_z) + 3.0f * smallc;
            } else {
                float c2v = 0.5f * (b2 * rinv + a2);
                float cfx = sqrt(c2v + sqrt(max(c2v * c2v - a2 * q.Bx * q.Bx * rinv, 0.0f)));
                float cfy = sqrt(c2v + sqrt(max(c2v * c2v - a2 * q.By * q.By * rinv, 0.0f)));
                float cfz = sqrt(c2v + sqrt(max(c2v * c2v - a2 * q.Bz * q.Bz * rinv, 0.0f)));
                ctot = abs(q.velocity_x) + cfx + abs(q.velocity_y) + cfy + abs(q.velocity_z) + cfz;
            }
            float grav = abs(constant_gravity[0]) + abs(constant_gravity[1]) + abs(constant_gravity[2]);
            grav = max(grav * dx / (ctot * ctot), 0.0001f);
            dtloc = dx / ctot * (sqrt(1.0f + 2.0f * courant_factor * grav) - 1.0f) / grav;
        }
    }
    threadgroup float tg_mass[32];
    threadgroup float tg_ekin[32];
    threadgroup float tg_eint[32];
    threadgroup float tg_emag[32];
    threadgroup float tg_dt[32];
    float wm = simd_sum(mass);
    float wk = simd_sum(ekin);
    float wi = simd_sum(eint);
    float wb = simd_sum(emag);
    float wd = simd_min(dtloc);
    if (lane == 0) {
        tg_mass[simd_index] = wm;
        tg_ekin[simd_index] = wk;
        tg_eint[simd_index] = wi;
        tg_emag[simd_index] = wb;
        tg_dt[simd_index] = wd;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_index == 0) {
        uint groups = (threads_per_group + simd_width - 1) / simd_width;
        float bm = lane < groups ? tg_mass[lane] : 0.0f;
        float bk = lane < groups ? tg_ekin[lane] : 0.0f;
        float bi = lane < groups ? tg_eint[lane] : 0.0f;
        float bb = lane < groups ? tg_emag[lane] : 0.0f;
        float bd = lane < groups ? tg_dt[lane] : HUGE_VALF;
        bm = simd_sum(bm);
        bk = simd_sum(bk);
        bi = simd_sum(bi);
        bb = simd_sum(bb);
        bd = simd_min(bd);
        if (lane == 0) {
            mr_atomic_add(&data_out[0], bm);
            mr_atomic_add(&data_out[1], bk);
            mr_atomic_add(&data_out[2], bi);
            mr_atomic_add(&data_out[3], bb);
            mr_atomic_min(&data_out[4], bd);
        }
    }
}

kernel void hydro_integrator_uct_kernel(
    device const oct_t *grid [[buffer(0)]],
    device const float *uold [[buffer(1)]],
    device float *unew [[buffer(2)]],
    device const float *bold [[buffer(3)]],
    device float *bnew [[buffer(4)]],
    device const int *father [[buffer(5)]],
    device const int *nbor [[buffer(6)]],
    constant int &head_idx [[buffer(7)]],
    constant int &num_subgrids [[buffer(8)]],
    constant int &ngridmax [[buffer(9)]],
    constant int &ilevel [[buffer(10)]],
    constant int &levelmin [[buffer(11)]],
    constant int &levelmax [[buffer(12)]],
    constant float &gamma [[buffer(13)]],
    constant float &smallr [[buffer(14)]],
    constant float &smallc2 [[buffer(15)]],
    constant float &dt [[buffer(16)]],
    constant float &dx [[buffer(17)]],
    constant int &slope [[buffer(18)]],
    constant int &slope_mag [[buffer(19)]],
    constant float &switch_llf_dmin [[buffer(20)]],
    constant float &switch_llf_pmin [[buffer(21)]],
    constant int &induction [[buffer(22)]],
    constant float &etamag [[buffer(23)]],
    constant float *constant_gravity [[buffer(24)]],
    uint block_idx [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]])
{
    if (int(block_idx) >= num_subgrids) return;
    threadgroup float smem[MR_SMEM_FLOATS];
    threadgroup bool refined[MR_REFINED_CELLS];
    int subgrid_idx = head_idx + int(block_idx);
    float dtdx = dt / dx;
    for (int work = int(tid); work < MR_LOCAL_CELLS; work += int(threads_per_group)) {
        int oct_lattice = work / MR_TWOTONDIM;
        int i_sg;
        int j_sg;
        int k_sg;
        mr_index3(oct_lattice, MR_NSUBGRIDP2, MR_NSUBGRIDP2, i_sg, j_sg, k_sg);
        int ind_nbor = oct_lattice + 1;
        int source_idx = mr_nbor_get(nbor, subgrid_idx, ind_nbor);
        int cell_idx = work % MR_TWOTONDIM + 1;
        int ib;
        int jb;
        int kb;
        mr_index3(cell_idx - 1, 2, 2, ib, jb, kb);
        int i = ib + 2 * i_sg;
        int j = jb + 2 * j_sg;
        int k = kb + 2 * k_sg;
        float b0[6];
        for (int component = 0; component < 6; ++component) b0[component] = mr_b_get(bold, source_idx, component + 1, cell_idx);
        if (i >= 1) {
            int face_source = ib == 0 ? mr_nbor_get(nbor, subgrid_idx, ind_nbor - 1) : source_idx;
            int face_cell = cell_idx + 1 - 2 * ib;
            mr_bf_set(smem, 0, i, j, k, 0.5f * (b0[0] + mr_b_get(bold, face_source, 4, face_cell)));
        }
        if (j >= 1) {
            int face_source = jb == 0 ? mr_nbor_get(nbor, subgrid_idx, ind_nbor - MR_NSUBGRIDP2) : source_idx;
            int face_cell = cell_idx + 2 * (1 - 2 * jb);
            mr_bf_set(smem, 1, i, j, k, 0.5f * (b0[1] + mr_b_get(bold, face_source, 5, face_cell)));
        }
        if (k >= 1) {
            int face_source = kb == 0 ? mr_nbor_get(nbor, subgrid_idx, ind_nbor - MR_NSUBGRIDP2 * MR_NSUBGRIDP2) : source_idx;
            int face_cell = cell_idx + 4 * (1 - 2 * kb);
            mr_bf_set(smem, 2, i, j, k, 0.5f * (b0[2] + mr_b_get(bold, face_source, 6, face_cell)));
        }
        mr_conserved_t u;
        u.density = mr_u_get(uold, source_idx, 1, cell_idx);
        u.momentum_x = mr_u_get(uold, source_idx, 2, cell_idx);
        u.momentum_y = mr_u_get(uold, source_idx, 3, cell_idx);
        u.momentum_z = mr_u_get(uold, source_idx, 4, cell_idx);
        u.energy = mr_u_get(uold, source_idx, 5, cell_idx);
        u.Bx = 0.5f * (b0[0] + b0[3]);
        u.By = 0.5f * (b0[1] + b0[4]);
        u.Bz = 0.5f * (b0[2] + b0[5]);
        mr_primitive_t q = mr_conserved_to_primitive(u, gamma, smallr, smallc2);
        q.velocity_x += 0.5f * dt * constant_gravity[0];
        q.velocity_y += 0.5f * dt * constant_gravity[1];
        q.velocity_z += 0.5f * dt * constant_gravity[2];
        mr_local_set(smem, 0, i, j, k, q.density);
        mr_local_set(smem, 1, i, j, k, q.velocity_x);
        mr_local_set(smem, 2, i, j, k, q.velocity_y);
        mr_local_set(smem, 3, i, j, k, q.velocity_z);
        mr_local_set(smem, 4, i, j, k, q.pressure);
        if (i >= 1 && i <= MR_TRACE && j >= 1 && j <= MR_TRACE && k >= 1 && k <= MR_TRACE) mr_refined_set(refined, i, j, k, grid[source_idx - 1].refined[cell_idx - 1] != 0);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < MR_REFINED_CELLS) {
        int i;
        int j;
        int k;
        mr_index3(int(tid), MR_TRACE, MR_TRACE, i, j, k);
        ++i;
        ++j;
        ++k;
        mr_trace_t trace = mr_predict_cell(smem, i, j, k, gamma, dtdx, slope, slope_mag, induction);
        if (i > 1 && j > 1 && j < MR_TRACE && k > 1 && k < MR_TRACE) {
            int index = mr_face_index(0, i - 2, j - 2, k - 2);
            mr_face_store(smem, 1, index, mr_traced_face(smem, trace, 0, -1, i, j, k, smallr, smallc2));
        }
        if (i < MR_TRACE && j > 1 && j < MR_TRACE && k > 1 && k < MR_TRACE) {
            int index = mr_face_index(0, i - 1, j - 2, k - 2);
            mr_face_store(smem, 0, index, mr_traced_face(smem, trace, 0, 1, i, j, k, smallr, smallc2));
        }
        if (i > 1 && i < MR_TRACE && j > 1 && k > 1 && k < MR_TRACE) {
            int index = mr_face_index(1, i - 2, j - 2, k - 2);
            mr_face_store(smem, 3, index, mr_traced_face(smem, trace, 1, -1, i, j, k, smallr, smallc2));
        }
        if (i > 1 && i < MR_TRACE && j < MR_TRACE && k > 1 && k < MR_TRACE) {
            int index = mr_face_index(1, i - 2, j - 1, k - 2);
            mr_face_store(smem, 2, index, mr_traced_face(smem, trace, 1, 1, i, j, k, smallr, smallc2));
        }
        if (i > 1 && i < MR_TRACE && j > 1 && j < MR_TRACE && k > 1) {
            int index = mr_face_index(2, i - 2, j - 2, k - 2);
            mr_face_store(smem, 5, index, mr_traced_face(smem, trace, 2, -1, i, j, k, smallr, smallc2));
        }
        if (i > 1 && i < MR_TRACE && j > 1 && j < MR_TRACE && k < MR_TRACE) {
            int index = mr_face_index(2, i - 2, j - 2, k - 1);
            mr_face_store(smem, 4, index, mr_traced_face(smem, trace, 2, 1, i, j, k, smallr, smallc2));
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mr_primitive_t left_face;
    mr_primitive_t right_face;
    int orientation = int(tid) / MR_FACE_CELLS;
    int face_local = int(tid) - orientation * MR_FACE_CELLS;
    if (tid < MR_ALL_FACES) {
        left_face = mr_face_load(smem, 2 * orientation, face_local);
        right_face = mr_face_load(smem, 2 * orientation + 1, face_local);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < MR_ALL_FACES) {
        left_face = mr_rotate_face(left_face, orientation);
        right_face = mr_rotate_face(right_face, orientation);
        bool use_llf = (switch_llf_dmin > 0.0f && min(left_face.density, right_face.density) < switch_llf_dmin) || (switch_llf_pmin > 0.0f && min(left_face.pressure, right_face.pressure) < switch_llf_pmin);
        mr_uct_record_t record;
        mr_conserved_t flux;
        if (use_llf) {
            flux = mr_hll_mhd_flux(left_face, right_face, gamma, smallr, smallc2);
            record = mr_llf_record(left_face, right_face, gamma, smallr, smallc2);
        } else {
            flux = mr_hlld_mhd_flux(left_face, right_face, gamma, smallr, smallc2, record);
        }
        flux = mr_unrotate_flux(flux, orientation);
        if (induction != 0) {
            flux.density = 0.0f;
            flux.momentum_x = 0.0f;
            flux.momentum_y = 0.0f;
            flux.momentum_z = 0.0f;
            flux.energy = 0.0f;
        }
        mr_flux_set(smem, 0, int(tid), flux.density);
        mr_flux_set(smem, 1, int(tid), flux.momentum_x);
        mr_flux_set(smem, 2, int(tid), flux.momentum_y);
        mr_flux_set(smem, 3, int(tid), flux.momentum_z);
        mr_flux_set(smem, 4, int(tid), flux.energy);
        mr_interior_record_store(smem, int(tid), record);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mr_uct_record_t shell_record;
    if (tid < MR_ALL_SHELLS) shell_record = mr_shell_solve(smem, int(tid), gamma, smallr, smallc2, dtdx, slope, slope_mag, induction, switch_llf_dmin, switch_llf_pmin);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < MR_ALL_SHELLS) mr_shell_record_store(smem, int(tid), shell_record);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int edge = int(tid); edge < MR_ALL_EDGES; edge += int(threads_per_group)) {
        int I;
        int J;
        int K;
        float emf;
        int edge_orientation;
        if (edge < MR_EDGE_CELLS) {
            mr_index3(edge, MR_M + 1, MR_M + 1, I, J, K);
            mr_uct_record_t f1lo = mr_get_record(smem, 0, I, J - 1, K);
            mr_uct_record_t f1hi = mr_get_record(smem, 0, I, J, K);
            mr_uct_record_t f2lo = mr_get_record(smem, 1, J, I - 1, K);
            mr_uct_record_t f2hi = mr_get_record(smem, 1, J, I, K);
            emf = mr_uct_edge(f1lo, f1hi, f2lo, f2hi, 1, 1);
            edge_orientation = 0;
            if (etamag > 0.0f) {
                int X = I + 2;
                int Y = J + 2;
                int Z = K + 2;
                emf -= etamag / dx * ((mr_bf_get(smem, 1, X, Y, Z) - mr_bf_get(smem, 1, X - 1, Y, Z)) - (mr_bf_get(smem, 0, X, Y, Z) - mr_bf_get(smem, 0, X, Y - 1, Z)));
            }
            if (ilevel < levelmax) {
                int X = I + 2;
                int Y = J + 2;
                int Z = K + 2;
                if (mr_refined_get(refined, X - 1, Y - 1, Z) || mr_refined_get(refined, X, Y - 1, Z) || mr_refined_get(refined, X - 1, Y, Z) || mr_refined_get(refined, X, Y, Z)) emf = 0.0f;
            }
        } else if (edge < 2 * MR_EDGE_CELLS) {
            mr_index3(edge - MR_EDGE_CELLS, MR_M + 1, MR_M, I, J, K);
            mr_uct_record_t f1lo = mr_get_record(smem, 2, K, I - 1, J);
            mr_uct_record_t f1hi = mr_get_record(smem, 2, K, I, J);
            mr_uct_record_t f2lo = mr_get_record(smem, 0, I, J, K - 1);
            mr_uct_record_t f2hi = mr_get_record(smem, 0, I, J, K);
            emf = mr_uct_edge(f1lo, f1hi, f2lo, f2hi, 1, 2);
            edge_orientation = 1;
            if (etamag > 0.0f) {
                int X = I + 2;
                int Y = J + 2;
                int Z = K + 2;
                emf -= etamag / dx * ((mr_bf_get(smem, 0, X, Y, Z) - mr_bf_get(smem, 0, X, Y, Z - 1)) - (mr_bf_get(smem, 2, X, Y, Z) - mr_bf_get(smem, 2, X - 1, Y, Z)));
            }
            if (ilevel < levelmax) {
                int X = I + 2;
                int Y = J + 2;
                int Z = K + 2;
                if (mr_refined_get(refined, X - 1, Y, Z - 1) || mr_refined_get(refined, X, Y, Z - 1) || mr_refined_get(refined, X - 1, Y, Z) || mr_refined_get(refined, X, Y, Z)) emf = 0.0f;
            }
        } else {
            mr_index3(edge - 2 * MR_EDGE_CELLS, MR_M, MR_M + 1, I, J, K);
            mr_uct_record_t f1lo = mr_get_record(smem, 1, J, I, K - 1);
            mr_uct_record_t f1hi = mr_get_record(smem, 1, J, I, K);
            mr_uct_record_t f2lo = mr_get_record(smem, 2, K, I, J - 1);
            mr_uct_record_t f2hi = mr_get_record(smem, 2, K, I, J);
            emf = mr_uct_edge(f1lo, f1hi, f2lo, f2hi, 2, 2);
            edge_orientation = 2;
            if (etamag > 0.0f) {
                int X = I + 2;
                int Y = J + 2;
                int Z = K + 2;
                emf -= etamag / dx * ((mr_bf_get(smem, 2, X, Y, Z) - mr_bf_get(smem, 2, X, Y - 1, Z)) - (mr_bf_get(smem, 1, X, Y, Z) - mr_bf_get(smem, 1, X, Y, Z - 1)));
            }
            if (ilevel < levelmax) {
                int X = I + 2;
                int Y = J + 2;
                int Z = K + 2;
                if (mr_refined_get(refined, X, Y - 1, Z - 1) || mr_refined_get(refined, X, Y, Z - 1) || mr_refined_get(refined, X, Y - 1, Z) || mr_refined_get(refined, X, Y, Z)) emf = 0.0f;
            }
        }
        mr_emf_set(smem, edge_orientation, I, J, K, emf);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < MR_ALL_FACES && ilevel < levelmax) {
        int i;
        int j;
        int k;
        if (orientation == 0) {
            mr_index3(face_local, MR_M + 1, MR_M, i, j, k);
            if (mr_refined_get(refined, i + 1, j + 2, k + 2) || mr_refined_get(refined, i + 2, j + 2, k + 2)) for (int field = 0; field < 5; ++field) mr_flux_set(smem, field, int(tid), 0.0f);
        } else if (orientation == 1) {
            mr_index3(face_local, MR_M, MR_M + 1, i, j, k);
            if (mr_refined_get(refined, i + 2, j + 1, k + 2) || mr_refined_get(refined, i + 2, j + 2, k + 2)) for (int field = 0; field < 5; ++field) mr_flux_set(smem, field, int(tid), 0.0f);
        } else {
            mr_index3(face_local, MR_M, MR_M, i, j, k);
            if (mr_refined_get(refined, i + 2, j + 2, k + 1) || mr_refined_get(refined, i + 2, j + 2, k + 2)) for (int field = 0; field < 5; ++field) mr_flux_set(smem, field, int(tid), 0.0f);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < MR_INTERIOR_CELLS) {
        int i_sg;
        int j_sg;
        int k_sg;
        mr_index3(int(tid) / MR_TWOTONDIM, MR_NSUBGRID, MR_NSUBGRID, i_sg, j_sg, k_sg);
        ++i_sg;
        ++j_sg;
        ++k_sg;
        int ind_nbor = 1 + i_sg + MR_NSUBGRIDP2 * j_sg + MR_NSUBGRIDP2 * MR_NSUBGRIDP2 * k_sg;
        int oct_idx = mr_nbor_get(nbor, subgrid_idx, ind_nbor);
        int cell_idx = int(tid) % MR_TWOTONDIM + 1;
        int i;
        int j;
        int k;
        mr_index3(cell_idx - 1, 2, 2, i, j, k);
        i += 2 * (i_sg - 1);
        j += 2 * (j_sg - 1);
        k += 2 * (k_sg - 1);
        int fx0 = mr_face_index(0, i, j, k);
        int fx1 = mr_face_index(0, i + 1, j, k);
        int fy0 = MR_FACE_CELLS + mr_face_index(1, i, j, k);
        int fy1 = MR_FACE_CELLS + mr_face_index(1, i, j + 1, k);
        int fz0 = 2 * MR_FACE_CELLS + mr_face_index(2, i, j, k);
        int fz1 = 2 * MR_FACE_CELLS + mr_face_index(2, i, j, k + 1);
        for (int field = 0; field < 5; ++field) {
            float update = (mr_flux_get(smem, field, fx0) - mr_flux_get(smem, field, fx1) + mr_flux_get(smem, field, fy0) - mr_flux_get(smem, field, fy1) + mr_flux_get(smem, field, fz0) - mr_flux_get(smem, field, fz1)) * dtdx;
            mr_u_set(unew, oct_idx, field + 1, cell_idx, mr_u_get(unew, oct_idx, field + 1, cell_idx) + update);
        }
        float dflux = ((mr_emf_get(smem, 1, i, j, k) - mr_emf_get(smem, 1, i, j, k + 1)) - (mr_emf_get(smem, 0, i, j, k) - mr_emf_get(smem, 0, i, j + 1, k))) * dtdx;
        mr_b_set(bnew, oct_idx, 1, cell_idx, mr_b_get(bnew, oct_idx, 1, cell_idx) + dflux);
        dflux = ((mr_emf_get(smem, 1, i + 1, j, k) - mr_emf_get(smem, 1, i + 1, j, k + 1)) - (mr_emf_get(smem, 0, i + 1, j, k) - mr_emf_get(smem, 0, i + 1, j + 1, k))) * dtdx;
        mr_b_set(bnew, oct_idx, 4, cell_idx, mr_b_get(bnew, oct_idx, 4, cell_idx) + dflux);
        dflux = ((mr_emf_get(smem, 0, i, j, k) - mr_emf_get(smem, 0, i + 1, j, k)) - (mr_emf_get(smem, 2, i, j, k) - mr_emf_get(smem, 2, i, j, k + 1))) * dtdx;
        mr_b_set(bnew, oct_idx, 2, cell_idx, mr_b_get(bnew, oct_idx, 2, cell_idx) + dflux);
        dflux = ((mr_emf_get(smem, 0, i, j + 1, k) - mr_emf_get(smem, 0, i + 1, j + 1, k)) - (mr_emf_get(smem, 2, i, j + 1, k) - mr_emf_get(smem, 2, i, j + 1, k + 1))) * dtdx;
        mr_b_set(bnew, oct_idx, 5, cell_idx, mr_b_get(bnew, oct_idx, 5, cell_idx) + dflux);
        dflux = ((mr_emf_get(smem, 2, i, j, k) - mr_emf_get(smem, 2, i, j + 1, k)) - (mr_emf_get(smem, 1, i, j, k) - mr_emf_get(smem, 1, i + 1, j, k))) * dtdx;
        mr_b_set(bnew, oct_idx, 3, cell_idx, mr_b_get(bnew, oct_idx, 3, cell_idx) + dflux);
        dflux = ((mr_emf_get(smem, 2, i, j, k + 1) - mr_emf_get(smem, 2, i, j + 1, k + 1)) - (mr_emf_get(smem, 1, i, j, k + 1) - mr_emf_get(smem, 1, i + 1, j, k + 1))) * dtdx;
        mr_b_set(bnew, oct_idx, 6, cell_idx, mr_b_get(bnew, oct_idx, 6, cell_idx) + dflux);
    }
    if (ilevel > levelmin) {
        mr_coarse_hydro_update(smem, grid, father, nbor, unew, subgrid_idx, ngridmax, int(tid), dtdx);
        if (tid == 0) mr_coarse_ct_update(smem, grid, father, nbor, bnew, subgrid_idx, ngridmax, dtdx);
    }
}
