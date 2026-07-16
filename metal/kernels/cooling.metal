/*
 * metal/kernels/cooling.metal
 *
 * Metal port of gpu_cooling.cuf.
 */

#ifndef NVAR
#define NVAR 5
#endif

#include <metal_stdlib>
#include "../metal_types.h"
using namespace metal;

struct CoolingParams {
    int table_n1;
    int table_n2;
    int head_idx;
    int num_octs;
    float dlog_nH;
    float dlog_T2;
    float X_frac;
    float dt;
    float scale_T2;
    float scale_nH;
    float z_ave;
    float T2max;
    float gamma;
    float smallr;
    float smallc2;
    int eos_type;
    float eos_T2;
    float eos_nH;
    float eos_index;
    int imetal;
    int cooling;
    int metal;
    int self_shielding;
    int isothermal;
};

// CGS constants
constant float kB_cgs = 1.380649e-16f;
constant float varmax = 4.0f;
constant float T2_max_fix = 1.0e9f;
constant float mH_cgs = 1.6605390e-24f;
constant float T2_min_fix = 1.0e-2f;

inline float interpolate_rate(
    int i_nH, int i_T2, int n1, float yy, float w1H, float w2H, float h, float h2, float h3, float tau,
    device const float* rate_tbl, device const float* prime_tbl, thread float &rate_prime
) {
    float fa      = rate_tbl[i_nH + i_T2 * n1] * w1H + rate_tbl[(i_nH + 1) + i_T2 * n1] * w2H;
    float fb      = rate_tbl[i_nH + (i_T2 + 1) * n1] * w1H + rate_tbl[(i_nH + 1) + (i_T2 + 1) * n1] * w2H;
    float fprimea = prime_tbl[i_nH + i_T2 * n1] * w1H + prime_tbl[(i_nH + 1) + i_T2 * n1] * w2H;
    float fprimeb = prime_tbl[i_nH + (i_T2 + 1) * n1] * w1H + prime_tbl[(i_nH + 1) + (i_T2 + 1) * n1] * w2H;

    float al = fprimea;
    float be = 3.0f * (fb - fa) / h2 - (2.0f * fprimea + fprimeb) / h;
    float ga = (fprimea + fprimeb) / h2 - 2.0f * (fb - fa) / h3;

    float yy2 = yy * yy;
    float yy3 = yy2 * yy;
    float val = pow(10.0f, fa + al * yy + be * yy2 + ga * yy3);
    rate_prime = val / tau * (al + 2.0f * be * yy + 3.0f * ga * yy2);
    return val;
}

inline float solve_cooling_metal(
    float nH_cell, float T2_cell, float Zsolar_cell, float boost_cell, float dt, float X_frac,
    int n1, int n2, float dlog_nH, float dlog_T2,
    device const float* nH_tbl, device const float* T2_tbl,
    device const float* cool_tbl, device const float* heat_tbl,
    device const float* cool_com_tbl, device const float* heat_com_tbl,
    device const float* metal_tbl, device const float* cool_prime_tbl,
    device const float* heat_prime_tbl, device const float* cool_com_prime_tbl,
    device const float* heat_com_prime_tbl, device const float* metal_prime_tbl
) {
    float h = 1.0f / dlog_T2;
    float h2 = h * h;
    float h3 = h2 * h;
    float precoeff = 2.0f * X_frac / (3.0f * kB_cgs);
    float logT2max = log10(T2_max_fix);

    float facH = clamp(log10(nH_cell / boost_cell), nH_tbl[0], nH_tbl[n1 - 1]);
    int i_nH = clamp(int((facH - nH_tbl[0]) * dlog_nH), 0, n1 - 2);
    float w1H = (nH_tbl[i_nH + 1] - facH) * dlog_nH;
    float w2H = (facH - nH_tbl[i_nH]) * dlog_nH;

    float tau = T2_cell;
    float tau_ini = T2_cell;
    float time_max = dt * precoeff * nH_cell;
    float time = 0.0f;
    float wmax = 1.0f / time_max;
    float tau_old = tau;
    float time_old = 0.0f;

    int iter = 0;
    while (time < time_max) {
        iter++;
        if (iter > 500) break;

        float facT = log10(tau);

        float lambda = 0.0f;
        float lambda_prime = 0.0f;

        if (facT <= logT2max) {
            int i_T2 = clamp(int((facT - T2_tbl[0]) * dlog_T2), 0, n2 - 2);
            float yy = facT - T2_tbl[i_T2];

            float cool_prime_val = 0.0f;
            float cool = interpolate_rate(i_nH, i_T2, n1, yy, w1H, w2H, h, h2, h3, tau, cool_tbl, cool_prime_tbl, cool_prime_val);

            float heat_prime_val = 0.0f;
            float heat = interpolate_rate(i_nH, i_T2, n1, yy, w1H, w2H, h, h2, h3, tau, heat_tbl, heat_prime_tbl, heat_prime_val);

            float cool_com_prime_val = 0.0f;
            float cool_com = interpolate_rate(i_nH, i_T2, n1, yy, w1H, w2H, h, h2, h3, tau, cool_com_tbl, cool_com_prime_tbl, cool_com_prime_val);

            float heat_com_prime_val = 0.0f;
            float heat_com = interpolate_rate(i_nH, i_T2, n1, yy, w1H, w2H, h, h2, h3, tau, heat_com_tbl, heat_com_prime_tbl, heat_com_prime_val);

            float metal_prime_val = 0.0f;
            float metal_val = interpolate_rate(i_nH, i_T2, n1, yy, w1H, w2H, h, h2, h3, tau, metal_tbl, metal_prime_tbl, metal_prime_val);

            lambda = cool + Zsolar_cell * metal_val - heat + (cool_com - heat_com) / nH_cell;
            lambda_prime = cool_prime_val + Zsolar_cell * metal_prime_val - heat_prime_val + (cool_com_prime_val - heat_com_prime_val) / nH_cell;
        } else {
            lambda = 1.42e-27f * sqrt(tau) * 1.1f;
            lambda_prime = lambda / (2.0f * tau);
        }

        float wcool = max(max(abs(lambda) / tau * varmax, wmax), -lambda_prime * varmax);

        tau_old = tau;
        time_old = time;
        tau = tau * (1.0f + lambda_prime / wcool - lambda / tau / wcool) / (1.0f + lambda_prime / wcool);
        time = time + 1.0f / wcool;
    }

    if (time > time_old) {
        tau = tau * (time_max - time_old) / (time - time_old) + tau_old * (time - time_max) / (time - time_old);
    }

    return tau - tau_ini;
}

kernel void cooling_kernel(
    device float*             uold           [[buffer(0)]],
    device const float*       bold           [[buffer(1)]],
    device const oct_t*       grid           [[buffer(2)]],
    device const CoolingParams& params       [[buffer(3)]],
    device const float*       nH_tbl         [[buffer(4)]],
    device const float*       T2_tbl         [[buffer(5)]],
    device const float*       cool_tbl       [[buffer(6)]],
    device const float*       heat_tbl       [[buffer(7)]],
    device const float*       cool_com_tbl   [[buffer(8)]],
    device const float*       heat_com_tbl   [[buffer(9)]],
    device const float*       metal_tbl      [[buffer(10)]],
    device const float*       cool_prime_tbl [[buffer(11)]],
    device const float*       heat_prime_tbl [[buffer(12)]],
    device const float*       cool_com_prime_tbl [[buffer(13)]],
    device const float*       heat_com_prime_tbl [[buffer(14)]],
    device const float*       metal_prime_tbl    [[buffer(15)]],
    uint2                     thread_idx     [[thread_position_in_threadgroup]],
    uint2                     group_idx      [[threadgroup_position_in_grid]],
    uint2                     group_dim      [[threads_per_threadgroup]]
) {
    int oct_idx = (group_idx.y * group_dim.y + thread_idx.y);
    if (oct_idx >= params.num_octs) return;
    oct_idx = params.head_idx + oct_idx;
    
    int cell_idx = thread_idx.x;

    if (grid[oct_idx].refined[cell_idx]) return;

    int nvar = NVAR;

    float density = max(uold[cell_idx + 8 * 0 + 8 * nvar * oct_idx], params.smallr);
    float velocity_x = uold[cell_idx + 8 * 1 + 8 * nvar * oct_idx] / density;
    float velocity_y = uold[cell_idx + 8 * 2 + 8 * nvar * oct_idx] / density;
    float velocity_z = uold[cell_idx + 8 * 3 + 8 * nvar * oct_idx] / density;
    float energy = uold[cell_idx + 8 * 4 + 8 * nvar * oct_idx] / density;
    float kinetic = 0.5f * (velocity_x * velocity_x + velocity_y * velocity_y + velocity_z * velocity_z);
    
    float emag = 0.0f;
#ifdef MHD
    for (int idim = 0; idim < 3; idim++) {
        emag += 0.125f * pow(bold[cell_idx + 8 * idim + 8 * 6 * oct_idx] + bold[cell_idx + 8 * (idim + 3) + 8 * 6 * oct_idx], 2.f);
    }
#endif

    float poverrho = (params.gamma - 1.0f) * (energy - kinetic - emag / density);
    poverrho = max(poverrho, params.smallc2 / params.gamma);
    float temperature = poverrho * params.scale_T2;
    float nH = density * params.scale_nH;
    float rho = nH * mH_cgs / params.X_frac;

    float T2min = 0.0f;
    if (params.eos_type == 1) {
        T2min = params.eos_T2;
    } else if (params.eos_type == 2) {
        T2min = params.eos_T2 * pow(nH / params.eos_nH, params.eos_index - 1.0f);
    } else if (params.eos_type == 3) {
        T2min = params.eos_T2 * (1.0f + pow(nH / params.eos_nH, params.eos_index - 1.0f));
    } else if (params.eos_type == 4) {
        if (nH < params.eos_nH) {
            T2min = params.eos_T2;
        } else {
            T2min = params.eos_T2 * pow(nH / params.eos_nH, params.eos_index - 1.0f);
        }
    } else if (params.eos_type == 5) {
        float factor1 = sqrt(1.0f + pow(rho / 3.866301516e-15f, 0.8f));
        float factor2 = pow(1.0f + rho / 3.866301516e-10f, -0.3f);
        float factor3 = pow(1.0f + rho / 3.866301516e-05f, 0.56667f);
        T2min = params.eos_T2 * factor1 * factor2 * factor3;
    }

    float T2_therm = 0.0f;
    float delta_T2 = 0.0f;

    if (params.cooling) {
        float Zsolar = params.z_ave;
        if (params.metal) {
            Zsolar = uold[cell_idx + 8 * params.imetal + 8 * nvar * oct_idx] / density / 0.02f;
        }

        float boost = 1.0f;
        if (params.self_shielding) {
            boost = max(exp(-nH / 0.01f), 1.0e-20f);
        }

        T2_therm = min(max(temperature - T2min, T2_min_fix), params.T2max);

        delta_T2 = solve_cooling_metal(
            nH, T2_therm, Zsolar, boost, params.dt, params.X_frac,
            params.table_n1, params.table_n2, params.dlog_nH, params.dlog_T2,
            nH_tbl, T2_tbl, cool_tbl, heat_tbl, cool_com_tbl, heat_com_tbl,
            metal_tbl, cool_prime_tbl, heat_prime_tbl, cool_com_prime_tbl,
            heat_com_prime_tbl, metal_prime_tbl
        );
    }

    if (params.isothermal) {
        energy = T2min / params.scale_T2 / (params.gamma - 1.0f) + kinetic + emag / density;
    } else {
        energy = (T2_therm + delta_T2 + T2min) / params.scale_T2 / (params.gamma - 1.0f) + kinetic + emag / density;
    }

    uold[cell_idx + 8 * 4 + 8 * nvar * oct_idx] = density * energy;
}
