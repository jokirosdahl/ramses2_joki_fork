The namelist block `&CR_GROUPS` is used to specify per-group properties for cosmic ray multigroup transport. The number of CR groups is set at compile time via the `NCRGRP` preprocessor variable (default: 1). Each array parameter below has one entry per CR group.

| Variable name | Fortran type | Default value  | Description      |
|:------------------- |:-------|:----- |:------------------------- |
| `cr_d`              | `real array`   | `1.0d29` (per group)  | Parallel CR diffusion coefficient (cm²/s) for each CR group. This controls the rate of spatial diffusion of cosmic rays along magnetic field lines. |
| `cr_d_perp_factors` | `real array`   | `1.0d-6` (per group)  | Perpendicular CR diffusion suppression factor for each group. The perpendicular diffusion coefficient is computed as `cr_d * cr_d_perp_factors`. A value of `1.0d-6` means perpendicular diffusion is strongly suppressed relative to parallel diffusion. |
| `fecr`              | `real array`   | `0.0` (per group)     | Fraction of supernova energy injected into cosmic rays for each group. A typical value used in galaxy formation simulations is `0.1` (10% of the SN energy goes into CRs). |
