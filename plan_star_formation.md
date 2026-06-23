# Objective
Port the CPU-based `star_formation` routine to CUDA to enable GPU-accelerated star formation, maintaining the existing architectural philosophy of the RAMSES CUDA port.

# Key Files & Context
- `gpu/gpu_runner.cuf`: The main dispatcher and device memory holder.
- `pm/init_part.f90`: Handles the allocation of particle-related arrays.
- `star/star_formation.f90`: The original CPU routine containing the physics and logic.
- `amr/amr_step.f90`: The main time-stepping loop where star formation is invoked.
- New file `gpu/gpu_star.cuf`: Will contain the actual CUDA kernels for star formation.

# Implementation Steps

## 0. Plan Documentation
- Save this entire plan to `plan_star_formation.md` in the root of the repository for reference. No other files will be modified until this is confirmed.

## 1. Infrastructure and Device Memory (`gpu/gpu_runner.cuf` & `pm/init_part.f90`)
- **`gpu_runner.cuf`**:
    - Import the `curand` module (`use curand_device`).
    - Declare device arrays specifically for star particles: `star_xp`, `star_vp`, `star_mp`, `star_tp`, `star_zp`, `star_levelp`, `star_idp`.
    - Note: No new temporary array is needed for star formation as we will reuse the existing `prefix_sum` array.
- **`pm/init_part.f90`**:
    - Modify `init_star` to allocate the newly declared `star_*` device arrays when `_CUDA` is defined.
    - Ensure `prefix_sum` capacity is sufficient for cell-level scans (at least `ngridmax * twotondim`).

## 2. Kernel Development (`gpu/gpu_star.cuf`)
- Create `gpu_star.cuf` and include it in `gpu_runner.cuf`.
- Implement a two-pass kernel strategy:
    - **Pass 1 (`count_star_formation_kernel`)**:
        - One thread per cell.
        - Check leaf cell status, density threshold (`d0`), and zoom mask.
        - Calculate SF efficiency (`sfr_ff`) based on local gas properties.
        - Initialize `curandState` locally in registers using the cell's unique ID and current time/step as the seed.
        - Use `curand_poisson` to determine `nstar` for each cell.
        - Store `nstar` in the existing `prefix_sum` array.
    - **Pass 2 (`star_formation_kernel`)**:
        - Takes the prefix-summed `prefix_sum` as input.
        - Updates the gas density (`uold(ind,1)`) and pressure (`uold(ind,5)`).
        - Writes the newly formed star properties (position, velocity, mass, age, metallicity) into the `star_*` arrays using the computed offsets.

## 3. Dispatcher Implementation (`gpu/gpu_runner.cuf`)
- Implement `gpu_star_formation(sim, ilevel, mstar_loc)`:
    - Launch `count_star_formation_kernel` (filling `prefix_sum` with counts).
    - Perform an inclusive prefix sum (`gpu_scan`) on `prefix_sum`.
    - Check if the new `npart` exceeds `nstarmax`.
    - Launch `star_formation_kernel` to finalize particle creation and gas updates.
    - Update `sim%star%npart` and `sim%star%npart_tot` on the host, and fetch `mstar_loc`.

## 4. Integration and Synchronization
- Update `star/star_formation.f90` (inside `r_star_formation`) to dispatch to `gpu_star_formation` when running on GPU. No changes needed in `amr_step.f90`.
- Ensure that the generated stars are transferred back to the host at the end of the time step if required for I/O or CPU-side tasks, using the logic in `gpu_manager.f90`.

# Verification & Testing
- Compile the code to ensure `curand` links correctly (`-lcurand` may be needed in the Makefile/CMake).
- Run a small test problem (e.g., `namelist/coeur.nml` or similar) with star formation enabled.
- Verify that the number of stars created and their mass distribution on the GPU statistically match the CPU implementation.
- Check that the total gas mass is correctly conserved after star formation.
