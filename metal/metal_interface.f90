module metal_interface
  use iso_c_binding
  implicit none

  interface

    ! Initialize Metal device and command queue.
    subroutine mtl_init() bind(C, name="mtl_init")
    end subroutine mtl_init

    ! Allocate Metal buffers for uold, unew, grid, nbor, hash_key, hash_val.
    ! No Fortran pointers needed; data is copied in mtl_set_grid_device /
    ! built on device in mtl_build_nbor.
    subroutine mtl_alloc_amr(ngridmax, nvar, twotondim, hash_size) &
        bind(C, name="mtl_alloc_amr")
      import c_int
      integer(c_int), value :: ngridmax, nvar, twotondim, hash_size
    end subroutine mtl_alloc_amr

    ! Copy host arrays into Metal buffers (H->D).
    ! Mirrors r_set_grid_device / cudaMemcpy in the CUDA path.
    ! On Apple Silicon the memcpy stays within DRAM; no PCIe transfer.
    subroutine mtl_set_grid_device(uold, unew, grid, ngridmax, nvar, twotondim) &
        bind(C, name="mtl_set_grid_device")
      import c_ptr, c_int
      type(c_ptr), value  :: uold   ! real(kind=4)(1:twotondim,1:nvar,1:ngridmax)
      type(c_ptr), value  :: unew   ! real(kind=4)(1:twotondim,1:nvar,1:ngridmax)
      type(c_ptr), value  :: grid   ! type(oct)(1:ngridmax)
      integer(c_int), value :: ngridmax, nvar, twotondim
    end subroutine mtl_set_grid_device

    ! Copy Metal buffers back into host arrays (D->H), called before I/O.
    ! Mirrors r_transfer_grid_host / cudaMemcpy in the CUDA path.
    ! Only uold is needed for output; grid is static for levelmin==levelmax.
    subroutine mtl_transfer_grid_host(uold, ngridmax, nvar, twotondim) &
        bind(C, name="mtl_transfer_grid_host")
      import c_ptr, c_int
      type(c_ptr), value  :: uold
      integer(c_int), value :: ngridmax, nvar, twotondim
    end subroutine mtl_transfer_grid_host

    ! Dispatch set_unew_kernel: unew(:,:,oct) = uold(:,:,oct) for octs at ilevel.
    subroutine mtl_set_unew(head_idx, num_octs) bind(C, name="mtl_set_unew")
      import c_int
      integer(c_int), value :: head_idx   ! 1-based index of first oct
      integer(c_int), value :: num_octs
    end subroutine mtl_set_unew

    ! Dispatch set_uold_kernel: uold(:,:,oct) = unew(:,:,oct) for octs at ilevel.
    subroutine mtl_set_uold(head_idx, num_octs) bind(C, name="mtl_set_uold")
      import c_int
      integer(c_int), value :: head_idx
      integer(c_int), value :: num_octs
    end subroutine mtl_set_uold

    ! Dispatch cmpdt_kernel and return integrated quantities + Courant dt.
    ! Mirrors gpu_cmpdt in gpu_runner.cuf; output args passed by reference.
    subroutine mtl_cmpdt(head_idx, num_octs, dx, gamma, smallr, smallc2, &
        courant_factor, constant_gravity, mass, ekin, eint, emag, dt) &
        bind(C, name="mtl_cmpdt")
      import c_int, c_float
      integer(c_int), value :: head_idx, num_octs
      real(c_float),  value :: dx, gamma, smallr, smallc2, courant_factor
      real(c_float), intent(in) :: constant_gravity(3)
      real(c_float), intent(out) :: mass, ekin, eint, emag, dt
    end subroutine mtl_cmpdt

    ! Clear hash table and insert all oct Hilbert keys.
    ! Mirrors insert_hash_kernel<<<>>> in r_set_grid_device (gpu_manager.cuf).
    subroutine mtl_insert_hash(head_idx, num_octs, hash_size, &
        ckey_max_l, key_off_l) &
        bind(C, name="mtl_insert_hash")
      import c_int, c_long
      integer(c_int), value :: head_idx, num_octs
      integer(c_int), value :: hash_size
      integer(c_int), value :: ckey_max_l
      integer(c_long), value :: key_off_l
    end subroutine mtl_insert_hash

    ! Build nbor array from the already-populated hash table.
    ! Mirrors the 27-launch update_nbor_array loop in r_set_grid_device
    ! (gpu_manager.cuf); a single dispatch replaces those 27 launches.
    subroutine mtl_build_nbor(head_idx, num_subgrids, hash_size, &
        ckey_max_l, key_off_l, &
        box_ckey_min, box_ckey_max, periodic_i) &
        bind(C, name="mtl_build_nbor")
      import c_int, c_long
      integer(c_int), value :: head_idx, num_subgrids
      integer(c_int), value :: hash_size
      integer(c_int), value :: ckey_max_l
      integer(c_long), value :: key_off_l
      integer(c_int), intent(in) :: box_ckey_min(3)
      integer(c_int), intent(in) :: box_ckey_max(3)
      integer(c_int), intent(in) :: periodic_i(3)
    end subroutine mtl_build_nbor

    ! Flush and wait for all pending Metal work to complete.
    ! Mirrors cudaDeviceSynchronize() in the CUDA path; called by m_timer
    ! before reading the wall clock so GPU time is captured correctly.
    subroutine mtl_device_sync() bind(C, name="mtl_device_sync")
    end subroutine mtl_device_sync

    ! Dispatch hydro_integrator_kernel (MUSCL-Hancock Godunov).
    ! Mirrors gpu_godunov in gpu_runner.cuf.
    ! constant_gravity: 3-element array (all zeros for GRAV=0).
    subroutine mtl_godunov(head_idx, num_subgrids, ngridmax, &
        ilevel, levelmin, levelmax, &
        gamma, smallr, smallc2, dt, dx, slope, riemann, &
        constant_gravity) &
        bind(C, name="mtl_godunov")
      import c_int, c_float
      integer(c_int), value :: head_idx, num_subgrids, ngridmax
      integer(c_int), value :: ilevel, levelmin, levelmax
      real(c_float),  value :: gamma, smallr, smallc2, dt, dx
      integer(c_int), value :: slope, riemann
      real(c_float), intent(in) :: constant_gravity(3)
    end subroutine mtl_godunov

  end interface

end module metal_interface
