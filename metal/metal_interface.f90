module metal_interface
  use iso_c_binding
  implicit none

  interface

    subroutine mtl_init() bind(C, name="mtl_init")
    end subroutine mtl_init

    subroutine mtl_alloc_amr(ngridmax, ncachemax, nvar, twotondim, hash_size) &
        bind(C, name="mtl_alloc_amr")
      import c_int
      integer(c_int), value :: ngridmax, ncachemax, nvar, twotondim, hash_size
    end subroutine mtl_alloc_amr

    subroutine mtl_alloc_refine(ngridmax, ncachemax, nlevelmax) &
        bind(C, name="mtl_alloc_refine")
      import c_int
      integer(c_int), value :: ngridmax, ncachemax, nlevelmax
    end subroutine mtl_alloc_refine

    subroutine mtl_upload_level_params(ckey_max, key_off, &
        box_ckey_min, box_ckey_max, periodic, nlevelmax) &
        bind(C, name="mtl_upload_level_params")
      import c_ptr, c_int
      type(c_ptr), value     :: ckey_max      ! integer(c_int)(1:nlevelmax+1)
      type(c_ptr), value     :: key_off        ! integer(c_long)(1:nlevelmax+1)
      type(c_ptr), value     :: box_ckey_min   ! integer(c_int)(1:ndim,1:nlevelmax+1)
      type(c_ptr), value     :: box_ckey_max   ! integer(c_int)(1:ndim,1:nlevelmax+1)
      integer(c_int), intent(in) :: periodic(3)
      integer(c_int), value  :: nlevelmax
    end subroutine mtl_upload_level_params

    subroutine mtl_set_grid_device(uold, unew, grid, ngridmax, nvar, twotondim) &
        bind(C, name="mtl_set_grid_device")
      import c_ptr, c_int
      type(c_ptr), value  :: uold   ! real(kind=4)(1:twotondim,1:nvar,1:ngridmax)
      type(c_ptr), value  :: unew   ! real(kind=4)(1:twotondim,1:nvar,1:ngridmax)
      type(c_ptr), value  :: grid   ! type(oct)(1:ngridmax)
      integer(c_int), value :: ngridmax, nvar, twotondim
    end subroutine mtl_set_grid_device

    subroutine mtl_upload_flag1(flag1, ngridmax) bind(C, name="mtl_upload_flag1")
      import c_ptr, c_int
      type(c_ptr),    value :: flag1
      integer(c_int), value :: ngridmax
    end subroutine mtl_upload_flag1

    subroutine mtl_transfer_grid_host(uold, ngridmax, nvar, twotondim) &
        bind(C, name="mtl_transfer_grid_host")
      import c_ptr, c_int
      type(c_ptr), value  :: uold
      integer(c_int), value :: ngridmax, nvar, twotondim
    end subroutine mtl_transfer_grid_host

    subroutine mtl_transfer_grid_struct_host(grid, ngridmax) &
        bind(C, name="mtl_transfer_grid_struct_host")
      import c_ptr, c_int
      type(c_ptr), value    :: grid
      integer(c_int), value :: ngridmax
    end subroutine mtl_transfer_grid_struct_host

    subroutine mtl_set_unew(head_idx, num_octs) bind(C, name="mtl_set_unew")
      import c_int
      integer(c_int), value :: head_idx   ! 1-based index of first oct
      integer(c_int), value :: num_octs
    end subroutine mtl_set_unew

    subroutine mtl_set_uold(head_idx, num_octs) bind(C, name="mtl_set_uold")
      import c_int
      integer(c_int), value :: head_idx
      integer(c_int), value :: num_octs
    end subroutine mtl_set_uold

    subroutine mtl_cmpdt(head_idx, num_octs, dx, gamma, smallr, smallc2, &
        courant_factor, constant_gravity, mass, ekin, eint, emag, dt) &
        bind(C, name="mtl_cmpdt")
      import c_int, c_float
      integer(c_int), value :: head_idx, num_octs
      real(c_float),  value :: dx, gamma, smallr, smallc2, courant_factor
      real(c_float), intent(in) :: constant_gravity(3)
      real(c_float), intent(out) :: mass, ekin, eint, emag, dt
    end subroutine mtl_cmpdt

    subroutine mtl_insert_hash(head_idx, num_octs, hash_size, &
        ckey_max_l, key_off_l) &
        bind(C, name="mtl_insert_hash")
      import c_int, c_long
      integer(c_int), value :: head_idx, num_octs
      integer(c_int), value :: hash_size
      integer(c_int), value :: ckey_max_l
      integer(c_long), value :: key_off_l
    end subroutine mtl_insert_hash

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

    subroutine mtl_prefix_scan(offset, n) bind(C, name="mtl_prefix_scan")
      import c_int
      integer(c_int), value :: offset, n
    end subroutine mtl_prefix_scan

    function mtl_get_prefix_total(offset, n) bind(C, name="mtl_get_prefix_total")
      import c_int
      integer(c_int)        :: mtl_get_prefix_total
      integer(c_int), value :: offset, n
    end function mtl_get_prefix_total


    function mtl_init_flag_batch(head_coarse, noct_coarse, head_fine, noct_fine) &
        bind(C, name="mtl_init_flag_batch")
      import c_int
      integer(c_int)        :: mtl_init_flag_batch
      integer(c_int), value :: head_coarse, noct_coarse, head_fine, noct_fine
    end function mtl_init_flag_batch

    function mtl_user_flag_batch(head_idx, num_octs, &
        gamma, smallr, smallc2, err_grad_d, err_grad_p, floor_d, floor_p) &
        bind(C, name="mtl_user_flag_batch")
      import c_int, c_float
      integer(c_int)        :: mtl_user_flag_batch
      integer(c_int), value :: head_idx, num_octs
      real(c_float),  value :: gamma, smallr, smallc2
      real(c_float),  value :: err_grad_d, err_grad_p, floor_d, floor_p
    end function mtl_user_flag_batch

    function mtl_smooth_flag_batch(head_idx, num_octs) &
        bind(C, name="mtl_smooth_flag_batch")
      import c_int
      integer(c_int)        :: mtl_smooth_flag_batch
      integer(c_int), value :: head_idx, num_octs
    end function mtl_smooth_flag_batch

    subroutine mtl_reset_flag1(head_idx, num_octs) bind(C, name="mtl_reset_flag1")
      import c_int
      integer(c_int), value :: head_idx, num_octs
    end subroutine mtl_reset_flag1

    subroutine mtl_reset_flag2(head_idx, num_octs) bind(C, name="mtl_reset_flag2")
      import c_int
      integer(c_int), value :: head_idx, num_octs
    end subroutine mtl_reset_flag2

    subroutine mtl_init_flag(head_idx, num_octs) bind(C, name="mtl_init_flag")
      import c_int
      integer(c_int), value :: head_idx, num_octs
    end subroutine mtl_init_flag

    function mtl_count_flag1(head_idx, num_octs) bind(C, name="mtl_count_flag1")
      import c_int
      integer(c_int)        :: mtl_count_flag1
      integer(c_int), value :: head_idx, num_octs
    end function mtl_count_flag1

    subroutine mtl_hydro_flag(head_idx, num_octs, &
        gamma, smallr, smallc2, err_grad_d, err_grad_p, floor_d, floor_p) &
        bind(C, name="mtl_hydro_flag")
      import c_int, c_float
      integer(c_int), value :: head_idx, num_octs
      real(c_float),  value :: gamma, smallr, smallc2
      real(c_float),  value :: err_grad_d, err_grad_p, floor_d, floor_p
    end subroutine mtl_hydro_flag

    subroutine mtl_count_neighbors(head_idx, num_octs) &
        bind(C, name="mtl_count_neighbors")
      import c_int
      integer(c_int), value :: head_idx, num_octs
    end subroutine mtl_count_neighbors

    subroutine mtl_flag_count(head_idx, num_octs, num_nbors) &
        bind(C, name="mtl_flag_count")
      import c_int
      integer(c_int), value :: head_idx, num_octs, num_nbors
    end subroutine mtl_flag_count

    subroutine mtl_enforce_rules(head_idx, num_octs, ngridmax) &
        bind(C, name="mtl_enforce_rules")
      import c_int
      integer(c_int), value :: head_idx, num_octs, ngridmax
    end subroutine mtl_enforce_rules

    subroutine mtl_build_father(head_idx, num_octs, hash_size, &
        ckey_max_l, key_off_l) bind(C, name="mtl_build_father")
      import c_int, c_long
      integer(c_int), value :: head_idx, num_octs, hash_size, ckey_max_l
      integer(c_long), value :: key_off_l
    end subroutine mtl_build_father


    subroutine mtl_set_ifree(val) bind(C, name="mtl_set_ifree")
      import c_int
      integer(c_int), value :: val
    end subroutine mtl_set_ifree

    function mtl_get_ifree() bind(C, name="mtl_get_ifree")
      import c_int
      integer(c_int) :: mtl_get_ifree
    end function mtl_get_ifree

    subroutine mtl_set_ifree_cache(val) bind(C, name="mtl_set_ifree_cache")
      import c_int
      integer(c_int), value :: val
    end subroutine mtl_set_ifree_cache

    function mtl_get_ifree_cache() bind(C, name="mtl_get_ifree_cache")
      import c_int
      integer(c_int) :: mtl_get_ifree_cache
    end function mtl_get_ifree_cache

    subroutine mtl_advance_ifree_cache(new_noct) &
        bind(C, name="mtl_advance_ifree_cache")
      import c_int
      integer(c_int), value :: new_noct
    end subroutine mtl_advance_ifree_cache

    subroutine mtl_refine_cells(head_idx, num_octs) &
        bind(C, name="mtl_refine_cells")
      import c_int
      integer(c_int), value :: head_idx, num_octs
    end subroutine mtl_refine_cells

    subroutine mtl_reset_hash(ifree, ngridmax, ifree_cache, hash_size) &
        bind(C, name="mtl_reset_hash")
      import c_int
      integer(c_int), value :: ifree, ngridmax, ifree_cache, hash_size
    end subroutine mtl_reset_hash

    subroutine mtl_derefine_cells(head_idx, num_octs, hash_size) &
        bind(C, name="mtl_derefine_cells")
      import c_int
      integer(c_int), value :: head_idx, num_octs, hash_size
    end subroutine mtl_derefine_cells

    subroutine mtl_free_hash_range(head_idx, num_octs, hash_size) &
        bind(C, name="mtl_free_hash_range")
      import c_int
      integer(c_int), value :: head_idx, num_octs, hash_size
    end subroutine mtl_free_hash_range

    subroutine mtl_update_hash_range(head_idx, num_octs, hash_size) &
        bind(C, name="mtl_update_hash_range")
      import c_int
      integer(c_int), value :: head_idx, num_octs, hash_size
    end subroutine mtl_update_hash_range

    subroutine mtl_insert_hash_all(head_idx, num_octs, hash_size) &
        bind(C, name="mtl_insert_hash_all")
      import c_int
      integer(c_int), value :: head_idx, num_octs, hash_size
    end subroutine mtl_insert_hash_all

    subroutine mtl_init_swap_table(head_idx, num_octs) &
        bind(C, name="mtl_init_swap_table")
      import c_int
      integer(c_int), value :: head_idx, num_octs
    end subroutine mtl_init_swap_table

    subroutine mtl_init_prefix_level(head_idx, num_octs, ilevel) &
        bind(C, name="mtl_init_prefix_level")
      import c_int
      integer(c_int), value :: head_idx, num_octs, ilevel
    end subroutine mtl_init_prefix_level

    subroutine mtl_hilbert_sort_level(head_idx, num_octs, num_bits) &
        bind(C, name="mtl_hilbert_sort_level")
      import c_int
      integer(c_int), value :: head_idx, num_octs, num_bits
    end subroutine mtl_hilbert_sort_level

    subroutine mtl_init_prefix_bit(head_idx, num_octs, ibit) &
        bind(C, name="mtl_init_prefix_bit")
      import c_int
      integer(c_int), value :: head_idx, num_octs, ibit
    end subroutine mtl_init_prefix_bit

    subroutine mtl_compute_local_swap(head_idx, num_octs) &
        bind(C, name="mtl_compute_local_swap")
      import c_int
      integer(c_int), value :: head_idx, num_octs
    end subroutine mtl_compute_local_swap

    subroutine mtl_update_global_swap(head_idx, num_octs) &
        bind(C, name="mtl_update_global_swap")
      import c_int
      integer(c_int), value :: head_idx, num_octs
    end subroutine mtl_update_global_swap

    subroutine mtl_sort_gather_grid(head_idx, num_octs) &
        bind(C, name="mtl_sort_gather_grid")
      import c_int
      integer(c_int), value :: head_idx, num_octs
    end subroutine mtl_sort_gather_grid

    subroutine mtl_sort_scatter_grid(head_idx, num_octs) &
        bind(C, name="mtl_sort_scatter_grid")
      import c_int
      integer(c_int), value :: head_idx, num_octs
    end subroutine mtl_sort_scatter_grid

    subroutine mtl_sort_gather_flag(head_idx, num_octs) &
        bind(C, name="mtl_sort_gather_flag")
      import c_int
      integer(c_int), value :: head_idx, num_octs
    end subroutine mtl_sort_gather_flag

    subroutine mtl_sort_scatter_flag(head_idx, num_octs) &
        bind(C, name="mtl_sort_scatter_flag")
      import c_int
      integer(c_int), value :: head_idx, num_octs
    end subroutine mtl_sort_scatter_flag

    subroutine mtl_sort_gather_hydro(head_idx, num_octs) &
        bind(C, name="mtl_sort_gather_hydro")
      import c_int
      integer(c_int), value :: head_idx, num_octs
    end subroutine mtl_sort_gather_hydro

    subroutine mtl_blit_unew_to_uold(head_idx, num_octs) &
        bind(C, name="mtl_blit_unew_to_uold")
      import c_int
      integer(c_int), value :: head_idx, num_octs
    end subroutine mtl_blit_unew_to_uold

    function mtl_nbor_scan(head_idx, num_subgrids, hash_size, input_ind) &
        bind(C, name="mtl_nbor_scan")
      import c_int
      integer(c_int)        :: mtl_nbor_scan
      integer(c_int), value :: head_idx, num_subgrids, hash_size, input_ind
    end function mtl_nbor_scan

    subroutine mtl_cache_fill(head_idx, num_subgrids, hash_size, input_ind, &
        ngridmax, ifree_cache, new_noct) bind(C, name="mtl_cache_fill")
      import c_int
      integer(c_int), value :: head_idx, num_subgrids, hash_size, input_ind
      integer(c_int), value :: ngridmax, ifree_cache, new_noct
    end subroutine mtl_cache_fill

    subroutine mtl_update_nbor_prefix(head_idx, num_subgrids, &
        hash_size, input_ind) bind(C, name="mtl_update_nbor_prefix")
      import c_int
      integer(c_int), value :: head_idx, num_subgrids, hash_size, input_ind
    end subroutine mtl_update_nbor_prefix

    subroutine mtl_compute_cache_swap(head_idx, num_subgrids) &
        bind(C, name="mtl_compute_cache_swap")
      import c_int
      integer(c_int), value :: head_idx, num_subgrids
    end subroutine mtl_compute_cache_swap

    subroutine mtl_make_cache_octs(head_idx, num_subgrids, hash_size, &
        ngridmax, ifree_cache, new_noct, input_ind) &
        bind(C, name="mtl_make_cache_octs")
      import c_int
      integer(c_int), value :: head_idx, num_subgrids, hash_size
      integer(c_int), value :: ngridmax, ifree_cache, new_noct, input_ind
    end subroutine mtl_make_cache_octs

    subroutine mtl_insert_hash_cache_r(hash_size, ngridmax, &
        ifree_cache, new_noct) bind(C, name="mtl_insert_hash_cache_r")
      import c_int
      integer(c_int), value :: hash_size, ngridmax, ifree_cache, new_noct
    end subroutine mtl_insert_hash_cache_r

    subroutine mtl_upload(head_idx, num_octs, internal_energy, &
        gamma, smallr, smallc2) bind(C, name="mtl_upload")
      import c_int, c_float
      integer(c_int), value :: head_idx, num_octs, internal_energy
      real(c_float),  value :: gamma, smallr, smallc2
    end subroutine mtl_upload

    subroutine mtl_device_sync() bind(C, name="mtl_device_sync")
    end subroutine mtl_device_sync

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
