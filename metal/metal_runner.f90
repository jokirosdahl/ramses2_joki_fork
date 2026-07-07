module metal_runner
  use metal_interface
  use iso_c_binding
  implicit none

contains

subroutine metal_allocate_amr(sim)
  use amr_parameters, only: twotondim, ndim
  use hydro_parameters, only: nvar
  use ramses_commons, only: ramses_t
  implicit none
  type(ramses_t) :: sim
  integer :: ilevel
  integer(c_int) :: periodic_i(3)

  sim%m%hash_size = 2 * (sim%m%ngridmax + sim%m%ncachemax)

  allocate(sim%m%key_off(1:sim%r%nlevelmax+1))
  sim%m%key_off(1) = 1_8
  do ilevel = 2, sim%r%nlevelmax + 1
     sim%m%key_off(ilevel) = sim%m%key_off(ilevel-1) + sim%m%hkey_max(1, ilevel-1)
  end do

  call mtl_alloc_amr( &
       int(sim%m%ngridmax,   c_int), &
       int(sim%m%ncachemax,  c_int), &
       int(nvar,             c_int), &
       int(twotondim,        c_int), &
       int(sim%m%hash_size,  c_int))

  call mtl_alloc_refine( &
       int(sim%m%ngridmax,  c_int), &
       int(sim%m%ncachemax, c_int), &
       int(sim%r%nlevelmax, c_int))

  sim%m%ifree_cache = 1

  periodic_i = merge(int(1, c_int), int(0, c_int), sim%r%periodic)
  call mtl_upload_level_params( &
       c_loc(sim%m%ckey_max(1)),          &
       c_loc(sim%m%key_off(1)),           &
       c_loc(sim%m%box_ckey_min(1,1)),    &
       c_loc(sim%m%box_ckey_max(1,1)),    &
       periodic_i,                         &
       int(sim%r%nlevelmax, c_int))

end subroutine metal_allocate_amr

recursive subroutine r_set_grid_device(pst)
  use mdl_module
  use mdl_parameters, only: MDL_SET_GRID_DEVICE
  use amr_parameters, only: twotondim
  use hydro_parameters, only: nvar
  use ramses_commons, only: pst_t
  implicit none
  type(pst_t) :: pst

  integer :: rID

  if(pst%nLower > 0) then
     rID = mdl_send_request(pst%s%mdl, MDL_SET_GRID_DEVICE, pst%iUpper+1)
     call r_set_grid_device(pst%pLower)
     call mdl_get_reply(pst%s%mdl, rID, 0)
  else
     call mtl_set_grid_device( &
          c_loc(pst%s%m%uold(1,1,1)), &
          c_loc(pst%s%m%unew(1,1,1)), &
          c_loc(pst%s%m%grid(1)),     &
          int(pst%s%m%ngridmax, c_int), &
          int(nvar,             c_int), &
          int(twotondim,        c_int))

     if (pst%s%r%nlevelmax > pst%s%r%levelmin) then
        call mtl_upload_flag1( &
             c_loc(pst%s%m%flag1(1,1)), &
             int(pst%s%m%ngridmax, c_int))
     end if

     call metal_set_nbor(pst%s, pst%s%r%levelmin)
     pst%s%m%data_on_device = .true.
  endif

end subroutine r_set_grid_device

recursive subroutine r_transfer_grid_host(pst)
  use mdl_module
  use mdl_parameters, only: MDL_TRANSFER_GRID_HOST
  use amr_parameters, only: twotondim
  use hydro_parameters, only: nvar
  use ramses_commons, only: pst_t
  implicit none
  type(pst_t) :: pst

  integer :: rID

  if(pst%nLower > 0) then
     rID = mdl_send_request(pst%s%mdl, MDL_TRANSFER_GRID_HOST, pst%iUpper+1)
     call r_transfer_grid_host(pst%pLower)
     call mdl_get_reply(pst%s%mdl, rID, 0)
  else
     call mtl_transfer_grid_host( &
          c_loc(pst%s%m%uold(1,1,1)), &
          int(pst%s%m%ngridmax, c_int), &
          int(nvar,             c_int), &
          int(twotondim,        c_int))
     if (pst%s%r%nlevelmax > pst%s%r%levelmin) then
        call mtl_transfer_grid_struct_host( &
             c_loc(pst%s%m%grid(1)), &
             int(pst%s%m%ngridmax, c_int))
     end if
  endif

end subroutine r_transfer_grid_host

subroutine metal_set_unew(sim, ilevel)
  use ramses_commons, only: ramses_t
  implicit none
  type(ramses_t), intent(inout) :: sim
  integer, intent(in) :: ilevel

  call mtl_set_unew( &
       int(sim%m%head(ilevel), c_int), &
       int(sim%m%noct(ilevel), c_int))

end subroutine metal_set_unew

subroutine metal_set_uold(sim, ilevel)
  use ramses_commons, only: ramses_t
  implicit none
  type(ramses_t), intent(inout) :: sim
  integer, intent(in) :: ilevel

  call mtl_set_uold( &
       int(sim%m%head(ilevel), c_int), &
       int(sim%m%noct(ilevel), c_int))

end subroutine metal_set_uold

subroutine metal_cmpdt(sim, ilevel, mass, ekin, eint, emag, dt)
  use ramses_commons, only: ramses_t
  implicit none
  type(ramses_t), intent(inout) :: sim
  integer,        intent(in)    :: ilevel
  real(kind=8),   intent(out)   :: mass, ekin, eint, emag, dt

  real(c_float) :: dx, gamma, smallr, smallc2, courant_factor
  real(c_float) :: constant_gravity(3)
  real(c_float) :: mass_f, ekin_f, eint_f, emag_f, dt_f

  dx             = real(sim%r%boxlen / 2**ilevel,    c_float)
  gamma          = real(sim%r%gamma,                 c_float)
  smallr         = real(sim%r%smallr,                c_float)
  smallc2        = real(sim%r%smallc**2,             c_float)
  courant_factor = real(sim%r%courant_factor,        c_float)
  constant_gravity = real(sim%r%constant_gravity,    c_float)

  call mtl_cmpdt(                      &
       int(sim%m%head(ilevel), c_int), &
       int(sim%m%noct(ilevel), c_int), &
       dx, gamma, smallr, smallc2,     &
       courant_factor,                 &
       constant_gravity,               &
       mass_f, ekin_f, eint_f, emag_f, dt_f)

  mass = real(mass_f, 8)
  ekin = real(ekin_f, 8)
  eint = real(eint_f, 8)
  emag = real(emag_f, 8)
  dt   = real(dt_f,   8)

end subroutine metal_cmpdt

subroutine metal_godunov(sim, ilevel)
  use ramses_commons, only: ramses_t
  implicit none
  type(ramses_t), intent(inout) :: sim
  integer,        intent(in)    :: ilevel

  real(c_float) :: gamma, smallr, smallc2, dt, dx
  real(c_float) :: constant_gravity(3)

  gamma            = real(sim%r%gamma,                   c_float)
  smallr           = real(sim%r%smallr,                  c_float)
  smallc2          = real(sim%r%smallc**2,               c_float)
  dt               = real(sim%g%dtnew(ilevel),           c_float)
  dx               = real(sim%r%boxlen / 2**ilevel,      c_float)
  constant_gravity = real(sim%r%constant_gravity,        c_float)

  if(sim%r%verbose .and. sim%g%myid==1) &
       write(*,'("   Entering metal_godunov for level ",I2)') ilevel

  call mtl_godunov(                    &
       int(sim%m%head(ilevel), c_int), &
       int(sim%m%noct(ilevel), c_int), &
       int(sim%m%ngridmax,     c_int), &
       int(ilevel,             c_int), &
       int(sim%r%levelmin,     c_int), &
       int(sim%r%nlevelmax,    c_int), &
       gamma, smallr, smallc2,         &
       dt, dx,                         &
       int(sim%r%slope_type,   c_int), &
       int(sim%r%riemann,      c_int), &
       constant_gravity)

end subroutine metal_godunov

subroutine metal_set_nbor(sim, ilevel)
  use amr_parameters, only: ndim
  use ramses_commons,  only: ramses_t
  use iso_c_binding
  implicit none
  type(ramses_t), intent(inout) :: sim
  integer,        intent(in)    :: ilevel

  integer(c_int)  :: hash_size_l, ckey_max_l
  integer(c_long) :: key_off_l
  integer(c_int)  :: bmin(3), bmax(3), periodic_i(3)

  hash_size_l = int(sim%m%hash_size,                   c_int)
  ckey_max_l  = int(sim%m%ckey_max(ilevel),            c_int)
  key_off_l   = int(sim%m%key_off(ilevel),             c_long)
  bmin        = int(sim%m%box_ckey_min(1:ndim,ilevel), c_int)
  bmax        = int(sim%m%box_ckey_max(1:ndim,ilevel), c_int)
  periodic_i  = merge(int(1, c_int), int(0, c_int), sim%r%periodic)

  call mtl_insert_hash_all( &
       int(1,             c_int), &
       int(sim%m%ifree-1, c_int), &
       hash_size_l)

  call mtl_build_nbor( &
       int(sim%m%head(ilevel), c_int), &
       int(sim%m%noct(ilevel), c_int), &
       hash_size_l, ckey_max_l, key_off_l, &
       bmin, bmax, periodic_i)

end subroutine metal_set_nbor


subroutine metal_init_flag(sim, ilevel, nflag)
  use ramses_commons, only: ramses_t
  use iso_c_binding
  implicit none
  type(ramses_t), intent(inout) :: sim
  integer,        intent(in)    :: ilevel
  integer,        intent(out)   :: nflag

  nflag = int(mtl_init_flag_batch( &
       int(sim%m%head(ilevel), c_int), &
       int(sim%m%noct(ilevel), c_int), &
       int(sim%m%head(ilevel+1), c_int), &
       int(merge(sim%m%noct(ilevel+1), 0, &
                 ilevel < sim%r%nlevelmax .and. sim%m%noct(ilevel+1) > 0), c_int)))

end subroutine metal_init_flag


subroutine metal_user_flag(sim, ilevel, nflag)
  use ramses_commons, only: ramses_t
  use iso_c_binding
  implicit none
  type(ramses_t), intent(inout) :: sim
  integer,        intent(in)    :: ilevel
  integer,        intent(out)   :: nflag

  real(c_float) :: gamma, smallr, smallc2
  real(c_float) :: err_grad_d, err_grad_p, floor_d, floor_p

  gamma      = real(sim%r%gamma,      c_float)
  smallr     = real(sim%r%smallr,     c_float)
  smallc2    = real(sim%r%smallc**2,  c_float)
  err_grad_d = real(sim%r%err_grad_d, c_float)
  err_grad_p = real(sim%r%err_grad_p, c_float)
  floor_d    = real(sim%r%floor_d,    c_float)
  floor_p    = real(sim%r%floor_p,    c_float)

  nflag = int(mtl_user_flag_batch( &
       int(sim%m%head(ilevel), c_int), &
       int(sim%m%noct(ilevel), c_int), &
       gamma, smallr, smallc2,          &
       err_grad_d, err_grad_p, floor_d, floor_p))

end subroutine metal_user_flag


subroutine metal_enforce_rules(sim, ilevel)
  use ramses_commons, only: ramses_t
  use iso_c_binding
  implicit none
  type(ramses_t), intent(inout) :: sim
  integer,        intent(in)    :: ilevel

  if (sim%m%noct(ilevel) > 0) then
     call mtl_enforce_rules(int(sim%m%head(ilevel), c_int), &
                            int(sim%m%noct(ilevel), c_int), &
                            int(sim%m%ngridmax,     c_int))
  end if

end subroutine metal_enforce_rules


subroutine metal_smooth_flag(sim, ilevel, nflag)
  use amr_parameters, only: ndim
  use ramses_commons,  only: ramses_t
  use iso_c_binding
  implicit none
  type(ramses_t), intent(inout) :: sim
  integer,        intent(in)    :: ilevel
  integer,        intent(out)   :: nflag

  nflag = int(mtl_smooth_flag_batch( &
       int(sim%m%head(ilevel), c_int), &
       int(sim%m%noct(ilevel), c_int)))

end subroutine metal_smooth_flag

subroutine metal_refine(sim, ilevel, nmake, nkill)
  use amr_parameters, only: ndim
  use ramses_commons, only: ramses_t
  use iso_c_binding
  implicit none
  type(ramses_t), intent(inout) :: sim
  integer,        intent(in)    :: ilevel
  integer,        intent(out)   :: nmake, nkill

  integer(c_int) :: hash_size_l, old_ifree, new_ifree
  integer(c_int) :: head_child, n_all, n_rem
  integer(c_int) :: ilev, ind
  integer(c_int) :: cur_head, nones, new_noct
  integer(c_int) :: total_valid, ifree_cache_now
  integer(c_int) :: cache_noct_lev

  hash_size_l = int(sim%m%hash_size, c_int)

  if (sim%m%ifree_cache > 1) then
     call mtl_free_hash_range( &
          int(sim%m%ngridmax + 1, c_int), &
          int(sim%m%ifree_cache - 1, c_int), &
          hash_size_l)
  end if

  sim%m%ifree_cache = 1

  old_ifree = int(sim%m%ifree, c_int)
  call mtl_set_ifree(old_ifree)

  call mtl_refine_cells( &
       int(sim%m%head(ilevel),        c_int), &
       old_ifree - int(sim%m%head(ilevel), c_int))

  new_ifree = mtl_get_ifree()
  nmake     = int(new_ifree - old_ifree)

  call mtl_insert_hash_all( &
       old_ifree, &
       int(nmake, c_int), &
       hash_size_l)

  do ilev = int(sim%r%nlevelmax, c_int), int(ilevel + 1, c_int), -1
     if (sim%m%noct(ilev) > 0) then
        call mtl_derefine_cells( &
             int(sim%m%head(ilev), c_int), &
             int(sim%m%noct(ilev), c_int), &
             hash_size_l)
     end if
  end do

  head_child = int(sim%m%head(ilevel + 1), c_int)
  n_all      = new_ifree - head_child   ! total slots in child region

  if (n_all > 0) then
     call mtl_init_swap_table(head_child, n_all)

     cur_head    = head_child
     total_valid = 0
     do ilev = int(ilevel + 1, c_int), int(sim%r%nlevelmax, c_int)
        n_rem = new_ifree - cur_head
        if (n_rem <= 0) exit

        call mtl_init_prefix_level(cur_head, n_rem, ilev)
        call mtl_prefix_scan(int(cur_head - 1, c_int), n_rem)
        nones   = mtl_get_prefix_total(int(cur_head - 1, c_int), n_rem)
        new_noct = n_rem - nones   ! octs at level ilev

        call mtl_compute_local_swap(cur_head, n_rem)
        call mtl_update_global_swap(cur_head, n_rem)

        sim%m%head(ilev) = int(cur_head)
        sim%m%noct(ilev) = int(new_noct)
        sim%m%tail(ilev) = int(cur_head + new_noct - 1)
        cur_head    = cur_head + new_noct
        total_valid = total_valid + int(new_noct)
     end do

     nkill          = int(n_all - total_valid)
     sim%m%noct_used = sim%m%tail(sim%r%nlevelmax)
     sim%m%ifree    = int(cur_head)   ! first free slot after valid octs

     do ilev = int(ilevel + 1, c_int), int(sim%r%nlevelmax, c_int)
        if (sim%m%noct(ilev) <= 0) cycle
        call mtl_hilbert_sort_level( &
             int(sim%m%head(ilev), c_int), &
             int(sim%m%noct(ilev), c_int), &
             int(ndim * ilev,      c_int))
     end do

     call mtl_sort_gather_grid(head_child, n_all)
     call mtl_sort_scatter_grid(head_child, n_all)
     call mtl_sort_gather_flag(head_child, n_all)
     call mtl_sort_scatter_flag(head_child, n_all)
     call mtl_sort_gather_hydro(head_child, n_all)
     call mtl_blit_unew_to_uold(head_child, n_all)

     call mtl_update_hash_range(head_child, n_all, hash_size_l)

     do ilev = int(ilevel + 1, c_int), int(sim%r%nlevelmax, c_int)
        if (sim%m%noct(ilev) <= 0) cycle
        call mtl_build_father( &
             int(sim%m%head(ilev),    c_int), &
             int(sim%m%noct(ilev),    c_int), &
             hash_size_l, &
             int(sim%m%ckey_max(ilev - 1), c_int), &
             int(sim%m%key_off(ilev - 1),  c_long))
     end do

  else
     nkill = 0
  end if

  ifree_cache_now = int(sim%m%ifree_cache, c_int)   ! starts at 1

  do ilev = int(ilevel + 1, c_int), int(sim%r%nlevelmax, c_int)
     if (sim%m%noct(ilev) <= 0) cycle

     do ind = 1_c_int, 27_c_int
        cache_noct_lev = mtl_nbor_scan( &
             int(sim%m%head(ilev), c_int), &
             int(sim%m%noct(ilev), c_int), &
             hash_size_l, ind)
        if (cache_noct_lev <= 0) cycle

        call mtl_cache_fill( &
             int(sim%m%head(ilev), c_int), &
             int(sim%m%noct(ilev), c_int), &
             hash_size_l, ind, &
             int(sim%m%ngridmax, c_int), &
             ifree_cache_now, cache_noct_lev)

        call mtl_advance_ifree_cache(cache_noct_lev)
        ifree_cache_now = ifree_cache_now + cache_noct_lev
     end do
  end do

  sim%m%ifree_cache = int(ifree_cache_now)

  if (ilevel == sim%r%levelmin) then
     call mtl_reset_hash( &
          int(sim%m%ifree,       c_int), &
          int(sim%r%ngridmax,    c_int), &
          int(sim%m%ifree_cache, c_int), &
          hash_size_l)
  end if

end subroutine metal_refine

subroutine metal_upload(sim, ilevel)
  use ramses_commons, only: ramses_t
  use iso_c_binding
  implicit none
  type(ramses_t), intent(inout) :: sim
  integer,        intent(in)    :: ilevel

  integer(c_int) :: internal_energy

  if (ilevel >= sim%r%nlevelmax) return
  if (sim%m%noct(ilevel+1) <= 0) return

  internal_energy = int(merge(1, 0, sim%r%interpol_var == 1), c_int)

  call mtl_upload( &
       int(sim%m%head(ilevel+1), c_int), &
       int(sim%m%noct(ilevel+1), c_int), &
       internal_energy,                   &
       real(sim%r%gamma,        c_float), &
       real(sim%r%smallr,       c_float), &
       real(sim%r%smallc**2,    c_float))

end subroutine metal_upload

end module metal_runner
