module metal_runner
  use metal_interface
  use iso_c_binding
  implicit none

contains

!###########################################################
!###########################################################
!###########################################################
!###########################################################
subroutine metal_allocate_amr(sim)
  use amr_parameters, only: twotondim
  use hydro_parameters, only: nvar
  use ramses_commons, only: ramses_t
  implicit none
  type(ramses_t) :: sim
  integer :: ilevel

  ! Set hash_size the same way gpu_allocate_amr does (gpu_manager.cuf:274).
  sim%m%hash_size = 2 * (sim%m%ngridmax + sim%m%ncachemax)

  ! Allocate key_off (mirrors gpu_allocate_amr, gpu_manager.cuf:333-336).
  ! init_amr populates hkey_max but never allocates key_off; the CUDA path
  ! does it in gpu_allocate_amr.  metal_set_nbor reads key_off(ilevel).
  allocate(sim%m%key_off(1:sim%r%nlevelmax+1))
  sim%m%key_off(1) = 1_8
  do ilevel = 2, sim%r%nlevelmax + 1
     sim%m%key_off(ilevel) = sim%m%key_off(ilevel-1) + sim%m%hkey_max(1, ilevel-1)
  end do

  ! Allocate Metal-owned buffers; data is copied in r_set_grid_device,
  ! and the nbor/hash buffers are populated by metal_set_nbor.
  call mtl_alloc_amr( &
       int(sim%m%ngridmax,   c_int), &
       int(nvar,             c_int), &
       int(twotondim,        c_int), &
       int(sim%m%hash_size,  c_int))

end subroutine metal_allocate_amr

!###########################################################
!###########################################################
!###########################################################
!###########################################################
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
     ! Copy host arrays into Metal buffers (mirrors H->D cudaMemcpy in gpu_manager.cuf).
     call mtl_set_grid_device( &
          c_loc(pst%s%m%uold(1,1,1)), &
          c_loc(pst%s%m%unew(1,1,1)), &
          c_loc(pst%s%m%grid(1)),     &
          int(pst%s%m%ngridmax, c_int), &
          int(nvar,             c_int), &
          int(twotondim,        c_int))

     ! Build nbor on device via insert_hash + build_nbor kernels
     ! (mirrors insert_hash_kernel + update_nbor_array in gpu_manager.cuf).
     call metal_set_nbor(pst%s, pst%s%r%levelmin)
     pst%s%m%data_on_device = .true.
  endif

end subroutine r_set_grid_device

!###########################################################
!###########################################################
!###########################################################
!###########################################################
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
     ! Copy Metal buffer back into host uold (mirrors D->H cudaMemcpy in gpu_manager.cuf).
     call mtl_transfer_grid_host( &
          c_loc(pst%s%m%uold(1,1,1)), &
          int(pst%s%m%ngridmax, c_int), &
          int(nvar,             c_int), &
          int(twotondim,        c_int))
  endif

end subroutine r_transfer_grid_host

!###########################################################
!###########################################################
!###########################################################
!###########################################################
subroutine metal_set_unew(sim, ilevel)
  use ramses_commons, only: ramses_t
  implicit none
  type(ramses_t), intent(inout) :: sim
  integer, intent(in) :: ilevel

  call mtl_set_unew( &
       int(sim%m%head(ilevel), c_int), &
       int(sim%m%noct(ilevel), c_int))

end subroutine metal_set_unew

!###########################################################
!###########################################################
!###########################################################
!###########################################################
subroutine metal_set_uold(sim, ilevel)
  use ramses_commons, only: ramses_t
  implicit none
  type(ramses_t), intent(inout) :: sim
  integer, intent(in) :: ilevel

  call mtl_set_uold( &
       int(sim%m%head(ilevel), c_int), &
       int(sim%m%noct(ilevel), c_int))

end subroutine metal_set_uold

!###########################################################
!###########################################################
!###########################################################
!###########################################################
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

!###########################################################
!###########################################################
!###########################################################
!###########################################################
subroutine metal_godunov(sim, ilevel)
  use ramses_commons, only: ramses_t
  implicit none
  type(ramses_t), intent(inout) :: sim
  integer,        intent(in)    :: ilevel

  real(c_float) :: gamma, smallr, smallc2, dt, dx
  real(c_float) :: constant_gravity(3)

  ! nsubgrid=1, so nsubgridtondim=1: head_idx and num_subgrids equal
  ! sim%m%head(ilevel) and sim%m%noct(ilevel) directly.
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

!###########################################################
!###########################################################
!###########################################################
!###########################################################
subroutine metal_set_nbor(sim, ilevel)
  ! Insert all oct Hilbert keys into the device hash table, then build
  ! the device nbor array by looking up the 27 neighbours per subgrid.
  ! Mirrors the two-step block in r_set_grid_device (gpu_manager.cuf):
  !   call insert_hash_kernel<<<...>>>(...)
  !   do ind = 1, subgridsize
  !     call update_nbor_array<<<...>>>(..., ind)
  !   end do
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
  ! Convert logical periodic(1:3) to 0/1 integers for C/Metal
  periodic_i  = merge(int(1, c_int), int(0, c_int), sim%r%periodic)

  ! Step 1: insert all allocated octs into the hash table (1 .. ifree-1)
  call mtl_insert_hash( &
       int(1,              c_int), &
       int(sim%m%ifree-1,  c_int), &
       hash_size_l, ckey_max_l, key_off_l)

  ! Step 2: build nbor for octs at ilevel (the subgrid level for the PoC)
  call mtl_build_nbor( &
       int(sim%m%head(ilevel), c_int), &
       int(sim%m%noct(ilevel), c_int), &
       hash_size_l, ckey_max_l, key_off_l, &
       bmin, bmax, periodic_i)

end subroutine metal_set_nbor

end module metal_runner
