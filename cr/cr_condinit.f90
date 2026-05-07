!================================================================
!================================================================
!================================================================
!================================================================
subroutine cr_condinit(r,g,x,q,dx,nn,ilevel)
  use amr_parameters, only: ndim, nvector
  use cr_parameters, only: ncrvar
  use amr_commons, only: run_t, global_t
  use cr_input_condinit_module, only: cr_region_condinit
  implicit none
  type(run_t)::r
  type(global_t)::g
  integer ::nn,ilevel
  real(kind=8)::dx                              ! Cell size
  real(kind=8),dimension(1:nvector,1:ncrvar)::q ! CR variables
  real(kind=8),dimension(1:nvector,1:ndim)::x   ! Cell center position.
  !================================================================
  ! This routine generates initial conditions for RAMSES.
  ! Positions are in user (aka code) units:
  ! x(i,1:ndim) are in [0,box_size]**ndim.
  ! Q is the CR variable vector in code units.
  !================================================================
  ! Call built-in initial condition generator
  call cr_region_condinit(r,g,x,q,dx,nn,ilevel)

  ! Add here, if you wish, some user-defined initial conditions
  ! ........

end subroutine cr_condinit
