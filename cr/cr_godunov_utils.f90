!###########################################################
!###########################################################
!###########################################################
subroutine get_cr_courant_dt(r,g,dt,ilevel)
!-------------------------------------------------------------------------
! Determine the coarse CR timestep length set by the Courant condition
!-------------------------------------------------------------------------
  use amr_parameters
  use cr_parameters
  use amr_commons, only: run_t, global_t
  implicit none
  type(run_t)::r
  type(global_t)::g
  integer::ilevel
  real(kind=8)::dt,dx
!-------------------------------------------------------------------------
  ! Mesh spacing at coarse level
  dx=r%boxlen/2**ilevel
  dt=r%courant_factor*dx/3d0/g%cr_c(ilevel)
end subroutine get_cr_courant_dt

!###########################################################
!###########################################################
!###########################################################
!###########################################################
subroutine cr_refine(r,ug,um,ud,ok)
  use amr_parameters, only: ndim
  use amr_commons, only: run_t
  use cr_parameters, only: ncrvar, ncrgrp
  implicit none
  ! routine arguments
  type(run_t)::r
  real(kind=8)::ug(1:ncrvar)
  real(kind=8)::um(1:ncrvar)
  real(kind=8)::ud(1:ncrvar)
  logical ::ok
  ! local variables
  integer::igroup, ivar
  real(kind=8)::ng,nm,nd,error

  ! Compute errors
  do igroup=1, ncrgrp
    if(r%cr_err_grad_e(igroup) >= 0.)then
      ivar = 1+(igroup-1)*(ndim+1)
      ng=ug(ivar); nm=um(ivar); nd=ud(ivar)
      error=2.0d0*MAX( &
          & ABS((nd-nm)/(nd+nm+r%cr_floor_e(igroup))) , &
          & ABS((nm-ng)/(nm+ng+r%cr_floor_e(igroup))) )
      ok = ok .or. error > r%cr_err_grad_e(igroup)
    end if
  end do

end subroutine cr_refine
