!###########################################################
!###########################################################
!###########################################################
subroutine get_cr_courant_dt(r,g,dt,ilevel)
!-------------------------------------------------------------------------
! Determine the coarse CR timestep length set by the Courant condition
!-------------------------------------------------------------------------
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
