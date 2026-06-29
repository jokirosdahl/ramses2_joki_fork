!############################################################
!############################################################
!############################################################
!############################################################
subroutine cr_boundana(r,g,x,cru,dx,ibound,ncell)
  use amr_parameters, only: ndim, nvector
  use amr_commons, only: run_t, global_t
  use cr_parameters, only: ncrvar
  implicit none
  type(run_t)::r
  type(global_t)::g
  integer ::ibound                                ! Index of boundary region
  integer ::ncell                                 ! Number of active cells
  real(kind=8)::dx                                ! Cell size
  real(kind=8),dimension(1:nvector,1:ncrvar)::cru ! CR vars
  real(kind=8),dimension(1:nvector,1:ndim)::x     ! Cell center position.
  !================================================================
  ! This routine generates boundary conditions for RAMSES.
  ! Positions are in user (aka code) units:
  ! x(i,1:ndim) are in [0,box_size]**ndim.
  ! CRU contains the CR variables in code units
  ! ibound is the index of the boundary region defined in the namelist.
  !================================================================
  integer::ivar,i
  if(r%cr_test_setup=='streaming_triangle') then
     do i=1,ncell
        cru(i,1)=2d0+r%cr_gamma(1)*g%t-abs(x(i,1)-r%box_size(1)*0.5d0)
        if(x(i,1)<r%box_size(1)*0.5d0)then
           cru(i,2)=-r%cr_gamma(1)*cru(i,1)
        else
           cru(i,2)= r%cr_gamma(1)*cru(i,1)
        endif
        if(ncrvar.gt.2) cru(i,3:)=0d0
     end do
  endif

  ! Add here, if you wish, some user-defined boudary conditions
  ! ........

end subroutine cr_boundana
