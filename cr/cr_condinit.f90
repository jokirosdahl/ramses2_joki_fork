!================================================================
!================================================================
!================================================================
!================================================================
subroutine cr_condinit(r,g,x,q,dx,nn)
  use amr_parameters, only: ndim, nvector
  use cr_parameters, only: ncruvar
  use amr_commons, only: run_t, global_t
  use cr_input_condinit_module, only: cr_region_condinit
  implicit none
  type(run_t)::r
  type(global_t)::g
  integer ::nn, i
  real(kind=8)::dx                              ! Cell size
  real(kind=8),dimension(1:nvector,1:ncruvar)::q ! CR variables
  real(kind=8),dimension(1:nvector,1:ndim)::x   ! Cell center position.
  real(kind=8)::xx,yy,zz,rr,theta,pi,xcenter,ttmin,ttmax
  !================================================================
  ! This routine generates initial conditions for RAMSES.
  ! Positions are in user (aka code) units:
  ! x(i,1:ndim) are in [0,box_size]**ndim.
  ! Q is the CR variable vector in code units.
  !================================================================
  ! Call built-in initial condition generator
  call cr_region_condinit(r,g,x,q,dx,nn)

  ! Add here, if you wish, some user-defined initial conditions
  ! ........
  if(r%cr_test_setup=='none') return
  if(r%cr_test_setup=='circular_diffusion') then
     ! CR energy: enhanced on one arc of the loop (around the z-axis).
     ! atan2 replaces atan(yy/xx) to avoid a divide-by-zero FPE at xx=0;
     ! it is identical in the xx>0 region that the arc condition selects.
     pi=acos(-1d0)
     ttmin=-pi/12d0
     ttmax= pi/12d0
     xcenter=r%box_size(1)*0.5d0
     do i=1,nn
        xx=x(i,1)-xcenter
        yy=x(i,2)-xcenter
        rr=sqrt(xx**2+yy**2)
        theta=atan2(yy,xx)
        if(rr>0.25d0*r%box_size(1) .and. rr<0.35d0*r%box_size(1) .and. theta>ttmin .and. &
            & theta<ttmax .and. xx>0d0 .and. zz>-0.1d0*r%box_size(1) .and. zz<0.1d0*r%box_size(1))then
          q(i,1)=1.2d1
        else
          q(i,1)=1.0d1
        endif
     end do
  endif

end subroutine cr_condinit
