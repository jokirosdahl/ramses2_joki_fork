!================================================================
!================================================================
!================================================================
!================================================================
subroutine vecpotentialinit(r,g,x,A,idim,nn)
  use amr_parameters, only: dp, ndim, nvector
  use amr_commons, only: run_t, global_t
  use merger_parameters
  use const
  implicit none
  type(run_t)::r
  type(global_t)::g
  integer::nn                             ! Number of cells
  integer::idim                           ! Direction of the component
  real(dp),dimension(1:nvector)::A        ! Vector potential component
  real(dp),dimension(1:nvector,1:ndim)::x ! Cell center position.
  !================================================================
  ! This routine generates initial conditions for RAMSES.
  ! Positions are in user (aka code) units:
  ! x(i,1:ndim) are in [0,boxlen]**ndim.
  ! A is the component of the vector potential corresponding
  ! to direction idim.
  ! A(:) is in user (aka code) units.
  !================================================================
  integer::i
  real(dp)::xx,yy,zz,rc,xc,yc,zc, rr, dnfw, eps, rs

#ifdef MHD
  ! Galaxy parameters from namelist
  xc = gal_center1(1)+r%boxlen/2
  yc = gal_center1(2)+r%boxlen/2
  zc = gal_center1(3)+r%boxlen/2

  eps = 0.01 ! small like 10 pc
  rs = 50.
  eps = eps/rs

  do i = 1,nn ! we are working in [kpc], see units

     xx = x(i,1)-xc
     yy = x(i,2)-yc
     zz = x(i,3)-zc
     rc = sqrt(xx**2+yy**2) ! kpc

     rr = sqrt(xx**2+yy**2+zz**2)/rs ! units of rs
     
     rr = max(rr,eps)
     rc = max(rc,eps*rs)

     dnfw = 1./rr/(1.+rr)**2
     !print*,xc,yc,zc, xx,yy,zz, rc, typ_radius1

     select case (mag_topology)
        
     case ('constant') ! constant done elsewhere
        if(idim==1) A(i)=0
        if(idim==2) A(i)=0
        if(idim==3) A(i)=0

     case ('toroidal')
        if(idim==1) A(i)=0
        if(idim==2) A(i)=0
        !if(idim==3) A(i)=B_ave * (exp(-abs(zz)/typ_height1) * exp(-rc/typ_radius1))**two3rd
        !if(idim==3) A(i)= -B_ave * rc/typ_radius1
        !if(idim==3) A(i)= B_ave * (exp(-rc/typ_radius1))**two3rd

        if(idim==3)A(i)=B_ave*rc*dnfw**two3rd

     case ('none')

     end select

  enddo
#endif  

end subroutine vecpotentialinit

