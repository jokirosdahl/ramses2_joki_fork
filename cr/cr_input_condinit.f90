module cr_input_condinit_module
contains
!###############################################
!###############################################
!###############################################
!###############################################
recursive subroutine r_cr_input_condinit(pst,ilevel,input_size)
  use mdl_module
  use ramses_commons, only: pst_t
  use mdl_parameters
  implicit none
  type(pst_t)::pst
  integer,VALUE::input_size

  integer::ilevel
  integer::rID
  if(pst%nLower>0)then
     rID = mdl_send_request(pst%s%mdl,MDL_CR_INPUT_CONDINIT,pst%iUpper+1,input_size,0,ilevel)
     call r_cr_input_condinit(pst%pLower,ilevel,input_size)
     call mdl_get_reply(pst%s%mdl,rID,0)
  else
     call cr_input_condinit(pst%s%r,pst%s%g,pst%s%m,ilevel)
  endif
  
end subroutine r_cr_input_condinit
!#########################################################################
!#########################################################################
!#########################################################################
!#########################################################################
subroutine cr_input_condinit(r,g,m,ilevel)
  use amr_parameters, only: ndim, twotondim, nvector
  use cr_parameters,only: ncruvar
  use amr_commons, only: run_t, global_t, mesh_t
  implicit none
  type(run_t)::r
  type(global_t)::g
  type(mesh_t)::m
  integer::ilevel
  
  ! Local variables
  integer::igrid,ngrid,ind,idim,nstride,i,ivar
  !integer::l
  real(kind=8),dimension(1:nvector,1:ndim)::xx
  real(kind=8),dimension(1:nvector,1:ncruvar)::qq
  real(kind=8)::dx

  if(m%noct(ilevel)==0)return
  !-----------------------------------------------------------
  ! Set CR ICs from user-defined conditions
  !-----------------------------------------------------------
  ! Mesh size at level ilevel in code units
  dx=r%boxlen/2**ilevel
  ! Loop over grids by vector sweeps
  do igrid=m%head(ilevel),m%tail(ilevel),nvector
     ngrid=MIN(nvector,m%tail(ilevel)-igrid+1)
     ! Loop over cells
     do ind=1,twotondim
        ! Compute cell centre position in code units
        do idim=1,ndim
           nstride=2**(idim-1)
           do i=1,ngrid
              xx(i,idim)=(2*m%grid(igrid+i-1)%ckey(idim)+MOD((ind-1)/nstride,2)+0.5)*dx-m%skip(idim)
           end do
        end do
        ! Call initial condition routine
        call cr_condinit(r,g,xx,qq,dx,ngrid)
        ! Scatter primitive variables to main memory
#ifdef CRS
        do ivar=1,ncruvar
           do i=1,ngrid
              m%cruold(ind, ivar, igrid+i-1) = qq(i, ivar)
           end do
        end do
#endif
      end do
     ! End loop over cells
  end do
  ! End loop over grids

end subroutine cr_input_condinit
!################################################################
!################################################################
!################################################################
!################################################################
subroutine cr_region_condinit(r,g,x,q,dx,nn)
  use amr_parameters, only: nvector, ndim
  use cr_parameters, only: ncruvar, ncrgrp
  use amr_commons, only: run_t, global_t
  implicit none
  type(run_t)::r
  type(global_t)::g
  integer ::nn,idim
  real(kind=8)::dx
  real(kind=8),dimension(1:nvector,1:ncruvar)::q
  real(kind=8),dimension(1:nvector,1:ndim)::x
  real(kind=8)::scale_nH,scale_T2,scale_l,scale_d,scale_t,scale_v
  real(kind=8)::dx_cgs,scale_e
  !----------------------------------------------------
  ! This routine sets simple pre-defined initial
  ! conditions, like points, squares, etc.
  !----------------------------------------------------
  integer::i,k,ivar,igroup
  real(kind=8)::vol,rad,weight,xn,yn,zn,en
  ! Units are needed for point regions
  call units(r,g,scale_l,scale_t,scale_d,scale_v,scale_nH,scale_T2)
  dx_cgs=dx*scale_l
  scale_e = scale_d*scale_v**2

  ! Set (tiny) default values outside of regions or if n_region=0
  do ivar = 1, ncruvar
     q(1:nn,ivar) = 0.0  ! CR flux variables
  end do

  ! Loop over initial conditions regions
  do k=1,r%cr_nregion
     if(r%cr_reg_group(k) .le. 0 .or. r%cr_reg_group(k) .gt. ncrgrp) cycle
     igroup = r%cr_reg_group(k)
     
     ! For "square" regions only:
     if(r%cr_region_type(k) .eq. 'square')then
        ! Exponent of choosen norm
        en=r%cr_exp_region(k)
        do i=1,nn
           ! Compute position in normalized coordinates
           xn=0.0d0; yn=0.0d0; zn=0.0d0
           xn=2.0d0*abs(x(i,1)-r%cr_reg_x_center(k))/r%cr_reg_length_x(k)
#if NDIM>1
           yn=2.0d0*abs(x(i,2)-r%cr_reg_y_center(k))/r%cr_reg_length_y(k)
#endif
#if NDIM>2
           zn=2.0d0*abs(x(i,3)-r%cr_reg_z_center(k))/r%cr_reg_length_z(k)
#endif
           ! Compute cell "radius" relative to region center
           if(r%cr_exp_region(k)<10)then
              rad=(xn**en+yn**en+zn**en)**(1.0/en)
           else
              rad=max(xn,yn,zn)
           end if
           ! If cell lies within region,
           if(rad<1.0)then
              q(i,1+(igroup-1)*ndim)=r%cr_fx_region(k)
#if NDIM>1 
              q(i,2+(igroup-1)*ndim)=r%cr_fy_region(k)
#endif
#if NDIM>2
              q(i,3+(igroup-1)*ndim)=r%cr_fz_region(k)
#endif
           end if
        end do
     end if
     
  end do

  return
end subroutine cr_region_condinit

end module cr_input_condinit_module
