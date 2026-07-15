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
  use cr_parameters,only: ncrvar, ncrgrp
  use amr_commons, only: run_t, global_t, mesh_t
  implicit none
  type(run_t)::r
  type(global_t)::g
  type(mesh_t)::m
  integer::ilevel
  
  ! Local variables
  integer::igrid,ngrid,ind,idim,nstride,i,ivar,igrp
  !integer::l
  real(kind=8),dimension(1:nvector,1:ndim)::xx
  real(kind=8),dimension(1:nvector,1:ncrvar)::qq
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
        do igrp=1,ncrgrp
           do i=1,ngrid
              m%uold(ind,r%iEcr+igrp-1,igrid+i-1) = qq(i,1+(igrp-1)*(ndim+1))
#ifdef CRS
              m%cruold(ind, 1+(igrp-1)*ndim:igrp*ndim, igrid+i-1) = &
                & qq(i, 2+(igrp-1)*(ndim+1):igrp*(ndim+1))
#endif
           end do
        end do
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
  use cr_parameters, only: ncrvar, ncrgrp
  use amr_commons, only: run_t, global_t
  implicit none
  type(run_t)::r
  type(global_t)::g
  integer ::nn,idim,igrp
  real(kind=8)::dx
  real(kind=8),dimension(1:nvector,1:ncrvar)::q
  real(kind=8),dimension(1:nvector,1:ndim)::x
  real(kind=8)::scale_nH,scale_T2,scale_l,scale_d,scale_t,scale_v
  real(kind=8)::dx_cgs,scale_e
  !----------------------------------------------------
  ! This routine sets simple pre-defined initial
  ! conditions, like points, squares, etc.
  !----------------------------------------------------
  integer::i,k,group_ind
  real(kind=8)::vol,rad,weight,xn,yn,zn,en
  ! Units are needed for point regions
  call units(r,g,scale_l,scale_t,scale_d,scale_v,scale_nH,scale_T2)
  dx_cgs=dx*scale_l
  scale_e = scale_d*scale_v**2

  ! Set (tiny) default values outside of regions or if n_region=0
  do igrp = 1, ncrgrp
     q(1:nn,1+(igrp-1)*(ndim+1)) = 0.0  ! CR densities
     do idim = 1, ndim
        q(1:nn,1+idim+(igrp-1)*(ndim+1)) = 0.0d0 ! photon fluxes
     end do
  end do

  ! Loop over initial conditions regions
  do k=1,r%cr_nregion
     if(r%cr_reg_group(k) .le. 0 .or. r%cr_reg_group(k) .gt. ncrgrp) cycle
     if(r%cr_e_region(k).le.0.D0) r%cr_e_region(k)=0.D0
     group_ind = 1+(r%cr_reg_group(k)-1)*(ndim+1)
     
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
              q(i,group_ind)=r%cr_e_region(k)
              q(i,group_ind+1)=r%cr_fx_region(k)
#if NDIM>1 
              q(i,group_ind+2)=r%cr_fy_region(k)
#endif
#if NDIM>2
              q(i,group_ind+3)=r%cr_fz_region(k)
#endif
           end if
        end do
     end if
     
     ! For "point" regions only:
     if(r%cr_region_type(k) .eq. 'point')then
        ! Volume elements
        vol=dx_cgs**ndim
        ! Compute CIC weights relative to region center
        do i=1,nn
           xn=1.0; yn=1.0; zn=1.0
           xn=max(1.0-abs(x(i,1)-r%cr_reg_x_center(k))/dx,0.0d0)
#if NDIM>1
           yn=max(1.0-abs(x(i,2)-r%cr_reg_x_center(k))/dx,0.0d0)
#endif
#if NDIM>2
           zn=max(1.0-abs(x(i,3)-r%cr_reg_x_center(k))/dx,0.0d0)
#endif
           weight=xn*yn*zn
           if(weight.gt.0) then
              ! If cell lies within CIC cloud, 
              ! Convert energy to energy density
              q(i,group_ind) = r%cr_e_region(k)/scale_e *weight/vol 
           endif
        end do
     end if
  end do

  return
end subroutine cr_region_condinit
!###############################################
!###############################################
!###############################################
!###############################################
recursive subroutine r_cr_input_source_regions(pst,ilevel,input_size)
  use mdl_module
  use ramses_commons, only: pst_t
  use mdl_parameters
  implicit none
  type(pst_t)::pst
  integer,VALUE::input_size

  integer::ilevel
  integer::rID
  if(pst%nLower>0)then
     rID = mdl_send_request(pst%s%mdl,MDL_CR_INPUT_SOURCE_REGIONS,pst%iUpper+1,input_size,0,ilevel)
     call r_cr_input_source_regions(pst%pLower,ilevel,input_size)
     call mdl_get_reply(pst%s%mdl,rID,0)
  else
     call cr_input_source_regions(pst%s%r,pst%s%g,pst%s%m,ilevel)
  endif
  
end subroutine r_cr_input_source_regions
!#########################################################################
!#########################################################################
!#########################################################################
!#########################################################################
subroutine cr_input_source_regions(r,g,m,ilevel)
  use amr_parameters, only: ndim, twotondim, nvector
  use cr_parameters,only: ncrvar, ncrgrp
  use amr_commons, only: run_t, global_t, mesh_t
  implicit none
  type(run_t)::r
  type(global_t)::g
  type(mesh_t)::m
  integer::ilevel
  
  ! Local variables
  integer::igrid,ngrid,ind,idim,nstride,i,igrp
  !integer::l
  real(kind=8),dimension(1:nvector,1:ndim)::xx
  real(kind=8),dimension(1:nvector,1:ncrvar)::qq
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

        ! Add what is already in the grid
        do igrp=1,ncrgrp
           do i=1,ngrid
              qq(i,1+(igrp-1)*(ndim+1)) = m%unew(ind,r%iEcr+igrp-1,igrid+i-1)
#ifdef CRS
              do idim=1,ndim
                qq(i,1+idim+(igrp-1)*(ndim+1)) = m%crunew(ind,idim+(igrp-1)*ndim,igrid+i-1)
              end do
#endif
           end do
        end do

        ! Inject sources
        call cr_source_regions_sweep(r,g,xx,qq,dx,g%dtnew(ilevel),ngrid)

        ! Scatter primitive variables to main memory
        do igrp=1,ncrgrp
           do i=1,ngrid
              m%unew(ind,r%iEcr+igrp-1,igrid+i-1)=qq(i,1+(igrp-1)*(ndim+1))
#ifdef CRS
              do idim=1,ndim
                m%crunew(ind,idim+(igrp-1)*ndim,igrid+i-1)=qq(i,1+idim+(igrp-1)*(ndim+1))
              end do
#endif
           end do
        end do

     end do
     ! End loop over cells
  end do
  ! End loop over grids

end subroutine cr_input_source_regions
!################################################################
!################################################################
!################################################################
!################################################################
subroutine cr_source_regions_sweep(r,g,x,q,dx,dt,nn)
  use amr_parameters, only: nvector, ndim
  use cr_parameters, only: ncrvar, ncrgrp
  use amr_commons, only: run_t, global_t
  implicit none
  type(run_t)::r
  type(global_t)::g
  integer ::nn
  real(kind=8)::dx,dt
  real(kind=8),dimension(1:nvector,1:ncrvar)::q
  real(kind=8),dimension(1:nvector,1:ndim)::x
  real(kind=8)::scale_nH,scale_T2,scale_l,scale_d,scale_t,scale_v
  real(kind=8)::dx_cgs, dt_cgs, scale_e, scale_fe
  !----------------------------------------------------
  ! This routine sets simple pre-defined initial
  ! conditions, like points, squares, etc.
  !----------------------------------------------------
  integer::i,k,group_ind
  real(kind=8)::vol,rad,weight,xn,yn,zn,en
  ! Units are needed for point source regions
  call units(r,g,scale_l,scale_t,scale_d,scale_v,scale_nH,scale_T2)
  dx_cgs=dx*scale_l
  dt_cgs=dt*scale_t
  scale_e = scale_d*scale_v**2
  scale_fe = scale_e*scale_v

  ! Loop over initial conditions regions
  do k=1,r%cr_nsource
     if(r%cr_src_group(k) .le. 0 .or. r%cr_src_group(k) .gt. ncrgrp) cycle
     if(r%cr_e_source(k).le.0D0) r%cr_e_source(k)=0D0
     group_ind = 1+(r%cr_src_group(k)-1)*(ndim+1)
     
     ! For "square" regions only:
     if(r%cr_source_type(k) .eq. 'square')then
        ! Exponent of choosen norm
        en=r%cr_exp_source(k)
        do i=1,nn
           ! Compute position in normalized coordinates
           xn=0.0d0; yn=0.0d0; zn=0.0d0
           xn=2.0d0*abs(x(i,1)-r%cr_src_x_center(k))/r%cr_src_length_x(k)
#if NDIM>1
           yn=2.0d0*abs(x(i,2)-r%cr_src_y_center(k))/r%cr_src_length_y(k)
#endif
#if NDIM>2
           zn=2.0d0*abs(x(i,3)-r%cr_src_z_center(k))/r%cr_src_length_z(k)
#endif
           ! Compute cell "radius" relative to region center
           if(r%cr_exp_source(k)<10)then
              rad=(xn**en+yn**en+zn**en)**(1.0/en)
           else
              rad=max(xn,yn,zn)
           end if
           ! If cell lies within region,
           if(rad<1.0)then
              q(i,group_ind)=r%cr_e_source(k)/scale_e
              ! The input flux is also given in cgs, i.e. erg/cm2/s
              q(i,group_ind+1)=r%cr_fx_source(k)/scale_fe
#if NDIM>1 
              q(i,group_ind+2)=r%cr_fy_source(k)/scale_fe
#endif
#if NDIM>2
              q(i,group_ind+3)=r%cr_fz_source(k)/scale_fe
#endif
           end if
        end do
     end if
     
     ! For "point" regions only:
     if(r%cr_source_type(k) .eq. 'point')then
        ! Volume elements
        vol=dx_cgs**ndim
        ! Compute CIC weights relative to region center
        do i=1,nn
           xn=1.0; yn=1.0; zn=1.0
           xn=max(1.0-abs(x(i,1)-r%cr_src_x_center(k))/dx,0.0d0)
#if NDIM>1
           yn=max(1.0-abs(x(i,2)-r%cr_src_x_center(k))/dx,0.0d0)
#endif
#if NDIM>2
           zn=max(1.0-abs(x(i,3)-r%cr_src_x_center(k))/dx,0.0d0)
#endif
           weight=xn*yn*zn
           if(weight.gt.0) then
              ! If cell lies within CIC cloud, 
              ! Convert photon number to photon number density
              q(i,group_ind) = q(i,group_ind) + &
                               r%cr_e_source(k)/scale_e *weight/vol * dt_cgs
           endif
        end do
     end if
  end do

  return
end subroutine cr_source_regions_sweep
!################################################################
!################################################################
!################################################################
!################################################################

end module cr_input_condinit_module
