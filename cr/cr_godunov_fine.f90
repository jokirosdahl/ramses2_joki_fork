module cr_godunov_fine_module
contains
!###########################################################
!###########################################################
!###########################################################
!###########################################################
recursive subroutine r_cr_godunov_fine(pst,ilevel,input_size)
  use mdl_module
  use ramses_commons, only: pst_t
  use mdl_parameters
  implicit none
  type(pst_t)::pst
  integer,VALUE::input_size
  integer::ilevel

  integer::rID

  if(pst%nLower>0)then
     rID = mdl_send_request(pst%s%mdl,MDL_CR_GODUNOV_FINE,pst%iUpper+1,input_size,0,ilevel)
     call r_cr_godunov_fine(pst%pLower,ilevel,input_size)
     call mdl_get_reply(pst%s%mdl,rID,0)
  else
     call cr_godunov_fine(pst%s,ilevel)
  endif

end subroutine r_cr_godunov_fine
!###########################################################
!###########################################################
!###########################################################
!###########################################################
subroutine cr_godunov_fine(s,ilevel)
  use ramses_commons, only: ramses_t
  use cache_commons
  use cache
  use marshal, only: pack_fetch_refine,unpack_fetch_refine
  use boundaries, only: init_bound_refine
  implicit none
  type(ramses_t)::s
  integer::ilevel
  type(msg_large_realdp)::dummy_large_realdp
  !--------------------------------------------------------------------------
  ! This routine is a wrapper to the second order Godunov solver.
  ! Small grids (2x2x2) are gathered from level ilevel and sent to the
  ! hydro solver. On entry, hydro variables are gathered from array cruold.
  ! On exit, crunew has been updated. 
  !--------------------------------------------------------------------------
  integer::igrid

  if(s%r%verbose.and.s%g%myid==1)write(*,'("   Entering godunov_fine for level ",I2)')ilevel

  associate(r=>s%r,g=>s%g,m=>s%m,mdl=>s%mdl)

  call open_cache(mdl, m, pack_size=storage_size(dummy_large_realdp)/32, &
       pack=pack_fetch_refine, unpack=unpack_fetch_refine, &
       init=init_flush_cr_godunov, flush=pack_flush_cr_godunov, &
       combine=unpack_flush_cr_godunov, bound=init_bound_refine)

  ! Loop over active grids by vector sweeps
  igrid=m%head(ilevel)
  do while(igrid.LE.m%tail(ilevel))
     SELECT CASE (m%grid(igrid)%superoct)
     ! For now just re-using the hydro kernel for CR. Might change this later.
     CASE(1)
        call cr_godfine1(s,igrid,ilevel,m%cr_w%kernel_1)
     CASE(2**ndim)
        call cr_godfine1(s,igrid,ilevel,m%cr_w%kernel_2)
     CASE(4**ndim)
        call cr_godfine1(s,igrid,ilevel,m%cr_w%kernel_4)
     CASE(8**ndim)
        call cr_godfine1(s,igrid,ilevel,m%cr_w%kernel_8)
     CASE(16**ndim)
        call cr_godfine1(s,igrid,ilevel,m%cr_w%kernel_16)
     CASE(32**ndim)
        call cr_godfine1(s,igrid,ilevel,m%cr_w%kernel_32)
     END SELECT
     igrid=igrid+m%grid(igrid)%superoct
  end do

  call close_cache(mdl)

  end associate

end subroutine cr_godunov_fine
!###########################################################
!###########################################################
!###########################################################
!###########################################################
recursive subroutine r_set_crunew(pst,ilevel,input_size)
  use mdl_module
  use ramses_commons, only: pst_t
  use mdl_parameters
  implicit none
  type(pst_t)::pst
  integer,VALUE::input_size
  integer::ilevel

  integer::rID

  if(pst%nLower>0)then
     rID = mdl_send_request(pst%s%mdl,MDL_SET_CRUNEW,pst%iUpper+1,input_size,0,ilevel)
     call r_set_crunew(pst%pLower,ilevel,input_size)
     call mdl_get_reply(pst%s%mdl,rID,0)
  else
     call set_crunew(pst%s%m,ilevel)
  endif

end subroutine r_set_crunew
!###########################################################
!###########################################################
!###########################################################
!###########################################################
subroutine set_crunew(m,ilevel)
  use amr_parameters, only: ndim, twotondim
  use amr_commons, only: run_t, global_t, mesh_t
  implicit none
  type(mesh_t)::m
  integer::ilevel
  !--------------------------------------------------------------------------
  ! This routine sets array crunew to its initial value cruold before calling
  ! the hydro scheme. crunew is set to zero in virtual boundaries.
  !--------------------------------------------------------------------------
  integer::i

  ! Set crunew to cruold for myid cells
#ifdef CR
  do i=m%head(ilevel),m%tail(ilevel)
     m%crunew(:,:,i)=m%cruold(:,:,i)
  end do
#endif

end subroutine set_crunew
!###########################################################
!###########################################################
!###########################################################
!###########################################################
recursive subroutine r_set_cruold(pst,ilevel,input_size)
  use mdl_module
  use ramses_commons, only: pst_t
  use mdl_parameters
  implicit none
  type(pst_t)::pst
  integer,VALUE::input_size
  integer::ilevel
  integer::rID

  if(pst%nLower>0)then
     rID = mdl_send_request(pst%s%mdl,MDL_SET_CRUOLD,pst%iUpper+1,input_size,0,ilevel)
     call r_set_cruold(pst%pLower,ilevel,input_size)
     call mdl_get_reply(pst%s%mdl,rID,0)
  else
     call set_cruold(pst%s%r, pst%s%g, pst%s%m, ilevel)
  endif

end subroutine r_set_cruold
!###########################################################
!###########################################################
!###########################################################
!###########################################################
subroutine set_cruold(r, g, m, ilevel)
  use cr_parameters, only: ncrgrp, smallcr
  use amr_parameters, only: ndim, twotondim
  use amr_commons, only: run_t, global_t, mesh_t
  implicit none
  type(run_t) :: r
  type(global_t) :: g
  type(mesh_t) :: m
  integer :: ilevel
  real(kind=8)::Ecrc,fred,sqrt3
  !---------------------------------------------------------
  ! This routine sets array cruold to its new value crunew 
  ! after the hydro step.
  !---------------------------------------------------------
  integer :: i, j, ig, iE

  if(r%isotropic_pressure)then
     sqrt3=sqrt(3d0)
  else
     sqrt3=1.0d0
  endif
   
 ! Set cruold to crunew
#ifdef CR
  if (r%reduced_CR_flux_correction) then
     ! Make a CR conservation fix (prevent CR explosions)
     do ig = 1, ncrgrp
        iE = 1 + (ig-1)*ndim
        do i = m%head(ilevel), m%tail(ilevel)
           do j = 1, twotondim
              ! No negative CR densities:
              m%crunew(j,iE,i) = max(m%crunew(j,iE,i),smallcr)
              Ecrc=m%crunew(j,iE,i)*g%cr_c(ilevel)
              ! Reduced flux, should always be .le. 1
              fred = sqrt(sum((m%crunew(j,iE+1:iE+ndim,i))**2))/Ecrc*sqrt3
              if(fred .gt. 1d0) then ! Too big so normalize flux to one
                 m%crunew(j,iE+1:iE+ndim,i) &
                      = m%crunew(j,iE+1:iE+ndim,i)/fred
              endif
           end do
        end do
     end do
     ! End CR conservation fix
  endif

  do i = m%head(ilevel), m%tail(ilevel)
     m%cruold(:,:,i) = m%crunew(:,:,i)
  end do

#endif

end subroutine set_cruold
!###########################################################
!###########################################################
!###########################################################
!###########################################################
subroutine cr_godfine1(s,ind_grid,ilevel,h)
  use mdl_module
  use amr_parameters, only: ndim, twondim, twotondim
  use cr_parameters, only: ncrvar
  use ramses_commons, only: ramses_t
  use nbors_utils
  use cr_commons
  use hash
  use cr_flux_module, only: cr_unsplit
  implicit none
  type(ramses_t)::s
  integer::ind_grid,ilevel
  type(cr_kernel_t)::h
  !-------------------------------------------------------------------
  ! This routine gathers first CR variables from neighboring grids
  ! to set initial conditions in a 6x6x6 grid. It interpolate from
  ! coarser level missing grid variables. It then calls the
  ! Godunov solver that computes fluxes. These fluxes are zeroed at 
  ! coarse-fine boundaries, since contribution from finer levels has
  ! already been taken into account. Conservative variables are updated 
  ! and stored in array crunew(:), both at the current level and at the 
  ! coarser level if necessary.
  !-------------------------------------------------------------------
  integer::ivar,idim,ind_son,ind_oct
  integer::icell,inbor,ipass
  integer::igrid,ichild
  integer::i0,j0,k0,i1,j1,k1,i2,j2,k2,i3,j3,k3
  integer::ii0,jj0,kk0,ii1,jj1,kk1
  integer::i1min,i1max,j1min,j1max,k1min,k1max
  integer::ii1min,ii1max,jj1min,jj1max,kk1min,kk1max
  integer::i2min,i2max,j2min,j2max,k2min,k2max
  integer::i3min,i3max,j3min,j3max,k3min,k3max
  integer,dimension(1:ndim)::ckey_corner,ckey
  integer(kind=8),dimension(0:ndim)::hash_nbor
  integer,dimension(0:twondim)::icell_nbor
  integer,dimension(0:twondim)::igrid_nbor
  real(kind=8)::dx,oneontwotondim,cr_c_diff
  real(kind=8),dimension(0:twondim  ,1:ncrvar)::u1
  real(kind=8),dimension(1:twotondim,1:ncrvar)::u2
  logical::okx,oky,okz,oknbor

  i2min=0; i2max=0; j2min=0; j2max=0; k2min=0; k2max=0
  i3min=1; i3max=1; j3min=1; j3max=1; k3min=1; k3max=1
  okx=.true.; oky=.true.; okz=.true.

  associate(r=>s%r,g=>s%g,m=>s%m,mdl=>s%mdl)

  oneontwotondim = 1.d0/dble(twotondim)

  ! Mesh spacing in that level
  dx=r%boxlen/2**ilevel

  ! Integer constants
  i1min=h%io1; i1max=h%io2; j1min=h%jo1; j1max=h%jo2; k1min=h%ko1; k1max=h%ko2
#if NDIM>0
  i2max=1; i3min=h%iu1+2; i3max=h%iu2-2
#endif
#if NDIM>1
  j2max=1; j3min=h%ju1+2; j3max=h%ju2-2
#endif
#if NDIM>2
  k2max=1; k3min=h%ku1+2; k3max=h%ku2-2
#endif

  !---------------------
  ! Gather hydro stencil
  !---------------------
  hash_nbor(0)=m%grid(ind_grid)%lev
  ckey_corner(1:ndim)=(m%grid(ind_grid)%ckey(1:ndim)/(i1max-1))*(i1max-1)
  ind_oct=ind_grid
  h%inkernel=.false.

  ! Loop over 3x3x3 neighboring father cells using 7 passes
  do ipass = 1, 1+2*ndim

  if(ipass == 1)then
     ii1min = i1min+1; ii1max = i1max-1
     jj1min = j1min; jj1max = j1max
#if NDIM>1
     jj1min = jj1min+1; jj1max = jj1max-1
#endif
     kk1min = k1min; kk1max = k1max;
#if NDIM>2
     kk1min = kk1min+1; kk1max = kk1max-1
#endif
  endif
  if(ipass == 2)then
     ii1min = i1min; ii1max = i1min
  endif
  if(ipass == 3)then
     ii1min = i1max; ii1max = i1max
  endif
  if(ipass == 4)then
     ii1min = i1min+1; ii1max = i1max-1
     jj1min = j1min; jj1max = j1min
  endif
  if(ipass == 5)then
     jj1min = j1max; jj1max = j1max
  endif
  if(ipass == 6)then
     jj1min = j1min+1; jj1max = j1max-1
     kk1min = k1min; kk1max = k1min
  endif
  if(ipass == 7)then
     kk1min = k1max; kk1max = k1max
  endif

  do k1=kk1min,kk1max
#if NDIM>2
     okz=(k1>k1min.and.k1<k1max)
#endif
     do j1=jj1min,jj1max
#if NDIM>1
        oky=(j1>j1min.and.j1<j1max)
#endif
        do i1=ii1min,ii1max
           okx=(i1>i1min.and.i1<i1max)

           !--------------------
           ! For inner octs only
           !--------------------
           if(okx.and.oky.and.okz)then

              ! Compute relative Cartesian key
              ckey(1:ndim)=m%grid(ind_oct)%ckey(1:ndim)-ckey_corner(1:ndim)

              ! Store grid index
              ii1=1; jj1=1; kk1=1
              ii1=ckey(1)+1
#if NDIM>1
              jj1=ckey(2)+1
#endif
#if NDIM>2
              kk1=ckey(3)+1
#endif
              h%childloc(ii1,jj1,kk1)=ind_oct
              h%inkernel(ii1,jj1,kk1)=.true.
              h%gridloc(ii1,jj1,kk1)=0
              h%cellloc(ii1,jj1,kk1)=0

              ! Loop over 2x2x2 cells
              do k2=k2min,k2max
                 do j2=j2min,j2max
                    do i2=i2min,i2max
                       ind_son=1+i2+2*j2+4*k2
                       i3=1; j3=1; k3=1
                       i3=1+2*(ii1-1)+i2
#if NDIM>1
                       j3=1+2*(jj1-1)+j2
#endif
#if NDIM>2
                       k3=1+2*(kk1-1)+k2
#endif             
                       ! Gather CR variables
                       do ivar=1,ncrvar
#ifdef CR
                          h%cruloc(i3,j3,k3,ivar)=m%cruold(ind_son,ivar,ind_oct)
#endif
                       end do
                       ! Gather refinement flag
                       h%okloc(i3,j3,k3)=m%grid(ind_oct)%refined(ind_son)
                    end do
                 end do
              end do

              ! Go to next inner oct
              ind_oct=ind_oct+1

           !-----------------------
           ! For boundary octs only
           !-----------------------
           else

              ! Compute neighboring grid Cartesian index
              hash_nbor(1)=ckey_corner(1)+i1-1
#if NDIM>1
              hash_nbor(2)=ckey_corner(2)+j1-1
#endif
#if NDIM>2
              hash_nbor(3)=ckey_corner(3)+k1-1
#endif
              ! Periodic boundary conditions
              do idim=1,ndim
                 if(r%periodic(idim))then
                    if(hash_nbor(idim)< m%box_ckey_min(idim,ilevel))hash_nbor(idim)=m%box_ckey_max(idim,ilevel)-1
                    if(hash_nbor(idim)>=m%box_ckey_max(idim,ilevel))hash_nbor(idim)=m%box_ckey_min(idim,ilevel)
                 endif
              enddo
              
              ! Set grid index to null
              icell=0
              igrid=0
              do inbor=0,twondim
                 igrid_nbor(inbor)=0
              end do
              ! Get neighboring grid index with read-only cache
              call get_grid(s,hash_nbor,ichild,flush_cache=.false.,fetch_cache=.true.,lock=.true.)

              !----------------------------------------------------
              ! If grid does not exist, interpolate CR variables
              !----------------------------------------------------
              if(ichild==0)then

                 ! Get parent father cell with read-write cache
                 call get_parent_cell(s,hash_nbor,igrid,icell,flush_cache=.true.,fetch_cache=.true.,lock=.true.)
                 if(igrid==0)then
                    write(*,*)'CR-GODUNOV: parent_cell should exist'
                    write(*,*)'PE ',g%myid,hash_nbor
                    call mdl_abort(mdl)
                 endif

                 ! Get 2ndim neighboring father cells with read-write cache
                 ! Note that possible cache grids are locked inside the routine
                 call get_twondim_nbor_parent_cell(s,hash_nbor,igrid_nbor,icell_nbor,flush_cache=.true.,fetch_cache=.true.)
                 oknbor=.true.
                 do inbor=0,twondim
                    oknbor=oknbor.and.(igrid_nbor(inbor)>0)
                 end do
                 if(.not. oknbor)then
                    write(*,*)"CR-GODUNOV: parent neighbors should exist"
                    write(*,*)'PE ',g%myid,hash_nbor
                    write(*,*)igrid_nbor(0)
                    do idim=1,ndim
                       write(*,*)igrid_nbor(2*idim-1)
                       write(*,*)igrid_nbor(2*idim)
                    end do
                    call mdl_abort(mdl)
                 endif

#ifdef CR
                 ! Gather CR variables
                 do inbor=0,twondim
                    do ivar=1,ncrvar
                       u1c(inbor,ivar)=m%cruold(icell_nbor(inbor),ivar,igrid_nbor(inbor))
                    end do
                 end do
                 ! Interpolate using cr variables
                 call interpol_cr(u1c,u2c,r%interpol_var,r%interpol_type)
#endif

                 ! Gather hydro variables
                 do inbor=0,twondim
                    do ivar=1,nvar
                       u1h(inbor,++ivar)=m%uold(icell_nbor(inbor),ivar,igrid_nbor(inbor))
                    end do
                 end do
#ifdef MHD
                 ! Gather B field variables
                 do inbor=0,twondim
                    do ivar=1,6
                       b1(inbor,ivar)=m%bold(icell_nbor(inbor),ivar,igrid_nbor(inbor))
                    end do
                 end do
                 ! Get neighboring children grids
                 hash_son_nbor(0)=ilevel
                 do inbor=1,twondim
                    hash_son_nbor(1:ndim)=hash_nbor(1:ndim)+shift(1:ndim,inbor)
                    ! Periodic boundary conditions
                    do idim=1,ndim
                       if(r%periodic(idim))then
                          if(hash_son_nbor(idim)< m%box_ckey_min(idim,ilevel))hash_son_nbor(idim)=m%box_ckey_max(idim,ilevel)-1
                          if(hash_son_nbor(idim)>=m%box_ckey_max(idim,ilevel))hash_son_nbor(idim)=m%box_ckey_min(idim,ilevel)
                       endif
                    enddo
                    call get_grid(s,hash_son_nbor,igridn,flush_cache=.false.,fetch_cache=.true.,lock=.true.)
                    igrid_son_nbor(inbor)=igridn
                    refined(inbor)=(igridn>0)
                    if(refined(inbor))then
                       do ind=1,twotondim
                          do ivar=1,6
                             b3(inbor,ind,ivar)=m%bold(ind,ivar,igridn)
                          end do
                       end do
                    endif
                 end do

                 ! Interpolate using MHD variables
                 call interpol_mhd(u1h,u2h,b1,b2,b3,refined,r%interpol_var,r%interpol_type,r%smallr)
#endif

!!$                 ! Interpolate using cr variables
!!$                 call interpol_hydro(u1h,u2h,r%interpol_var,r%interpol_type)

!!$                 ! Wrap up in the same u2 array
!!$                 u2(:,1     :nvar         )=u2h(:,1:nvar  )
!!$                 u2(:,nvar+1:nvar+1+ncrvar)=u2c(:,1:ncrvar)
              endif

              ! Store grid index
              h%childloc(i1,j1,k1)=ichild
              h%inkernel(i1,j1,k1)=.true.
              h%gridloc (i1,j1,k1)=igrid
              h%cellloc (i1,j1,k1)=icell
              do inbor=1,twondim
                 h%nborloc(i1,j1,k1,inbor)=igrid_nbor(inbor)
              end do

              ! Loop over 2x2x2 cells
              do k2=k2min,k2max
                 do j2=j2min,j2max
                    do i2=i2min,i2max
                       ind_son=1+i2+2*j2+4*k2
                       i3=1; j3=1; k3=1
                       i3=1+2*(i1-1)+i2
#if NDIM>1
                       j3=1+2*(j1-1)+j2
#endif
#if NDIM>2
                       k3=1+2*(k1-1)+k2
#endif
                       !----------------------------------------------------
                       ! If neighboring grid exists, use refined grid values
                       !----------------------------------------------------
                       if(ichild>0)then

                          ! Gather refinement flag
                          h%okloc(i3,j3,k3)=m%grid(ichild)%refined(ind_son)
                          ! Gather CR variables
                          do ivar=1,ncrvar
#ifdef CR
                             h%cruloc(i3,j3,k3,ivar)=m%cruold(ind_son,ivar,ichild)
#endif
                          end do

                          ! Gather hydro variables
                          do ivar=1,nvar
                             h%uloc(i3,j3,k3,ivar)=m%uold(ind_son,ivar,ichild)
                          end do
#ifdef MHD
                          ! Gather MHD variables
                          do ivar=1,6
                             h%bloc(i3,j3,k3,ivar)=m%bold(ind_son,ivar,ichild)
                          end do
#endif

                       !-----------------------------------------------------------
                       ! If neighboring grid doesn't exist, use interpolated values
                       !-----------------------------------------------------------
                       else

                          ! Gather refinement flag
                          h%okloc(i3,j3,k3)=.false.
                          ! Gather interpolated hydro variables
                          do ivar=1,ncrvar
#ifdef CR
                             h%cruloc(i3,j3,k3,ivar)=u2c(ind_son,ivar)
                             if(mod(ivar,ndim+1)==1) then
                                ! Variable light speed correction for coarser level
                                h%cruloc(i3,j3,k3,ivar)           &
                                  = h%cruloc(i3,j3,k3,ivar)       &
                                  * g%cr_c(ilevel-1)/g%cr_c(ilevel)
                             endif
#endif
                          end do

                          ! Gather interpolated hydro variables
                          do ivar=1,nvar
                             h%uloc(i3,j3,k3,ivar)=u2h(ind_son,ivar)
                          end do
#ifdef MHD
                          ! Gather interpolated MHD variables
                          do ivar=1,6
                             h%bloc(i3,j3,k3,ivar)=b2(ind_son,ivar)
                          end do
#endif
                          
                       end if

                    end do
                 end do
              end do
              ! End loop over 2x2x2 cells
           endif
        end do
     end do
  end do
  ! End over octs
  end do

  !-------------------------------------------------
  ! Compute flux using second-order Godunov method
  !-------------------------------------------------
#ifdef CR
  call cr_unsplit(r,h%uloc,u%bloc,h%cruloc,h%crflux,h%cFlx, &
       & g%cr_c(ilevel),dx,g%dtnew(ilevel),                 &
       & h%iu1,h%iu2,h%ju1,h%ju2,h%ku1,h%ku2,               &
       & h%if1,h%if2,h%jf1,h%jf2,h%kf1,h%kf2)
#endif

  !-------------------------------------------------
  ! Reset flux along direction at refined interfaces
  !-------------------------------------------------
  if(r%cr_nsubcycle.eq.1) then
  do idim=1,ndim
     i0=0; j0=0; k0=0
     if(idim==1)i0=1
     if(idim==2)j0=1
     if(idim==3)k0=1
     do k3=k3min,k3max+k0
        do j3=j3min,j3max+j0
           do i3=i3min,i3max+i0
              if(h%okloc(i3-i0,j3-j0,k3-k0) .or. h%okloc(i3,j3,k3))then
                 do ivar=1,ncrvar
#ifdef CR
                    h%crflux(i3,j3,k3,ivar,idim)=0.0d0
#endif
                 end do
              end if
           end do
        end do
     end do
  end do
  endif
  !--------------------------------------
  ! Conservative update at level ilevel
  !--------------------------------------
  ! Loop over dimensions
  do idim=1,ndim
     i0=0; j0=0; k0=0
     ii0=0; jj0=0; kk0=0
     if(idim==1)i0=1
     if(idim==2)j0=1
     if(idim==3)k0=1
     ii0=1
#if NDIM>1
     jj0=1
#endif
#if NDIM>2
     kk0=1
#endif
     ! Loop over inner octs
     do k1=k1min+kk0,k1max-kk0
        do j1=j1min+jj0,j1max-jj0
           do i1=i1min+ii0,i1max-ii0
              ! Get oct index
              ichild=h%childloc(i1,j1,k1)
              ! Loop over cells
              do k2=k2min,k2max
                 do j2=j2min,j2max
                    do i2=i2min,i2max
                       ind_son=1+i2+2*j2+4*k2
                       i3=1; j3=1; k3=1
                       i3=1+2*(i1-1)+i2
#if NDIM>1
                       j3=1+2*(j1-1)+j2
#endif
#if NDIM>2
                       k3=1+2*(k1-1)+k2
#endif
                       ! Update conservative variables new state vector
                       do ivar=1,ncrvar
#ifdef CR
                          m%crunew(ind_son,ivar,ichild)=m%crunew(ind_son,ivar,ichild)+ &
                               & (h%crflux(i3   ,j3   ,k3   ,ivar,idim) &
                               & -h%crflux(i3+i0,j3+j0,k3+k0,ivar,idim))
#endif
                       end do
                    end do
                 end do
              end do
           end do
        end do
     end do
  end do

  ! If sitting in coarsest level, skip.
  if(ilevel>r%levelmin)then

  !--------------------------------------
  ! Conservative update at level ilevel-1
  !--------------------------------------
  if(r%cr_nsubcycle.eq.1) then
  ! Loop over dimensions
  do idim=1,ndim
     i0=0; j0=0; k0=0
     ii0=0; jj0=0; kk0=0
     if(idim==1)i0=1
     if(idim==2)j0=1
     if(idim==3)k0=1
     ii0=1
#if NDIM>1
     jj0=1
#endif
#if NDIM>2
     kk0=1
#endif
     ii1min=i1min+ii0; ii1max=i1max-ii0
     jj1min=j1min+jj0; jj1max=j1max-jj0
     kk1min=k1min+kk0; kk1max=k1max-kk0
     !----------------------
     ! Left flux at boundary
     !----------------------     
     if(idim==1)then
        ii1min=i1min; ii1max=i1min
     endif
     if(idim==2)then
        jj1min=j1min; jj1max=j1min
     endif
     if(idim==3)then
        kk1min=k1min; kk1max=k1min
     endif
     ! Loop over outer octs on left face
     do k1=kk1min,kk1max
        do j1=jj1min,jj1max
           do i1=ii1min,ii1max
              ! Get oct index
              ichild=h%childloc(i1,j1,k1)
              ! Check that parent cell is not refined
              if(ichild==0)then
                 ! Get parent cell index
                 igrid=h%gridloc(i1,j1,k1)
                 icell=h%cellloc(i1,j1,k1)
                 ! Loop over inner cell left faces
                 do k2=k2min,k2max-k0
                    do j2=j2min,j2max-j0
                       do i2=i2min,i2max-i0
                          i3=1; j3=1; k3=1
                          i3=1+2*(i1+i0-1)+i2
#if NDIM>1
                          j3=1+2*(j1+j0-1)+j2
#endif
#if NDIM>2
                          k3=1+2*(k1+k0-1)+k2
#endif
                          ! Conservative update of new state variables
                          do ivar=1,ncrvar
                             ! For VSLA, when updating coarser level, rescale radiation flux
                             ! to the expression which would be seen from there.
                             cr_c_diff = g%cr_c(ilevel-1)/g%cr_c(ilevel)
                             if (mod(ivar,ndim+1)==1) then
                                cr_c_diff=1.d0
                             end if
#ifdef CR
                             m%crunew(icell,ivar,igrid)=m%crunew(icell,ivar,igrid) &
                                  & -h%crflux(i3,j3,k3,ivar,idim)              &
                                  & * oneontwotondim * cr_c_diff
#endif
                          end do
                       end do
                    end do
                 end do
              endif
           end do
        end do
     end do
     !-----------------------
     ! Right flux at boundary
     !-----------------------     
     if(idim==1)then
        ii1min=i1max; ii1max=i1max
     endif
     if(idim==2)then
        jj1min=j1max; jj1max=j1max
     endif
     if(idim==3)then
        kk1min=k1max; kk1max=k1max
     endif
     ! Loop over outer octs on right face
     do k1=kk1min,kk1max
        do j1=jj1min,jj1max
           do i1=ii1min,ii1max
              ! Get oct index
              ichild=h%childloc(i1,j1,k1)
              ! Check that parent cell is not refined
              if(ichild==0)then
                 ! Get parent cell index
                 igrid=h%gridloc(i1,j1,k1)
                 icell=h%cellloc(i1,j1,k1)
                 ! Loop over inner cell right faces
                 do k2=k2min+k0,k2max
                    do j2=j2min+j0,j2max
                       do i2=i2min+i0,i2max
                          i3=1; j3=1; k3=1
                          i3=1+2*(i1-i0-1)+i2
#if NDIM>1
                          j3=1+2*(j1-j0-1)+j2
#endif
#if NDIM>2
                          k3=1+2*(k1-k0-1)+k2
#endif
                          ! Conservative update of new state variables
                          do ivar=1,ncrvar
                             ! Rescale for VSLA, as above
                             cr_c_diff = g%cr_c(ilevel-1)/g%cr_c(ilevel)
                             if (mod(ivar,ndim+1)==1) then
                                cr_c_diff=1.d0
                             end if
#ifdef CR
                             m%crunew(icell,ivar,igrid)=m%crunew(icell,ivar,igrid) &
                                  & +h%crflux(i3+i0,j3+j0,k3+k0,ivar,idim)     &
                                  & *oneontwotondim * cr_c_diff
#endif
                          end do
                       end do
                    end do
                 end do
                 ! End loop over faces
              endif
           end do
        end do
     end do
     ! End loop over boundary octs
  end do
  ! End loop over dimensions
  endif ! if(r%cr_nsubcycle.eq.1)

  endif

  !----------------
  ! Unlock all octs
  !----------------
  do k1=k1min,k1max
     do j1=j1min,j1max
        do i1=i1min,i1max     
           ! Check if in kernel
           if(.not.h%inkernel(i1,j1,k1))cycle
           ! Get oct index
           ichild=h%childloc(i1,j1,k1)
           ! Check that parent cell is not refined
           if(ichild>0)then
              call unlock_cache(m,ichild)
           else
              ! Get parent cell index
              igrid=h%gridloc(i1,j1,k1)
              call unlock_cache(m,igrid)
              ! Get neighbouring parent oct index
              do inbor=1,twondim
                 igrid=h%nborloc(i1,j1,k1,inbor)
                 call unlock_cache(m,igrid)
              end do
           endif
        end do
     end do
  end do

  end associate

end subroutine cr_godfine1
!###########################################################
!###########################################################
!###########################################################
!###########################################################
subroutine init_flush_cr_godunov(mesh,igrid,hash_key)
  use amr_parameters, only: ndim, twotondim
  use cr_parameters, only: ncrvar
  use amr_commons, only: mesh_t
  type(mesh_t)::mesh
  integer::igrid
  integer(kind=8),dimension(0:ndim)::hash_key

  integer::ind,ivar

  mesh%grid(igrid)%lev=hash_key(0)
  mesh%grid(igrid)%ckey(1:ndim)=hash_key(1:ndim)

#ifdef CR
  do ivar=1,ncrvar
     do ind=1,twotondim
        mesh%crunew(ind,ivar,igrid)=0.0d0
     enddo
  enddo
#endif

end subroutine init_flush_cr_godunov
!###########################################################
!###########################################################
!###########################################################
!###########################################################
subroutine pack_flush_cr_godunov(mesh,igrid,msg_size,msg_array)
  use amr_parameters, only: twotondim
  use cr_parameters, only: ncrvar
  use amr_commons, only: mesh_t
  use cache_commons, only: msg_large_realdp
  type(mesh_t)::mesh
  integer::igrid
  integer::msg_size
  integer,dimension(1:msg_size),optional::msg_array

  integer::ind,ivar
  type(msg_large_realdp)::msg

#ifdef CR
  do ivar=1,ncrvar
     do ind=1,twotondim
        msg%realdp_cr(ind,ivar)=mesh%crunew(ind,ivar,igrid)
     end do
  end do
#endif

  msg_array=transfer(msg,msg_array)

end subroutine pack_flush_cr_godunov
!###########################################################
!###########################################################
!###########################################################
!###########################################################
subroutine unpack_flush_cr_godunov(mesh,igrid,msg_size,msg_array,hash_key)
  use amr_parameters, only: ndim, twotondim
  use cr_parameters, only: ncrvar
  use amr_commons, only: mesh_t
  use cache_commons, only: msg_large_realdp
  type(mesh_t)::mesh
  integer::igrid
  integer::msg_size
  integer,dimension(1:msg_size),optional::msg_array
  integer(kind=8),dimension(0:ndim)::hash_key

  integer::ind,ivar
  type(msg_large_realdp)::msg

  mesh%grid(igrid)%lev=hash_key(0)
  mesh%grid(igrid)%ckey(1:ndim)=hash_key(1:ndim)
  msg=transfer(msg_array,msg)

#ifdef CR
  do ivar=1,ncrvar
     do ind=1,twotondim
        mesh%crunew(ind,ivar,igrid)=mesh%crunew(ind,ivar,igrid)+msg%realdp_cr(ind,ivar)
     end do
  end do
#endif

end subroutine unpack_flush_cr_godunov
!###########################################################
!###########################################################
!###########################################################
!###########################################################
end module cr_godunov_fine_module
