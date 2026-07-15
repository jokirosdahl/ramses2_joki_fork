module cr_flag_module
contains
!#####################################################################
!#####################################################################
!#####################################################################
!#####################################################################
subroutine cr_flag(s,ilevel)
  use amr_parameters, only: ndim, twotondim, twondim
  use ramses_commons, only: ramses_t
  use cr_parameters, only: ncrvar, ncrgrp
  use cache_commons
  use cache
  use nbors_utils
  use boundaries, only: init_bound_refine
  implicit none
  type(ramses_t)::s
  integer::ilevel
  ! -------------------------------------------------------------------
  ! This routine flag for refinement cells that satisfies
  ! some user-defined physical criteria at the level ilevel. 
  ! -------------------------------------------------------------------
  integer,dimension(1:3,1:8),save::iii=reshape(&
       & (/0,0,0,1,0,0,0,1,0,1,1,0,0,0,1,1,0,1,0,1,1,1,1,1/),(/3,8/))
  integer,dimension(1:3,1:6),save::shift=reshape(&
       & (/-1,0,0,1,0,0,0,-1,0,0,1,0,0,0,-1,0,0,1/),(/3,6/))
  integer::igrid,ind,idim,iidim,igrp,i_nbor
  integer::icelld,icellg,igridg,igridd,icellp,igridp,igroup
  integer,dimension(1:twondim)::icelln
  integer(kind=8),dimension(0:ndim)::hash_key,hash_nbor
  real(kind=8),dimension(1:ncrvar)::uug,uum,uud
  logical::ok, do_cr_refine
  integer,dimension(1:twondim)::igridn
  type(msg_realdp)::dummy_realdp

  associate(r=>s%r,g=>s%g,m=>s%m,mdl=>s%mdl)

  do_cr_refine=.false.
  do igroup=1, ncrgrp
    if( r%cr_err_grad_ecr(igroup) .ne. -1.0 ) do_cr_refine=.true.
  end do
  if(.not. do_cr_refine) return ! No refinement done on radiation vars

  hash_key(0)=ilevel+1

  call open_cache(mdl, m, pack_size=storage_size(dummy_realdp)/32, &
       pack=pack_fetch_cr, unpack=unpack_fetch_cr, &
       bound=init_bound_refine)

  ! Loop over active grids
  do igrid=m%head(ilevel),m%tail(ilevel)

     ! Loop over cells
     do ind=1,twotondim

        ! Compute cell hash key
        hash_key(1:ndim)=2*m%grid(igrid)%ckey(1:ndim)+iii(1:ndim,ind)

        ! Initialize refinement to false
        ok=.false.

        ! If a neighbor cell does not exist,
        ! replace it by its father cell
        do i_nbor=1,twondim
           hash_nbor(0)=hash_key(0)
           ! Periodic boundary conditions
           do idim=1,ndim
              hash_nbor(idim)=hash_key(idim)+shift(idim,i_nbor)
              if(r%periodic(idim))then
                 if(hash_nbor(idim)< m%box_ckey_min(idim,ilevel+1))hash_nbor(idim)=m%box_ckey_max(idim,ilevel+1)-1
                 if(hash_nbor(idim)>=m%box_ckey_max(idim,ilevel+1))hash_nbor(idim)=m%box_ckey_min(idim,ilevel+1)
              endif
           enddo
           call get_parent_cell(s,hash_nbor,igridp,icellp,flush_cache=.false.,fetch_cache=.true.,lock=.true.)
           if(igridp>0)then
              igridn(i_nbor)=igridp
              icelln(i_nbor)=icellp
           else
              hash_nbor(0)=hash_nbor(0)-1
              hash_nbor(1:ndim)=hash_nbor(1:ndim)/2
              call get_parent_cell(s,hash_nbor,igridp,icellp,flush_cache=.false.,fetch_cache=.true.,lock=.true.)
              igridn(i_nbor)=igridp
              icelln(i_nbor)=icellp
           endif
        end do

        ! Loop over dimensions
        do idim=1,ndim
           ! Gather cr variables
           do igrp=1,ncrgrp
              icellg=icelln(2*idim-1)
              icelld=icelln(2*idim  )
              igridg=igridn(2*idim-1)
              igridd=igridn(2*idim  )
#ifdef CRS
              uug(1+(igrp-1)*(ndim+1))=m%uold(icellg,r%iEcr+igrp-1,igridg)
              uum(1+(igrp-1)*(ndim+1))=m%uold(ind   ,r%iEcr+igrp-1,igrid )
              uud(1+(igrp-1)*(ndim+1))=m%uold(icelld,r%iEcr+igrp-1,igridd)
              do iidim=1,ndim
                uug(1+iidim+(igrp-1)*(ndim+1))=m%cruold(icellg,iidim+(igrp-1)*ndim,igridg)
                uum(1+iidim+(igrp-1)*(ndim+1))=m%cruold(ind   ,iidim+(igrp-1)*ndim,igrid )
                uud(1+iidim+(igrp-1)*(ndim+1))=m%cruold(icelld,iidim+(igrp-1)*ndim,igridd)
              end do
#endif 
           end do
           call cr_refine(r,uug,uum,uud,ok)
        end do

        do i_nbor=1,twondim
           call unlock_cache(m,igridn(i_nbor))
        end do

        ! Count only newly flagged cells
        if(m%flag1(ind,igrid)==0.and.ok)g%nflag=g%nflag+1
        if(ok)m%flag1(ind,igrid)=1

     end do
     ! End loop over cells
  end do
  ! End loop over grids

  call close_cache(mdl)

  end associate

end subroutine cr_flag
!#####################################################################
!#####################################################################
!#####################################################################
!#####################################################################
subroutine pack_fetch_cr(mesh,igrid,msg_size,msg_array)
  use amr_parameters, only: ndim, twotondim
  use hydro_parameters, only: nvar
  use cr_parameters, only: ncruvar
  use amr_commons, only: mesh_t
  use cache_commons, only: msg_realdp
  type(mesh_t)::mesh
  integer::igrid
  integer::msg_size
  integer,dimension(1:msg_size),optional::msg_array

  integer::ind,ivar
  type(msg_realdp)::msg

  do ind=1,twotondim
     if(mesh%grid(igrid)%refined(ind))then
        msg%int4(ind)=1
     else
        msg%int4(ind)=0
     endif
  end do

#ifdef HYDRO
  do ivar=1,nvar
     do ind=1,twotondim
        msg%realdp(ind,ivar)=mesh%uold(ind,ivar,igrid)
     end do
  end do
#endif
#ifdef CRS
  do ivar=1,ncruvar
     do ind=1,twotondim
        msg%realdp_cr(ind,ivar)=mesh%cruold(ind,ivar,igrid)
     end do
  end do
#endif

  msg_array=transfer(msg,msg_array)

end subroutine pack_fetch_cr
!#####################################################################
!#####################################################################
!#####################################################################
!#####################################################################
subroutine unpack_fetch_cr(mesh,igrid,msg_size,msg_array,hash_key)
  use amr_parameters, only: ndim, twotondim
  use hydro_parameters, only: nvar
  use cr_parameters, only: ncruvar
  use amr_commons, only: mesh_t
  use cache_commons, only: msg_realdp
  type(mesh_t)::mesh
  integer::igrid
  integer::msg_size
  integer,dimension(1:msg_size),optional::msg_array
  integer(kind=8),dimension(0:ndim)::hash_key

  integer::ind,ivar
  type(msg_realdp)::msg

  mesh%grid(igrid)%lev=hash_key(0)
  mesh%grid(igrid)%ckey(1:ndim)=hash_key(1:ndim)
  msg=transfer(msg_array,msg)

  do ind=1,twotondim
     if(msg%int4(ind)==1)then
        mesh%grid(igrid)%refined(ind)=.true.
     else
        mesh%grid(igrid)%refined(ind)=.false.
     endif
  end do

#ifdef HYDRO
  do ivar=1,nvar
     do ind=1,twotondim
        mesh%uold(ind,ivar,igrid)=msg%realdp(ind,ivar)
     end do
  end do
#endif

#ifdef CRS
  do ivar=1,ncruvar
     do ind=1,twotondim
        mesh%cruold(ind,ivar,igrid)=msg%realdp_cr(ind,ivar)
     end do
  end do
#endif

end subroutine unpack_fetch_cr
!#####################################################################
!#####################################################################
!#####################################################################
!#####################################################################
end module cr_flag_module
