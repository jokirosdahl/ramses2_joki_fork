!#########################################################################
!#########################################################################
!#########################################################################
!#########################################################################
subroutine m_cr_init_flow_fine(pst,ilevel)
  use amr_commons, only: ndim
  use amr_parameters, only: nvector, twotondim
  use ramses_commons, only: pst_t
  use cr_input_condinit_module, only: r_cr_input_condinit
  use cr_parameters, only: ncrgrp
  implicit none
  type(pst_t)::pst
  integer::ilevel
  !--------------------------------------------------------------------
  ! This routine is the master procedure to input a given initial 
  ! condition for CR variables into the exisiting AMR structure from an
  ! analytical model given as an external routine.
  !--------------------------------------------------------------------
  character(len=80)::filename
  logical::ok_file1,ok_file2,ok_file

  associate(s=>pst%s,r=>pst%s%r,m=>pst%s%m)

  if(s%m%noct_tot(ilevel)==0)return
  if(s%r%verbose)write(*,111)ilevel
111 format(' Entering init_cr_fine for level ',I2)

  ! Use internal-defined or user-defined functions.
  ! We always call the condinit routine, even in cosmological simulations, 
  ! just to initialisethe RT variables (to small values)
  if(s%r%verbose)write(*,*)'Initialising CR variables'
  call r_cr_input_condinit(pst,ilevel,1)

  filename=TRIM(s%r%initfile(ilevel))//'/ic_d'
  INQUIRE(file=filename,exist=ok_file1)
  filename=TRIM(s%r%initfile(ilevel))//'/ic_deltab'
  INQUIRE(file=filename,exist=ok_file2)
  ok_file=ok_file1.or.ok_file2
  if(ok_file)then
     ! No initialization necessary for photons
  else if (s%r%filetype=='gadget')then
     ! No initialization necessary for photons
  endif

  end associate

end subroutine m_cr_init_flow_fine
!###############################################
!###############################################
!###############################################
!###############################################
module update_cr_c_module
  type :: in_broadcast_cr_c
     real(kind=8)::rt_c
  end type in_broadcast_cr_c
contains
!##############################################################
!##############################################################
!##############################################################
!##############################################################
recursive subroutine r_cr_updates(pst, nstep_coarse, input_size)
  use mdl_module
  use ramses_commons, only: pst_t
  use mdl_parameters
  implicit none
  type(pst_t)::pst
  integer,VALUE::input_size
  integer::nstep_coarse, rID

  associate(s=>pst%s,r=>pst%s%r,m=>pst%s%m,g=>pst%s%g)

  if(pst%nLower>0)then
     rID = mdl_send_request(pst%s%mdl,MDL_CR_UPDATES,pst%iUpper+1,input_size,0,nstep_coarse)
     call r_cr_updates(pst%pLower, nstep_coarse, input_size)
     call mdl_get_reply(pst%s%mdl,rID,0)
  else
     ! Update reduced speed of light
     if(r%cr .and. r%cosmo) call update_cr_c(r, g)
  endif

  end associate

end subroutine r_cr_updates
!##############################################################
!##############################################################
!##############################################################
!##############################################################
SUBROUTINE update_cr_c(r, g)
  ! Update CR speed of light in code units
  !-------------------------------------------------------------------------
  use amr_commons, only: run_t, global_t
  implicit none
  type(run_t) :: r
  type(global_t) :: g
  real(kind=8)::scale_nH,scale_T2,scale_l,scale_d,scale_t,scale_v
  integer::i
  !-------------------------------------------------------------------------
  call units(r,g,scale_l,scale_t,scale_d,scale_v,scale_nH,scale_T2)
  do i=r%nlevelmax,r%levelmin,-1
    g%cr_vmax_cgs(i) = c_cgs * r%cr_c_fraction(i)
    g%cr_vmax(i) = g%cr_vmax_cgs(i) / scale_v
  enddo

END SUBROUTINE update_cr_c


end module update_cr_c_module
