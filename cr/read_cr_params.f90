module cr_params_module
contains
!###########################################################
!###########################################################
!###########################################################
!###########################################################
subroutine m_read_cr_params(pst)
  use amr_parameters
  use constants,only:c_cgs
  use hydro_parameters
  use cr_parameters
  use ramses_commons, only: pst_t
  use mdl_module
  use movie_module, only: set_movie_vars
  implicit none
  type(pst_t)::pst

  !--------------------------------------------------
  ! Local variables
  !--------------------------------------------------
  character(LEN=80)::infile
  logical::nml_ok

  !--------------------------------------------------
  ! CR namelist variables
  !--------------------------------------------------

  logical::cr_advect=.false.              ! Advection of cosmic rays?                       !
  logical::cr_streaming_diffusion=.false. ! Streaming diffusion of cosmic rays?             !
  logical::cr_streaming_heating=.false.   ! Streaming heating of cosmic rays?               !
  logical::cr_cooling=.false.             ! CR cooling?                                     !
  logical::cr_isotropic_pressure=.true.   ! Isotropic CR pressure?                          !
  logical::cr_varc=.false.                ! Vary the speed of light for CRs?                !
  logical::cr_varc_vdvs=.false.           ! Use diffusion and Alfven speed for cr_c         !
  logical::cr_reduced_flux_correction=.false.  ! Make sure F<c*E always?                    !
  real(kind=8)::cr_c_fraction=1.0
  real(kind=8)::cr_dmax=1.0               ! Max CR streaming diffusion coefficient in cgs   !
  integer::cr_nsubcycle=1                 ! Maximum number of CR subcycles per hydro step   !
  real(kind=8)::cr_courant_factor=0.8d0   ! Courant factor for CR timesteps                 !
  real(kind=8)::cr_varc_fudge=3.0
  real(kind=8)::cr_smallr_decouple=1d-4   ! Density (over smallr) at which to decouple CRs  !
  real(kind=8),dimension(1:ncrgrp)::cr_d=1.0d29
  real(kind=8),dimension(1:ncrgrp)::cr_d_perp_factors=1d-6 ! perp CR diffusion suppression  !
  real(kind=8),dimension(1:ncrgrp)::cr_gamma=4d0/3d0
  real(kind=8),dimension(1:ncrgrp)::fecr=0d0               ! SN fraction of CR energy       !
  real(kind=8),dimension(1:ncrgrp)::v_alfven=0.0 ! For idealised tests

  ! CR source regions parameters----------------------------------------------------------
  integer                           ::cr_nsource=0
  character(LEN=10),dimension(1:MAXREGION)::cr_source_type='square'
  real(kind=8),dimension(1:MAXREGION)   ::cr_src_x_center=0.
  real(kind=8),dimension(1:MAXREGION)   ::cr_src_y_center=0.
  real(kind=8),dimension(1:MAXREGION)   ::cr_src_z_center=0.
  real(kind=8),dimension(1:MAXREGION)   ::cr_src_length_x=1.E10
  real(kind=8),dimension(1:MAXREGION)   ::cr_src_length_y=1.E10
  real(kind=8),dimension(1:MAXREGION)   ::cr_src_length_z=1.E10
  real(kind=8),dimension(1:MAXREGION)   ::cr_exp_source=2.0
  integer, dimension(1:MAXREGION)       ::cr_src_group=1  
  real(kind=8),dimension(1:MAXREGION)   ::cr_e_source=0.                      ! CR density
  real(kind=8),dimension(1:MAXREGION)   ::cr_fx_source=0.                     ! CR flux
  real(kind=8),dimension(1:MAXREGION)   ::cr_fy_source=0.                     ! CR flux
  real(kind=8),dimension(1:MAXREGION)   ::cr_fz_source=0.                     ! CR flux

  !--------------------------------------------------
  ! Namelist definitions
  !--------------------------------------------------
  namelist/cr_params/ &
       &  cr_advect, cr_streaming_diffusion, cr_streaming_heating        &
       & ,cr_cooling, cr_isotropic_pressure, cr_varc, cr_varc_vdvs       &
       & ,cr_reduced_flux_correction, cr_c_fraction, cr_dmax             &
       & ,cr_nsubcycle, cr_courant_factor, cr_varc_fudge                 &
       & ,cr_smallr_decouple

  namelist/cr_groups/cr_d, cr_d_perp_factors, cr_gamma, fecr, v_alfven 

  namelist/cr_sources/cr_nsource, cr_source_type                         &
       & ,cr_src_x_center, cr_src_y_center, cr_src_z_center              &
       & ,cr_src_length_x, cr_src_length_y, cr_src_length_z              &
       & ,cr_exp_source, cr_src_group                                    &
       & ,cr_e_source, cr_fx_source, cr_fy_source, cr_fz_source

  associate(s=>pst%s)

  write(*,'(" Working with ",I2," CR groups ")') ncrgrp

  if(ncrgrp .le. 0) then
     s%r%cr = .false.
     write(*,'(" Turning off CRs, since no CR groups ")')
     return
  endif

  !--------------------------------------------------
  ! Set defaults for CR groups
  !--------------------------------------------------

  ! ...

  !-------------------------------------------------
  ! Read the namelist file
  !-------------------------------------------------
  CALL getarg(1,infile)
  namelist_file=TRIM(infile)
  INQUIRE(file=infile,exist=nml_ok)
  if(.not. nml_ok)then
     write(*,*)'File '//TRIM(infile)//' does not exist'
     call mdl_abort(s%mdl)
  end if
  open(1,file=infile)
  rewind(1)
  read(1,NML=cr_params,END=113)
113 continue
  rewind(1)
  read(1,NML=cr_groups,END=114)
114 continue
  rewind(1)
  read(1,NML=cr_sources,END=115)
115 continue
  close(1)

  ! Fill in all run parameters in corresponding structure
  s%r%cr_advect=cr_advect
  s%r%cr_streaming_diffusion=cr_streaming_diffusion
  s%r%cr_streaming_heating=cr_streaming_heating
  s%r%cr_cooling=cr_cooling
  s%r%cr_isotropic_pressure=cr_isotropic_pressure
  s%r%cr_varc=cr_varc
  s%r%cr_varc_vdvs=cr_varc_vdvs
  s%r%cr_reduced_flux_correction=cr_reduced_flux_correction
  s%r%cr_c_fraction=cr_c_fraction
  s%r%cr_dmax=cr_dmax
  s%r%cr_nsubcycle=cr_nsubcycle
  s%r%cr_courant_factor=cr_courant_factor
  s%r%cr_varc_fudge=cr_varc_fudge
  s%r%cr_smallr_decouple=cr_smallr_decouple

  s%r%cr_d=cr_d
  s%r%cr_d_perp_factors=cr_d_perp_factors
  s%r%cr_gamma=cr_gamma
  s%r%fecr=fecr
  s%r%v_alfven=v_alfven

  s%r%cr_nsource=cr_nsource
  s%r%cr_source_type=cr_source_type
  s%r%cr_src_x_center=cr_src_x_center
  s%r%cr_src_y_center=cr_src_y_center
  s%r%cr_src_z_center=cr_src_z_center
  s%r%cr_src_length_x=cr_src_length_x
  s%r%cr_src_length_y=cr_src_length_y
  s%r%cr_src_length_z=cr_src_length_z
  s%r%cr_exp_source=cr_exp_source
  s%r%cr_src_group=cr_src_group
  s%r%cr_e_source=cr_e_source
  s%r%cr_fx_source=cr_fx_source
  s%r%cr_fy_source=cr_fy_source
  s%r%cr_fz_source=cr_fz_source

  end associate

end subroutine m_read_cr_params
!#########################################################################
!#########################################################################
!#########################################################################
!#########################################################################
end module cr_params_module
