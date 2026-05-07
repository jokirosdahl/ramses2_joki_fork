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

  !--------------------------------------------------
  ! CR namelist variables
  !--------------------------------------------------

  logical::cr_advect=.false.              ! Advection of cosmic rays?                       !
  logical::cr_streaming_diffusion=.false. ! Streaming diffusion of cosmic rays?             !
  logical::cr_streaming_heating=.false.   ! Streaming heating of cosmic rays?               !
  logical::cr_isotropic_pressure=.false.  ! Isotropic CR pressure?                          !
  real(dp)::cr_c_fraction=1.0       
  real(dp)::cr_dmax=1.0                   ! Maximum allowed CR streaming diffusion coefficient in cgs
  real(dp),dimension(1:ncrgrp)::cr_d=1.0d29 ! Classical value, in cm^2/s (e.g., Jockipii 1999)
  real(dp),dimension(1:ncrgrp)::cr_d_perp_factors=1d-6 ! perpendicular diffusion suppression of CRs
  real(dp),dimension(1:ncrgrp)::v_alfven=0.0 ! For idealised tests
  real(dp),dimension(1:ncrgrp)::fecr=0d0             ! SN fraction of CR energy

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
  namelist/mhd_params/ &
       &  cr_advect, cr_streaming_diffusion, cr_streaming_heating &
       & ,cr_dmax, cooling_cr, cr_c_fraction, cr_varvmax &
       & ,cr_varvmax_fudge, cr_varvmax_vdvs, cr_nsubcycle, cr_smallr_decouple &
       & ,reduced_CR_flux_correction, &
       & ,cr_isotropic_pressure &
  namelist/cr_sources/cr_nsource, cr_source_type                  &
       & ,cr_src_x_center, cr_src_y_center, cr_src_z_center              &
       & ,cr_src_length_x, cr_src_length_y, cr_src_length_z              &
       & ,cr_exp_source, cr_src_group                                    &
       & ,cr_n_source, cr_u_source, cr_v_source, cr_w_source
  namelist/cr_groups/cr_d, cr_d_perp_factor, gamma_cr, fecr, v_alfven 

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


  namelist/cr_sources/cr_nsource, cr_source_type                  &
       & ,cr_src_x_center, cr_src_y_center, cr_src_z_center              &
       & ,cr_src_length_x, cr_src_length_y, cr_src_length_z              &
       & ,cr_exp_source, cr_src_group                                    &
       & ,cr_n_source, cr_u_source, cr_v_source, cr_w_source
  namelist/cr_groups/cr_d, cr_d_perp_factor, gamma_cr, fecr, v_alfven 

  ! Fill in all run parameters in corresponding structure
  s%r%cr_advect=cr_advect
  s%r%cr_streaming_diffusion=cr_streaming_diffusion
  s%r%cr_streaming_heating=cr_streaming_heating
  s%r%cr_dmax=cr_dmax
  s%r%cooling_cr=cooling_cr
  s%r%cr_c_fraction=cr_c_fraction
  s%r%cr_varvmax=cr_varvmax
  s%r%cr_varvmax_fudge=cr_varvmax_fudge
  s%r%cr_nsubcycle=cr_nsubcycle
  s%r%cr_smallr_decouple=cr_smallr_decouple
  s%r%reduced_CR_flux_correction=reduced_CR_flux_correction
  s%r%cr_HLLE=cr_HLLE
  s%r%cr_isotropic_pressure=cr_isotropic_pressure

  s%r%cr_d=cr_d
  s%r%cr_d_perp_factors=cr_d_perp_factors
  s%r%gamma_cr=gamma_cr
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
