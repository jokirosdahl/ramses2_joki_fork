module cr_step_module
contains
!#####################################################################
!#####################################################################
!#####################################################################
!#####################################################################
subroutine m_cr_step(pst,ilevel)
  use ramses_commons, only: pst_t
  use cr_godunov_fine_module, only: r_cr_godunov_fine,r_set_crunew,r_set_cruold
  use cr_source_terms_module, only: r_cr_source_terms
  use cr_upload_module, only: m_cr_upload_fine
  use cr_input_condinit_module, only: r_cr_input_source_regions  
  type(pst_t)::pst
  integer::ilevel

  real(kind=8) :: dt_save, t_save
  real(kind=8) :: dt_cr, t_cr
  integer  :: i, i_substep

  associate(r=>pst%s%r,g=>pst%s%g,m=>pst%s%m,mdl=>pst%s%mdl)

  ! Store hydro timestep length and hydro time
  dt_save = g%dtnew(ilevel)
  t_save = g%t

  ! We shift the time backwards one hydro-dt, just in case,
  ! though not really needed for now.
  t_cr = t_save - dt_save

  ! Get CR courant time step at coarse level
  call get_cr_courant_dt(r,g,dt_cr,ilevel)
  ! Compute CR courant time step at fine level
  dt_cr = MIN(dt_cr,dt_save)

  ! Compute number of subcycles
  i_substep = ceiling(dt_save/dt_cr-1e-12)
  dt_cr = dt_save/dble(i_substep)

  ! CR sub-cycle loop
  do i = 1, i_substep
     if(i_substep .gt. r%cr_nsubcycle) then
         print*,'Doing CR substeps but should not!! ',i_substep, r%cr_nsubcycle, dt_save
         stop
     endif

     ! Shift the CR time forwards one dt_cr
     t_cr = t_cr + dt_cr

     ! Update time variables
     call m_cr_update_time(pst,ilevel,t_cr,dt_cr)

     ! Set crunew equal to cruold
     if(i>1)call r_set_crunew(pst,ilevel,1)

     ! Hyperbolic CR solver
     if(r%cr_advect)call r_cr_godunov_fine(pst,ilevel,1)
     if(r%cr_advect)call r_cr_source_terms(pst,ilevel,1)

     ! Add anisotropic radiation from other sources
     if(r%cr_nsource>0)call r_cr_input_source_regions(pst,ilevel,1)

     ! Set cruold equal to crunew
     call r_set_cruold(pst,ilevel,1)

  end do
  ! End CR subcycle loop

  ! Restore original hydro timestep and time
  call m_cr_update_time(pst,ilevel,t_save,dt_save)

  ! Restriction operator to update coarser level split cells
  call m_cr_upload_fine(pst,ilevel)

  if (g%myid==1 .and. r%cr_nsubcycle .gt. 1) write(*,901) ilevel, i_substep
901 format (' Performed level', I3, ' CR-step with ', I5, ' subcycles')

  end associate

end subroutine m_cr_step
!#####################################################################
!#####################################################################
!#####################################################################
!#####################################################################
subroutine m_cr_update_time(pst,ilevel,t,dt)
  use amr_parameters, only: n_frw
  use ramses_commons, only: pst_t
  use update_time_module, only: in_broadcast_aexp_t, r_broadcast_aexp
  use newdt_fine_module, only: in_broadcast_dt_t, r_broadcast_dt
  use mdl_module
  implicit none
  type(pst_t)::pst
  integer::ilevel
  real(kind=8)::dt,t

  type(in_broadcast_dt_t)::in_broadcast_dt
  type(in_broadcast_aexp_t)::in_broadcast_aexp
  integer::i

  associate(r=>pst%s%r,g=>pst%s%g,m=>pst%s%m,p=>pst%s%p,mdl=>pst%s%mdl)

  ! Store time step to global variable
  g%dtnew(ilevel)=dt

  ! Broadcast dtnew to all CPUs
  in_broadcast_dt%ilevel=ilevel
  in_broadcast_dt%dtnew=g%dtnew(ilevel)
  in_broadcast_dt%dtold=g%dtold(ilevel)
  call r_broadcast_dt(pst,in_broadcast_dt,storage_size(in_broadcast_dt)/32)

  ! Store time to global variable
  g%t=t

  ! Compute expansion factor, hubble rate and proper time
  if(r%cosmo)then
     ! Find neighboring times
     i=1
     do while(g%tau_frw(i)>g%t.and.i<n_frw)
        i=i+1
     end do
     ! Interpolate expansion factor
     g%aexp = g%aexp_frw(i  )*(g%t-g%tau_frw(i-1))/(g%tau_frw(i  )-g%tau_frw(i-1))+ &
            & g%aexp_frw(i-1)*(g%t-g%tau_frw(i  ))/(g%tau_frw(i-1)-g%tau_frw(i  ))
     g%hexp = g%hexp_frw(i  )*(g%t-g%tau_frw(i-1))/(g%tau_frw(i  )-g%tau_frw(i-1))+ &
            & g%hexp_frw(i-1)*(g%t-g%tau_frw(i  ))/(g%tau_frw(i-1)-g%tau_frw(i  ))
     g%texp =    g%t_frw(i  )*(g%t-g%tau_frw(i-1))/(g%tau_frw(i  )-g%tau_frw(i-1))+ &
            &    g%t_frw(i-1)*(g%t-g%tau_frw(i  ))/(g%tau_frw(i-1)-g%tau_frw(i  ))
  else
     g%aexp = 1.0
     g%hexp = 0.0
     g%texp = g%t
  end if

  ! Broadcast t, aexp, texp and hexp to all CPUs
  in_broadcast_aexp%t=g%t
  in_broadcast_aexp%texp=g%texp
  in_broadcast_aexp%aexp=g%aexp
  in_broadcast_aexp%hexp=g%hexp
  call r_broadcast_aexp(pst,in_broadcast_aexp,storage_size(in_broadcast_aexp)/32)

  end associate

end subroutine m_cr_update_time
!#####################################################################
!#####################################################################
!#####################################################################
!#####################################################################
end module cr_step_module
