module cr_parameters
  use amr_parameters, only: ndim

#ifdef NCRGRP
  integer,parameter::ncrgrp=NCRGRP             ! # of CR groups (set in Makefile)
#else
  integer,parameter::ncrgrp=1
#endif
  integer,parameter::ncrvar=ncrgrp*(1+ndim)    ! # of CR variables (CR density and flux)

  real(kind=8),parameter::smallEcr=1d-50        !  Minimum CR density

end module cr_parameters
