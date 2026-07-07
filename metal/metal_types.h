#pragma once
/*
 * C/MSL mirror of Fortran type oct (oct_commons.f90), compiled for NDIM=3.
 *
 * Fortran layout (gfortran, NDIM=3, twotondim=8, nhilbert=1):
 *   integer(kind=8) hkey(1)       ! offset  0, 8 bytes
 *   integer(kind=4) ckey(3)       ! offset  8, 12 bytes
 *   logical         refined(8)    ! offset 20, 32 bytes  (logical = 4-byte default)
 *   integer(kind=4) lev           ! offset 52, 4 bytes
 *   integer(kind=4) superoct      ! offset 56, 4 bytes
 *                                 ! total : 60 bytes, no padding
 */
typedef struct {
    long hkey;        /* integer(kind=8), nhilbert=1 */
    int  ckey[3];     /* integer(kind=4), ndim=3     */
    int  refined[8];  /* logical (4-byte), twotondim=8 */
    int  lev;
    int  superoct;
} oct_t;
