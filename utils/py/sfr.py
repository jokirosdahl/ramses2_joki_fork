#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import argparse
import miniramses as ram

parser = argparse.ArgumentParser()
parser.add_argument("nout", help="enter output number")
parser.add_argument("--path", help="specify a path")
parser.add_argument("--log", help="plot log SFR",action="store_true")
parser.add_argument("--out", help="output a png image")
parser.add_argument("--xcen", help="specify the region center x-coordinate")
parser.add_argument("--ycen", help="specify the region center y-coordinate")
parser.add_argument("--zcen", help="specify the region center z-coordinate")
parser.add_argument("--rad", help="specify the region radius")
parser.add_argument("--bin", help="specify the bin size in Gyr")
args = parser.parse_args()
# path the the file
path = args.path
radius = args.rad
xcenter = args.xcen
ycenter = args.ycen
zcenter = args.zcen
log = args.log
dt = args.bin

if path==None:
    path="./"
else:
    path=path+"/"
if xcenter==None:
    xcenter=None
else:
    xcenter=float(xcenter)
if ycenter==None:
    ycenter=None
else:
    ycenter=float(ycenter)
if zcenter==None:
    zcenter=None
else:
    zcenter=float(zcenter)
if radius==None:
    radius=None
else:
    radius=float(radius)
center=np.array([xcenter,ycenter,zcenter])
if dt==None:
    bin_size=0.1  #in Gyr
else:
    bin_size=float(dt)

nout = args.nout
print("Reading output number ",nout)

s=ram.rd_part(nout,path=path,prefix='star',center=center,radius=radius)
i=ram.rd_info(nout,path=path)
is_cosmo = bool(np.any(s.birth_date < 0))

if is_cosmo:
    time = abs(s.birth_date * i.unit_t / i.aexp**2 / (365 * 24 * 3600 * 1e9))
    xlabel = 'Lookback time [Gyr]'
else:
    time = s.birth_date * i.unit_t / (365 * 24 * 3600 * 1e9)
    xlabel = 'Time [Gyr]'

if np.max(time) < bin_size:
    print("Reduce bin size, bin=", bin_size, " max(time)=", np.max(time))
    exit()

n_bin = int(np.max(time) / bin_size)
bins = np.linspace(0, np.max(time), n_bin)

if is_cosmo:
    hist, _ = np.histogram(time, bins=bins)
    imin = np.where(hist > 0)[0][0]
    bins = bins[imin:]

unit_m = i.unit_d * i.unit_l**3 / 2e33 / (bins[1] - bins[0]) / 1e9

fig, ax1 = plt.subplots()
sfr, _, _ = ax1.hist(time, weights=s.mass * unit_m, bins=bins)
if log:
    ax1.set_yscale("log")
ax1.set_xlabel(xlabel)
ax1.set_ylabel('SFR [Msol/yr]')

if is_cosmo:
    cum_mass = np.cumsum(sfr[::-1])[::-1] * (bins[1] - bins[0]) * 1e9
else:
    cum_mass = np.cumsum(sfr) * (bins[1] - bins[0]) * 1e9

time_bins = 0.5 * (bins[:-1] + bins[1:])

ax2 = ax1.twinx()
ax2.plot(time_bins, cum_mass, color='r')
ax2.set_ylabel('Cumulative Mass [Msol]')
if log:
    ax2.set_yscale("log")

ax1.set_xlim(bins[0], bins[-1])

if args.out:
    plt.savefig(args.out)

plt.show()
