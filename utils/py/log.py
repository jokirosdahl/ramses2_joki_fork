#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import FortranFile
from astropy.io import ascii
import argparse
import os
import re

parser = argparse.ArgumentParser()
parser.add_argument("file", help="enter filename run.log")
parser.add_argument("--log", help="plot log variable",action="store_true")
parser.add_argument("--sym", help="use a circle for each cell",action="store_true")
parser.add_argument("--out", help="output a png image")
args = parser.parse_args()
print("Reading "+args.file)

# path the the file
path_to_output = args.file

# read log file
emag_vals = []
time_vals = []
dt_vals = []
aexp_vals = []

do_mag = False
do_time = False

main_step_seen = False
emag_val = None
lines_since_main = 0

with open(path_to_output, "r") as f:
    for line in f:
        if "Main step" in line:
            main_step_seen = True
            lines_since_main = 0
            emag_match = re.search(r"\bemag\s*=\s*(\S+)", line)
            if emag_match:
                try:
                    emag_val = float(emag_match.group(1).replace('D', 'E').replace('d', 'e'))
                except ValueError:
                    emag_val = None
            else:
                emag_val = None
            continue

        if main_step_seen:
            lines_since_main += 1
            if "Fine step" in line:
                t_match = re.search(r"\bt\s*=\s*(\S+)", line)
                dt_match = re.search(r"\bdt\s*=\s*(\S+)", line)
                a_match = re.search(r"\ba\s*=\s*(\S+)", line)

                if t_match and dt_match and a_match:
                    try:
                        t_val = float(t_match.group(1).replace('D', 'E').replace('d', 'e'))
                        dt_val = float(dt_match.group(1).replace('D', 'E').replace('d', 'e'))
                        a_val = float(a_match.group(1).replace('D', 'E').replace('d', 'e'))

                        time_vals.append(t_val)
                        dt_vals.append(dt_val)
                        aexp_vals.append(a_val)
                        if emag_val is not None:
                            emag_vals.append(emag_val)
                            do_mag = True
                        do_time = True
                    except ValueError:
                        pass
                main_step_seen = False
            elif lines_since_main >= 4:
                main_step_seen = False

emag = np.array(emag_vals)
time = np.array(time_vals)
dt = np.array(dt_vals)
aexp = np.array(aexp_vals)

if len(emag_vals) == 0:
    print("no magnetic energy")
    do_mag = False
if len(time_vals) == 0:
    print("no proper time")
    do_time = False

if do_mag:
    if args.sym:
        plt.plot(time,emag,"o")
    else:
        plt.plot(time,emag)
    if args.log:
        plt.yscale("log")
    plt.xlabel('time')
    plt.ylabel('magnetic energy')

    if args.out:
        plt.savefig(args.out)
    plt.show()

if do_time:
    if args.sym:
        plt.plot(time,dt,"o")
    else:
        plt.plot(time,dt)
else:
    if args.sym:
        plt.plot(aexp,dt,"o")
    else:
        plt.plot(aexp,dt)
    
if args.log:
    plt.yscale("log")
plt.xlabel('time')
plt.ylabel('time step')

if args.out:
    plt.savefig(args.out)
plt.show()


