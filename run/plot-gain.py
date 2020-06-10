#!/usr/bin/env python3

import sys
import pandas as pd
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['figure.dpi'] = 400
plt.rcParams["legend.loc"] = 'lower right'

# fontsize = 10
# plt.tick_params(labelsize=fontsize)
plt.tick_params()

plt.locator_params(axis='y', nbins=6)
plt.locator_params(axis='x', nbins=3)


reuse_window = 0
reuse_rate = 0.9
lossy_window = 100000000

# usage
# ./plot-gain.py agrawal abrupt/poisson10

generator = sys.argv[1]# "agrawal"
generator_traits = sys.argv[2] # "abrupt/poisson10"


seq_len=0
backtrack_window=0
pro_drift_window=0
stability=0
hybrid=0

if generator == "agrawal":
    seed = 7
    kappa = 0.1
    ed = 100

    if "abrupt" in generator_traits:
        plt.ylim(0, 6500)

        # "agrawal/abrupt/poisson10" : ['8', '25', '200', '0.01', '0.9'],
        seq_len=8
        backtrack_window=25
        pro_drift_window=200
        stability=0.01
        hybrid=0.9
        print(f"generator_traits: agrawal/abrupt")
    elif "gradual" in generator_traits:
        plt.ylim(0, 4500)

        # "agrawal/gradual/poisson10" : ['8', '25', '400', '0.01', '0.9']
        seq_len=8
        backtrack_window=25
        pro_drift_window=400
        stability=0.01
        hybrid=0.9
        print(f"generator_traits: agrawal/gradual")
    else:
        print("Unknow agrawal generator_traits")
elif generator == "tree":
    plt.ylim(0, 1100)

    seed = 7
    kappa = 0.0
    ed = 100
    if "abrupt" in generator_traits:
        # "tree/abrupt/poisson10": ['8', '25', '300', '0.01', '0.9'],
        seq_len=8
        backtrack_window=25
        pro_drift_window=300
        stability=0.01
        hybrid=0.9

    elif "gradual" in generator_traits:
        # "tree/gradual/poisson10": ['8', '25', '300', '0.1', '0.9'],
        seq_len=8
        backtrack_window=25
        pro_drift_window=300
        stability=0.1
        hybrid=0.9

    else:
        print("Unknow tree generator_traits")

else:
    print("Unknow generator")
    exit()

arf_result_path = f"{generator}/{generator_traits}"
result_path = f"{generator}/{generator_traits}/" \
              f"k{kappa}-e{ed}/r{reuse_rate}-r{reuse_rate}-w{reuse_window}/" \
              f"lossy-{lossy_window}/"
poisson_lambda = 6

arf_path = f"{arf_result_path}/result-{seed}-{poisson_lambda}.csv"
pearl_path = f"{result_path}/result-{seed}-{poisson_lambda}.csv"

propearl_path = f"{result_path}/nacre/{seq_len}/{backtrack_window}/" \
                f"{pro_drift_window}/{stability}/{hybrid}/result-pro-{seed}-{poisson_lambda}.csv"

arf = pd.read_csv(arf_path, index_col=0)
pearl = pd.read_csv(pearl_path, index_col=0)
propearl = pd.read_csv(propearl_path, index_col=0)


total_gain = 0
for i in range(len(propearl)):
    propearl["accuracy"].iloc[i] = \
        propearl["accuracy"].iloc[i] - arf["accuracy"].iloc[i] \
        + total_gain
    total_gain = propearl["accuracy"].iloc[i]

total_gain = 0
for i in range(len(propearl)):
    pearl["accuracy"].iloc[i] = \
        pearl["accuracy"].iloc[i] - arf["accuracy"].iloc[i] \
        + total_gain
    total_gain = pearl["accuracy"].iloc[i]

propearl["accuracy"] = propearl["accuracy"] * 100
pearl["accuracy"] = pearl["accuracy"] * 100


plt.plot(propearl["accuracy"], label="Nacre", linestyle="--")
plt.plot(pearl["accuracy"], label="PEARL")
plt.legend()

# plt.title("")
plt.xlabel("no. of instances") #, size=fontsize)
plt.ylabel("Cumulative Accuracy Gain %") #, size=fontsize)

# plt.xlim(0, 10)

# plt.legend(prop={'size': fontsize})
plt.legend(loc="upper left")
plt.tight_layout()
# plt.show()

plt.savefig(f'{generator}-gain/{generator_traits}.eps', dpi = (400))
