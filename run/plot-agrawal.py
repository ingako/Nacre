#!/usr/bin/env python

import pandas as pd
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams["backend"] = "Qt4Agg"
plt.rcParams["figure.figsize"] = (20, 10)

kappa = 0.1
ed = 100

reuse_window = 0
reuse_rate = 0.9
lossy_window = 100000000
generator_traits = "abrupt"
arf_result_path = f"agrawal/{generator_traits}"
result_path = f"agrawal/{generator_traits}/k{kappa}-e{ed}/r{reuse_rate}-r{reuse_rate}-w{reuse_window}/lossy-{lossy_window}/"
poisson_lambda = 6

arf_path = f"{arf_result_path}/result-0-{poisson_lambda}.csv"
pearl_path = f"{result_path}/result-0-{poisson_lambda}.csv"
propearl_path = f"{result_path}/result-pro-0-{poisson_lambda}.csv"

# left, right = 0, 500
# arf = pd.read_csv(arf_path, index_col=0).iloc[left:right]
# pearl = pd.read_csv(pearl_path, index_col=0).iloc[left:right]
# propearl = pd.read_csv(propearl_path, index_col=0).iloc[left:right]
arf = pd.read_csv(arf_path, index_col=0)
pearl = pd.read_csv(pearl_path, index_col=0)
propearl = pd.read_csv(propearl_path, index_col=0)

pearl_gain = sum(propearl["accuracy"]) - sum(pearl["accuracy"])
arf_gain = sum(propearl["accuracy"]) - sum(arf["accuracy"])
gain = sum(pearl["accuracy"]) - sum(arf["accuracy"])

print(f'| propearl-pearl | {int(round(pearl_gain * 100))}% | {int(round(propearl["time"].iloc[-1]))} |')
print(f'| propearl-arf | {int(round(arf_gain * 100))}% | {int(round(propearl["time"].iloc[-1]))} |')
print(f'| pearl-arf | {int(round(gain * 100))}% | {int(round(propearl["time"].iloc[-1]))} |')

plt.plot(arf["accuracy"], label="arf", linestyle='-')
plt.plot(pearl["accuracy"], label="PEARL", linestyle="--")
plt.plot(propearl["accuracy"], label="ProPEARL")

plt.legend()
plt.xlabel("no. of instances")
plt.ylabel("accuracy")

plt.show()
# plt.savefig('covtype-results.png', bbox_inches='tight', dpi=100)
