#!/usr/bin/env python

import pandas as pd
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams["backend"] = "Qt4Agg"
plt.rcParams["figure.figsize"] = (20, 10)

reuse_window=0
reuse_rate=0.18
lossy_window=100000000
poisson_lambda=1

arf_cpp_path = "covtype/result-0.csv"
result_path = f"covtype/k0.4-e90/r{reuse_rate}-r{reuse_rate}-w{reuse_window}/lossy-{lossy_window}/"
pearl_path = f"{result_path}/result-0-{poisson_lambda}.csv"

propearl_path = f"{result_path}/result-pro-0-{poisson_lambda}.csv"

# arf_cpp = pd.read_csv(arf_cpp_path, index_col=0)
pearl = pd.read_csv(pearl_path, index_col=0)
propearl = pd.read_csv(propearl_path, index_col=0)

propearl_gain = sum(propearl["accuracy"]) - sum(pearl["accuracy"])
print(f"PEARL's cumulative accuracy gain: {propearl_gain}")

plt.plot(pearl["accuracy"], label="PEARL")
plt.plot(propearl["accuracy"], label="ProPEARL", linestyle="--")

plt.legend()
plt.xlabel("no. of instances")
plt.ylabel("accuracy")

plt.show()
# plt.savefig('covtype-results.png', bbox_inches='tight', dpi=100)
