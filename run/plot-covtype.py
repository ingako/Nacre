#!/usr/bin/env python

import pandas as pd
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams["backend"] = "Qt4Agg"
plt.rcParams["figure.figsize"] = (20, 10)

dataset="covtype"
reuse_window=0
reuse_rate=0.18
lossy_window=100000000
poisson_lambda=1

arf_path = f"{dataset}/result-0-{poisson_lambda}.csv"
result_path = f"{dataset}/k0.4-e90/r{reuse_rate}-r{reuse_rate}-w{reuse_window}/lossy-{lossy_window}/"
pearl_path = f"{result_path}/result-0-{poisson_lambda}.csv"
propearl_path = f"{result_path}/result-pro-0-{poisson_lambda}.csv"

arf = pd.read_csv(arf_path, index_col=0)
pearl = pd.read_csv(pearl_path, index_col=0)
propearl = pd.read_csv(propearl_path, index_col=0)


pearl_gain = sum(pearl["accuracy"]) - sum(arf["accuracy"])
pearl_runtime = round(pearl["time"].iloc[-1]/60, 2)
pearl_tree = pearl["tree_pool_size"].iloc[-1]

propearl_gain = sum(propearl["accuracy"]) - sum(arf["accuracy"])
propearl_runtime = round(propearl["time"].iloc[-1]/60, 2)
propearl_tree = propearl["tree_pool_size"].iloc[-1]

arf_runtime = round(arf["time"].iloc[-1]/60, 2)

print(f" - & {arf_runtime} & 60")
print(f"{int(round(pearl_gain * 100))}\% & {pearl_runtime} & {pearl_tree}")
print(f"{int(round(propearl_gain * 100))}\% & {propearl_runtime} & {propearl_tree}")

propearl_pearl_gain = sum(propearl["accuracy"]) - sum(pearl["accuracy"])
print(f"ProPEARL - PEARL: {propearl_pearl_gain}")


# propearl_gain = sum(propearl["accuracy"]) - sum(pearl["accuracy"])
# print(f"PEARL's cumulative accuracy gain: {propearl_gain}")

plt.plot(pearl["accuracy"], label="PEARL")
plt.plot(propearl["accuracy"], label="ProPEARL", linestyle="--")

plt.legend()
plt.xlabel("no. of instances")
plt.ylabel("accuracy")

plt.show()
# plt.savefig('covtype-results.png', bbox_inches='tight', dpi=100)
