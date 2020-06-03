#!/usr/bin/env python3

import os
import sys
import math
import pandas as pd
import numpy as np
from dataclasses import dataclass
from pprint import PrettyPrinter


param_strs = ["seq", "backtrack", "adapt_window", "stability", "hybrid"]
metric_strs = ["Acc", "Kappa", "Gain per Drift", "Cum. Acc. Gain", "Runtime", "#Trees"]


@dataclass
class Param:
    generator: str = "covtype"
    seed: int = 0
    kappa: float = 0.0
    ed: int = 90
    reuse_window_size: int = 0
    reuse_rate: float = 0.18
    lossy_window_size: int = 100000000
    poisson_lambda: int = 1
    kappa_window: int = 50


def is_empty_file(fpath):
    return False if os.path.isfile(fpath) and os.path.getsize(fpath) > 0 else True

def get_metrics(df, gain_per_drift, gain):
    return [df["accuracy"].mean()*100, df["kappa"].mean()*100,
            gain_per_drift * 100, gain*100,
            df["time"].iloc[-1]/60, df["tree_pool_size"].iloc[-1]]

def eval_nacre_output(cur_data_dir, param_values, nacre_metrics_dict,
                      arf_acc, pearl_acc, pearl_acc_per_drift_mean, p):

    if len(param_values) != len(param_strs):
        # recurse
        params = [f for f in os.listdir(cur_data_dir) if os.path.isdir(os.path.join(cur_data_dir, f))]
        print(f"evaluating {params}...")

        for cur_param in params:
            param_values.append(cur_param)
            metrics = eval_nacre_output(f"{cur_data_dir}/{cur_param}",
                                        param_values,
                                        nacre_metrics_dict,
                                        arf_acc,
                                        pearl_acc,
                                        pearl_acc_per_drift_mean,
                                        p)
            param_values.pop()

    else:

        nacre_acc_per_drift = pd.read_csv(
                f"{cur_data_dir}/acc-per-drift-{seed}.log", header=None)
        nacre_gain_per_drift= nacre_acc_per_drift.mean() - pearl_acc_per_drift_mean

        nacre_output = f"{cur_data_dir}/result-pro-{seed}-{p.poisson_lambda}.csv"

        if is_empty_file(nacre_output):
            print(f"{nacre_output} is empty")
            return

        nacre_df = pd.read_csv(nacre_output)
        nacre_acc = nacre_df["accuracy"]

        num_instances = nacre_df["count"]

        #end = min(int(sys.argv[2]), min(len(nacre_acc), len(arf_acc)))

        nacre_arf_gain, nacre_pearl_gain = 0, 0
        for i in range(0, int(sys.argv[2])):
            nacre_arf_gain += nacre_acc[i] - arf_acc[i]
            nacre_pearl_gain += nacre_acc[i] - pearl_acc[i]

        nacre_metrics = get_metrics(nacre_df, nacre_gain_per_drift, nacre_pearl_gain)

        key = tuple(v for v in param_values)
        if key in nacre_metrics_dict:
            for i in range(len(metric_strs)):
                nacre_metrics_dict[key][i].append(nacre_metrics[i])
        else:
            nacre_metrics_dict[key] = [[v] for v in nacre_metrics]


def get_metrics_for_seed(seed,
                         arf_metrics_dict,
                         pearl_metrics_dict,
                         nacre_metrics_dict):

    base_dir = os.getcwd()
    generator = sys.argv[1]

    agrawal_params = Param(
            generator = generator,
            seed = seed,
            kappa=0.1,
            ed=100,
            reuse_window_size=0,
            reuse_rate=0.9,
            lossy_window_size=100000000,
            poisson_lambda=6,
            kappa_window=50)
    p = agrawal_params

    arf_data_dir = f"{base_dir}/{generator}"
    pearl_data_dir = f"{arf_data_dir}/k{p.kappa}-e{p.ed}/r{p.reuse_rate}-r{p.reuse_rate}-w{p.reuse_window_size}/lossy-{p.lossy_window_size}/"

    arf_output  = f"{arf_data_dir}/result-{seed}-{p.poisson_lambda}.csv"
    pearl_output = f"{pearl_data_dir}/result-{seed}-{p.poisson_lambda}.csv"

    arf_acc_per_drift  = f"{arf_data_dir}/acc-per-drift-{seed}.csv"
    pearl_acc_per_drift = f"{pearl_data_dir}/acc-per-drift-{seed}.csv"

    # metrics for ARF and PEARL
    arf_df = pd.read_csv(arf_output)
    pearl_df = pd.read_csv(pearl_output)
    arf_acc = arf_df["accuracy"]
    pearl_acc = pearl_df["accuracy"]

    pearl_arf_gain = 0
    for i in range(0, int(sys.argv[2])):
        pearl_arf_gain += pearl_acc[i] - arf_acc[i]

    arf_metrics = get_metrics(arf_df, 0, 0)
    pearl_metrics = get_metrics(pearl_df, 0, pearl_arf_gain)

    # gain per drift for ARF and PEARL
    arf_acc_per_drift = pd.read_csv(
            f"{arf_data_dir}/acc-per-drift-{seed}.log", header=None)
    pearl_acc_per_drift = pd.read_csv(
            f"{pearl_data_dir}/acc-per-drift-{seed}.log", header=None)
    arf_acc_per_drift_mean = arf_acc_per_drift[0].mean()
    pearl_acc_per_drift_mean = arf_acc_per_drift[0].mean()

    cur_data_dir = f"{pearl_data_dir}/nacre/"
    print(f"evaluating {generator}...")
    print("evaluating params...")

    eval_nacre_output(cur_data_dir, [], nacre_metrics_dict,
                      arf_acc, pearl_acc, pearl_acc_per_drift_mean, p)


arf_metrics_dict = {}
pearl_metrics_dict = {}
nacre_metrics_dict = {}

for seed in range(0, 10):
    cur_metric = get_metrics_for_seed(seed,
                                      arf_metrics_dict,
                                      pearl_metrics_dict,
                                      nacre_metrics_dict)

highest_gain = 0
result = None
for (key, vals) in nacre_metrics_dict.items():
    for i in range(len(metric_strs)):
        mean = np.mean(nacre_metrics_dict[key][i])
        std = np.std(nacre_metrics_dict[key][i])

        if metric_strs[i] == "#Trees":
            mean, std = int(round(mean)), int(round(std))
            nacre_metrics_dict[key][i] = f"${mean}\pm{std}$"
        else:
            mean, std = round(mean, 2), round(std, 2)
            nacre_metrics_dict[key][i] = f"${mean:.2f}\pm{std:.2f}$"

        if metric_strs[i] == "Cum. Acc. Gain":
            if highest_gain < mean:
                highest_gain = mean
                result = [key, vals]

pp = PrettyPrinter()
pp.pprint(nacre_metrics_dict)
pp.pprint(result)
print(" & ".join(result[1]))
