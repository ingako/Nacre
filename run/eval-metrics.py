#!/usr/bin/env python3

import os
import sys
import math
import pandas as pd
import numpy as np
from dataclasses import dataclass

@dataclass
class Param:
    generator: str = "covtype"
    seed: int = 0
    kappa: float = 0.0
    ed: int =90
    reuse_window_size: int = 0
    reuse_rate: float = 0.18
    lossy_window_size: int = 100000000
    poisson_lambda: int = 1
    kappa_window: int = 50


def is_empty_file(fpath):
    return False if os.path.isfile(fpath) and os.path.getsize(fpath) > 0 else True

def get_metrics(df, gain):
    return [round(df["accuracy"].mean(), 2), round(df["kappa"].mean(), 2), \
            round(gain, 2), round(df["time"].iloc[-1]/60, 2), df["tree_pool_size"].iloc[-1]]


for seed in range(0, 1):
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

    arf_df = pd.read_csv(arf_output)
    pearl_df = pd.read_csv(pearl_output)

    cur_data_dir = f"{pearl_data_dir}/nacre/"
    print(f"evaluating {generator}...")
    print("evaluating params...")

    param_strs = ["seq", "backtrack", "adapt_window", "stability", "hybrid"]
    metric_strs = ["Acc", "Kappa", "Gain per Drift", "Cum. Acc. Gain", "Runtime", "#Trees"]

    def eval_nacre_output(cur_data_dir, param_values):

        if len(param_values) != len(param_strs):
            # recurse
            params = [f for f in os.listdir(cur_data_dir) if os.path.isdir(os.path.join(cur_data_dir, f))]
            print(f"evaluating {params}...")

            highest_gain = -1
            arf_metrics,  pearl_metrics, nacre_metrics = [], [], []
            for cur_param in params:
                param_values.append(cur_param)
                metrics = eval_nacre_output(f"{cur_data_dir}/{cur_param}", param_values)
                param_values.pop()
                if highest_gain < metrics[0]:
                   highest_gain = metrics[0]
                   arf_metrics = metrics[1]
                   pearl_metrics = metrics[2]
                   nacre_metrics = metrics[3]

            return [highest_gain, arf_metrics, pearl_metrics, nacre_metrics]


        else:
            nacre_output = f"{cur_data_dir}/result-pro-{seed}-{p.poisson_lambda}.csv"

            arf_acc = arf_df["accuracy"]
            pearl_acc = pearl_df["accuracy"]

            if is_empty_file(nacre_output):
                return

            nacre_df = pd.read_csv(nacre_output)
            nacre_acc = nacre_df["accuracy"]

            num_instances = nacre_df["count"]

            # end = int(sys.argv[2])
            # if len(nacre_acc) < end or len(arf_acc) < end or len(pearl_acc) < end:
            #     exit()
            end = min(int(sys.argv[2]), min(len(nacre_acc), len(arf_acc)))

            pearl_arf_gain, nacre_arf_gain, nacre_pearl_gain = 0, 0, 0

            for i in range(0, end):
                pearl_arf_gain += pearl_acc[i] - arf_acc[i]
                nacre_arf_gain += nacre_acc[i] - arf_acc[i]
                nacre_pearl_gain += nacre_acc[i] - pearl_acc[i]

            # if  highest_gain < nacre_pearl_gain:
            highest_gain = round(nacre_pearl_gain, 2)
            arf_metrics = get_metrics(arf_df, 0)
            pearl_metrics = get_metrics(pearl_df, pearl_arf_gain)
            nacre_metrics = get_metrics(nacre_df, nacre_arf_gain)

            return [highest_gain, arf_metrics, pearl_metrics, nacre_metrics]

    metrics = eval_nacre_output(cur_data_dir, [])
    print(metrics)
