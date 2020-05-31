#!/usr/bin/env python3

import os
import sys
import math
import pandas as pd
import numpy as np

def is_empty_file(fpath):
    return False if os.path.isfile(fpath) and os.path.getsize(fpath) > 0 else True

base_dir = os.getcwd()
generator = "covtype"
seed = 0

kappa=0.4
ed=90

reuse_window_size=0
reuse_rate=0.18
lossy_window_size=100000000
poisson_lambda=1

kappa_window=50

arf_data_dir = f"{base_dir}/{generator}"
pearl_data_dir = f"{arf_data_dir}/k{kappa}-e{ed}/r{reuse_rate}-r{reuse_rate}-w{reuse_window_size}/lossy-{lossy_window_size}/"
gain_report_path = f"{arf_data_dir}/gain-pro-report.txt"

arf_output  = f"{arf_data_dir}/result-{seed}-{poisson_lambda}.csv"
pearl_output = f"{pearl_data_dir}/result-{seed}-{poisson_lambda}.csv"

arf_df = pd.read_csv(arf_output)
pearl_df = pd.read_csv(pearl_output)

cur_data_dir = f"{pearl_data_dir}/nacre/"
print(f"evaluating {generator}...")
print("evaluating params...")

gain_report_out = open(gain_report_path, "w")
gain_report_out.write("param,reuse-param,lossy-win,#instances,pearl-arf,nacre-arf,nacre-pearl\n")
param_strs = ["seq", "backtrack", "adapt_window", "stability", "nacre"]

def eval_nacre_output(cur_data_dir, param_values, gain_report_out):

    if len(param_values) != len(param_strs):
        # recurse
        params = [f for f in os.listdir(cur_data_dir) if os.path.isdir(os.path.join(cur_data_dir, f))]
        print(f"evaluating {params}...")
        for cur_param in params:
            param_values.append(cur_param)
            eval_nacre_output(f"{cur_data_dir}/{cur_param}", param_values, gain_report_out)
            param_values.pop()

    else:
        nacre_output = f"{cur_data_dir}/result-pro-{seed}-{poisson_lambda}.csv"
        gain_output = f"{cur_data_dir}/gain.csv"
        with open(gain_output, "w") as out:

            arf_acc = arf_df["accuracy"]
            pearl_acc = pearl_df["accuracy"]

            if is_empty_file(nacre_output):
                return

            nacre_df = pd.read_csv(nacre_output)
            nacre_acc = nacre_df["accuracy"]

            num_instances = nacre_df["count"]

            out.write("#count,pearl-arf-gain,nacre-arf-gain,nacre-pearl-gain\n")

            end = min(len(nacre_acc), len(arf_acc))

            pearl_arf_gain = 0
            nacre_arf_gain = 0
            nacre_pearl_gain = 0

            pearl_arf_gain_list = []
            nacre_arf_gain_list = []
            nacre_pearl_gain_list = []

            for i in range(0, end):
                # pearl_arf_gain += pearl_acc[i] - arf_acc[i]
                nacre_arf_gain += nacre_acc[i] - arf_acc[i]
                nacre_pearl_gain += nacre_acc[i] - pearl_acc[i]

                if i == (end - 1):
                    gain_report_out.write(f"{param_values[0]},{param_values[1]},{param_values[2]},"
                                          f"{param_values[3]},{param_values[4]},"
                                          f"{num_instances[i]},"
                                          f"{pearl_arf_gain},"
                                          f"{nacre_arf_gain},"
                                          f"{nacre_pearl_gain}\n")

                    pearl_arf_gain_list.append(pearl_arf_gain)
                    nacre_arf_gain_list.append(nacre_arf_gain)
                    nacre_pearl_gain_list.append(nacre_pearl_gain)

                out.write(f"{num_instances[i]},"
                          f"{pearl_arf_gain},"
                          f"{nacre_arf_gain},"
                          f"{nacre_pearl_gain}\n")

                out.flush()

eval_nacre_output(cur_data_dir, [], gain_report_out)

gain_report_out.close()
