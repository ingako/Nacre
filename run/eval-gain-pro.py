#!/usr/bin/env python3

import os
import sys
import math
import pandas as pd
import numpy as np
from dataclasses import dataclass

def is_empty_file(fpath):
    return False if os.path.isfile(fpath) and os.path.getsize(fpath) > 0 else True

base_dir = os.getcwd()
generator = sys.argv[1]
seed = sys.argv[2]

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

covtype_params = Param(
        generator = generator,
        seed = seed,
        kappa=0.4,
        ed=90,
        reuse_window_size=0,
        reuse_rate=0.18,
        lossy_window_size=100000000,
        poisson_lambda=1,
        kappa_window=50)

insect_params = Param(
        generator = generator,
        seed = seed,
        kappa=0.0,
        ed=110,
        reuse_window_size=0,
        reuse_rate=0.9,
        lossy_window_size=100000000,
        poisson_lambda=1,
        kappa_window=50)

sensor_params = Param(
        generator = generator,
        seed = seed,
        kappa=0.0,
        ed=100,
        reuse_window_size=0,
        reuse_rate=0.9,
        lossy_window_size=100000000,
        poisson_lambda=1,
        kappa_window=50)

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

tree_params = Param(
        generator = generator,
        seed = seed,
        kappa=0.0,
        ed=100,
        reuse_window_size=0,
        reuse_rate=0.9,
        lossy_window_size=100000000,
        poisson_lambda=6,
        kappa_window=50)

if generator == "covtype":
    p = covtype_params
elif generator == "sensor":
    p = sensor_params
elif generator[:6] == "insect":
    p = insect_params
elif generator[:4] == "tree":
    p = tree_params
elif generator[:7] == "agrawal":
    p = agrawal_params

arf_data_dir = f"{base_dir}/{generator}"
pearl_data_dir = f"{arf_data_dir}/k{p.kappa}-e{p.ed}/r{p.reuse_rate}-r{p.reuse_rate}-w{p.reuse_window_size}/lossy-{p.lossy_window_size}/"
gain_report_path = f"{arf_data_dir}/gain-pro-report.txt"

arf_output  = f"{arf_data_dir}/result-{seed}-{p.poisson_lambda}.csv"
pearl_output = f"{pearl_data_dir}/result-{seed}-{p.poisson_lambda}.csv"

arf_df = pd.read_csv(arf_output)
pearl_df = pd.read_csv(pearl_output)

cur_data_dir = f"{pearl_data_dir}/nacre/"
print(f"evaluating {generator}...")
print("evaluating params...")

gain_report_out = open(gain_report_path, "w")
gain_report_out.write("param,reuse-param,lossy-win,#instances,pearl-arf,nacre-arf,nacre-pearl\n")
param_strs = ["seq", "backtrack", "adapt_window", "stability", "hybrid"]

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
        nacre_output = f"{cur_data_dir}/result-pro-{seed}-{p.poisson_lambda}.csv"
        gain_output = f"{cur_data_dir}/gain.csv"
        with open(gain_output, "w") as out:

            arf_acc = arf_df["accuracy"]
            pearl_acc = pearl_df["accuracy"]

            if is_empty_file(nacre_output):
                return

            nacre_df = pd.read_csv(nacre_output)
            nacre_acc = nacre_df["accuracy"]

            num_instances = nacre_df["count"]

            out.write("#count,seq,backtrack," \
                      "adapt_win,stability," \
                      "hybrid,pearl-arf-gain,nacre-arf-gain,nacre-pearl-gain\n")

            end = min(int(sys.argv[3]), min(len(nacre_acc), len(arf_acc)))

            pearl_arf_gain = 0
            nacre_arf_gain = 0
            nacre_pearl_gain = 0

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

eval_nacre_output(cur_data_dir, [], gain_report_out)

gain_report_out.close()
