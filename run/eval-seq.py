#!/usr/bin/env python3

import os
import sys
import math
import pandas as pd
import numpy as np
from dataclasses import dataclass
from pprint import PrettyPrinter

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


# ./eval-seq.py agrawal/abrupt/poisson10 600
base_dir = os.getcwd()
generator = sys.argv[1]
end = int(sys.argv[2]) * 1000

data_dir = f"../data/{generator}/"

param_strs = ["seq", "backtrack", "adapt_window", "stability", "hybrid"]

agrawal_abrupt_poisson10_params = ['8', '25', '200', '0.01', '0.9']
agrawal_abrupt_poisson3_params = ['8', '25', '400', '0.001', '0.9']
agrawal_abrupt_uniform_params = ['8', '25', '500', '0.001', '0.9']

agrawal_gradual_poisson10_params = ['8', '25', '400', '0.01', '0.9']
agrawal_gradual_poisson3_params = ['8', '25', '300', '0.1', '0.9']
agrawal_gradual_uniform_params = ['8', '25', '400', '0.1', '0.9']


# nacre_param_path = "/".join(agrawal_abrupt_poisson10_params)
nacre_param_path = "/".join(agrawal_abrupt_poisson3_params)
#nacre_param_path = "/".join(agrawal_abrupt_uniform_params)


agrawal_params = Param(
        generator = generator,
        kappa=0.1,
        ed=100,
        reuse_window_size=0,
        reuse_rate=0.9,
        lossy_window_size=100000000,
        poisson_lambda=6,
        kappa_window=50)

tree_params = Param(
        generator = generator,
        kappa=0.0,
        ed=100,
        reuse_window_size=0,
        reuse_rate=0.9,
        lossy_window_size=100000000,
        poisson_lambda=6,
        kappa_window=50)

if generator[:7] == "agrawal":
    p = agrawal_params
elif generator[:4] == "tree":
    p = tree_params

for seed in range(0, 10):

    drift_log_path = f"{data_dir}/drift-{seed}.log"
    arf_data_dir = f"{base_dir}/{generator}"
    pearl_data_dir = f"{arf_data_dir}/" \
                     f"k{p.kappa}-e{p.ed}/" \
                     f"r{p.reuse_rate}-r{p.reuse_rate}-w{p.reuse_window_size}/" \
                     f"lossy-{p.lossy_window_size}/"
    nacre_data_dir = f"{pearl_data_dir}/nacre/{nacre_param_path}"

    accepted_predictions_path = f"{nacre_data_dir}/accepted-predicted-drifts-{seed}.log"
    all_predictions_path = f"{nacre_data_dir}/all-predicted-drifts-{seed}.log"

    all_predictions = []
    with open(all_predictions_path) as f:
        for line in f:
            if line == "\n":
                all_predictions.append([])
            else:
                all_predictions.append([int(v) for v in line.split(",")])

    actual_drifts = []
    with open(drift_log_path) as f:
        for line in f:
            actual_drifts.append(int(line))
            if int(line) > end:
                break

    for tree_idx in range(60):
        all_prediction = all_predictions[tree_idx]
        i, j = 0, 0
        true_positive, false_positive = 0, 0

        offset = 1000
        while i < len(all_prediction) and j < len(actual_drifts):
            if all_prediction[i] < actual_drifts[j]:
                if all_prediction[i] <= actual_drifts[j] - offset:
                    false_positive += 1
                else:
                    true_positive += 1
                    j += 1
                i += 1
            else:
                if all_prediction[i] >= actual_drifts[j] + offset:
                    false_positive += 1
                else:
                    true_positive += 1
                    i += 1
                j += 1

        # offset = 1000
        # while i < len(all_prediction) and j < len(actual_drifts):
        #     if all_prediction[i] < actual_drifts[j]:
        #         false_positive += 1
        #         i += 1
        #     else:
        #         if all_prediction[i] >= actual_drifts[j] + offset:
        #             false_positive += 1
        #         else:
        #             true_positive += 1
        #             i += 1
        #         j += 1

        false_negative  = len(actual_drifts) - true_positive
        print(f"tree {tree_idx} : true_positive={true_positive}, false_positive={false_positive}, false_negative={false_negative}")

        if true_positive + false_positive == 0 or true_positive + false_negative == 0:
            print(f"precision=0 recall=0")
            continue
        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive + false_negative)
        print(f"precision={precision}, recall={recall}")
