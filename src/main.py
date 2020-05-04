#!/usr/bin/env python3

import argparse
import math
import random
import pathlib
import time
import logging
import os.path

import numpy as np

from evaluator import Evaluator

import sys
path = r'../'

if path not in sys.path:
    sys.path.append(path)

from build.pro_pearl import pearl, pro_pearl

formatter = logging.Formatter('%(message)s')

def setup_logger(name, log_file, level=logging.INFO):
    handler = logging.FileHandler(log_file, mode='w')
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # propearl specific params
    parser.add_argument("--proactive",
                        dest="proactive", action="store_true",
                        help="Enable ProPearl")
    parser.set_defaults(proactive=False)
    parser.add_argument("--proactive_percentage",
                        dest="proactive_percentage", default=100, type=int,
                        help="The percentage of triggering proactive drift detection")

    parser.add_argument("--dataset_name",
                        dest="dataset_name", default="", type=str,
                        help="dataset name")
    parser.add_argument("--data_format",
                        dest="data_format", default="", type=str,
                        help="dataset format {csv|arff}")

    # pre-generated synthetic datasets
    parser.add_argument("-g", "--is_generated_data",
                        dest="is_generated_data", action="store_true",
                        help="Handle dataset as pre-generated synthetic dataset")
    parser.set_defaults(is_generator_data=False)
    parser.add_argument("--generator_name",
                        dest="generator_name", default="agrawal", type=str,
                        help="name of the synthetic data generator")
    parser.add_argument("--generator_traits",
                        dest="generator_traits", default="abrupt/0", type=str,
                        help="Traits of the synthetic data")
    parser.add_argument("--generator_seed",
                        dest="generator_seed", default=0, type=int,
                        help="Seed used for generating synthetic data")

    # pearl params
    parser.add_argument("-t", "--tree",
                        dest="num_trees", default=60, type=int,
                        help="number of trees in the forest")
    parser.add_argument("-c", "--candidate_tree",
                        dest="max_num_candidate_trees", default=60, type=int,
                        help="max number of candidate trees in the forest")
    # parser.add_argument("--pool",
    #                     dest="tree_pool_size", default=180, type=int,
    #                     help="number of trees in the online tree repository")
    parser.add_argument("-w", "--warning",
                        dest="warning_delta", default=0.0001, type=float,
                        help="delta value for drift warning detector")
    parser.add_argument("-d", "--drift",
                        dest="drift_delta", default=0.00001, type=float,
                        help="delta value for drift detector")
    parser.add_argument("--drift_tension",
                        dest="drift_tension", default=-1.0, type=float,
                        help="delta value for drift tension")
    parser.add_argument("--max_samples",
                        dest="max_samples", default=200000, type=int,
                        help="total number of samples")
    parser.add_argument("--sample_freq",
                        dest="sample_freq", default=1000, type=int,
                        help="log interval for performance")
    parser.add_argument("--kappa_window",
                        dest="kappa_window", default=50, type=int,
                        help="number of instances must be seen for calculating kappa")
    parser.add_argument("--random_state",
                        dest="random_state", default=0, type=int,
                        help="Seed used for adaptive hoeffding tree")

    parser.add_argument("-s", "--enable_state_adaption",
                        dest="enable_state_adaption", action="store_true",
                        help="enable the state adaption algorithm")
    parser.set_defaults(enable_state_adaption=False)
    parser.add_argument("-p", "--enable_state_graph",
                        dest="enable_state_graph", action="store_true",
                        help="enable state transition graph")
    parser.set_defaults(enable_state_graph=False)

    parser.add_argument("--cd_kappa_threshold",
                        dest="cd_kappa_threshold", default=0.2, type=float,
                        help="Kappa value that the candidate tree needs to outperform both"
                             "background tree and foreground drifted tree")
    parser.add_argument("--bg_kappa_threshold",
                        dest="bg_kappa_threshold", default=0.00, type=float,
                        help="Kappa value that the background tree needs to outperform the "
                             "foreground drifted tree to prevent from false positive")
    parser.add_argument("--edit_distance_threshold",
                        dest="edit_distance_threshold", default=100, type=int,
                        help="The maximum edit distance threshold")
    parser.add_argument("--lossy_window_size",
                        dest="lossy_window_size", default=5, type=int,
                        help="Window size for lossy count")
    parser.add_argument("--reuse_window_size",
                        dest="reuse_window_size", default=0, type=int,
                        help="Window size for calculating reuse rate")
    parser.add_argument("--reuse_rate_upper_bound",
                        dest="reuse_rate_upper_bound", default=0.4, type=float,
                        help="The reuse rate threshold for switching from "
                             "pattern matching to graph transition")
    parser.add_argument("--reuse_rate_lower_bound",
                        dest="reuse_rate_lower_bound", default=0.1, type=float,
                        help="The reuse rate threshold for switching from "
                             "pattern matching to graph transition")

    args = parser.parse_args()

    if args.reuse_rate_upper_bound < args.reuse_rate_lower_bound:
        exit("reuse rate upper bound must be greater than or equal to the lower bound")

    if args.enable_state_graph:
        args.enable_state_adaption = True

    # prepare data
    if args.is_generated_data:
        data_file_path = f"../data/{args.generator_name}/" \
                         f"{args.generator_traits}/" \
                         f"{args.generator_seed}.{args.data_format}"
        result_directory = f"{args.generator_name}/{args.generator_traits}/"

    else:
        data_file_path = f"../third_party/PEARL/data/" \
                         f"{args.dataset_name}/{args.dataset_name}.{args.data_format}"
        result_directory = args.generator

    if not os.path.isfile(data_file_path):
        print(f"Cannot locate file at {data_file_path}")
        exit()

    print(f"Preparing stream from file {data_file_path}...")


    if args.enable_state_graph:
        result_directory = f"{result_directory}/" \
                           f"k{args.cd_kappa_threshold}-e{args.edit_distance_threshold}/" \
                           f"r{args.reuse_rate_upper_bound}-r{args.reuse_rate_lower_bound}-" \
                           f"w{args.reuse_window_size}/" \
                           f"lossy-{args.lossy_window_size}"

    elif args.enable_state_adaption:
        result_directory = f"{result_directory}/" \
                           f"k{args.cd_kappa_threshold}-e{args.edit_distance_threshold}/"

    pathlib.Path(result_directory).mkdir(parents=True, exist_ok=True)

    metric_output_file = "result"
    time_output_file = "time"
    if args.proactive:
        metric_output_file = f"{result_directory}/{metric_output_file}-pro-{args.generator_seed}.csv"
        time_output_file = f"{result_directory}/{time_output_file}-pro-{args.generator_seed}.log"
    else:
        metric_output_file = f"{result_directory}/{metric_output_file}-{args.generator_seed}.csv"
        time_output_file = f"{result_directory}/{time_output_file}-{args.generator_seed}.log"


    configs = (
        f"metric_output_file: {metric_output_file}\n"
        f"warning_delta: {args.warning_delta}\n"
        f"drift_delta: {args.drift_delta}\n"
        f"max_samples: {args.max_samples}\n"
        f"sample_freq: {args.sample_freq}\n"
        f"kappa_window: {args.kappa_window}\n"
        f"random_state: {args.random_state}\n"
        f"enable_state_adaption: {args.enable_state_adaption}\n"
        f"enable_state_graph: {args.enable_state_graph}\n")

    print(configs)
    with open(f"{result_directory}/config", 'w') as out:
        out.write(configs)
        out.flush()

    # other params for pearl/propearl
    arf_max_features = -1
    num_features = -1

    # repo_size = args.num_trees * 160
    repo_size = args.num_trees * 1600
    np.random.seed(args.random_state)
    random.seed(0)

    if args.enable_state_adaption:
        with open(f"{result_directory}/reuse-rate-{args.generator_seed}.log", 'w') as out:
            out.write("background_window_count,candidate_window_count,reuse_rate\n")

    metrics_logger = setup_logger('metrics', metric_output_file)
    process_logger = setup_logger('process', f'{result_directory}/processes-{args.generator_seed}.info')
    seq_logger = setup_logger('seq', f'{result_directory}/seq-{args.generator_seed}.log')

    if args.proactive:
        pearl = pro_pearl(args.num_trees,
                          args.max_num_candidate_trees,
                          repo_size,
                          args.edit_distance_threshold,
                          args.kappa_window,
                          args.lossy_window_size,
                          args.reuse_window_size,
                          arf_max_features,
                          args.bg_kappa_threshold,
                          args.cd_kappa_threshold,
                          args.reuse_rate_upper_bound,
                          args.warning_delta,
                          args.drift_delta,
                          args.drift_tension)
        eval_func = Evaluator.prequential_evaluation_proactive

    else:
        pearl = pearl(args.num_trees,
                      args.max_num_candidate_trees,
                      repo_size,
                      args.edit_distance_threshold,
                      args.kappa_window,
                      args.lossy_window_size,
                      args.reuse_window_size,
                      arf_max_features,
                      args.bg_kappa_threshold,
                      args.cd_kappa_threshold,
                      args.reuse_rate_upper_bound,
                      args.warning_delta,
                      args.drift_delta,
                      args.enable_state_adaption,
                      args.enable_state_graph)
        eval_func = Evaluator.prequential_evaluation

    expected_drift_locs = []
    # expected_drift_locs_log = "../data/agrawal/abrupt/5/drift-0.log"
    # with open(f"{expected_drift_locs_log}", 'r') as f:
    #     expected_drift_locs.append(int(f.readline()))
    # print(expected_drift_locs)

    start = time.process_time()
    eval_func(classifier=pearl,
              stream=data_file_path,
              max_samples=args.max_samples,
              sample_freq=args.sample_freq,
              expected_drift_locs=expected_drift_locs,
              metrics_logger=metrics_logger,
              seq_logger=seq_logger,
              proactive_percentage=args.proactive_percentage)
    elapsed = time.process_time() - start

    with open(f"{time_output_file}", 'w') as out:
        out.write(str(elapsed) + '\n')
