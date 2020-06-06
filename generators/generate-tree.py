#!/usr/bin/env python3

import os
import sys
import logging
from random import randrange

path = r'../'
if path not in sys.path:
    sys.path.append(path)

from stream_generator import RecurrentDriftStream

formatter = logging.Formatter('%(message)s')

def setup_logger(name, log_file, level=logging.INFO):
    handler = logging.FileHandler(log_file, mode='w')
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger

max_samples = 600001

generator = 'tree'
drift_type = sys.argv[1]
data_dir_prefix = '../data/'


for param in [(3, "poisson"), (10, "poisson"), (-1, "uniform")]:
    dir_suffix = f'{generator}/{drift_type}/{param[1]}{str(param[0])}/'
    data_dir = f'{data_dir_prefix}/{dir_suffix}'

    for seed in range(0, 10):
        print(f"generating seed {seed}")

        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        logger = setup_logger(f'seq-{drift_type}-{param[1]}{str(param[0])}-{seed}', f'{data_dir}/drift-{seed}.log')

        if drift_type == "abrupt":
            stream = RecurrentDriftStream(generator=generator,
                                          concepts = [2, 3, 4, 5, 6, 7, 8, 9, 10],
                                          has_noise=False,
                                          stable_period_lam=param[0],
                                          stable_period_start=1000,
                                          stable_period_base=200,
                                          stable_period_logger=logger,
                                          drift_interval_distr=param[1],
                                          random_state=seed)
        elif drift_type == "gradual":
            stream = RecurrentDriftStream(generator=generator,
                                          width=1000,
                                          concepts = [2, 3, 4, 5, 6, 7, 8, 9, 10],
                                          has_noise=False,
                                          stable_period_lam=param[0],
                                          stable_period_start=1000,
                                          stable_period_base=200,
                                          stable_period_logger=logger,
                                          drift_interval_distr=param[1],
                                          random_state=seed)
        else:
            print(f"Unknown drift type {drift_type}")
            exit()

        stream.prepare_for_use()
        print(stream.get_data_info())

        output_filename = os.path.join(data_dir, f'{seed}.arff')
        print(f'generating {output_filename}...')

        with open(output_filename, 'w') as out:
            out.write(stream.get_arff_header())

            for _ in range(max_samples):
                X, y = stream.next_sample()

                out.write(','.join(str(v) for v in X[0]))
                out.write(f',{y[0]}')
                out.write('\n')
