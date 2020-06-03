#!/usr/bin/env python3

import os
import sys
import logging
from random import randrange

path = r'../'
if path not in sys.path:
    sys.path.append(path)

# from third_party.PEARL.src.stream_generator import RecurrentDriftStream
from stream_generator import RecurrentDriftStream

formatter = logging.Formatter('%(message)s')

def setup_logger(name, log_file, level=logging.INFO):
    handler = logging.FileHandler(log_file, mode='w')
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger

max_samples = 1000001

generator = 'agrawal'
data_dir_prefix = '../data/'
logger_dir_prefix = '../stable-period-logs/'

for param in [(-1, "uniform"), (3, "poisson")]:
    #dir_suffix = 'agrawal/gradual/'
    #dir_suffix = 'agrawal/gradual/uniform/'
    dir_suffix = f'agrawal/gradual/{param[1]}{str(param[0])}/'

    data_dir = f'{data_dir_prefix}/{dir_suffix}'
    logger_dir = f'{logger_dir_prefix}/{dir_suffix}'

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    if not os.path.exists(logger_dir):
        os.makedirs(logger_dir)


    for seed in range(0, 10):

        print(f"generating seed {seed}")
        logger = setup_logger('seq', f'{data_dir}/drift-{seed}.log')

        stream = RecurrentDriftStream(generator=generator,
                                      width=1000,
                                      concepts=[4, 0, 8, 6, 2, 1, 3, 5, 7, 9],
                                      has_noise=False,
                                      stable_period_lam=param[0],
                                      stable_period_start=1000,
                                      stable_period_base=200,
                                      stable_period_logger=logger,
                                      drift_interval_distr=param[1],
                                      random_state=seed)
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
