#!/usr/bin/env python3

import os
import sys
import logging

path = r'../'
if path not in sys.path:
    sys.path.append(path)

from PEARL.src.stream_generator import RecurrentDriftStream

# max_samples = 10000001
max_samples = 100000

generator = 'agrawal'
data_dir_prefix = '../data/'
logger_dir_prefix = '../stable-period-logs/'
dir_suffix = 'agrawal/abrupt/5/'

data_dir = f'{data_dir_prefix}/{dir_suffix}'
logger_dir = f'{logger_dir_prefix}/{dir_suffix}'

if not os.path.exists(data_dir):
    os.makedirs(data_dir)

if not os.path.exists(logger_dir):
    os.makedirs(logger_dir)

for seed in range(0, 10):
    logging.basicConfig(
            filename=f'{logger_dir}/{seed}.csv',
            format='%(message)s',
            filemode='w')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    stream = RecurrentDriftStream(generator=generator,
                                  concepts=[4, 0, 8, 6, 2],
                                  has_noise=False,
                                  stable_period_lam=10,
                                  stable_period_start=1000,
                                  stable_period_base=200,
                                  stable_period_logger=logger,
                                  random_state=seed)
    stream.prepare_for_use()
    print(stream.get_data_info())

    output_filename = os.path.join(data_dir, f'{seed}.csv')
    print(f'generating {output_filename}...')

    with open(output_filename, 'w') as out:
        for _ in range(max_samples):
            X, y = stream.next_sample()

            out.write(','.join(str(v) for v in X[0]))
            out.write(f',{y[0]}')
            out.write('\n')
