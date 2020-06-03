#!/usr/bin/env bash

MAX_SAMPLES=10000000

mkdir -p logs

generator_name=agrawal
generator_traits=abrupt
data_format=arff
kappa=0.1
ed=100

reuse_window_size=0
reuse_rate=0.9
lossy_window_size=100000000

lambda=6

# ARF
nohup ../src/main.py --max_samples $MAX_SAMPLES --data_format $data_format --is_generated_data \
    --generator_name $generator_name --generator_traits $generator_traits --generator_seed 0 \
    --data_format $data_format --poisson_lambda $lambda \
    -t 60 &

# PEARL
# for kappa in 0 0.1 0.2 0.3 0.4 ; do
#     for ((ed=60;ed<=120;ed+=10)); do
nohup ../src/main.py --max_samples $MAX_SAMPLES --data_format $data_format --is_generated_data \
    --generator_name $generator_name --generator_traits $generator_traits --generator_seed 0 \
    -t 60 -c 120 --poisson_lambda $lambda \
    -s --cd_kappa_threshold $kappa --edit_distance_threshold $ed \
    -p \
    --reuse_rate_upper_bound $reuse_rate \
    --reuse_rate_lower_bound $reuse_rate \
    --reuse_window_size $reuse_window_size \
    --kappa_window 300 \
    --lossy_window_size $lossy_window_size &
#     done
# done

# ProPEARL

# valgrind --tool=memcheck --suppressions=python.supp \
#                                           python -E -tt \

grpc_pids=""
propearl_pids=""

for lambda in 6 ; do
    port=50101
    echo "starting grpc server at port $port"
    nohup ../grpc/build/install/seqprediction/bin/seq-prediction-server $port &
    grpc_pids+=" $!"

    ../src/main.py --max_samples $MAX_SAMPLES --data_format $data_format --is_generated_data \
        --generator_name $generator_name --generator_traits $generator_traits --generator_seed 0 \
        -t 60 -c 120 \
        -s --cd_kappa_threshold $kappa --edit_distance_threshold $ed \
        --reuse_rate_upper_bound $reuse_rate \
        --reuse_rate_lower_bound $reuse_rate \
        --reuse_window_size $reuse_window_size \
        --lossy_window_size $lossy_window_size \
        --kappa_window 300 \
        --poisson_lambda $lambda --random_state 0 \
        -p --proactive --grpc_port $port &

    propearl_pids+=" $!"
done

echo "wait $propearl_pids"
wait $propearl_pids

echo "kill grpc servers listening at $grpc_pids"
kill $grpc_pids
