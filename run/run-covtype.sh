#!/usr/bin/env bash

MAX_SAMPLES=10000000

mkdir -p logs

dataset_name=covtype
data_format=arff
kappa=0.4
ed=90

reuse_window_size=0
reuse_rate=0.18
lossy_window_size=100000000
lambda=1

kappa_window=50

ports=($(seq 50010 10 50200))
for port in $ports ; do
    port_info=$(lsof -i:$port)
    if [ -z "$port_info" ] ; then 
        echo "running..."
    else
        echo "port $port already in use"
        exit 1
    fi
done

# ARF
nohup ../src/main.py --max_samples $MAX_SAMPLES --dataset_name $dataset_name --data_format $data_format \
    -t 60 --poisson_lambda $lambda &


# PEARL
nohup ../src/main.py --max_samples $MAX_SAMPLES --dataset_name $dataset_name --data_format $data_format \
    -t 60 -c 120 \
    -s --cd_kappa_threshold $kappa --edit_distance_threshold $ed \
    -p \
    --reuse_rate_upper_bound $reuse_rate \
    --reuse_rate_lower_bound $reuse_rate \
    --reuse_window_size $reuse_window_size \
    --kappa_window $kappa_window --poisson_lambda $lambda \
    --lossy_window_size $lossy_window_size &

# ProPEARL

port_idx=0
seq_len=8
backtrack_window=25
pro_drift_window=300
stability=0.001
hybrid=0.001

# valgrind --tool=memcheck --suppressions=python.supp \
#                                           python -E -tt \
for seq_len in 8 12 16 20 ; do
    for backtrack_window in 25 50 75 100 ; do
        for pro_drift_window in 200 300 400 500 ; do
            for stability in 0.1 0.01 0.001 ; do
                for hybrid in 0.1 0.01 0.001 ; do


    port=${ports[$port_idx]}
    echo "starting grpc server at port $port"
    nohup ../grpc/build/install/seqprediction/bin/seq-prediction-server $port &
    grpc_pids+=" $!"

    port_idx=$((port_idx+1))

    ../src/main.py --max_samples $MAX_SAMPLES --dataset_name $dataset_name --data_format $data_format \
        -t 60 -c 120 \
        -s --cd_kappa_threshold $kappa --edit_distance_threshold $ed \
        --reuse_rate_upper_bound $reuse_rate \
        --reuse_rate_lower_bound $reuse_rate \
        --reuse_window_size $reuse_window_size \
        --lossy_window_size $lossy_window_size \
        --poisson_lambda $lambda \
        --kappa_window $kappa_window \
        --sequence_len $seq_len \
        --backtrack_window $backtrack_window \
        --pro_drift_window $pro_drift_window \
        --stability $stability \
        --hybrid $hybrid \
        -p --proactive --grpc_port $port &

    propearl_pids+=" $!"

    done
done

wait $propearl_pids

kill $grpc_pids
