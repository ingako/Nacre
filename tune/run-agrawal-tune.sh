#!/usr/bin/env bash

MAX_SAMPLES=600001

mkdir -p logs

generator_name=agrawal

generator_traits=abrupt/poisson10
data_format=arff


kappa=0.1
ed=100

reuse_window_size=0
reuse_rate=0.9
lossy_window_size=100000000
lambda=6
kappa_window=200


ports=($(seq 50051 1 50091))
for port in $ports ; do
    port_info=$(lsof -i:$port)
    if [ -z "$port_info" ] ; then
        echo "running..."
    else
        echo "port $port already in use"
        exit 1
    fi
done


# "agrawal/abrupt/poisson10" : ['8', '25', '200', '0.01', '0.9'],
port_idx=0
seq_len=8
backtrack_window=25
pro_drift_window=200
stability=0.01
hybrid=0.9

grpc_pids=""
propearl_pids=""


for seq_len in 4 8 12 16 ; do
# for backtrack_window in 25 50 75 100 ; do
# for pro_drift_window in 200 300 400 500 ; do
# for stability in 0.1 0.01 0.001 ; do
# for hybrid in 0.1 0.3 0.5 0.7 0.9 ; do

for seed in {0..9} ; do

    echo "running $seed ..."


    port=${ports[$port_idx]}
    echo "starting grpc server at port $port"
    nohup ../grpc/build/install/seqprediction/bin/seq-prediction-server $port &
    grpc_pids+=" $!"

    sleep 5

    port_idx=$((port_idx+1))

    nohup ../src/main.py --max_samples $MAX_SAMPLES --data_format $data_format --is_generated_data \
        --generator_name $generator_name --generator_traits $generator_traits --generator_seed $seed \
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

                done # seed
    #         done
    #     done

    sleep 1

    echo "wait $propearl_pids"
    wait $propearl_pids

    echo "kill grpc servers listening at $grpc_pids"
    kill $grpc_pids

    wait


done # params
