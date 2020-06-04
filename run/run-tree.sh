#!/usr/bin/env bash

MAX_SAMPLES=600001

mkdir -p logs

generator_name=tree
generator_trait_list=( abrupt/uniform-1 abrupt/poisson10 abrupt/poisson3 gradual/poisson10 gradual/poisson3 gradual/uniform-1 )

data_format=arff

kappa=0.0
# kappa=0.2 # for lambda 1
ed=100

reuse_window_size=0
reuse_rate=0.9
lossy_window_size=100000000
lambda=6


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


# for generator_traits in "abrupt/uniform-1" ; do
for generator_traits in ${generator_trait_list[@]} ; do
for seed in {1..9} ; do
for kappa_window in 200 ; do

    echo "running $seed ..."

# ARF
nohup ../src/main.py --max_samples $MAX_SAMPLES --data_format $data_format --is_generated_data \
    --generator_name $generator_name --generator_traits $generator_traits --generator_seed $seed \
    --data_format $data_format --poisson_lambda $lambda \
    -t 60 &

# PEARL
# for kappa in 0 0.1 0.2 0.3 0.4 ; do
    # for ((ed=60;ed<=120;ed+=10)); do
nohup ../src/main.py --max_samples $MAX_SAMPLES --data_format $data_format --is_generated_data \
    --generator_name $generator_name --generator_traits $generator_traits --generator_seed $seed \
    -t 60 -c 120 --poisson_lambda $lambda \
    -s --cd_kappa_threshold $kappa --edit_distance_threshold $ed \
    -p \
    --reuse_rate_upper_bound $reuse_rate \
    --reuse_rate_lower_bound $reuse_rate \
    --reuse_window_size $reuse_window_size \
    --kappa_window $kappa_window \
    --lossy_window_size $lossy_window_size &
#     done
# done


# ProPEARL

port_idx=0
seq_len=8
backtrack_window=25
pro_drift_window=500
stability=0.01
hybrid=0.1

grpc_pids=""
propearl_pids=""

# valgrind --tool=memcheck --suppressions=python.supp \
#                                           python -E -tt \

#for seq_len in 8 12 16 20 ; do
#    for backtrack_window in 25 50 75 100 ; do
        # for pro_drift_window in 200 300 400 500 ; do
            # for stability in 0.1 0.01 0.001 ; do
            #     for hybrid in 0.1 0.5 0.9 ; do
        for pro_drift_window in 200 300 ; do
            for stability in 0.1 0.01 ; do
                for hybrid in 0.1 0.5 0.9 ; do

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

                done
            done
        done

    sleep 1

    echo "wait $propearl_pids"
    wait $propearl_pids

    echo "kill grpc servers listening at $grpc_pids"
    kill $grpc_pids

    wait

done # for kappa_window

done # for seed

done # for generator_traits_lists
