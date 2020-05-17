import copy
from collections import deque
import math
from random import randrange
import time

import numpy as np
from sklearn.metrics import cohen_kappa_score

import sys
paths = [r'..', r'../third_party']

for path in paths:
    if path not in sys.path:
        sys.path.append(path)

from build.pro_pearl import pearl, pro_pearl

class Evaluator:

    @staticmethod
    def prequential_evaluation(classifier,
                               stream,
                               max_samples,
                               sample_freq,
                               expected_drift_locs,
                               metrics_logger,
                               seq_logger,
                               grpc_port):
        correct = 0
        window_actual_labels = []
        window_predicted_labels = []
        if isinstance(classifier, pearl):
            print("is an instance of pearl, turn on log_size")

        log_size = isinstance(classifier, pearl)

        metrics_logger.info("count,accuracy,candidate_tree_size,tree_pool_size,time")
        start_time = time.process_time()

        classifier.init_data_source(stream);

        for count in range(0, max_samples):
            if not classifier.get_next_instance():
                break

            # test
            prediction = classifier.predict()

            actual_label = classifier.get_cur_instance_label()
            if prediction == actual_label:
                correct += 1

            window_actual_labels.append(actual_label)
            window_predicted_labels.append(prediction)

            # classifier.handle_drift(count)

            # train
            classifier.train()

            classifier.delete_cur_instance()

            if count % sample_freq == 0 and count != 0:
                accuracy = correct / sample_freq
                elapsed_time = time.process_time() - start_time

                candidate_tree_size = 0
                tree_pool_size = 60
                if log_size:
                    candidate_tree_size = classifier.get_candidate_tree_group_size()
                    tree_pool_size = classifier.get_tree_pool_size()

                print(f"{count},{accuracy},{candidate_tree_size},{tree_pool_size},{elapsed_time}")
                metrics_logger.info(f"{count},{accuracy}," \
                                    f"{candidate_tree_size},{tree_pool_size},{elapsed_time}")

                correct = 0
                window_actual_labels = []
                window_predicted_labels = []

    @staticmethod
    def prequential_evaluation_proactive(classifier,
                                         stream,
                                         max_samples,
                                         sample_freq,
                                         expected_drift_locs,
                                         metrics_logger,
                                         seq_logger,
                                         grpc_port):
        num_trees = 60
        np.random.seed(0)

        import grpc
        import seqprediction_pb2
        import seqprediction_pb2_grpc

        from denstream.DenStream import DenStream

        clusterer = DenStream(lambd=0.1, eps=10, beta=0.5, mu=3)

        def fit_predict(clusterer, interval):
            x = [np.array([interval])]
            label = clusterer.fit_predict(x)[0]

            if label == -1:
                return interval

            return int(round(clusterer.p_micro_clusters[label].center()[0]))

        correct = 0
        window_actual_labels = []
        window_predicted_labels = []

        # proactive drift point prediction
        drift_interval_seq_len = 8
        num_request = 0
        cpt_runtime = 0

        next_adapt_state_locs = [-1 for v in range(num_trees)]
        predicted_drift_locs = [-1 for v in range(num_trees)]
        drift_interval_sequences = [deque(maxlen=drift_interval_seq_len) for v in range(num_trees)]
        last_actual_drift_points = [0 for v in range(num_trees)]

        expected_drift_loc_indices = [0 for i in range(num_trees)]

        metrics_logger.info("count,accuracy,candidate_tree_size,tree_pool_size,time")
        start_time = time.process_time()

        classifier.init_data_source(stream)

        with grpc.insecure_channel(f'localhost:{grpc_port}') as channel:
            print(f'Sequence prediction server is listening at {grpc_port}...')

            stub = seqprediction_pb2_grpc.PredictorStub(channel)

            stub.setNumTrees(seqprediction_pb2.SetNumTreesMessage(numTrees=num_trees))

            for count in range(0, max_samples):
                if not classifier.get_next_instance():
                    break

                # set expected drift probability
                # for idx in range(num_trees):
                #     if predicted_drift_locs[idx] > 0:
                #         t_r = (count - last_actual_drift_points[idx]) / (predicted_drift_locs[idx] - last_actual_drift_points[idx])
                #         expected_drift_prob = 1 - math.sin(math.pi * t_r)
                #         classifier.set_expected_drift_prob(idx, expected_drift_prob)

                # test
                prediction = classifier.predict()

                actual_label = classifier.get_cur_instance_label()
                if prediction == actual_label:
                    correct += 1

                window_actual_labels.append(actual_label)
                window_predicted_labels.append(prediction)

                # Generate new sequences for the actual drifted trees
                for idx in classifier.get_stable_tree_indices():

                    # find actual drift point at num_instances_before
                    num_instances_before = classifier.find_last_actual_drift_point(idx)

                    if num_instances_before > -1:
                        interval = count - num_instances_before - last_actual_drift_points[idx]
                        if interval < 0:
                            print("Failed to find the actual drift point")
                            # exit()
                        else:
                            interval = fit_predict(clusterer, interval)
                            drift_interval_sequences[idx].append(interval)
                            last_actual_drift_points[idx] = count - num_instances_before

                    # train CPT with the new sequence
                    if len(drift_interval_sequences[idx]) >= drift_interval_seq_len:
                        num_request += 1
                        seq_logger.info(f"Tree {idx}: {drift_interval_sequences[idx]}")
                        cpt_response = stub.train(seqprediction_pb2
                                          .SequenceMessage(seqId=num_request,
                                                           treeId=idx,
                                                           seq=drift_interval_sequences[idx]))
                        if cpt_response.result:
                            cpt_runtime += cpt_response.runtimeInSeconds
                        else:
                            print("CPT training failed")
                            exit()

                    # predict the next drift points
                    if len(drift_interval_sequences[idx]) >= drift_interval_seq_len:
                        drift_interval_sequences[idx].popleft()

                        response = stub.predict(seqprediction_pb2
                                                    .SequenceMessage(seqId=count,
                                                                     treeId=idx,
                                                                     seq=drift_interval_sequences[idx]))
                        cpt_runtime += cpt_response.runtimeInSeconds

                        # print(f"Predicted next drift point: {response.seq[0]}")
                        if len(response.seq) > 0:
                            interval = fit_predict(clusterer, response.seq[0])

                            predicted_drift_locs[idx] = count + interval
                            next_adapt_state_locs[idx] = count + interval + 300 # TODO currently set to the same as kappa window
                            # next_adapt_state_locs[idx] = count + interval + 50 # TODO currently set to the same as kappa window

                            drift_interval_sequences[idx].append(interval)
                            last_actual_drift_points[idx] = count

                # check if hit predicted drift locations
                transition_tree_pos_list = []
                adapt_state_tree_pos_list = []

                for idx in range(num_trees):
                    predicted_drift_loc = predicted_drift_locs[idx]
                    # find potential drift trees for candidate selection
                    if count >= predicted_drift_loc and predicted_drift_loc != -1:
                        predicted_drift_locs[idx] = -1
                        # TODO if not classifier.actual_drifted_trees[idx]:
                        transition_tree_pos_list.append(idx)

                    # find trees with actual drifts
                    next_adapt_state_loc = next_adapt_state_locs[idx]
                    if count >= next_adapt_state_loc and next_adapt_state_loc != -1:
                        next_adapt_state_locs[idx] = -1
                        if not classifier.has_actual_drift(idx):
                            adapt_state_tree_pos_list.append(idx)

                if len(transition_tree_pos_list) > 0:
                    # select candidate_trees
                    classifier.select_candidate_trees(transition_tree_pos_list)
                if len(adapt_state_tree_pos_list) > 0:
                    # update actual drifted trees
                    classifier.update_drifted_tree_indices(adapt_state_tree_pos_list)

                # adapt state for both drifted tree and predicted drifted trees
                actual_drifted_tree_indices = classifier.adapt_state_with_proactivity()
                # print(f"actual_drifted_tree_indices: {actual_drifted_tree_indices}")

                # train
                classifier.train()

                # classifier.delete_cur_instance()

                # log performance
                if count % sample_freq == 0 and count != 0:
                    accuracy = correct / sample_freq
                    candidate_tree_size = classifier.get_candidate_tree_group_size()
                    tree_pool_size = classifier.get_tree_pool_size()
                    elapsed_time = time.process_time() - start_time + cpt_runtime

                    print(f"{count},{accuracy},{candidate_tree_size},{tree_pool_size},{str(elapsed_time)}")
                    metrics_logger.info(f"{count},{accuracy}," \
                                        f"{candidate_tree_size},{tree_pool_size},{str(elapsed_time)}")

                    correct = 0
                    window_actual_labels = []
                    window_predicted_labels = []
