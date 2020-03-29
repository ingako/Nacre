import copy
from collections import deque
import math

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
    def prequential_evaluation_cpp(classifier,
                                   stream,
                                   max_samples,
                                   sample_freq,
                                   metrics_logger):
        correct = 0
        window_actual_labels = []
        window_predicted_labels = []
        if isinstance(classifier, pearl):
            print("is an instance of pearl, turn on log_size")

        log_size = isinstance(classifier, pearl)

        metrics_logger.info("count,accuracy,candidate_tree_size,tree_pool_size")

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

            if count % sample_freq == 0 and count != 0:
                accuracy = correct / sample_freq

                candidate_tree_size = 0
                tree_pool_size = 60
                if log_size:
                    candidate_tree_size = classifier.get_candidate_tree_group_size()
                    tree_pool_size = classifier.get_tree_pool_size()

                print(f"{count},{accuracy},{candidate_tree_size},{tree_pool_size}")
                metrics_logger.info(f"{count},{accuracy}," \
                                    f"{candidate_tree_size},{tree_pool_size}")

                correct = 0
                window_actual_labels = []
                window_predicted_labels = []

            # train
            classifier.train()

            classifier.delete_cur_instance()

    @staticmethod
    def prequential_evaluation_proactive_(classifier,
                                         stream,
                                         max_samples,
                                         sample_freq,
                                         metrics_logger):
        np.random.seed(0)

        import grpc
        import seqprediction_pb2
        import seqprediction_pb2_grpc

        correct = 0

        # proactive drift point prediction
        drift_interval_seq_len = 8
        drift_interval_sequence = deque(maxlen=drift_interval_seq_len)
        last_actual_drift_point = 0
        num_request = 0

        metrics_logger.info("count,accuracy,candidate_tree_size,tree_pool_size")

        classifier.init_data_source(stream);

        with grpc.insecure_channel('localhost:50051') as channel:

            stub = seqprediction_pb2_grpc.PredictorStub(channel)

            for count in range(0, max_samples):
                if not classifier.get_next_instance():
                    break

                # test
                prediction = classifier.predict()

                actual_label = classifier.get_cur_instance_label()
                if prediction == actual_label:
                    correct += 1

                if classifier.drift_detected:
                    if classifier.is_state_graph_stable():
                        exit()

                # else:
                #     stub.train(seqprediction_pb2.SequenceMessage(seqId=num_request, seq=drift_interval_sequence))

                if count % sample_freq == 0 and count != 0:
                    accuracy = correct / sample_freq
                    candidate_tree_size = classifier.get_candidate_tree_group_size()
                    tree_pool_size = classifier.get_tree_pool_size()

                    print(f"{count},{accuracy},{candidate_tree_size},{tree_pool_size}")
                    metrics_logger.info(f"{count},{accuracy}," \
                                        f"{candidate_tree_size},{tree_pool_size}")

                    correct = 0

                # train
                classifier.train()

    @staticmethod
    def prequential_evaluation_proactive(classifier,
                                         stream,
                                         max_samples,
                                         sample_freq,
                                         metrics_logger):
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

        next_adapt_state_locs = [-1 for v in range(num_trees)]
        predicted_drift_locs = [-1 for v in range(num_trees)]
        drift_interval_sequences = [deque(maxlen=drift_interval_seq_len) for v in range(num_trees)]
        last_actual_drift_points = [0 for v in range(num_trees)]

        metrics_logger.info("count,accuracy,candidate_tree_size,tree_pool_size")

        classifier.init_data_source(stream)

        with grpc.insecure_channel('localhost:50051') as channel:

            stub = seqprediction_pb2_grpc.PredictorStub(channel)

            stub.setNumTrees(seqprediction_pb2.SetNumTreesMessage(numTrees=num_trees))

            for count in range(0, max_samples):
                if not classifier.get_next_instance():
                    break

                # if classifier.is_state_graph_stable():
                #     t_r = (count - last_actual_drift_point) / (predicted_drift_loc - last_actual_drift_point)
                #     expected_drift_prob = 1 - math.sin(math.pi * t_r)
                #     classifier.set_expected_drift_prob(expected_drift_prob)

                # test
                prediction = classifier.predict()

                actual_label = classifier.get_cur_instance_label()
                if prediction == actual_label:
                    correct += 1

                window_actual_labels.append(actual_label)
                window_predicted_labels.append(prediction)

                drifted_tree_positions = classifier.get_drifted_tree_positions()
                if len(drifted_tree_positions) > 0:
                    # print(f"drifted_tree_positions: {drifted_tree_positions}")

                    if classifier.is_state_graph_stable():
                        # predict the next drift point

                        for idx in drifted_tree_positions:
                            if len(drift_interval_sequences[idx]) >= drift_interval_seq_len:
                                drift_interval_sequences[idx].popleft()

                        response = stub.predict(
                                            seqprediction_pb2
                                                .SequenceMessage(seqId=count,
                                                                 treeId=drifted_tree_positions[0],
                                                                 seq=drift_interval_sequences[0]))

                        # print(f"Predicted next drift point: {response.seq[0]}")
                        if len(response.seq) > 0:
                            interval = fit_predict(clusterer, response.seq[0])

                            for idx in drifted_tree_positions:
                                predicted_drift_locs[idx] = count + interval
                                next_adapt_state_locs[idx] = count + interval + 50

                                drift_interval_sequences[idx].append(interval)
                                last_actual_drift_points[idx] = count

                    else:
                        # find actual drift point at num_instances_before
                        num_instances_before = classifier.find_last_actual_drift_point()

                        for idx in drifted_tree_positions:
                            if num_instances_before > -1:
                                interval = count - num_instances_before - last_actual_drift_points[idx]
                                if interval < 0:
                                    print("Failed to find the actual drift point")
                                    # exit()
                                else:
                                    interval = fit_predict(clusterer, interval)
                                    drift_interval_sequences[idx].append(interval)
                                    last_actual_drift_points[idx] = count - num_instances_before

                                    # train CPT
                                    if len(drift_interval_sequences[idx]) >= drift_interval_seq_len:
                                        num_request += 1
                                        print(f"gRPC train request {num_request}: {drift_interval_sequences[idx]}")
                                        if stub.train(
                                                    seqprediction_pb2
                                                        .SequenceMessage(seqId=num_request,
                                                                         treeId=idx,
                                                                         seq=drift_interval_sequences[idx])
                                                ).result:
                                            pass
                                        else:
                                            print("CPT training failed")
                                            exit()

                if classifier.is_state_graph_stable():
                    transition_tree_pos_list = []
                    adapt_state_tree_pos_list = []

                    for idx in range(num_trees):
                        predicted_drift_loc = predicted_drift_locs[idx]
                        if count >= predicted_drift_loc and predicted_drift_loc != -1:
                            predicted_drift_locs[idx] = -1
                            if not classifier.drift_detected:
                                transition_tree_pos_list.append(idx)

                        next_adapt_state_loc = next_adapt_state_locs[idx]
                        if count >= next_adapt_state_loc and next_adapt_state_loc != -1:
                            next_adapt_state_locs[idx] = -1
                            if not classifier.drift_detected:
                                adapt_state_tree_pos_list.append(idx)

                    if len(transition_tree_pos_list) > 0:
                        classifier.tree_transition(transition_tree_pos_list)
                    if len(adapt_state_tree_pos_list) > 0:
                        classifier.adapt_state(adapt_state_tree_pos_list)

                else:
                    for idx in range(num_trees):
                        predicted_drift_locs[idx] = -1
                        next_adapt_state_locs[idx] = -1


                if count % sample_freq == 0 and count != 0:
                    accuracy = correct / sample_freq
                    candidate_tree_size = classifier.get_candidate_tree_group_size()
                    tree_pool_size = classifier.get_tree_pool_size()

                    print(f"{count},{accuracy},{candidate_tree_size},{tree_pool_size}")
                    metrics_logger.info(f"{count},{accuracy}," \
                                        f"{candidate_tree_size},{tree_pool_size}")

                    correct = 0
                    window_actual_labels = []
                    window_predicted_labels = []

                # train
                classifier.train()

                # classifier.delete_cur_instance()
