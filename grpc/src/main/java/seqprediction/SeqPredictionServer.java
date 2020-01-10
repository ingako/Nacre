/*
 * Copyright 2015 The gRPC Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package seqprediction;

import io.grpc.Server;
import io.grpc.ServerBuilder;
import io.grpc.stub.StreamObserver;
import java.io.IOException;
import java.util.logging.Logger;

import java.util.ArrayList;
import java.util.List;

import ipredict.database.Item;
import ipredict.database.Sequence;
import ipredict.predictor.CPT.CPTPlus.CPTPlusPredictor;
import ipredict.predictor.CPT.CPT.CPTPredictor;
import ipredict.predictor.Markov.MarkovAllKPredictor;
import ipredict.predictor.profile.DefaultProfile;

/**
 * Server that manages startup/shutdown of a {@code Predictor} server.
 */
public class SeqPredictionServer {
    private static final Logger logger = Logger.getLogger(SeqPredictionServer.class.getName());

    private Server server;

    private void start() throws IOException {
        /* The port on which the server should run */
        int port = 50051;
        server = ServerBuilder.forPort(port)
            .addService(new PredictorImpl())
            .build()
            .start();

        logger.info("Server started, listening on " + port);
        Runtime.getRuntime().addShutdownHook(new Thread() {
                @Override
                public void run() {
                // Use stderr here since the logger may have been reset by its JVM shutdown hook.
                System.err.println("*** shutting down gRPC server since JVM is shutting down");
                SeqPredictionServer.this.stop();
                System.err.println("*** server shut down");
                }
                });
    }

    private void stop() {
        if (server != null) {
            server.shutdown();
        }
    }

    /**
     * Await termination on the main thread since the grpc library uses daemon threads.
     */
    private void blockUntilShutdown() throws InterruptedException {
        if (server != null) {
            server.awaitTermination();
        }
    }

    /**
     * Main launches the server from the command line.
     */
    public static void main(String[] args) throws IOException, InterruptedException {
        final SeqPredictionServer server = new SeqPredictionServer();
        server.start();
        server.blockUntilShutdown();
    }

    static class PredictorImpl extends PredictorGrpc.PredictorImplBase {

        CPTPredictor cpt;
        DefaultProfile profile;

        private void init() {
            // initializing the CPT predictor
            if (cpt == null) {
                cpt = new CPTPredictor();

                // setting the experiment parameters
                profile = new DefaultProfile();
                profile.Apply();
            }
        }

        private Sequence getSequenceFromRequest(SequenceMessage request) {
            int count = request.getSeqCount();
            Sequence seq = new Sequence(request.getSeqId());

            for (int i = 0; i < count; i++) {
                seq.addItem(new Item(request.getSeq(i)));
            }

            return seq;
        }

        @Override
        public void predict(SequenceMessage request, StreamObserver<SequenceMessage> responseObserver) {

            init();

            // prepare sequence
            Sequence seq = getSequenceFromRequest(request);

            // predicting a sequence
            Sequence predicted = cpt.Predict(seq);
            System.out.println("Predicted symbol: " + predicted);

            SequenceMessage.Builder builder = SequenceMessage.newBuilder();
            for (Item item : predicted.getItems()) {
                builder.addSeq(item.val);
            }

            SequenceMessage reply = builder.build();

            responseObserver.onNext(reply);
            responseObserver.onCompleted();
        }

        @Override
        public void train(seqprediction.SequenceMessage request,
                          io.grpc.stub.StreamObserver<seqprediction.TrainResponse> responseObserver) {

            init();

            // prepare sequence
            Sequence seq = getSequenceFromRequest(request);

            List<Sequence> trainingSet = new ArrayList<Sequence>();
            trainingSet.add(seq);
            cpt.Train(trainingSet);

            TrainResponse reply = TrainResponse.newBuilder().setResult(true).build();

            responseObserver.onNext(reply);
            responseObserver.onCompleted();
        }
    }

}
