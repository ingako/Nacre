#!/usr/bin/env python3

from __future__ import print_function
import logging

import grpc

import seqprediction_pb2
import seqprediction_pb2_grpc

def run():
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = seqprediction_pb2_grpc.PredictorStub(channel)
        response = stub.predict(seqprediction_pb2.SeqMsg(seqId=1, seq=[1, 2, 3, 4]))
    result = ",".join([str(v) for v in response.seq])
    print(f"seqID = {response.seqId}")
    print(f"Seq prediction client received: {result}" )

if __name__ == '__main__':
    logging.basicConfig()
    run()
