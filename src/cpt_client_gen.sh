#!/usr/bin/env bash
python -m grpc_tools.protoc -I../grpc/src/main/proto --python_out=. --grpc_python_out=. seqprediction.proto
