#!/usr/bin/venv bash
python3 -m grpc_tools.protoc -I./src/main/proto --python_out=. --grpc_python_out=. seqprediction.proto
