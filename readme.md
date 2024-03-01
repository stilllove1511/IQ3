generate grpc file
```
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. ops/ml_api.proto
```