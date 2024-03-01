import grpc
from concurrent import futures

from ops.ml_api_pb2_grpc import MLServiceStub
from ops.ml_api_pb2 import UploadDataRequest, GetStatusRequest, ActionRequest

token = "right_token"

with grpc.insecure_channel("localhost:50051") as channel:
    stub = MLServiceStub(channel)

    status_response = stub.GetStatus(GetStatusRequest(token=token))
    print("Status response: ", status_response.status)
