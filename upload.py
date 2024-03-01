import grpc
from concurrent import futures

from ops.ml_api_pb2_grpc import MLServiceStub
from ops.ml_api_pb2 import UploadDataRequest, GetStatusRequest, ActionRequest

token = "right_token"
upload_data = open("./data/1.jsonl", "rb").read()

with grpc.insecure_channel("localhost:50051") as channel:
    stub = MLServiceStub(channel)

    upload_data_request = UploadDataRequest(token=token, file_data=[upload_data])
    upload_response = stub.UploadData(upload_data_request)

    print(f"Upload Id: {upload_response.upload_id}")

    status_response = stub.GetStatus(GetStatusRequest(token=token))
    print(f"Status: {status_response.status}")
