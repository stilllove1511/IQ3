import grpc
from concurrent import futures

from ops.ml_api_pb2_grpc import MLServiceStub
from ops.ml_api_pb2 import UploadDataRequest, GetStatusRequest, ActionRequest

token = "right_token"

with grpc.insecure_channel("localhost:50051") as channel:
    stub = MLServiceStub(channel)

    action_request = ActionRequest()
    action_request.token = token
    action_request.action_type = 1
    action_request.payload = '{"upload_id": "1"}'

    action_response = stub.ExecuteAction(action_request)
    print(f"Action response: {action_response.status}")
