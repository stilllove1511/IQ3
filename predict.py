import grpc
from concurrent import futures

from ops.ml_api_pb2_grpc import MLServiceStub
from ops.ml_api_pb2 import UploadDataRequest, GetStatusRequest, ActionRequest

token = "right_token"

with grpc.insecure_channel("localhost:50051") as channel:
    stub = MLServiceStub(channel)
    prompt = (
        "You are a smart assistant. Question: What is the capital of France? Answer:"
    )
    action_request = ActionRequest()
    action_request.token = token
    action_request.action_type = 0
    action_request.payload = (
        '{"model_path": "/workspaces/IQ3/logs/blomz-3b-custom-finetune-1/checkpoint-500", "prompt": "'
        + prompt
        + '"}'
    )

    action_response = stub.ExecuteAction(action_request)
    print("Action response: ", action_response.response)
