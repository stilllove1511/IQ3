import grpc
from concurrent import futures

from ops.ml_api_pb2_grpc import MLServiceStub
from ops.ml_api_pb2 import DownloadRequest, DownloadResponse

token = "right_token"

with grpc.insecure_channel("localhost:50051") as channel:
    stub = MLServiceStub(channel)

    download_request = DownloadRequest()
    download_request.token = token
    download_request.file_type = 1
    download_request.upload_id = "1"
    response_iterator = stub.Download(download_request)

    with open(f"downloaded_logs_{download_request.upload_id}.zip", "wb") as file:
        for response in response_iterator:
            file.write(response.file_data)
