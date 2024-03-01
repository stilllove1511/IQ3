import grpc
import mlflow
from concurrent import futures
import json
from datasets import load_dataset

from ops import ml_api_pb2_grpc
from ops.ml_api_pb2 import (
    UploadDataResponse,
    GetStatusRequest,
    GetStatusResponse,
    ActionType,
    ActionRequest,
    ActionResponse,
    DownloadRequest,
    DownloadResponse,
)


def authenticate(token):
    return True if token == "right_token" else False


class MLServiceImpl(ml_api_pb2_grpc.MLServiceServicer):

    def UploadData(self, request, context):
        # Authenticate
        if not authenticate(request.token):
            context.aset_code(grpc.StatusCode.UNAUTHENTICATED)
            return UploadDataResponse()

        upload_id = "1"

        print(f"File data: {request.file_data}")

        with open(f"data/custom_dataset_{upload_id}.json1", "wb") as file:
            file.write(request.file_data[0])

        return UploadDataResponse(upload_id=upload_id)

    def GetStatus(self, request, context):
        # Authenticate
        if not authenticate(request.token):
            context.aset_code(grpc.StatusCode.UNAUTHENTICATED)
            return GetStatusResponse()

        active_run = mlflow.active_run()
        if active_run is not None:
            print(f"Active run: {active_run.info.run_id}")
            status = "TRAINING"
        else:
            status = "IDLE"

        return GetStatusResponse(status=status)

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    ml_service = MLServiceImpl()
    ml_api_pb2_grpc.add_MLServiceServicer_to_server(ml_service, server)
    server.add_insecure_port("[::]:50051")
    server.start()
    server.wait_for_termination()
    
if __name__ == "__main__":
    serve()