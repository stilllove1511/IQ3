import os
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
from services.predictor import Predictor
from services.trainer import Trainer


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

        with open(f"data/custom_dataset_{upload_id}.jsonl", "wb") as file:
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

    def ExecuteAction(self, request, context):
        if not authenticate(request.token):
            context.set_code(grpc.StatusCode.UNAUTHENTICATED)
            return ActionResponse()

        payload = json.loads(request.payload)

        if request.action_type == 0:
            predictor = Predictor(payload["model_path"])
            response = predictor.predict(payload["prompt"])
            return ActionResponse(status="SUCCESS", payload=response)
        elif request.action_type == 1:
            # check if training is already running
            active_run = mlflow.active_run()
            if active_run is not None:
                return ActionResponse(
                    status="TRAINING", response="Training is already running"
                )

            train_dataset = load_dataset(
                "json",
                data_files=f"data/custom_dataset_{payload['upload_id']}.jsonl",
                split="train",
            )
            validation_dataset = load_dataset(
                "json",
                data_files=f"data/custom_dataset_{payload['upload_id']}.jsonl",
                split="train",
            )
            trainer = Trainer(payload["upload_id"], train_dataset, validation_dataset)
            trainer.train(is_logged=True)
            return ActionResponse(status="SUCCESS")

    def Download(self, request, context):
        if not authenticate(request.token):
            context.set_code(grpc.StatusCode.UNAUTHENTICATED)
            return DownloadResponse()

        import zipfile

        if request.file_type == 0:
            with zipfile.ZipFile(
                f"./logs/bloomz-3b-custom-finetune-{request.upload_id}.zip", "w"
            ) as zipf:
                for root, dirs, files in os.walk(
                    f"./logs/bloomz-3b-custom-finetune-{request.upload_id}"
                ):
                    for file in files:
                        zipf.write(os.path.join(root, file))
            with zipfile.ZipFile(
                f"./logs/bloomz-3b-custom-finetune-{request.upload_id}.zip", "r"
            ) as zipf:
                for file_info in zipf.infolist():
                    with zipf.open(file_info) as file:
                        chunk = file.read(1024 * 1024)
                        while chunk:
                            yield DownloadResponse(file_data=chunk)
                            chunk = file.read(1024 * 1024)

        else:
            with zipfile.ZipFile(f"./mlruns/0/{request.upload_id}.zip", "w") as zipf:
                for root, dirs, files in os.walk(f"./mlruns/0/{request.upload_id}"):
                    for file in files:
                        zipf.write(os.path.join(root, file))
            with zipfile.ZipFile(f"./mlruns/0/{request.upload_id}.zip", "r") as zipf:
                for file_info in zipf.infolist():
                    with zipf.open(file_info) as file:
                        chunk = file.read(1024 * 1024)
                        while chunk:
                            yield DownloadResponse(file_data=chunk)
                            chunk = file.read(1024 * 1024)

        return DownloadResponse(
            file_data=open(f"data/custom_dataset_{request.upload_id}.zip", "rb").read()
        )


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    ml_service = MLServiceImpl()
    ml_api_pb2_grpc.add_MLServiceServicer_to_server(ml_service, server)
    server.add_insecure_port("[::]:50051")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
