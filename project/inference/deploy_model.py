import sagemaker
from sagemaker.pytorch import PyTorchModel
import logging
import sys

from constants import (
    S3_BUCKET_PATH,
    SAVE_PATH_TAR,
    MODEL_NAME,
    S3_SAGEMAKER_ARTIFACT_BUCKET,
)
from constants import INSTANCE_TYPE

SAGEMAKER_ROLE = "AmazonSageMaker-ExecutionRole-20220219T085820"
PYTORCH_FRAMEWORK_VERSION = "1.7"
PYTORCH_PYTHON_VERSION = "py36"
SAGEMAKER_INFERENCE_ENTRY_POINT = "predict.py"


def create_and_deploy_model(trained_model_path):
    try:
        logging.info(f"Started {trained_model_path} deployment")
        sagemaker_session = sagemaker.Session()

        model = PyTorchModel(
            model_data=trained_model_path,
            name=MODEL_NAME,
            role=SAGEMAKER_ROLE,
            framework_version=PYTORCH_FRAMEWORK_VERSION,
            py_version=PYTORCH_PYTHON_VERSION,
            entry_point=SAGEMAKER_INFERENCE_ENTRY_POINT,
            code_location=f"s3://{S3_SAGEMAKER_ARTIFACT_BUCKET}/{MODEL_NAME}",
            source_dir="source_dir",
        )

        model.sagemaker_session = sagemaker_session

        container_def = model.prepare_container_def(instance_type=INSTANCE_TYPE)

        logging.info(f"Definition of container: {container_def}")

        sagemaker_session.create_model(MODEL_NAME, SAGEMAKER_ROLE, container_def)
        logging.info(f"Model {MODEL_NAME} deployed successfully")

    except Exception as ex:
        logging.error(ex)
        raise ex


if __name__ == "__main__":
    logging.basicConfig(filename="deploy_model.log", level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    model_path = f"s3://{S3_BUCKET_PATH}/{SAVE_PATH_TAR}"
    create_and_deploy_model(model_path)

