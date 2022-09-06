#%%
import boto3
import numpy as np

from constants import INSTANCE_TYPE
from sagemaker.transformer import Transformer


OUTPUT_BUCKET = "s3://misha-data-bucket/cv/output"
INPUT_BUCKET = "s3://misha-data-bucket/cv/input/list_videos.csv"
ACCEPT = "text/csv"
# ACCEPT = 'application/json'
# INPUT_BUCKET = (
# "s3://misha-data-bucket/input/"  # "s3://misha-data-bucket/919-3_med_2.csv"
# )
MAX_TRANSFORMS = 2
#%%
# import pandas as pd
# df = pd.read_csv(INPUT_BUCKET)
# df.to_json('s3://misha-data-bucket/919-3.json')
#%%
random_suffix = "".join([str(x) for x in np.random.randint(1, 1000, 5)])
request = {
    "TransformJobName": f"image-classifier-job-{random_suffix}",
    "ModelName": "image-classifier",
    "MaxConcurrentTransforms": MAX_TRANSFORMS,
    "MaxPayloadInMB": 32,
    "BatchStrategy": "SingleRecord",
    "DataProcessing": {"InputFilter": "$[" + "0" + "]", "JoinSource": "Input"},
    "TransformOutput": {
        "S3OutputPath": OUTPUT_BUCKET,
        "Accept": ACCEPT,
        "AssembleWith": "Line",
    },
    "TransformInput": {
        "DataSource": {
            "S3DataSource": {"S3DataType": "S3Prefix", "S3Uri": INPUT_BUCKET}
        },
        "ContentType": ACCEPT,
        "SplitType": "Line",
        "CompressionType": "None",
    },
    "TransformResources": {"InstanceType": INSTANCE_TYPE, "InstanceCount": 1},
    "ModelClientConfig": {
        "InvocationsMaxRetries": 2,
        "InvocationsTimeoutInSeconds": 3600,
    },
}


if __name__ == "__main__":
    model_client_config = {
        "InvocationsTimeoutInSeconds": 3600,
        "InvocationsMaxRetries": 2,
    }

    client = boto3.client("sagemaker")
    client.create_transform_job(**request)
    print(f'Job {request["TransformJobName"]} is sent')

    # sagemaker_session = sagemaker.Session()
    # xgb_transformer = Transformer(
    #     model_name=MODEL_NAME,
    #     instance_count=1,
    #     instance_type=INSTANCE_TYPE,
    #     output_path=OUTPUT_BUCKET,
    #     strategy="MultiRecord",
    #     max_concurrent_transforms=16,
    #     assemble_with="Line",
    #     accept=ACCEPT,
    #     sagemaker_session=sagemaker_session,
    #     max_payload=4,
    # )

    # # Call transform on your test data
    # # Test data (new data) should be in S3
    # print("started...")
    # model_client_config = {
    #     "InvocationsTimeoutInSeconds": 3600,
    #     "InvocationsMaxRetries": 2,
    # }
    # xgb_transformer.transform(
    #     INPUT_BUCKET,
    #     content_type=ACCEPT,
    #     split_type="Line",
    #     wait=True,
    #     join_source="Input",
    #     model_client_config=model_client_config,
    # )
