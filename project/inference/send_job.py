import boto3
import numpy as np

OUTPUT_BUCKET = "s3://misha-data-bucket/processed"
INPUT_BUCKET = "s3://misha-data-bucket/919-3_small.csv"
INSTANCE_TYPE = "ml.m5.large"
MAX_TRANSFORMS = 4


random_suffix = "".join([str(x) for x in np.random.randint(1, 1000, 5)])
request = {
    "TransformJobName": f"sentiment-job-{random_suffix}",
    "ModelName": "sentiment-classifier",
    "MaxConcurrentTransforms": MAX_TRANSFORMS,
    "BatchStrategy": "SingleRecord",
    "DataProcessing": {"JoinSource": "Input"},
    "TransformOutput": {
        "S3OutputPath": OUTPUT_BUCKET,
        "Accept": "text/csv",
        "AssembleWith": "Line",
    },
    "TransformInput": {
        "DataSource": {
            "S3DataSource": {"S3DataType": "S3Prefix", "S3Uri": INPUT_BUCKET}
        },
        "ContentType": "text/csv",
        "SplitType": "Line",
        "CompressionType": "None",
    },
    "TransformResources": {"InstanceType": INSTANCE_TYPE, "InstanceCount": 1},
}


if __name__ == "__main__":
    client = boto3.client("sagemaker")
    client.create_transform_job(**request)
