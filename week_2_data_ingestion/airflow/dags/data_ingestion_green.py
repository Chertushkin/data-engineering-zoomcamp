import logging
import os
from datetime import datetime

import pyarrow.csv as pv
import pyarrow.parquet as pq
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from google.cloud import storage

AIRFLOW_HOME = os.environ.get("AIRFLOW_HOME", "/opt/airflow/")
PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
BUCKET = os.environ.get("GCP_GCS_BUCKET")
URL_PREFIX = "https://s3.amazonaws.com/nyc-tlc/trip+data"
URL_TEMPLATE = (
    URL_PREFIX + "/green_tripdata_{{ execution_date.strftime('%Y-%m') }}.csv"
)
OUTPUT_FILE_TEMPLATE = (
    AIRFLOW_HOME + "/green_tripdata_{{ execution_date.strftime('%Y-%m') }}.csv"
)

OUTPUT_FILE_PARQUET = OUTPUT_FILE_TEMPLATE.replace(".csv", ".parquet")
SHORT_FILE_PARQUET = os.path.basename(OUTPUT_FILE_PARQUET)

def format_to_parquet(src_file):
    if not src_file.endswith(".csv"):
        logging.error("Can only accept source files in CSV format, for the moment")
        return
    table = pv.read_csv(src_file)
    pq.write_table(table, src_file.replace(".csv", ".parquet"))


# NOTE: takes 20 mins, at an upload speed of 800kbps. Faster if your internet has a better upload speed
def upload_to_gcs(bucket, object_name, local_file):
    """
    Ref: https://cloud.google.com/storage/docs/uploading-objects#storage-upload-object-python
    :param bucket: GCS bucket name
    :param object_name: target path & file-name
    :param local_file: source path & file-name
    :return:
    """
    # WORKAROUND to prevent timeout for files > 6 MB on 800 kbps upload speed.
    # (Ref: https://github.com/googleapis/python-storage/issues/74)
    storage.blob._MAX_MULTIPART_SIZE = 5 * 1024 * 1024  # 5 MB
    storage.blob._DEFAULT_CHUNKSIZE = 5 * 1024 * 1024  # 5 MB
    # End of Workaround

    client = storage.Client()
    bucket = client.bucket(bucket)

    blob = bucket.blob(object_name)
    blob.upload_from_filename(local_file)


with DAG(
    dag_id="data_ingestion_green_2019_2020_v2",
    schedule_interval="0 6 2 * *",
    start_date=datetime(2019, 1, 1),
    end_date=datetime(2020, 12, 31),
    catchup=True,
    max_active_runs=3,
    tags=["dtc-de"],
) as dag:

    wget_task = BashOperator(
        task_id="wget",
        bash_command=f"curl -sSLf {URL_TEMPLATE} > {OUTPUT_FILE_TEMPLATE}",
    )

    print(f"Finished wget for {URL_TEMPLATE}")
    format_to_parquet_task = PythonOperator(
        task_id="format_to_parquet_task",
        python_callable=format_to_parquet,
        op_kwargs={"src_file": f"{OUTPUT_FILE_TEMPLATE}"},
    )
    print(f"Finished parquet for {OUTPUT_FILE_PARQUET}")
    # TODO: Homework - research and try XCOM to communicate output values between 2 tasks/operators
    local_to_gcs_task = PythonOperator(
        task_id="local_to_gcs_task",
        python_callable=upload_to_gcs,
        op_kwargs={
            "bucket": BUCKET,
            "object_name": f"raw/{SHORT_FILE_PARQUET}",
            "local_file": f"{OUTPUT_FILE_PARQUET}",
        },
    )
    print(f"Finished upload for {SHORT_FILE_PARQUET}")
    wget_task >> format_to_parquet_task >> local_to_gcs_task
