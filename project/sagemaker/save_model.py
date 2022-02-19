from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
import tarfile
import os
import boto3
import shutil as sh


S3_BUCKET_PATH = "misha-ml-models"
checkpoint = "cardiffnlp/twitter-roberta-base-sentiment"
save_path = "sentiment_classifier"
save_path_tar = f"{save_path}.tar.gz"


class ModelSaver:
    def __enter__(self):
        print("Entered")
        self.cleanup()
        return self

    def __exit__(self, type, value, traceback):
        print("Exited")
        self.cleanup()

    def cleanup(self):
        if os.path.exists(save_path):
            print(f"Cleaning {save_path}")
            sh.rmtree(save_path)
        if os.path.exists(save_path_tar):
            print(f"Cleaning {save_path_tar}")
            os.remove(save_path_tar)

    def make_tarfile(self, output_filename, source_dir):
        with tarfile.open(output_filename, "w:gz") as tar:
            tar.add(source_dir, arcname=os.path.basename(source_dir))

    def upload_s3_file(self, filename, local_filename):
        s3 = boto3.client("s3")
        with open(local_filename, "rb") as f:
            s3.upload_fileobj(f, S3_BUCKET_PATH, filename)


with ModelSaver() as mv:
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
    trainer = Trainer(model=model, tokenizer=tokenizer)
    trainer.save_model(save_path)

    mv.make_tarfile(save_path_tar, save_path)
    mv.upload_s3_file(save_path_tar, save_path_tar)

    print(f"File {save_path_tar} uploaded to {S3_BUCKET_PATH}/{save_path_tar}")
