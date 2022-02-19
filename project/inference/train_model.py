from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
import tarfile
import os
import boto3
import shutil as sh


from constants import S3_BUCKET_PATH, SAVE_PATH, SAVE_PATH_TAR

checkpoint = "cardiffnlp/twitter-roberta-base-sentiment"


class ModelSaver:
    def __enter__(self):
        print("Entered")
        self.cleanup()
        return self

    def __exit__(self, type, value, traceback):
        print("Exited")
        self.cleanup()

    def cleanup(self):
        if os.path.exists(SAVE_PATH):
            print(f"Cleaning {SAVE_PATH}")
            sh.rmtree(SAVE_PATH)
        if os.path.exists(SAVE_PATH_TAR):
            print(f"Cleaning {SAVE_PATH_TAR}")
            os.remove(SAVE_PATH_TAR)

    def make_tarfile(self, output_filename, source_dir):
        with tarfile.open(output_filename, "w:gz") as tar:
            tar.add(source_dir, arcname='.')
            # tar.add(source_dir, arcname=os.path.basename(source_dir))

    def upload_s3_file(self, filename, local_filename):
        s3 = boto3.client("s3")
        with open(local_filename, "rb") as f:
            s3.upload_fileobj(f, S3_BUCKET_PATH, filename)



with ModelSaver() as mv:
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
    trainer = Trainer(model=model, tokenizer=tokenizer)
    trainer.save_model(SAVE_PATH)

    mv.make_tarfile(SAVE_PATH_TAR, SAVE_PATH)
    mv.upload_s3_file(SAVE_PATH_TAR, SAVE_PATH_TAR)

    print(f"File {SAVE_PATH_TAR} uploaded to {S3_BUCKET_PATH}/{SAVE_PATH_TAR}")
