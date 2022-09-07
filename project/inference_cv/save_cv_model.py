from torchvision import models
import torch
import tarfile
import os
import boto3


from constants import S3_BUCKET_PATH, SAVE_PATH, SAVE_PATH_TAR


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
            os.remove(SAVE_PATH)
        if os.path.exists(SAVE_PATH_TAR):
            print(f"Cleaning {SAVE_PATH_TAR}")
            os.remove(SAVE_PATH_TAR)

    def make_tarfile(self, output_filename, source_dir):
        with tarfile.open(output_filename, "w:gz") as tar:
            tar.add(source_dir)
            # tar.add(source_dir, arcname=os.path.basename(source_dir))

    def upload_s3_file(self, filename, local_filename):
        s3 = boto3.client("s3")
        with open(local_filename, "rb") as f:
            s3.upload_fileobj(f, S3_BUCKET_PATH, filename)


with ModelSaver() as mv:
    model = models.resnet18(pretrained=True)
    torch.save(model.state_dict(), SAVE_PATH)

    mv.make_tarfile(SAVE_PATH_TAR, SAVE_PATH)
    mv.upload_s3_file(SAVE_PATH_TAR, SAVE_PATH_TAR)

    print(f"File {SAVE_PATH_TAR} uploaded to {S3_BUCKET_PATH}/{SAVE_PATH_TAR}")


# FPR = FP / (FP + TN)
# Sensitivity = Recall = TPR = TP / (TP + FN)
# Specificity = TN / (FP + TN) = 1 - FPR != Precision
# Precision = TP / (TP + FP)
# Accuracy = (TP + TN) / (TP + TN + FP + FN)
