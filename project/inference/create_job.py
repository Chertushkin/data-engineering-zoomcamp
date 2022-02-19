from cv2 import log
import numpy as np
import pandas as pd
import sagemaker
from sagemaker.pytorch import PyTorchModel
import logging
import sys

def create_and_deploy_model(trained_model_path):
    try:
        logging.info('Started deployment')
    except Exception as e:
        logging.error(e)

if __name__ == '__main__':
    logging.basicConfig(filename='create_job.log', level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    create_and_deploy_model('hh')

