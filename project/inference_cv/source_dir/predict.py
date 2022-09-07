import io
import torch
import logging
import numpy as np
import pandas as pd
import os
from torchvision import models


def model_fn(model_dir):
    """
    Loads model and its weights,and return the loaded model
    for inference
    Args:
        model_dir: the path to the S3 bucket containing the model file
    Returns:
        loaded model transferred to the appropriate device
    """

    logging.info("in model_fn()")

    # compute processor
    device = torch.device(
        "cuda:" + str(torch.cuda.current_device())
        if torch.cuda.is_available()
        else "cpu"
    )

    logging.info(f"Loading model from {model_dir}")
    model = models.resnet18(pretrained=False)
    path = os.path.join(model_dir, "image_classifier.pth")
    # with open(os.path.join(model_dir, "image_classifier.pth"), "rb") as f:
    model.load_state_dict(torch.load(path))

    model = model.to(device)
    return model


def input_fn(serialized_input_data, input_content_type):
    """
    Takes request data and deserialises the data into an object for prediction
    Args:
        request_body: a byte buffer array
        content_type: a python string, written as "application/x"
        where x is the content type csv.
    Returns:
        dataframe series
    """

    logging.info("in input_fn()")

    # validate input content type
    if input_content_type == input_content_type:
        # if isinstance(serialized_input_data, (bytes, bytearray)):
        # logging.info('MISHA: Converting from bytes array')
        # logging.info("Misha:", serialized_input_data)
        df = pd.read_csv(io.StringIO(serialized_input_data), dtype=str, header=None)[0]
        logging.info(f"Misha: {df.head()}")
        logging.info(f"Misha: len of df is: {len(df)}")
        return df

        # if not isinstance(serialized_input_data, str):
        #     serialized_input_data = str(serialized_input_data, "utf-8")
        # serialized_input_data = io.StringIO(serialized_input_data)
        # # logging.info(serialized_input_data)
        # df = pd.read_csv(serialized_input_data)
        # logging.info(df.head())
        # logging.info(f"Successfully read csv, payload item {df}")
        # return df["truncatedText"].values
    else:
        raise ValueError(f"Unsupported content type:{input_content_type}")


def predict_fn(input_data, model_artifacts):
    """
    Customises how the model server gets predictions from the loaded model
    Args:
        input_data:data loaded via input_fn()(i.e. each row)
        model: model loaded via model_fn() above
    Returns:
        a numpy array (2D) where each row is an entry
    """

    logging.info("in predict_fn()")
    logging.info(f"MISHA: len - {len(input_data)}")

    return len(input_data)

def output_fn(prediction_output, accept):
    """
    Serializes the prediction result into the desired response content type
    Args:
        predictions: a list of predictions generated by predict_fn()
    Returns:
        an output list, saved with the .out extension in an S3 bucket
    """

    logging.info("in output_fn()")

    return prediction_output
