import io
import torch
import logging
import numpy as np
import pandas as pd
import torch.nn.functional as F
import multiprocessing as mp
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer
from datasets import Dataset

max_len = 128
pretrained_model_name = "cardiffnlp/twitter-roberta-base-sentiment"
class_names = ["negative", "neutral", "positive"]


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

    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    tokenizer = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
    except EnvironmentError as e:
        logging.error(e)
        logging.error(
            f"{model_dir} does not have pretrained Tokenizer. Fallback to standard..."
        )
    finally:
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)

    model = model.to(device)
    return model, tokenizer


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

    input_ids = []
    attention_masks = []
    model, tokenizer = model_artifacts

    def softmax(z):
        assert len(z.shape) == 2
        s = np.max(z, axis=1)
        s = s[:, np.newaxis]  # necessary step to do broadcasting
        e_x = np.exp(z - s)
        div = np.sum(e_x, axis=1)
        div = div[:, np.newaxis]  # dito
        return e_x / div

    # limit number of CPU threads to be used per worker
    cpu_count = mp.cpu_count()
    if cpu_count > 1:
        torch.set_num_threads(cpu_count // 2)

    # encode input text
    model.eval()
    with torch.no_grad():
        # for row in input_data:
        #     encoded_row = tokenizer.encode_plus(
        #         text=row,
        #         add_special_tokens=True,
        #         max_length=max_len,
        #         pad_to_max_length=True,
        #         return_attention_mask=True,
        #         truncation=True,
        #     )
        #     input_ids.append(encoded_row.get("input_ids"))
        #     attention_masks.append(encoded_row.get("attention_mask"))
        ds = Dataset.from_dict({"text": input_data})
        predict_df = ds.map(
            lambda x: tokenizer(
                text=x["text"],
                add_special_tokens=True,
                max_length=max_len,
                pad_to_max_length=True,
                return_attention_mask=True,
                truncation=True,
            ),
            batched=True,
        )
        trainer = Trainer(model=model, tokenizer=tokenizer)
        trainer.args.per_device_eval_batch_size = 16
        trainer.args.output_dir = None
        trainer.args.overwrite_output_dir = True

        logits = trainer.predict(predict_df)
        probs = softmax(logits[0])

        # return the class with the highest prob with corresponding index
        # input_ids = torch.tensor(input_ids)
        # attention_masks = torch.tensor(attention_masks)
        # logits = model(input_ids, attention_masks)
        # probs = F.softmax(logits[0], dim=1)
        logging.info("Prediction done...")
        return probs


def output_fn(prediction_output, accept):
    """
    Serializes the prediction result into the desired response content type
    Args:
        predictions: a list of predictions generated by predict_fn()
    Returns:
        an output list, saved with the .out extension in an S3 bucket
    """

    logging.info("in output_fn()")

    final_predictions = []

    # associate class with predictions
    for prediction in prediction_output:
        final_predictions.append(
            (
                class_names[prediction.argmax()],
                np.max(prediction),
                prediction[0],
                prediction[1],
                prediction[2],
            )
        )

    logging.info(f"MISHA: Generated prediction len: {len(final_predictions)}")

    return pd.DataFrame(final_predictions).to_csv(index=False, header=None)
    # return pd.DataFrame(
    #     final_predictions,
    #     columns=[
    #         "label",
    #         "major_probability",
    #         "negative_probability",
    #         "neutral_probability",
    #         "positive_probability",
    #     ],
    # ).to_csv(index=False)
