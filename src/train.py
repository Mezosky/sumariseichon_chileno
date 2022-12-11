# Importing libraries
import logging

import numpy as np
import pandas as pd
import torch

from trainer import mT5_trainer

# Loggger
logging.basicConfig(
    filename="finetunning.log",
    encoding="utf-8",
    level=logging.INFO,
    format="%(asctime)s :  %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
)

# set up logging to console
console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
# set a format which is simpler for console use
formatter = logging.Formatter("%(asctime)s %(message)s", datefmt="%m/%d/%Y %I:%M:%S %p")
console.setFormatter(formatter)
# add the handler to the root logger
logging.getLogger("").addHandler(console)
logger = logging.getLogger(__name__)

# Model Hyperparams
model_params = {
    "MODEL": "google/mt5-small",  # model_type: mt5-base/mt5-large
    "TRAIN_BATCH_SIZE": 16,  # training batch size
    "VALID_BATCH_SIZE": 16,  # validation batch size
    "TRAIN_EPOCHS": 10,  # number of training epochs
    "LEARNING_RATE": 2e-3,  # learning rate
    "MAX_SOURCE_TEXT_LENGTH": 512,  # max length of source text
    "MAX_TARGET_TEXT_LENGTH": 150,  # max length of target text
    "SEED": 42,  # set seed for reproducibility
}

# Set random seeds and deterministic pytorch for reproducibility
torch.manual_seed(model_params["SEED"])
np.random.seed(model_params["SEED"])
torch.backends.cudnn.deterministic = True


def preprocess_dataset(
    df: pd.DataFrame,
    min_text_tokens: int = 20,
    max_text_tokens: int = 100,
    min_title_tokens: int = 5,
    min_subhead_tokens: int = 10,
) -> pd.DataFrame:

    # dropna
    df = df.dropna()

    # delete all text, title and subhead nulls
    df = df.loc[df["text"] != "", :]
    df = df.loc[df["title"] != "", :]
    df = df.loc[df["subhead"] != "", :]

    # drop every article that has less or more text tokens than the allowed
    text_n_tokens = df["text"].str.split(" ").apply(len)
    df = df.loc[
        (text_n_tokens >= min_text_tokens) & (text_n_tokens <= max_text_tokens), :
    ]

    # drop every article that has less title tokens than the allowed
    title_n_tokens = df["title"].str.split(" ").apply(len)
    df = df.loc[(title_n_tokens >= min_title_tokens), :]

    # drop every article that has less subhead tokens than the allowed
    subhead_n_tokens = df["subhead"].str.split(" ").apply(len)
    df = df.loc[(subhead_n_tokens >= min_subhead_tokens), :]

    return df


def main():
    """Load data and start the finetune procces."""

    logger.info("[Dataset] Loading dataset...")
    df = pd.read_parquet(
        "/data/imeza/text_datasets/data_summarization_with_title.parquet"
    ).sample(40000, random_state=42)
    # add summarize instruction to t5 to the main text.
    df["text"] = "summarize: " + df["text"]

    logger.info("[Dataset] Dataset load successfully completed.")

    mT5_trainer(
        source_text=df.loc[:, "text"],
        target_text=df.loc[:, "headlines"],
        model_params=model_params,
        output_dir="/data/imeza/text_datasets/outputs_mT5",
    )


if __name__ == "__main__":
    main()
