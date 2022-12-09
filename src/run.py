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
    "TRAIN_BATCH_SIZE": 8,  # training batch size
    "VALID_BATCH_SIZE": 8,  # validation batch size
    "TRAIN_EPOCHS": 10,  # number of training epochs
    "LEARNING_RATE": 1e-5,  # learning rate
    "MAX_SOURCE_TEXT_LENGTH": 512,  # max length of source text
    "MAX_TARGET_TEXT_LENGTH": 60,  # max length of target text
    "SEED": 42,  # set seed for reproducibility
}

# Set random seeds and deterministic pytorch for reproducibility
torch.manual_seed(model_params["SEED"])
np.random.seed(model_params["SEED"])
torch.backends.cudnn.deterministic = True


def main():
    """Load data and start the finetune procces."""

    logger.info("[Dataset] Loading dataset...")
    df = pd.read_parquet(
        "/data/imeza/text_datasets/data_summarization_with_title.parquet"
    ).sample(1000, random_state=42)
    # add summarize instruction to t5 to the main text.
    df["text"] = "summarize: " + df["text"]

    logger.info("[Dataset] Dataset load successfully completed.")

    mT5_trainer(
        source_text=df.loc[:, "text"],
        target_text=df.loc[:, "title"],
        model_params=model_params,
        output_dir="/data/imeza/text_datasets/outputs_mT5",
    )

if __name__ == "__main__":
    main()
