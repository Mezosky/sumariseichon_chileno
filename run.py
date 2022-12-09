# Importing libraries
import os
import logging
import numpy as np
import pandas as pd
import torch
from torch import cuda
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import os

# Importing the mT5 modules from huggingface/transformers
from transformers import T5Tokenizer, MT5Model, MT5Config, MT5ForConditionalGeneration
from dataset import SummaryDataSet
from trainer import mT5Trainer

# Loggger
logging.basicConfig(
    filename = 'finetunning.log', 
    encoding = 'utf-8', 
    level    = logging.INFO, 
    format   = '%(asctime)s :  %(message)s', 
    datefmt  = '%m/%d/%Y %I:%M:%S %p'
)

# set up logging to console
console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
# set a format which is simpler for console use
formatter = logging.Formatter(
    '%(asctime)s %(message)s',
    datefmt  = '%m/%d/%Y %I:%M:%S %p'
)
console.setFormatter(formatter)
# add the handler to the root logger
logging.getLogger('').addHandler(console)
logger = logging.getLogger(__name__)

# Model Hyperparams
model_params={
    "MODEL":"google/mt5-small",     # model_type: mt5-base/mt5-large
    "TRAIN_BATCH_SIZE":8,           # training batch size
    "VALID_BATCH_SIZE":8,           # validation batch size
    "TRAIN_EPOCHS":2,               # number of training epochs
    "VAL_EPOCHS":1,                 # number of validation epochs
    "LEARNING_RATE":2e-4,           # learning rate
    "MAX_SOURCE_TEXT_LENGTH":512,   # max length of source text
    "MAX_TARGET_TEXT_LENGTH":258,   # max length of target text
    "SEED": 42                      # set seed for reproducibility 
    }

# Set random seeds and deterministic pytorch for reproducibility
torch.manual_seed(model_params["SEED"])
np.random.seed(model_params["SEED"]) 
torch.backends.cudnn.deterministic = True


def main():
    df = pd.read_parquet("/data/imeza/text_datasets/data_summarization_with_title.parquet")
    df["text"] = "summarize: "+df["text"]

    print("[Dataset] Loading data...")

    mT5Trainer(dataframe=df, source_text="text", target_text="title", model_params=model_params)

if __name__ == "__main__":
    main()
