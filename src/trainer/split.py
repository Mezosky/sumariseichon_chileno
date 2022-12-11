import logging
from typing import Any

import pandas as pd

from dataset import SummaryDataSet

logger = logging.getLogger(__name__)


def split_data(
    source_text: pd.Series,
    target_text: pd.Series,
    tokenizer,
    params: dict[str, Any],
) -> tuple[SummaryDataSet, SummaryDataSet]:

    # Creation of Dataset and Dataloader
    # Defining the train size. So 80% of the data will be used for training
    # and the rest for validation.
    train_size = params["test_size"]

    train_index = source_text.sample(frac=train_size, random_state=params["SEED"]).index
    val_index = source_text.drop(train_index).index

    train_source_text = source_text.loc[train_index].values
    train_target_text = target_text.loc[train_index].values

    val_source_text = source_text.loc[val_index].values
    val_target_text = target_text.loc[val_index].values

    logger.info(f"[Split] Full Dataset : {len(source_text)}")
    logger.info(f"[Split] Train Dataset: {len(train_index)}")
    logger.info(f"[Split] Test Dataset : {len(val_index)}\n")

    # Creating the Training and Validation dataset for further creation of Dataloader
    training_set = SummaryDataSet(
        train_source_text,
        train_target_text,
        tokenizer,
        params["MAX_SOURCE_TEXT_LENGTH"],
        params["MAX_TARGET_TEXT_LENGTH"],
    )
    val_set = SummaryDataSet(
        val_source_text,
        val_target_text,
        tokenizer,
        params["MAX_SOURCE_TEXT_LENGTH"],
        params["MAX_TARGET_TEXT_LENGTH"],
    )
    return training_set, val_set
