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

class SummaryDataSet(Dataset):
    """
    Dataloader to Finetune a mT5 model focused in a summarization task.

    """
    def __init__(
        self, 
        dataframe: pd.DataFrame, 
        tokenizer: T5Tokenizer, 
        source_len: int, 
        target_len: int, 
        source_text: str, 
        target_text: str,
    ) -> None:
        
        self.tokenizer   = tokenizer
        self.source_len  = source_len
        self.summ_len    = target_len
        self.target_text = dataframe[target_text]
        self.source_text = dataframe[source_text]
    
    def __getitem__(
        self, index: int
    ) -> dict[str: torch.Tensor]:
        
        source_text = str(self.source_text[index])
        target_text = str(self.target_text[index])

        #cleaning data so as to ensure data is in string type
        source_text = ' '.join(source_text.split())
        target_text = ' '.join(target_text.split())

        source = self.tokenizer.batch_encode_plus(
            [source_text], 
            max_length        = self.source_len, 
            pad_to_max_length = True, 
            truncation        = True, 
            padding           = "max_length", 
            return_tensors    = 'pt'
        )
        target = self.tokenizer.batch_encode_plus(
            [target_text], 
            max_length        = self.summ_len, 
            pad_to_max_length = True, 
            truncation        = True, 
            padding           = "max_length", 
            return_tensors    = 'pt'
        )

        source_ids  = source['input_ids'].squeeze()
        source_mask = source['attention_mask'].squeeze()
        target_ids  = target['input_ids'].squeeze()
        target_mask = target['attention_mask'].squeeze()

        return {
            'source_ids': source_ids.to(dtype=torch.long), 
            'source_mask': source_mask.to(dtype=torch.long), 
            'target_ids': target_ids.to(dtype=torch.long),
            'target_ids_y': target_ids.to(dtype=torch.long)
        }
    
    def __len__(self) -> int:
        return len(self.target_text)