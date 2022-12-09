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

from dataset import SummaryDataSet

# Importing the mT5 modules from huggingface/transformers
from transformers import T5Tokenizer, MT5Model, MT5Config, MT5ForConditionalGeneration

logger = logging.getLogger(__name__)

def validate(
    model: MT5ForConditionalGeneration,
    tokenizer: T5Tokenizer,
    epoch: int, 
    loader: SummaryDataSet,
    device: str = 'cuda' if cuda.is_available() else 'cpu'
) -> (list, list):    
    """
    Function to evaluate model for predictions

    """
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for it, data in enumerate(loader, 0):
            y    = data['target_ids'].to(device, dtype = torch.long)
            ids  = data['source_ids'].to(device, dtype = torch.long)
            mask = data['source_mask'].to(device, dtype = torch.long)

            generated_ids = model.generate(
              input_ids          = ids,
              attention_mask     = mask, 
              max_length         = 150, 
              num_beams          = 2,
              repetition_penalty = 2.5, 
              length_penalty     = 1.0, 
              early_stopping     = True
            )

            preds  = [
                tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) 
                for g in generated_ids
            ]
            target = [
                tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True) 
                for t in y
            ]

            if _%10==0:
                logger.info(f'Completed {it}')

            predictions.extend(preds)
            actuals.extend(target)
            
    return predictions, actuals


def train(
    model: MT5ForConditionalGeneration, 
    tokenizer: T5Tokenizer, 
    epoch: int, 
    loader: SummaryDataSet, 
    optimizer: torch.optim, 
    device: str = 'cuda' if cuda.is_available() else 'cpu'
) -> None:
    """
    Function to be called for training with the parameters passed from main function

    """
    
    model.train()
    for _, data in enumerate(loader, 0):
    
        y     = data['target_ids'].to(device, dtype = torch.long)
        y_ids = y[:, :-1].contiguous()
        
        lm_labels = y[:, 1:].clone().detach()
        lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100
        
        ids  = data['source_ids'].to(device, dtype = torch.long)
        mask = data['source_mask'].to(device, dtype = torch.long)

        outputs = model(
            input_ids         = ids, 
            attention_mask    = mask, 
            decoder_input_ids = y_ids, 
            labels            = lm_labels
        )
        loss = outputs[0]
        
        if _%10==0:
            logger.info(f'Epoch: {epoch+1} | Loss: {str(round(float(outputs[0]), 3))}')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def mT5Trainer(
    dataframe: pd.DataFrame, 
    source_text: str, 
    target_text: str, 
    model_params: dict, 
    output_dir: str ="/data/imeza/text_datasets/outputs_mT5/",
    device: str = 'cuda' if cuda.is_available() else 'cpu'
) -> None:
    """
    mT5 trainer

    """

    # logging
    logger.info(f"""[Model]: Loading {model_params["MODEL"]}...\n""")

    # tokenzier for encoding the text
    tokenizer = T5Tokenizer.from_pretrained(model_params["MODEL"])

    # Defining the model. We are using t5-base model and added a Language model layer on top for generation of Summary. 
    # Further this model is sent to device (GPU/TPU) for using the hardware.
    model = MT5ForConditionalGeneration.from_pretrained(model_params["MODEL"])
    model = model.to(device)

    # logging
    logger.info(f"[Data]: Reading data...\n")

    # Importing the raw dataset
    dataframe = dataframe[[source_text,target_text]]


    # Creation of Dataset and Dataloader
    # Defining the train size. So 80% of the data will be used for training and the rest for validation. 
    train_size = 0.8
    train_dataset=dataframe.sample(frac=train_size,random_state = model_params["SEED"])
    val_dataset=dataframe.drop(train_dataset.index).reset_index(drop=True)
    train_dataset = train_dataset.reset_index(drop=True)

    logger.info(f"FULL Dataset : {dataframe.shape}")
    logger.info(f"TRAIN Dataset: {train_dataset.shape}")
    logger.info(f"TEST Dataset : {val_dataset.shape}\n")

    # Creating the Training and Validation dataset for further creation of Dataloader
    training_set = SummaryDataSet(
        train_dataset, 
        tokenizer, 
        model_params["MAX_SOURCE_TEXT_LENGTH"], 
        model_params["MAX_TARGET_TEXT_LENGTH"], 
        source_text, target_text
    )
    val_set      = SummaryDataSet(
        val_dataset, tokenizer, 
        model_params["MAX_SOURCE_TEXT_LENGTH"], 
        model_params["MAX_TARGET_TEXT_LENGTH"], 
        source_text, target_text
    )


    # Defining the parameters for creation of dataloaders
    train_params = {
      'batch_size': model_params["TRAIN_BATCH_SIZE"],
      'shuffle': True,
      'num_workers': 0
      }


    val_params = {
      'batch_size': model_params["VALID_BATCH_SIZE"],
      'shuffle': False,
      'num_workers': 0
      }


    # Creation of Dataloaders for testing and validation. This will be used down for training and validation stage for the model.
    training_loader = DataLoader(training_set, **train_params)
    val_loader      = DataLoader(val_set, **val_params)


    # Defining the optimizer that will be used to tune the weights of the network in the training session. 
    optimizer = torch.optim.Adam(params =  model.parameters(), lr=model_params["LEARNING_RATE"])


    # Training loop
    logger.info(f'[Initiating Fine Tuning]...\n')

    for epoch in range(model_params["TRAIN_EPOCHS"]):
        train(model, tokenizer, epoch, training_loader, optimizer, device)

    logger.info(f"[Saving Model]...\n")
    #Saving the model after training
    path = os.path.join(output_dir, "model_files")
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)


    # evaluating test dataset
    logger.info(f"[Initiating Validation]...\n")
    for epoch in range(model_params["VAL_EPOCHS"]):
        predictions, actuals = validate(model, tokenizer, epoch, val_loader, device)
        final_df = pd.DataFrame({'Generated Text':predictions,'Actual Text':actuals})
        final_df.to_csv(os.path.join(output_dir,'predictions.csv'))

    logger.info(f"[Validation Completed.]\n")
    logger.info(f"[Model] Model saved @ {os.path.join(output_dir, 'model_files')}\n")
    logger.info(f"[Validation] Generation on Validation data saved @ {os.path.join(output_dir,'predictions.csv')}\n")
    logger.info(f"[Logs] Logs saved @ {os.path.join(output_dir,'logs.txt')}\n")
