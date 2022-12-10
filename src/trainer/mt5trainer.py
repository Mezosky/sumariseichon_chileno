import logging
import os
from datetime import datetime
from typing import Final

import evaluate
import mlflow
import pandas as pd
import torch
from torch import cuda
from torch.utils.data import DataLoader
from transformers import MT5ForConditionalGeneration, T5Tokenizer



from trainer.split import split_data

logger = logging.getLogger(__name__)
rouge: Final = evaluate.load("rouge")
device: Final = "cuda" if cuda.is_available() else "cpu"


def validate(
    model: MT5ForConditionalGeneration,
    tokenizer: T5Tokenizer,
    dataloader: DataLoader,
    epoch: int,
    output_dir: str,
):
    """
    Perform the evaluation using the eval data at the end of a training epoch.

    """
    logger.info("[Model validation]...\n")

    model.eval()
    predictions = []
    references = []
    with torch.no_grad():
        for _, data in enumerate(dataloader, 0):
            y = data["target_ids"].to(device, dtype=torch.long)
            ids = data["source_ids"].to(device, dtype=torch.long)
            mask = data["source_mask"].to(device, dtype=torch.long)

            generated_ids = model.generate(
                input_ids=ids,
                attention_mask=mask,
                max_length=150,
                num_beams=2,
                repetition_penalty=2.5,
                length_penalty=1.0,
                early_stopping=True,
            )

            preds = [
                tokenizer.decode(
                    g, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                for g in generated_ids
            ]
            target = [
                tokenizer.decode(
                    t, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                for t in y
            ]

            predictions.extend(preds)
            references.extend(target)

    # calculate and log rouge metrics using the validation dataset
    results = rouge.compute(predictions=predictions, references=references)
    results = {f"eval_{k}": v for k, v in results.items()}
    mlflow.log_metrics(results, len(dataloader) * epoch)

    # save predictions dataframe
    pd.DataFrame({"Predictions": predictions, "References": references}).to_csv(
        os.path.join(output_dir, f"predictions_{epoch}.csv")
    )

    logger.info("[Model validation] Validation Completed\n")

    logger.info(
        "[Model validation] Validation data saved @ "
        f"{os.path.join(output_dir,f'predictions_{epoch}.csv')}\n"
    )


def train(
    model: MT5ForConditionalGeneration,
    tokenizer: T5Tokenizer,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    epoch: int,
) -> None:
    """
    Train the model with the specified parameters.
    """
    logger.info(f"[Model training] Starting training epoch {epoch+1}")

    model.train()
    for batch_idx, data in enumerate(dataloader, 0):

        optimizer.zero_grad()
        y = data["target_ids"].to(device, dtype=torch.long)
        y_ids = y[:, :-1].contiguous()

        lm_labels = y[:, 1:].clone().detach()
        lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100

        ids = data["source_ids"].to(device, dtype=torch.long)
        mask = data["source_mask"].to(device, dtype=torch.long)

        outputs = model(
            input_ids=ids,
            attention_mask=mask,
            decoder_input_ids=y_ids,
            labels=lm_labels,
        )
        loss = outputs[0]

        if batch_idx % 1 == 0:
            mlflow.log_metric("Loss", loss, batch_idx+len(dataloader)*epoch)

            logger.info(
                f"[Model training] Batch: {batch_idx+1}/{len(dataloader)} | "
                f"Loss: {str(round(float(loss), 3))}"
            )


        loss.backward()
        optimizer.step()


def mT5_trainer(
    source_text: pd.Series,
    target_text: pd.Series,
    model_params: dict,
    output_dir: str,
) -> None:
    """mT5 Training function.

    Parameters
    ----------
    source_text : pd.Series
        A series containing the source text (aka, text to summarize).
    target_text : pd.Series
        A series containing the target text (aka, the summarized text).
    model_params : dict
        Model parameters.
    output_dir : str
        An output path to store the model and intermediate results.
    """

    # ----------------------------------------------------------------------------------
    # Load pre-trained model and tokenizer

    logger.info(f"[Model]: Loading {model_params['MODEL']} pre-trained model.")
    # tokenizer for encoding the text
    tokenizer = T5Tokenizer.from_pretrained(model_params["MODEL"])

    # Define and load the model.
    # We are using t5-base model and added a Language model layer on top for
    # summarization.
    model = MT5ForConditionalGeneration.from_pretrained(model_params["MODEL"])
    # send the to device (GPU/TPU).
    model = model.to(device)
    logger.info("[Model] Pre-trained model load successfully completed.\n")

    # ----------------------------------------------------------------------------------
    # Load the datasets and dataloaders

    logger.info("[Data]: Splitting Datasets.\n")
    training_dataset, val_dataset = split_data(
        source_text,
        target_text,
        tokenizer,
        model_params,
    )

    logger.info("[Data]: Generating Dataloaders.\n")
    training_dataloader = DataLoader(
        training_dataset,
        batch_size=model_params["TRAIN_BATCH_SIZE"],
        shuffle=True,
        num_workers=0,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=model_params["VALID_BATCH_SIZE"],
        shuffle=False,
        num_workers=0,
    )

    # ----------------------------------------------------------------------------------
    # Define the optimizer

    # Defining the optimizer and scheduler that will be used to tune the weights in training session.
    optimizer = torch.optim.AdamW(
        params=model.parameters(), lr=model_params["LEARNING_RATE"]
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=1, gamma=0.1
        )

    # ----------------------------------------------------------------------------------
    # Train-eval loop

    logger.info("[Model training] Starting model training...\n")

    with mlflow.start_run(run_name=str(datetime.now().isoformat())):
        mlflow.log_params(model_params)

        for epoch in range(model_params["TRAIN_EPOCHS"]):
            train(
                model=model,
                tokenizer=tokenizer,
                dataloader=training_dataloader,
                optimizer=optimizer,
                epoch=epoch,
            )
            validate(
                model=model,
                tokenizer=tokenizer,
                dataloader=val_dataloader,
                epoch=epoch,
                output_dir=output_dir,
            )

            scheduler.step()
            logger.info("[Model training] Saving Model...\n")

            # Saving the model after training
            model_path = os.path.join(output_dir, "model_files", str(epoch))
            model.save_pretrained(model_path)
            tokenizer.save_pretrained(model_path)

            logger.info(f"[Model training] Model saved @ {model_path}\n")

    logger.info(f"[Logs] Logs saved @ {os.path.join(output_dir,'logs.txt')}\n")
