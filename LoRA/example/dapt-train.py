import argparse

import datasets
import torch
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, DataCollatorForLanguageModeling, Trainer,
    TrainingArguments,
)

from example import config


def load_datasets(tokenizer):
    corpus_datasets = datasets.load_dataset("text", data_files=str(config.CORPUS_TEXT), split="train")

    def tokenize_function(example):
        tokens = tokenizer(example["text"])
        return {"input_ids": tokens["input_ids"]}

    dataset_tokenized = corpus_datasets.map(tokenize_function, remove_columns=["text"])

    block_size = 2048

    def chunk_batched(examples):
        concatenated = sum(examples["input_ids"], [])
        total_len = (len(concatenated) // block_size) * block_size
        chunks = [concatenated[i: i + block_size] for i in range(0, total_len, block_size)]
        return {"input_ids": chunks}

    dataset_chunks = dataset_tokenized.map(chunk_batched, batched=True, remove_columns=["input_ids"])
    return dataset_chunks


def load_model(env: str):
    is_gpu = env == "gpu"

    if is_gpu:
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(config.BASE_MODEL, quantization_config=bnb_cfg, device_map="auto")
    else:
        model = AutoModelForCausalLM.from_pretrained(config.BASE_MODEL)

    # Freeze early layers to minimise drift
    freeze_layers = 8
    for layer in model.model.layers[: freeze_layers]:
        for param in layer.parameters():
            param.requires_grad = False

    if is_gpu:
        model = prepare_model_for_kbit_training(model)

    lora_cfg = LoraConfig(
        r=32,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)
    return model


def main(env: str):
    print("Loading tokenizer:", config.BASE_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    print("Loading base model:", config.BASE_MODEL)
    model = load_model(env)

    print("Loading datasets", config.CORPUS_TEXT)
    dataset_chunks = load_datasets(tokenizer)
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    train_args = TrainingArguments(
        output_dir=config.MODEL_OUTPUT_DIR,
        num_train_epochs=8 if env == "gpu" else 1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=16,
        learning_rate=1e-5,
        fp16=False,
        bf16=True,
        logging_steps=20,
        save_steps=200,
        warmup_ratio=0.05,
        max_grad_norm=0.3,
        lr_scheduler_type="cosine",
    )

    trainer = Trainer(
        model=model,
        train_dataset=dataset_chunks,
        data_collator=collator,
        args=train_args,
    )

    print("Starting training")
    trainer.train()
    print("Saving model to", config.MODEL_OUTPUT_DIR)
    trainer.save_model()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model with DAPT")
    parser.add_argument("-e", "--env", type=str, choices=["cpu", "gpu"], help="Environment: gpu or cpu")
    args = parser.parse_args()
    main(args.env)
