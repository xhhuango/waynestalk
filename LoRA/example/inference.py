import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig, pipeline

from example import config

QUESTION = "Summarise the key idea of the paper \"Forgetting Transformer\""


def load_model(env, model: str):
    is_gpu = env == "gpu"

    print("Loading tokenizer:", config.BASE_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL, use_fast=True, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token

    model_name = config.BASE_MODEL if model == "base" else config.MERGED_MODEL_OUTPUT_DIR
    print("Loading model:", model_name)

    if is_gpu:
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_cfg,
            device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

    gen_cfg = GenerationConfig(
        max_new_tokens=512,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    model.generation_config = gen_cfg
    model.eval()

    generator = pipeline(task="text-generation", model=model, tokenizer=tokenizer)
    return generator


def main(env: str, model_type: str):
    print("env:", env)

    generator = load_model(env, model_type)

    print("Question:", QUESTION)
    messages = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": QUESTION}]
    prompt = generator.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    print("Inferencing...")
    outputs = generator(prompt)
    output_text = outputs[0]["generated_text"]
    output_text = output_text[len(prompt):].strip()

    print("Answer:\n", output_text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference with the model.")
    parser.add_argument("-e", "--env", type=str, choices=["cpu", "gpu"], help="Environment: gpu or cpu")
    parser.add_argument(
        "-m", "--model", type=str, choices=["base", "fine-tuned"], required=True, help="Model Type"
    )
    args = parser.parse_args()
    main(args.env, args.model)
