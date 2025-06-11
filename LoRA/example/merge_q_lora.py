from peft import PeftModel
from transformers import AutoModelForCausalLM

from example import config

merged = AutoModelForCausalLM.from_pretrained(config.BASE_MODEL, torch_dtype="bfloat16")
model = PeftModel.from_pretrained(merged, config.MODEL_OUTPUT_DIR)
model = model.merge_and_unload()
model.save_pretrained(config.MERGED_MODEL_OUTPUT_DIR, safe_serialization=True)
