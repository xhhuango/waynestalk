from pathlib import Path

BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
MODEL_OUTPUT_DIR = Path(__file__).parent.parent / "model_output"
MERGED_MODEL_OUTPUT_DIR = Path(__file__).parent.parent / "model_merged"

DATA_DIR = Path(__file__).parent.parent / "data"

CORPUS_PDF = DATA_DIR / "FORGETTING TRANSFORMER- SOFTMAX ATTENTION WITH A FORGET GATE.pdf"

DATA_RAW_DIR = DATA_DIR / "raw"
CORPUS_TEXT = DATA_RAW_DIR / "corpus.txt"
