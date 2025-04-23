from pathlib import Path

from datasets import load_dataset

ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
out_path = "wikitext2_train.txt"
out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)

with out_path.open("w", encoding="utf‑8") as f:
    for line in ds["text"]:
        line = line.strip()
        if line:                       # skip blank lines
            f.write(line + "\n")
print("Wrote", out_path, "with", len(ds), "rows")