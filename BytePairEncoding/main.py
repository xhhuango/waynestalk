from datasets import load_dataset

from bpe import BPE


def main():
    print("Loading WikiText-2...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    train_text = dataset["train"]["text"]
    print(f"Number of training lines: {len(train_text)}")

    print("Training BPE...")
    bpe = BPE()
    bpe.train(corpus=train_text, vocab_size=2000)

    print("Training finished")
    print("Vocab size:", len(bpe.token_to_id))
    print("Number of merges:", len(bpe.merges))

    print("\nFirst 20 merges (as bytes):")
    for i, (a, b) in enumerate(bpe.merges[:20]):
        ta = bpe.id_to_token[a]
        tb = bpe.id_to_token[b]
        print(f"{i:02d}: {ta!r} + {tb!r}")

    example = "Natural language processing is interesting"
    encoded = bpe.encode(example)

    print()
    print(f"Example text: {example}")
    print(f"Encoded token ids: {encoded}")
    print(f"Decoded tokens: {[bpe.id_to_token[i] for i in encoded]}")


if __name__ == "__main__":
    main()
