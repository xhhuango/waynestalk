import pypdf

from example import config


def main():
    print("Reading PDF file", config.CORPUS_PDF)
    with pypdf.PdfReader(config.CORPUS_PDF) as pdf:
        text = "\n".join(p.extract_text() for p in pdf.pages)

    print("Writing text to", config.CORPUS_TEXT)
    config.DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.CORPUS_TEXT, "w") as f:
        f.write(text)


if __name__ == "__main__":
    main()
