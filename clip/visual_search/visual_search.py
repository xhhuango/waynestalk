import os
from pathlib import Path

import clip
import faiss
import numpy as np
import torch
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

PRODUCT_IMAGES = Path(__file__).parent.parent / "product_images"


def build_database_embeddings(database_folder):
    image_files = [f for f in os.listdir(database_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
    embeddings = []
    filenames = []

    for img_file in image_files:
        img_path = os.path.join(database_folder, img_file)
        image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)

        with torch.no_grad():
            embedding = model.encode_image(image).cpu().numpy()

        embeddings.append(embedding / np.linalg.norm(embedding))
        filenames.append(img_file)

    embeddings = np.vstack(embeddings).astype('float32')
    return embeddings, filenames


def build_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner product similarity
    index.add(embeddings)
    return index


def search_similar_product(user_image_path, index, filenames):
    image = preprocess(Image.open(user_image_path)).unsqueeze(0).to(device)

    with torch.no_grad():
        query_embedding = model.encode_image(image).cpu().numpy()

    query_embedding /= np.linalg.norm(query_embedding)
    distances, indices = index.search(query_embedding, k=3)

    return [(filenames[i], distances[0][idx]) for idx, i in enumerate(indices[0])]


if __name__ == '__main__':
    print("Building database embeddings...")
    db_embeddings, filenames = build_database_embeddings(PRODUCT_IMAGES)

    print("Creating FAISS index...")
    faiss_index = build_faiss_index(db_embeddings)

    SEARCH_IMAGE = "./search-image-1.jpg"

    print("Searching for similar products...")
    results = search_similar_product(SEARCH_IMAGE, faiss_index, filenames)

    print("Top matched products:")
    for filename, score in results:
        print(f"Product: {filename}, Similarity Score: {score:.4f}")
