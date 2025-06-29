from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset

from vit_example.vit import VisionTransformer


class DogCatDataset(Dataset):
    def __init__(self, data_dir: Path, image_size=224):
        self.num_classes = 2  # 0 for dog, 1 for cat
        self.image_paths = list(data_dir.glob("*.png"))
        self.transform = T.Compose([
            T.ToTensor(),
            T.Resize((image_size, image_size))
        ])

    def __len__(self):
        return len(self.image_paths) * 2

    def __getitem__(self, idx):
        base_idx = idx // 2
        flip = idx % 2 == 1

        image_path = self.image_paths[base_idx]
        image = Image.open(image_path).convert("RGB")

        if flip:
            image = ImageOps.mirror(image)

        image = self.transform(image)
        label = 0 if "dog" in image_path.name.lower() else 1
        return image, label


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pretrain():
    model = VisionTransformer(embed_dim=768, num_classes=1000).to(device)

    dataset = DogCatDataset(data_dir=Path(__file__).parent.parent / "data" / "train")
    dataloader = DataLoader(dataset, batch_size=20, shuffle=True)

    EPOCHS = 50
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    for epoch in range(EPOCHS):
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Epoch {epoch} Loss: {loss.item():.4f}")

    return model


def inference(model):
    model.eval()
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
    ])

    cat_image = Image.open(Path(__file__).parent.parent / "data" / "val" / "cat-val.png").convert("RGB")
    cat_image_tensor = transform(cat_image).unsqueeze(0).to(device)

    dog_image = Image.open(Path(__file__).parent.parent / "data" / "val" / "dog-val.png").convert("RGB")
    dog_image_tensor = transform(dog_image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(cat_image_tensor)
        probs = F.softmax(logits, dim=-1)
        pred = probs.argmax(dim=-1).item()
        print(f"Predicted class index for cat: {pred} (should be 1)")

        logits = model(dog_image_tensor)
        probs = F.softmax(logits, dim=-1)
        pred = probs.argmax(dim=-1).item()
        print(f"Predicted class index for dog: {pred} (should be 0)")


if __name__ == "__main__":
    model = pretrain()
    inference(model)
    torch.save(model.state_dict(), "vit_dog_cat.pth")
    print("Model saved as vit_dog_cat.pth")
