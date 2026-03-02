# Fall detection using CoCa model

import os
from PIL import Image
import logging

import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader

from transformers import CoCaProcessor, CoCaModel

# configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# dataset loader
class FallImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.samples = []
        for label in ["fall","nofall"]:
            dirpath = os.path.join(root_dir, label)
            if not os.path.isdir(dirpath):
                continue
            for fname in os.listdir(dirpath):
                if fname.lower().endswith(('.png','.jpg','.jpeg')):
                    self.samples.append((os.path.join(dirpath,fname), 1 if label=="fall" else 0))
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        path,label = self.samples[idx]
        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image,label


def main():
    # parse command-line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Train/evaluate fall detection using CoCa model")
    parser.add_argument("--root_dir", type=str, default="images",
                        help="dataset root with 'fall' and 'nofall' subfolders")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--pred_image", type=str, default=None,
                        help="path to a single image for prediction and exit")
    args = parser.parse_args()

    # configuration
    root_dir = args.root_dir  # dataset path with subfolders 'fall' and 'nofall'
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr

    # prepare transforms and dataset
    transform = T.Compose([
        T.Resize((224,224)),
        T.ToTensor(),
    ])
    dataset = FallImageDataset(root_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    logger.info(f"Dataset size: {len(dataset)}")

    # initialize CoCa model and classifier
    processor = CoCaProcessor.from_pretrained("openai/coca-small")
    model = CoCaModel.from_pretrained("openai/coca-small")

    class FallClassifier(nn.Module):
        def __init__(self, embed_dim, num_classes=2):
            super().__init__()
            self.fc = nn.Linear(embed_dim, num_classes)
        def forward(self, x):
            return self.fc(x)

    classifier = FallClassifier(embed_dim=model.config.projection_dim)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    classifier.to(device)

    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    logger.info(f"Using device: {device}")

    for epoch in range(epochs):
        classifier.train()
        total_loss = 0.0
        for imgs, labels in dataloader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                outputs = model.vision_model(pixel_values=imgs)
                emb = outputs.pooler_output
            logits = classifier(emb)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        logger.info(f"Epoch {epoch+1}, loss {total_loss/len(dataloader):.4f}")
    # save checkpoints
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(model.state_dict(), 'checkpoints/coca_model.pth')
    torch.save(classifier.state_dict(), 'checkpoints/classifier.pth')
    logger.info("Saved model and classifier checkpoints to checkpoints/")

    # inference helper
    def predict_image(path):
        img = Image.open(path).convert('RGB')
        img = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = model.vision_model(pixel_values=img).pooler_output
            logits = classifier(emb)
            pred = logits.argmax(dim=-1).item()
        return 'fall' if pred == 1 else 'nofall'

    # example usage
    sample_path = os.path.join(root_dir, 'fall', 'example1.jpg')
    if os.path.exists(sample_path):
        logger.info(f"Prediction for {sample_path}: {predict_image(sample_path)}")

if __name__ == '__main__':
    main()
