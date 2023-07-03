import os
import json
from pathlib import Path
import torchvision.transforms.functional as F
import torchvision.transforms as transforms

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import matplotlib
import matplotlib.pyplot as plt

from utils import *
from models import EfficientNetEncoder, GENetEncoder, EncoderGlobalFeatures

torch.autograd.set_detect_anomaly(False)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DATA_DIR = "/scratch/as216/amur/"

def main():
    if not torch.cuda.is_available():
        raise ValueError("No CUDA available")

    device = torch.device("cuda:0")
    has_multiple_devices = torch.cuda.device_count() > 1

    train_batch_size = 128
    valid_batch_size = 128
    n_epochs = 40

    # loaders
    train_transforms = transforms.Compose([transforms.Resize(256), transforms.RandomCrop(224), transforms.RandomHorizontalFlip(),  transforms.ToTensor(), transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),])
    val_transforms = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

    with open(DATA_DIR + 'labels_train.json', 'r') as f:
      json_dict = json.load(f)
    
    labels_dict = json_dict["labels"]

    enum = list(enumerate(list(set(labels_dict.values()))))
    class_to_target = dict((j, i) for i, j in list(enum))
    target_to_class = dict(enum)
    
    images = []
    targets = []
    
    for key in labels_dict.keys():
      images.append(DATA_DIR + "train/" + key + ".jpg")
      targets.append(class_to_target[labels_dict[key]])
        
    split = int(len(images)*0.8)
    images_train = images[:split]
    images_val = images[split:]
    
    targets_train = targets[:split]
    targets_val = targets[split:]
    
    train_dataset = ImagesDataset(images_train, targets_train, transforms=train_transforms )
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=os.cpu_count(),
        drop_last=True,
    )

    val_dataset = ImagesDataset(images_val, targets_val, transforms=val_transforms )

    val_loader = DataLoader(
        val_dataset,
        batch_size=valid_batch_size,
        shuffle=False,
        num_workers=os.cpu_count(),
        drop_last=False,
    )

    # init model & stuff
    seed_all(2020)

    encoder = EfficientNetEncoder("efficientnet-b1")
    # encoder = GENetEncoder("normal", "pretrains")
    model = EncoderGlobalFeatures(encoder, emb_dim=3, num_classes=107)

    model = model.to(device)
    if has_multiple_devices:
        model = nn.DataParallel(model)

    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, n_epochs + 1):
        model.train()
        train_loss = 0.0
        for idx, batch in enumerate(train_loader):
            x, y = t2d(batch, device)
            # with autograd.detect_anomaly():
            output = model(x, y)
            loss = criterion(output, y)
            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_loss /= idx + 1

        model.eval()
        valid_loss = 0.0
        accuracy = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x, y = t2d(batch, device)
                output = model(x, y)
                loss = criterion(output, y)
                torch.argmax(output)
                valid_loss += loss.item()
                acc = (torch.argmax(output, -1) == y).sum().detach().item()
                acc /= y.size(0)
                accuracy += acc
        valid_loss /= idx + 1
        accuracy /= idx + 1

        print(
            "Epoch {}/{}: train - {:.5f}, valid - {:.5f} (accuracy - {:.5f})".format(
                epoch, n_epochs, train_loss, valid_loss, accuracy
            )
        )

    out_file = "global_features.pth"
    torch.save({"model_state_dict": model.state_dict()}, out_file)
    print(f"Saved model to '{out_file}'")

    out_file = "encoder.pth"
    torch.save({"encoder_state_dict": model.encoder.state_dict()}, out_file)
    print(f"Saved encoder to '{out_file}'")


if __name__ == "__main__":
    main()
