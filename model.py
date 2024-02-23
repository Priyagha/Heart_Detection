import torch
import torchvision
import numpy as np
import cv2
from pathlib import Path 
from dataset import CardiacDataset


# train_dataset = CardiacDataset("rsna_heart_detection.csv", "train_subjects.npy", "Processed_Heart_Detection/train", 0.49, 0.25)
# val_dataset = CardiacDataset("rsna_heart_detection.csv", "val_subjects.npy", "Processed_Heart_Detection/val", 0.49, 0.25)

# batch_size = 16
# num_workers = 4

# train_loader = torch.util.data.DataLoader(train_dataset, batch_size = batch_size, num_workers = num_workers, shuffle = True)
# val_loader = torch.util.data.DataLoader(val_dataset, batch_size = batch_size, num_workers = num_workers, shuffle = False)

class CardiacDetectionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.model = torchvision.models.resnet18(pretrained = True)

        for params in self.model.parameters():
            params.requires_grad = False

        self.model.conv1 =  torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.model.fc = torch.nn.Linear(in_features=512, out_features=4)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = 1e-4)
        self.loss = torch.nn.MSELoss()

    def forward(self, x):
        return self.model(x)
    
    
    
