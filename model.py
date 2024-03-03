import torch
import torchvision
import numpy as np
import cv2
from pathlib import Path 
from dataset import CardiacDataset

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
    
    def load_model(self, model_path):
        self.load_state_dict(torch.load(model_path))

    def save_model(self, model_path):
        torch.save(self.state_dict(), model_path)


    
