import torch
import torchvision
import numpy as np
import cv2
from pathlib import Path 
from dataset import CardiacDataset


train_dataset = CardiacDataset("rsna_heart_detection.csv", "train_subjects.npy", "Processed_Heart_Detection/train", 0.49, 0.25)
val_dataset = CardiacDataset("rsna_heart_detection.csv", "val_subjects.npy", "Processed_Heart_Detection/val", 0.49, 0.25)

batch_size = 16
num_workers = 4

train_loader = torch.util.data.DataLoader(train_dataset, batch_size = batch_size, num_workers = num_workers, shuffle = True)
val_loader = torch.util.data.DataLoader(val_dataset, batch_size = batch_size, num_workers = num_workers, shuffle = False)

class CardiacDetectionModel()
