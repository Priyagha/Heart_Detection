import pandas as pd
import numpy as np
import torch
from pathlib import Path

class CardiacDataset(torch.utils.data.Dataset):
    def __init__(self, labels_csv, patient_ids, path_to_images, mean, std):
        self.labels_csv = pd.read_csv(labels_csv)
        self.patient_ids = np.load(patient_ids)
        self.path_to_images = Path(path_to_images)
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.patient_ids)
    
    def __getitem__(self,idx):
        patient_id = self.patient_ids[idx]
        patient_data = self.labels_csv[self.labels_csv["name"] == patient_id]
        x_min = patient_data["x0"].item()
        y_min = patient_data["y0"].item()
        x_max = patient_data["w"].item() + x_min
        y_max = patient_data["h"].item() + y_min

        image_path = self.path_to_images/patient_id
        image = np.load(f"{image_path}.npy").astype(np.float32)
        image = (image - self.mean) / (self.std)
        image = torch.tensor(image).unsqueeze(0)

        return image, torch.tensor((x_min, y_min, x_max, y_max))
