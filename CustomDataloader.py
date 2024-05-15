import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import normalize

import tqdm as notebook_tqdm
#from tqdm.auto import tqdm
from torchmetrics import Accuracy

import pytorch_lightning as pl
import seaborn as sns
from pylab import rcParams
from matplotlib import rc
from matplotlib.ticker import MaxNLocator

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger


from multiprocessing import cpu_count
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


class SurfaceDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences
    def __len__(self):
        return len(self.sequences)
    def __getitem__(self, idx):
        sequence, label = self.sequences[idx]
        return dict(
            sequence=torch.Tensor(sequence),
            label=torch.tensor(label).long()
        )

# Create a PyTorch Lightning Data Module
class SurfaceDatasetModule(pl.LightningDataModule):
    
    def __init__(self, train_sequences, val_sequences, test_sequences, batch_size, num_workers=0, presistent_workers=True):
        super().__init__()
        self.train_sequences = train_sequences
        self.val_sequences = val_sequences
        self.test_sequences = test_sequences
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.presistent_workers = presistent_workers

    def setup(self, stage: str):

        if stage == "fit" or stage is None:
            self.train_dataset = SurfaceDataset(self.train_sequences)
            self.val_dataset = SurfaceDataset(self.val_sequences)
        if stage == "test":
            self.test_dataset = SurfaceDataset(self.test_sequences)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.presistent_workers,
            pin_memory=True
        )
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.presistent_workers,
            pin_memory=True

        )
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.presistent_workers,
            pin_memory=True
        )
    
class SequenceModel(nn.Module):
        def __init__(self, n_features, n_classes, n_hidden=512, n_layers=6):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=n_features,
                hidden_size=n_hidden,
                num_layers=n_layers,
                batch_first=True,
                dropout=0.75
            )
            self.classifier = nn.Linear(n_hidden, n_classes)

        def forward(self, x):
            self.lstm.flatten_parameters()
            _, (hidden, _) = self.lstm(x)
            out = hidden[-1]
            return self.classifier(out)
        
class SurfacePredictor(pl.LightningModule):

    def __init__(self, n_features: int, n_classes: int, lr: float = 1e-3):
        super().__init__()
        self.model = SequenceModel(n_features, n_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.lr = lr
        if n_classes == 2:
            self.accuracy = Accuracy(task="BINARY", num_classes=n_classes).to(device, non_blocking=True)
        else:
            self.accuracy = Accuracy(task="multiclass", num_classes=n_classes).to(device, non_blocking=True)

    def forward(self, x, labels=None):
        output = self.model(x)
        loss = 0
        if labels is not None:
            loss = self.criterion(output, labels)
        return loss, output
    
    def training_step(self, batch, batch_idx):
        sequences = batch["sequence"]
        labels = batch["label"]
        loss, output = self(sequences, labels)
        predictions = torch.argmax(output, dim=1)
        step_accuracy = self.accuracy(predictions, labels)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_accuraacy", step_accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss, "accuracy": step_accuracy}
    
    def validation_step(self, batch, batch_idx):
        sequences = batch["sequence"]
        labels = batch["label"]
        loss, output = self(sequences, labels)
        predictions = torch.argmax(output, dim=1)
        step_accuracy = self.accuracy(predictions, labels)

        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_accuraacy", step_accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss, "accuracy": step_accuracy}
    
    def test_step(self, batch, batch_idx):
        sequences = batch["sequence"]
        labels = batch["label"]
        loss, output = self(sequences, labels)
        predictions = torch.argmax(output, dim=1)
        step_accuracy = self.accuracy(predictions, labels)

        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_accuraacy", step_accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss, "accuracy": step_accuracy}
    
    def configure_optimizers(self):
        #optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9)
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    

class ModelWrapper:
    def __init__(self, FEATURE_COLUMNS, N_CLASSES, EPOCHS, BATCH_SIZE, N_WORKERS, PRESISTANT_WORKERS, train_seq, val_seq, test_seq, TensorBoardLogName = "surface", lr=1e-3):
        self.model = SurfacePredictor(n_features=len(FEATURE_COLUMNS), n_classes=N_CLASSES, lr=lr)
        self.data_module = SurfaceDatasetModule(train_seq, val_seq, test_seq, BATCH_SIZE, N_WORKERS, PRESISTANT_WORKERS)
        self.checkpoint_callback = ModelCheckpoint(
            dirpath="checkpoints",
            filename="best-checkpoint",
            save_top_k=1,
            verbose=True,
            monitor="val_loss",
            mode="min"
        )

        self.logger = TensorBoardLogger("lightning_logs", name=TensorBoardLogName)

        self.trainer = pl.Trainer(
            logger=self.logger,
            callbacks=[self.checkpoint_callback],
            max_epochs=EPOCHS,
            enable_progress_bar=True,
            log_every_n_steps=1
        )