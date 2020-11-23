import os

from torch.nn.modules.dropout import Dropout
from data.createDataset import TrashData
from data.createDataset import trainTransform
from data.createDataset import testTransform
import pytorch_lightning as pl
from pytorch_lightning.metrics import ConfusionMatrix
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torchvision.models import MobileNetV2
from torchvision.models.resnet import resnet34
from torchvision.models.mobilenet import mobilenet_v2
import numpy as np
import pandas as pd

class Solver(pl.LightningModule):

    def __init__(self, pretrained=False, num_classes=5):
        super().__init__()
        # Pretrained model
        self.datapath = os.path.join(os.getcwd(), 'data')

        if pretrained:
            self.model = mobilenet_v2(pretrained=True)
            for params in self.model.features.parameters(recurse=True):
                params.requires_grad = False
            output_features_nb = 1280
            fc_layer = nn.Linear(output_features_nb, num_classes, bias=True)
            self.model.classifier = nn.Sequential(
                                           nn.Dropout(0.2),
                                            fc_layer
                                        )
            torch.nn.init.xavier_normal_(fc_layer.weight, gain=1.0)
            torch.nn.init.zeros_(fc_layer.bias)

            # self.model = resnet34(pretrained=True)
            # for params in self.model.parameters(recurse=True):
            #     params.requires_grad = False
            # output_features_nb = self.model.fc.weight.size(1)
            # self.model.fc = nn.Linear(output_features_nb, num_classes, bias=True)
            # for params in self.model.fc.parameters(recurse=True):
            #     params.requires_grad = True
            # torch.nn.init.xavier_normal_(self.model.fc.weight, gain=1.0)
            # torch.nn.init.zeros_(self.model.fc.bias)

        # From scratch model
        else:
            features = MobileNetV2(num_classes=num_classes)
            # Freeze the network except the classifier
            for params in features.features.parameters(recurse=True):
                params.requires_grad = False
            self.model = features
        
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.model(x)

        if self.train:
            return x
        return self.softmax(x)

    def training_step(self, batch, batch_nb):
        x, y = batch
        yhat = self(x)

        loss = torch.nn.functional.cross_entropy(yhat, y)
        tensorboard_logs = {'train_loss': loss}
        return {'loss':loss, 'log':tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        x, y = batch
        yhat = self(x)
        val_loss = torch.nn.functional.cross_entropy(yhat, y)
        tp = (self.softmax(yhat).argmax(dim=1) == y).sum()
        count = y.size(0)
        return{'val_loss':val_loss, 'tp':tp, 'count':count}

    def test_step(self, batch, batch_nb):
        x, y = batch
        yhat = self(x)
        test_loss = torch.nn.functional.cross_entropy(yhat, y)

        ypred = self.softmax(yhat).argmax(dim=1)
        confusion_matrix = np.zeros((yhat.size(1),yhat.size(1)))

        for y_, ypred_ in zip(y, ypred):
            confusion_matrix[y_, ypred_] += 1

        tp = (ypred == y).sum()
        count = y.size(0)
        return{'test_loss':test_loss, 'tp':tp, 'count':count, 'cmatrix':confusion_matrix}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        val_acc = torch.stack([x['tp'] for x in outputs]).sum() / float(sum([x['count'] for x in outputs])) * 100.0

        logs = {'val_loss': avg_loss, 'val_acc':val_acc}
        return {'val_loss':avg_loss, 'log':logs}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        test_acc = torch.stack([x['tp'] for x in outputs]).sum() / float(sum([x['count'] for x in outputs])) * 100.0
        cmatrix = np.sum(np.stack([x['cmatrix'] for x in outputs], axis=0), axis=0)
        
        logs = {'test_loss': avg_loss, 'test_acc': test_acc}

        return {'test_loss':avg_loss, 'log':logs, 'progress_bar':logs, 'cmatrix':cmatrix}

    def train_dataloader(self) -> DataLoader:
        trainSet = TrashData(self.datapath, set='train', transform=trainTransform)
        return DataLoader(trainSet, batch_size=32, shuffle=True)
    
    def val_dataloader(self):
        valSet = TrashData(self.datapath, set='val', transform=testTransform)
        return DataLoader(valSet, batch_size=32)

    def test_dataloader(self):
        testSet = TrashData(self.datapath, set='val', transform=testTransform)
        return DataLoader(testSet, batch_size=32)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.00005, weight_decay=0.0001)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=5e-7)
        return [optimizer]

if __name__ == '__main__':
    solver = Solver(pretrained=True, num_classes=6)
    trainer = pl.Trainer(gpus=1, max_epochs=5)
    trainer.fit(solver)
    trainer.test(solver, test_dataloaders=DataLoader(TrashData(solver.datapath, set='test', transform=testTransform), batch_size=32))