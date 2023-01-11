import pytorch_lightning as pl
from .schduler import WarmupCosineLR
import torch
from pytorch_lightning.metrics import Accuracy
import torch.nn as nn

import torchvision.models as models


class Imagenet_Models(pl.LightningModule):
  """Abstract base class for CIFAR10 Models

  """

  def __init__(self, hparams):
    super().__init__()
    self.hparams = hparams

  def forward(x):
    raise NotImplementedError

  def training_step():
    raise NotImplementedError

  def calibration():
    """Calculates binned calibration metrics given

    """

  def validation_step(self, batch, batch_nb):
    loss, accuracy = self.forward(batch)
    self.log("loss/val", loss)
    self.log("acc/val", accuracy)

  def test_step(self, batch, batch_nb):
    loss, accuracy = self.forward(batch)
    self.log("acc/test", accuracy)

  def setup_scheduler(self, optimizer, total_steps):
    """Chooses between the cosine learning rate scheduler that came with the repo, or step scheduler based on wideresnet training.

    """
    if self.hparams.scheduler in [None, "cosine"]:
      scheduler = {
        "scheduler": WarmupCosineLR(
          optimizer, warmup_epochs=total_steps * 0.3, max_epochs=total_steps
        ),
        "interval": "step",
        "name": "learning_rate",
      }
    elif self.hparams.scheduler == "step":
      scheduler = {
        "scheduler": torch.optim.lr_scheduler.MultiStepLR(
          optimizer, milestones=[60, 120, 160], gamma=0.2, last_epoch=-1
        ),
        "interval": "epoch",
        "frequency": 1,
        "name": "learning_rate",
      }
    return scheduler



class ImagenetModule(Imagenet_Models):
    def __init__(self, hparams):
        super().__init__(hparams)
        print(hparams)
        print(self.hparams)
        self.label_smoothing = self.hparams.get('label_smoothing', 0.0)
        self.criterion = torch.nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        self.accuracy = Accuracy()

        self.num_classes = hparams.get('num_classes', 200)
        pretrained = self.hparams.get('pretrained', False)
        self.model = models.__dict__[self.hparams.classifier](pretrained=pretrained,
                                                              num_classes=self.num_classes)
        # Change the last layer
        #in_f = self.model.classifier[-1].in_features
        #self.model.classifier[-1] = nn.Linear(in_f, self.num_classes)

        #self.model.avgpool = nn.AdaptiveAvgPool2d(1)
        #num_ftrs = self.model.fc.in_features
        #self.model.fc = nn.Linear(num_ftrs, 200)

    def forward(self, batch):
        images, labels = batch
        predictions = self.model(images)
        loss = self.criterion(predictions, labels)
        accuracy = self.accuracy(predictions, labels)
        return loss, accuracy * 100

    def calibration(self,batch,use_softmax = True):
        """Like forward, but just exit with the softmax predictions and labels. .
        """
        softmax = torch.nn.Softmax(dim = 1)
        images, labels = batch
        predictions = self.model(images)
        if use_softmax:
            smpredictions = softmax(predictions)
        else:
            smpredictions = predictions
        return smpredictions,labels

    def training_step(self, batch, batch_nb):
        loss, accuracy = self.forward(batch)
        self.log("loss/train", loss)
        self.log("acc/train", accuracy)
        return loss

    def validation_step(self, batch, batch_nb):
        loss, accuracy = self.forward(batch)
        self.log("loss/val", loss)
        self.log("acc/val", accuracy)

    def test_step(self, batch, batch_nb):
        loss, accuracy = self.forward(batch)
        self.log("acc/test", accuracy)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
            momentum=0.9,
            nesterov=True,
        )
        total_steps = self.hparams.max_epochs * len(self.train_dataloader())
        scheduler = self.setup_scheduler(optimizer,total_steps)
        return [optimizer], [scheduler]
