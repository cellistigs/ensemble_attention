import pytorch_lightning as pl
from .schduler import WarmupCosineLR
import torch
from pytorch_lightning.metrics import Accuracy
import torch.nn as nn
from .metrics import Model_D_KL,Model_Ortega_Variance,Model_JS_Unif,Model_JS_Avg,Model_DKL_Avg,Regression_Var

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
          optimizer, warmup_epochs=(total_steps * 0.3)/self.hparams.gpus, max_epochs=total_steps/self.hparams.gpus
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



class TinyImagenetModule(Imagenet_Models):
    def __init__(self, hparams):
        super().__init__(hparams)
        print(hparams)
        print(self.hparams)
        self.label_smoothing = self.hparams.get('label_smoothing', 0.0)
        self.criterion = torch.nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        self.accuracy = Accuracy()

        self.num_classes = hparams.get('num_classes', 200)
        pretrained = self.hparams.get('pretrained', False)
        self.model = models.__dict__[self.hparams.classifier](pretrained=pretrained)
        # Change the last layer
        #in_f = self.model.classifier[-1].in_features
        #self.model.classifier[-1] = nn.Linear(in_f, self.num_classes)

        self.model.avgpool = nn.AdaptiveAvgPool2d(1)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, self.num_classes)

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


class TinyImagenetEnsembleModule(Imagenet_Models):
  """Customized module to train an ensemble of models independently

  """

  def __init__(self, hparams):
    super().__init__(hparams)
    self.nb_models = hparams.nb_models

    self.label_smoothing = self.hparams.get('label_smoothing', 0.0)
    self.criterion = torch.nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
    self.fwd_criterion = torch.nn.NLLLoss()
    self.accuracy = Accuracy()

    self.num_classes = hparams.get('num_classes', 200)
    self.pretrained = self.hparams.get('pretrained', False)

    self.models = torch.nn.ModuleList([models.__dict__[self.hparams.classifier](pretrained=self.pretrained) for i in
                                       range(self.nb_models)])  ## now we add several different instances of the model.
    # del self.model

  def forward(self, batch):
    """for forward pass, we want to take the softmax,
    aggregate the ensemble output, take log(\bar{f}) and apply NNLoss.
    prediction  = \bar{f}
    """
    images, labels = batch
    softmax = torch.nn.Softmax(dim=1)

    losses = []
    accs = []
    softmaxes = []
    for m in self.models:
      predictions = m(images)
      normed = softmax(predictions)
      softmaxes.append(normed)
    mean = torch.mean(torch.stack(softmaxes), dim=0)
    ## we can pass this  through directly to the accuracy function.
    logoutput = torch.log(mean)
    tloss = self.fwd_criterion(logoutput,
                               labels)  ## beware: this is a transformed input, don't evaluate on test loss of ensembles.
    accuracy = self.accuracy(mean, labels)
    return tloss, accuracy * 100

  def calibration(self, batch):
    """Like forward, but just exit with the predictions and labels. .
    """
    images, labels = batch
    softmax = torch.nn.Softmax(dim=1)

    losses = []
    accs = []
    softmaxes = []
    for m in self.models:
      predictions = m(images)
      normed = softmax(predictions)
      softmaxes.append(normed)
    # gmean = torch.exp(torch.mean(torch.log(torch.stack(softmaxes)),dim = 0)) ## implementation from https://stackoverflow.com/questions/59722983/how-to-calculate-geometric-mean-in-a-differentiable-way
    mean = torch.mean(torch.stack(softmaxes), dim=0)
    return mean, labels

  def training_step(self, batch, batch_nb):
    """When we train, we want to train independently.
    Loss is the  average single model loss
    Loss = 1/M sum_i L( f_i, y), where f_i is the model output for the ith model.
    """

    images, labels = batch
    losses = []
    accs = []
    for m in self.models:
      predictions = m(images)  ## this just a bunch of unnormalized scores?
      mloss = self.criterion(predictions, labels)
      accuracy = self.accuracy(predictions, labels)
      losses.append(mloss)
      accs.append(accuracy)
    loss = sum(losses) / self.nb_models  ## calculate the sum with pure python functions.
    avg_accuracy = sum(accs) / self.nb_models

    self.log("loss/train", loss)
    self.log("acc/train", avg_accuracy * 100)
    return loss

  def configure_optimizers(self):
    optimizer = torch.optim.SGD(
      self.models.parameters(),
      # lr=self.hparams.learning_rate*len(self.models), ## when jointly training, we need to multiply the learning rate times the number of ensembles to make sure that the effective learning rate for each model stays the same.
      lr=self.hparams.learning_rate,
      ## when jointly training, we need to multiply the learning rate times the number of ensembles to make sure that the effective learning rate for each model stays the same.
      weight_decay=self.hparams.weight_decay,
      momentum=0.9,
      nesterov=True,
    )
    total_steps = self.hparams.max_epochs * len(self.train_dataloader())
    scheduler = self.setup_scheduler(optimizer, total_steps)
    return [optimizer], [scheduler]

  def setup_scheduler(self, optimizer, total_steps):
    """Chooses between the cosine learning rate scheduler that came with the repo, or step scheduler based on wideresnet training.For the ensemble, we need to manually set the warmup and eta_min parameters to maintain the right scaling for individual models.
    """
    if self.hparams.scheduler in [None, "cosine"]:
      scheduler = {
        "scheduler": WarmupCosineLR(
          optimizer,
          warmup_epochs=(total_steps * 0.3)/self.hparams.gpus,
          max_epochs=total_steps/self.hparams.gpus,
          warmup_start_lr=1e-8,
          eta_min=1e-8
          # warmup_start_lr = 1e-8*len(self.models),
          # eta_min = 1e-8*len(self.models)
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


class TinyImagenetEnsembleDKLModule(TinyImagenetEnsembleModule):
  """Formulation of the ensemble as a regularized single model with variable weight on regularization.

  """

  def __init__(self, hparams):
    super().__init__(hparams)

    self.traincriterion = NLLLoss_label_smooth(self.num_classes, self.label_smoothing)
    self.kl = Model_D_KL("torch")
    self.gamma = hparams.gamma

  def training_step(self, batch, batch_nb):
    """When we train, we want to train independently.
    Loss = NLL(log \bar{f}, y ) + gamma*JGAP(softmaxes, label)
    JGAP = 1/M sum_i^M CE(f_i,y) - NLL(log \bar{f}, y )
    """
    softmax = torch.nn.Softmax(dim=1)

    images, labels = batch
    losses = []
    accs = []
    softmaxes = []
    for m in self.models:
      # get logits
      predictions = m(images)
      normed = softmax(predictions)
      softmaxes.append(normed)
      mloss = self.criterion(predictions, labels)
      # accuracy = self.accuracy(predictions,labels)
      losses.append(mloss)
      # accs.append(accuracy)
    logoutput = torch.log(torch.mean(torch.stack(softmaxes), dim=0))
    mloss = self.traincriterion(logoutput, labels)
    dklloss = torch.mean(self.kl.kl(softmaxes,labels))
    loss = (mloss + self.gamma*dklloss)
    accuracy = self.accuracy(logoutput,labels)


    lr = self.trainer.lr_schedulers[0]["scheduler"].get_last_lr()[-1]
    self.log("lr/lr", lr)
    self.log("loss/train", loss)
    self.log("acc/train", accuracy * 100)
    self.log("reg/mloss", mloss)
    self.log("reg/jgap", dklloss)
    #self.log("reg/avg_sm_loss", avg_sm_loss)
    return loss

class TinyImagenetEnsembleJGAPModule(TinyImagenetEnsembleModule):
  """Formulation of the ensemble as a regularized single model with variable weight on regularization.

  """

  def __init__(self, hparams):
    super().__init__(hparams)

    self.traincriterion = NLLLoss_label_smooth(self.num_classes, self.label_smoothing)
    self.gamma = hparams.gamma

  def training_step(self, batch, batch_nb):
    """When we train, we want to train independently.
    Loss = NLL(log \bar{f}, y ) + gamma*JGAP(softmaxes, label)
    JGAP = 1/M sum_i^M CE(f_i,y) - NLL(log \bar{f}, y )
    """
    softmax = torch.nn.Softmax(dim=1)

    images, labels = batch
    losses = []
    accs = []
    softmaxes = []
    for m in self.models:
      # get logits
      predictions = m(images)
      normed = softmax(predictions)
      softmaxes.append(normed)
      mloss = self.criterion(predictions, labels)
      # accuracy = self.accuracy(predictions,labels)
      losses.append(mloss)
      # accs.append(accuracy)
    logoutput = torch.log(torch.mean(torch.stack(softmaxes), dim=0))
    mloss = self.traincriterion(logoutput, labels)

    # jensen gap
    avg_sm_loss = sum(losses) / self.nb_models
    jgaploss = avg_sm_loss - mloss

    loss = (
      mloss + self.gamma * jgaploss)  ## with gamma equal to 1, this is the same as the standard ensemble training loss (independent).
    accuracy = self.accuracy(logoutput, labels)

    lr = self.trainer.lr_schedulers[0]["scheduler"].get_last_lr()[-1]
    self.log("lr/lr", lr)
    self.log("loss/train", loss)
    self.log("acc/train", accuracy * 100)
    self.log("reg/mloss", mloss)
    self.log("reg/jgap", jgaploss)
    self.log("reg/avg_sm_loss", avg_sm_loss)
    return loss


class NLLLoss_label_smooth(torch.nn.Module):
  def __init__(self, num_classes, label_smoothing=0.1):
    super(NLLLoss_label_smooth, self).__init__()
    self.negative = label_smoothing / (num_classes - 1)
    self.positive = (1 - label_smoothing)

  def forward(self, log_softmax, target):
    true_dist = torch.zeros_like(log_softmax)
    true_dist.fill_(self.negative)
    true_dist.scatter_(1, target.data.unsqueeze(1), self.positive)
    return torch.sum(-true_dist * log_softmax, dim=1).mean()
