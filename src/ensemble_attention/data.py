import pytorch_lightning as pl
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torch
import os


class TinyImagenetData(pl.LightningDataModule):
  def __init__(self, args):
    super().__init__()
    self.hparams = args
    self.mean = (0.485, 0.456, 0.406)
    self.std = (0.229, 0.224, 0.225)
    ## if softmax targets are given, parse.
    if args.get("custom_targets_train" ,False):
      # self.set_targets_train = parse_softmax(args.softmax_targets_train)
      ## training targets should be softmax! others should be binary.
      raise NotImplementedError('custom train targets NA')
    else:
      self.set_targets_train = None
    if args.get("custom_targets_eval_ind" ,False):
      raise NotImplementedError('custom eval targets NA')
    else:
      self.set_targets_eval_ind = None


  def train_dataloader(self ,shuffle = True ,aug=True):
    """added optional shuffle parameter for generating random labels.
    added optional aug parameter to apply augmentation or not.

    Note: preprocessing is different for imagenet vs cifar10 models.
    """
    if aug is True:
      transform = transforms.Compose(
        [
          #transforms.Resize(256),
          #transforms.RandomResizedCrop(224),
          transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
          transforms.Normalize(self.mean, self.std),
        ]
      )
    else:
      transform = transforms.Compose(
        [
          transforms.ToTensor(),
          transforms.Normalize(self.mean, self.std),
        ]
      )
    train_dir = os.path.join(self.hparams.data_dir, "train")
    train_dataset = datasets.ImageFolder(
      train_dir,
      transform,
    )

    if self.set_targets_train is not None:
      raise NotImplementedError('custom train targets NA')

    dataloader = DataLoader(
      train_dataset,
      batch_size=self.hparams.batch_size,
      num_workers=self.hparams.num_workers,
      shuffle=shuffle,
      drop_last=False,
      #pin_memory=True,
    )
    return dataloader

  def val_dataloader(self):
    val_dir = os.path.join(self.hparams.data_dir, "val")
    val_loader = torch.utils.data.DataLoader(
      datasets.ImageFolder(
        val_dir,
        transforms.Compose(
          [
           #transforms.Resize(256),
           #transforms.CenterCrop(224),
           transforms.ToTensor(),
           transforms.Normalize(self.mean, self.std),
           ]
        ),
      ),
      batch_size=self.hparams.batch_size,
      shuffle=False,
      num_workers=self.hparams.num_workers,
    )
    return val_loader

  def test_dataloader(self):
    return self.val_dataloader()


class ImagenetData(pl.LightningDataModule):
  def __init__(self, args):
    super().__init__()
    self.hparams = args
    self.mean = (0.485, 0.456, 0.406)
    self.std = (0.229, 0.224, 0.225)
    ## if softmax targets are given, parse.
    if args.get("custom_targets_train" ,False):
      # self.set_targets_train = parse_softmax(args.softmax_targets_train)
      ## training targets should be softmax! others should be binary.
      raise NotImplementedError('custom train targets NA')
    else:
      self.set_targets_train = None
    if args.get("custom_targets_eval_ind" ,False):
      raise NotImplementedError('custom eval targets NA')
    else:
      self.set_targets_eval_ind = None


  def train_dataloader(self ,shuffle = True ,aug=True):
    """added optional shuffle parameter for generating random labels.
    added optional aug parameter to apply augmentation or not.

    Note: preprocessing is different for imagenet vs cifar10 models.
    """
    if aug is True:
      transform = transforms.Compose(
        [
          #transforms.Resize(256),
          transforms.RandomResizedCrop(224),
          transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
          transforms.Normalize(self.mean, self.std),
        ]
      )
    else:
      transform = transforms.Compose(
        [
          transforms.ToTensor(),
          transforms.Normalize(self.mean, self.std),
        ]
      )
    train_dir = os.path.join(self.hparams.data_dir, "train")
    train_dataset = datasets.ImageFolder(
      train_dir,
      transform,
    )

    if self.set_targets_train is not None:
      raise NotImplementedError('custom train targets NA')

    dataloader = DataLoader(
      train_dataset,
      batch_size=self.hparams.batch_size,
      num_workers=self.hparams.num_workers,
      shuffle=shuffle,
      drop_last=False,
      #pin_memory=True,
    )
    return dataloader

  def val_dataloader(self):
    val_dir = os.path.join(self.hparams.data_dir, "val")
    val_loader = torch.utils.data.DataLoader(
      datasets.ImageFolder(
        val_dir,
        transforms.Compose(
          [
           transforms.Resize(256),
           transforms.CenterCrop(224),
           transforms.ToTensor(),
           transforms.Normalize(self.mean, self.std),
           ]
        ),
      ),
      batch_size=self.hparams.batch_size,
      shuffle=False,
      num_workers=self.hparams.num_workers,
    )
    return val_loader

  def test_dataloader(self):
    return self.val_dataloader()
