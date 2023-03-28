"""
callbacks to check gradients. from
https://www.pytorchlightning.ai/blog/3-simple-tricks-that-will-change-the-way-you-debug-pytorch 
"""
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
import torch

class Check_GradNorm(Callback):
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        if (batch_idx + 1) % trainer.log_every_n_steps == 0:
            x,y  = batch[0].to(pl_module.device),batch[1].to(pl_module.device)
            x.requires_grad=True
            [b.cuda() for b in batch]
            pl_module.zero_grad()
            loss = pl_module.training_step([x,y],0)
            loss.backward()

            logger = trainer.logger

            logger.experiment.add_histogram("grad_norm", torch.norm(x.grad), global_step=trainer.global_step)



class GradNormCallbackSplit(Callback):
  """
  Logs the gradient norm.
  Edited from https://github.com/Lightning-AI/lightning/issues/1462
  """

  def on_after_backward(self, trainer, model):
    model.log("model/grad_norm", gradient_norm(model))
    has_models = getattr(model, "models", 0)
    if has_models != 0:
      for model_idx, model_ in enumerate(model.models):
        model.log(f"model/grad_norm_{model_idx}", gradient_norm(model_))


class GradNormCallback(Callback):
  """
  Logs the gradient norm.
  Edited from https://github.com/Lightning-AI/lightning/issues/1462
  """

  def on_after_backward(self, trainer, model):
    model.log("my_model/grad_norm", gradient_norm(model))

def gradient_norm(model):
  total_norm = 0.0
  for p in model.parameters():
    if p.grad is not None:
      param_norm = p.grad.detach().data.norm(2)
      total_norm += param_norm.item() ** 2
  total_norm = total_norm ** (1. / 2)
  return total_norm
