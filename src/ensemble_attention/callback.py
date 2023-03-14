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

