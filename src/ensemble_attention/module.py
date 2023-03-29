import pytorch_lightning as pl
import torch
from pytorch_lightning.metrics import Accuracy,ExplainedVariance

from .cifar10_models.densenet import densenet121, densenet161, densenet169
from .cifar10_models.googlenet import googlenet
from .cifar10_models.inception import inception_v3
from .cifar10_models.mobilenetv2 import mobilenet_v2
from .cifar10_models.resnet import resnet18, resnet34, resnet50, wideresnet18, wideresnet18_4, widesubresnet18,wideresnet18_4_grouplinear, resnet101
from .cifar10_models.wideresnet_28 import wideresnet28_10
from .cifar10_models.vgg import vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn
from .cifar10_models.rff import rff_regress_1000_wine,rff_regress_10000_wine,rff_regress_100000_wine,linreg_wine,rff_casregress_1000_mnist,rff_casregress_8000_mnist,rff_casregress_10000_mnist,rff_casregress_100000_mnist,rff_100_mnist,rff_10000_mnist,rff_100000_mnist
from .cifar10_models.lenet import lenet5
from .cifar10_models.shake_shake import shake_resnet26_2x96d,shake_resnet26_2x32d
from .cifar100_models.resnet import resnet18_cifar100,wideresnet18_cifar100
from .cifar100_models.vgg import vgg13_bn_cifar100
from .cifar100_models.densenet import densenet121_cifar100
from .cifar100_models.shake_shake import shake_resnet26_2x32d_cifar100
from .cifar10_models.wideresnet import wideresnet28_20
# ----------------
# missing models in public codebase
from .cifar10_models.resnet_cifar import resnet8_cf
from .cifar10_models.efficientnet import efficientnet_b2,efficientnet_b1,efficientnet_b0
from .cifar10_models.rff import rff_casregress_8000_mnist
from .cifar100_models.vgg import vgg13_bn_cifar100
from .cifar100_models.densenet import densenet121_cifar100
from .cifar100_models.shake_shake import shake_resnet26_2x32d_cifar100

# ----------------
from .cifar10_models.shake_shake import shake_resnet26_2x96d,shake_resnet26_2x32d
from .cifar100_models.resnet import resnet18_cifar100

from .schduler import WarmupCosineLR
from .layers import AttnComparison,PosEncodings,PosEncodingsSq,PosEncodingsSin
from .metrics import Model_D_KL,Model_Ortega_Variance,Model_JS_Unif,Model_JS_Avg,Model_DKL_Avg,Regression_Var

all_classifiers = {
    "vgg11_bn": vgg11_bn,
    "vgg13_bn": vgg13_bn,
    "vgg16_bn": vgg16_bn,
    "vgg19_bn": vgg19_bn,
    "wideresnet18":wideresnet18,
    "wideresnet18_4":wideresnet18_4,
    "wideresnet18_4_grouplinear":wideresnet18_4_grouplinear,
    "wideresnet28_10":wideresnet28_10,
    "resnet18": resnet18,
    "resnet34": resnet34,
    "resnet50": resnet50,
    "resnet101": resnet101,
    "shake_26_32": shake_resnet26_2x32d,
    "shake_26_96": shake_resnet26_2x96d,
    "densenet121": densenet121,
    "densenet161": densenet161,
    "densenet169": densenet169,
    "mobilenet_v2": mobilenet_v2,
    "googlenet": googlenet,
    "inception_v3": inception_v3,
    "lenet5" : lenet5,
    "efficientnet_b2": efficientnet_b2,
    "efficientnet_b1":efficientnet_b1,
    "efficientnet_b0":efficientnet_b0,
    "resnet18_cifar100": resnet18_cifar100,
    "vgg13_cifar100": vgg13_bn_cifar100,
    "densenet121_cifar100": densenet121_cifar100,
    "shake_26_32_cifar100": shake_resnet26_2x32d_cifar100,
    "rff_100":rff_100_mnist,
    "rff_10000":rff_10000_mnist,
    "rff_100000":rff_100000_mnist
    "wideresnet18_cifar100": wideresnet18_cifar100,
    "wideresnet28_20" : wideresnet28_20,
    # missing from public codebase
    "resnet8_cifar": resnet8_cf,
    #"efficientnet_b2": efficientnet_b2,
    #"efficientnet_b1":efficientnet_b1,
    #"efficientnet_b0":efficientnet_b0,
    "resnet18_cifar100": resnet18_cifar100,
    #"vgg13_cifar100": vgg13_bn_cifar100,
    #"densenet121_cifar100": densenet121_cifar100,
    #"shake_26_32_cifar100": shake_resnet26_2x32d_cifar100
}

all_regressors = {
        "rff_1000": rff_regress_1000_wine,
        "rff_10000": rff_regress_10000_wine,
        "rff_100000": rff_regress_100000_wine,
        "linear_reg": linreg_wine,
        "rff_1000_casf": rff_casregress_1000_mnist,
        "rff_10000_casf": rff_casregress_10000_mnist,
        "rff_100000_casf": rff_casregress_100000_mnist,
        # missing from public codebase
        # "rff_8000_casf": rff_casregress_8000_mnist,

}

class Regression_Models(pl.LightningModule):
    """Base class for regression models. 

    """
    def __init__(self,hparams):
        super().__init__()
        self.hparams = hparams
    def forward(x):    
        raise NotImplementedError
    def training_step():
        raise NotImplementedError

    def validation_step(self, batch, batch_nb):
        loss, accuracy = self.forward(batch)
        self.log("loss/val", loss)
        self.log("acc/val", accuracy)

    def test_step(self, batch, batch_nb):
        loss, accuracy = self.forward(batch)
        self.log("acc/test", accuracy)

    def setup_scheduler(self,optimizer,total_steps):
        """Chooses between the cosine learning rate scheduler that came with the repo, or step scheduler based on wideresnet training. 

        """
        if self.hparams.scheduler in [None,"cosine"]: 
            scheduler = {
                "scheduler": WarmupCosineLR(
                    optimizer, warmup_epochs=total_steps*0.3, max_epochs=total_steps
                ),
                "interval": "step",
                "name": "learning_rate",
            }
        elif self.hparams.scheduler == "step":    
            scheduler = {
                "scheduler": torch.optim.lr_scheduler.MultiStepLR(
                    optimizer, milestones = [60,120,160], gamma = 0.2, last_epoch=-1
                ),
                "interval": "epoch",
                "frequency":1,
                "name": "learning_rate",
                }
        elif self.hparams.scheduler == "lambdalr":
            scheduler = {
                "scheduler": torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                          lambda epoch: 0.1 ** (epoch // 30)
                                                              ),
                "interval": "epoch",
                "frequency": 1,
                "name": "learning_rate",
                }

        return scheduler    
#####regression

class RegressionSingleModel(Regression_Models):
    def __init__(self, hparams):
        super().__init__(hparams)
        print(hparams)
        print(self.hparams)

        self.criterion = torch.nn.MSELoss()
        self.ev = ExplainedVariance()#Accuracy()

        self.model = all_regressors[self.hparams.classifier]()

    def forward(self, batch):
        images, labels = batch
        predictions = self.model(images)
        loss = self.criterion(predictions, labels)
        ev = self.ev(predictions, labels)
        return loss, ev 

    def calibration(self,batch,use_softmax = True):
        """Like forward, but just exit with the softmax predictions and labels. . 
        """
        images, labels = batch
        predictions = self.model(images)
        return predictions,labels

    def training_step(self, batch, batch_nb):
        loss, ev = self.forward(batch)
        self.log("loss/train", loss)
        self.log("acc/train", ev)
        return loss

    def validation_step(self, batch, batch_nb):
        loss, ev = self.forward(batch)
        self.log("loss/val", loss)
        self.log("acc/val", ev)

    def test_step(self, batch, batch_nb):
        loss, ev = self.forward(batch)
        self.log("acc/test", ev)

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

class RegressionEnsembleModel(Regression_Models):
    def __init__(self,hparams):
        super().__init__(hparams)
        self.nb_models = hparams.nb_models

        self.criterion = torch.nn.MSELoss()
        self.ev = ExplainedVariance()#Accuracy()

        self.models = torch.nn.ModuleList([all_regressors[self.hparams.classifier]() for i in range(self.nb_models)]) ## now we add several different instances of the model. 
        #del self.model
    
    def forward(self,batch):
        """for forward, we want to take the softmax, aggregate the ensemble output, and then take the logit.  

        """
        images, labels = batch

        all_predictions = []
        for m in self.models:
            predictions = m(images)
            all_predictions.append(predictions)
        mean = torch.mean(torch.stack(all_predictions),dim = 0) 
        ## we can pass this  through directly to the accuracy function. 
        tloss = self.criterion(mean,labels)## beware: this is a transformed input, don't evaluate on test loss of ensembles. 
        accuracy = self.ev(mean,labels)
        return tloss,accuracy

    def calibration(self,batch):
        """Like forward, but just exit with the predictions and labels. . 
        """
        images, labels = batch
        softmax = torch.nn.Softmax(dim = 1)

        all_predictions = []
        for m in self.models:
            predictions = m(images)
            all_predictions.append(predictions)
        #gmean = torch.exp(torch.mean(torch.log(torch.stack(softmaxes)),dim = 0)) ## implementation from https://stackoverflow.com/questions/59722983/how-to-calculate-geometric-mean-in-a-differentiable-way   
        mean = torch.mean(torch.stack(all_predictions),dim = 0) 
        return mean,labels

    def training_step(self, batch, batch_nb):
        """When we train, we want to train independently. 
        """
        
        images, labels = batch
        losses = []
        accs = []
        for m in self.models:
            predictions = m(images) ## this just a bunch of unnormalized scores? 
            mloss = self.criterion(predictions, labels)
            accuracy = self.ev(predictions,labels)
            losses.append(mloss)
            accs.append(accuracy) 
        loss = sum(losses)/self.nb_models ## calculate the sum with pure python functions.    
        avg_accuracy = sum(accs)/self.nb_models

        self.log("loss/train", loss)
        self.log("acc/train", avg_accuracy)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.models.parameters(),
            #lr=self.hparams.learning_rate*len(self.models), ## when jointly training, we need to multiply the learning rate times the number of ensembles to make sure that the effective learning rate for each model stays the same. 
            lr=self.hparams.learning_rate, ## when jointly training, we need to multiply the learning rate times the number of ensembles to make sure that the effective learning rate for each model stays the same. 
            weight_decay=self.hparams.weight_decay,
            momentum=0.9,
            nesterov=True,
        )
        total_steps = self.hparams.max_epochs * len(self.train_dataloader())
        scheduler = self.setup_scheduler(optimizer,total_steps)
        return [optimizer], [scheduler]

    def setup_scheduler(self,optimizer,total_steps):
        """Chooses between the cosine learning rate scheduler that came with the repo, or step scheduler based on wideresnet training.For the ensemble, we need to manually set the warmup and eta_min parameters to maintain the right scaling for individual models.  
        """
        if self.hparams.scheduler in [None,"cosine"]: 
            scheduler = {
                "scheduler": WarmupCosineLR(
                    optimizer,
                    warmup_epochs=total_steps*0.3,
                    max_epochs=total_steps,
                    warmup_start_lr = 1e-8,
                    eta_min = 1e-8
                    #warmup_start_lr = 1e-8*len(self.models),
                    #eta_min = 1e-8*len(self.models)
                ),
                "interval": "step",
                "name": "learning_rate",
            }
        elif self.hparams.scheduler == "step":    
            scheduler = {
                "scheduler": torch.optim.lr_scheduler.MultiStepLR(
                    optimizer, milestones = [60,120,160], gamma = 0.2, last_epoch=-1
                ),
                "interval": "epoch",
                "frequency":1,
                "name": "learning_rate",
                }
        return scheduler    

class RegressionEnsemble_JGModel(RegressionEnsembleModel):
    def __init__(self,hparams):
        super().__init__(hparams)
        self.traincriterion = torch.nn.MSELoss()
        self.jg = Regression_Var("torch")
        self.gamma = hparams.gamma

    def training_step(self, batch, batch_nb):
        """When we train, we want to train independently. 
        """
        
        images, labels = batch
        all_predictions = []
        for m in self.models:
            predictions = m(images) ## this just a bunch of unnormalized scores? 
            all_predictions.append(predictions)
        output = torch.mean(torch.stack(all_predictions),dim = 0)
        mloss = self.traincriterion(output, labels)
        varloss = torch.mean(self.jg.var(all_predictions))
        loss = (mloss + self.gamma*varloss) ## with gamma equal to 1, this is the same as the standard ensemble training loss (independent). 
        accuracy = self.ev(output,labels)

        lr = self.trainer.lr_schedulers[0]["scheduler"].get_last_lr()[-1]
        self.log("lr/lr",lr)
        self.log("loss/train", loss)
        self.log("acc/train", accuracy)
        self.log("reg/dkl",varloss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.models.parameters(),
            lr=self.hparams.learning_rate, ## when jointly training, we need to multiply the learning rate times the number of ensembles to make sure that the effective learning rate for each model stays the same. 
            #lr=self.hparams.learning_rate*len(self.models), ## when jointly training, we need to multiply the learning rate times the number of ensembles to make sure that the effective learning rate for each model stays the same. 
            weight_decay=self.hparams.weight_decay,
            momentum=0.9,
            nesterov=True,
        )
        total_steps = self.hparams.max_epochs * len(self.train_dataloader())
        scheduler = self.setup_scheduler(optimizer,total_steps)
        return [optimizer], [scheduler]

    def setup_scheduler(self,optimizer,total_steps):
        """Chooses between the cosine learning rate scheduler that came with the repo, or step scheduler based on wideresnet training.For the ensemble, we need to manually set the warmup and eta_min parameters to maintain the right scaling for individual models.  
        """
        if self.hparams.scheduler in [None,"cosine"]: 
            scheduler = {
                "scheduler": WarmupCosineLR(
                    optimizer,
                    warmup_epochs=total_steps*0.3,
                    max_epochs=total_steps,
                    #warmup_start_lr = 1e-8*len(self.models),
                    #eta_min = 1e-8*len(self.models)
                    warmup_start_lr = 1e-8,
                    eta_min = 1e-8
                    ),
                "interval": "step",
                "name": "learning_rate",
            }
        elif self.hparams.scheduler == "step":    
            scheduler = {
                "scheduler": torch.optim.lr_scheduler.MultiStepLR(
                    optimizer, milestones = [60,120,160], gamma = 0.2, last_epoch=-1
                ),
                "interval": "epoch",
                "frequency":1,
                "name": "learning_rate",
                }
        return scheduler    
#####
    
#####Classification As retression
class ClassasRegressionSingleModel(Regression_Models):
    def __init__(self, hparams):
        super().__init__(hparams)
        print(hparams)
        print(self.hparams)

        self.criterion = torch.nn.MSELoss()
        self.acc = Accuracy()

        self.model = all_regressors[self.hparams.classifier]()

    def forward(self, batch):
        images, labels = batch
        predictions = self.model(torch.flatten(images,start_dim=1))
        loss = self.criterion(predictions, labels)
        acc = self.acc(predictions, labels)
        return loss, acc 

    def calibration(self,batch,use_softmax = True):
        """Like forward, but just exit with the softmax predictions and labels. . 
        """
        images, labels = batch
        predictions = self.model(torch.flatten(images,start_dim=1))
        return predictions,labels

    def training_step(self, batch, batch_nb):
        loss, acc = self.forward(batch)
        self.log("loss/train", loss)
        self.log("acc/train", acc)
        return loss

    def validation_step(self, batch, batch_nb):
        loss, acc = self.forward(batch)
        self.log("loss/val", loss)
        self.log("acc/val", acc)

    def test_step(self, batch, batch_nb):
        loss, acc = self.forward(batch)
        self.log("acc/test", acc)

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


class ClassasRegressionSingleModelOneHot(Regression_Models):
    def __init__(self, hparams):
        super().__init__(hparams)
        print(hparams)
        print(self.hparams)

        self.acc = Accuracy()
        self.num_classes = hparams.get('num_classes', 10)
        self.model = all_classifiers[self.hparams.classifier](num_classes=self.num_classes)
        self.criterion = MSELoss_classification(num_classes=self.num_classes)

    def forward(self, batch):
        images, labels = batch
        predictions = self.model(images)
        loss = self.criterion(predictions, labels)
        acc = self.acc(predictions.max(1)[1], labels)
        return loss, acc*100

    def calibration(self,batch,use_softmax = False, store_split=False):
        """Like forward, but just exit with the softmax predictions and labels. .
        """
        images, labels = batch
        predictions = self.model(images)
        return predictions,labels

    def training_step(self, batch, batch_nb):
        loss, acc = self.forward(batch)
        self.log("loss/train", loss)
        self.log("acc/train", acc)

        lr = self.trainer.lr_schedulers[0]["scheduler"].get_last_lr()[-1]
        self.log("lr/lr", lr)

        return loss

    def validation_step(self, batch, batch_nb):
        loss, acc = self.forward(batch)
        self.log("loss/val", loss)
        self.log("acc/val", acc)

    def test_step(self, batch, batch_nb):
        loss, acc = self.forward(batch)
        self.log("acc/test", acc)

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


class ClassasRegressionEnsembleModelOneHot(Regression_Models):
    def __init__(self, hparams):
        super().__init__(hparams)
        self.nb_models = hparams.nb_models

        self.accuracy = Accuracy()
        self.num_classes = hparams.get('num_classes', 10)
        self.criterion = MSELoss_classification(num_classes=self.num_classes)

        self.models = torch.nn.ModuleList([all_classifiers[self.hparams.classifier](num_classes=self.num_classes) for i in range(
            self.nb_models)])  ## now we add several different instances of the model.
        # del self.model


    def forward(self, batch):
        """for forward, we want to take the softmax, aggregate the ensemble output, and then take the logit.

        """
        images, labels = batch
        all_predictions = []
        for m in self.models:
            predictions = m(images)
            all_predictions.append(predictions)
        mean = torch.mean(torch.stack(all_predictions), dim=0)
        ## we can pass this  through directly to the accuracy function.
        tloss = self.criterion(mean, labels)
        # accuracy = self.acc(mean, labels)
        accuracy = self.accuracy(mean.max(1)[1], labels)
        return tloss, accuracy*100

    def calibration(self, batch, store_split=False):
        """Like forward, but just exit with the predictions and labels. .
        """
        images, labels = batch

        all_predictions = []
        for m in self.models:
            predictions = m(images)
            all_predictions.append(predictions)
        # gmean = torch.exp(torch.mean(torch.log(torch.stack(softmaxes)),dim = 0)) ## implementation from https://stackoverflow.com/questions/59722983/how-to-calculate-geometric-mean-in-a-differentiable-way
        if store_split:
            mean = torch.stack(all_predictions, 1)
        else:
            mean = torch.mean(torch.stack(all_predictions), dim=0)
        return mean, labels


    def training_step(self, batch, batch_nb):
        """When we train, we want to train independently.
        """

        images, labels = batch
        losses = []
        accs = []
        for m in self.models:
            predictions = m(images)
            mloss = self.criterion(predictions, labels)
            accuracy = self.accuracy(predictions.max(1)[1], labels)
            losses.append(mloss)
            accs.append(accuracy)
        loss = sum(losses) / self.nb_models  ## calculate the sum with pure python functions.
        avg_accuracy = sum(accs) / self.nb_models

        self.log("loss/train", loss)
        self.log("acc/train", avg_accuracy)
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
                    warmup_epochs=total_steps * 0.3,
                    max_epochs=total_steps,
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
        elif self.hparams.scheduler == "lambdalr":
            scheduler = {
                "scheduler": torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                          lambda epoch: 0.1 ** (epoch // 30)
                ),
                "interval": "epoch",
                "frequency": 1,
                "name": "learning_rate",
                }

        return scheduler


class ClassasRegressionEnsembleJGAPModelOneHot(ClassasRegressionEnsembleModelOneHot):
    """Formulation of the ensemble as a regularized single model with variable weight on regularization.

    """

    def __init__(self, hparams):
        super().__init__(hparams)
        self.traincriterion = MSELoss_classification(num_classes=self.num_classes)
        self.gamma = hparams.gamma
        self.jgap_weight = 1 # (self.nb_models - 1)/self.nb_models

    def training_step(self, batch, batch_nb):
        """When we train, we want to train independently.
        Loss = NLL(log \bar{f}, y ) + gamma*JGAP(predictions, label)
        JGAP = avg_sm_loss - mloss
        """

        images, labels = batch
        losses = []
        accs = []
        softmaxes = []
        for m in self.models:
            # get logits
            predictions = m(images)
            softmaxes.append(predictions)
            mloss = self.criterion(predictions, labels)
            # accuracy = self.accuracy(predictions,labels)
            losses.append(mloss)
            # accs.append(accuracy)
        logoutput = torch.mean(torch.stack(softmaxes), dim=0)
        mloss = self.traincriterion(logoutput, labels)

        # jensen gap
        avg_sm_loss = sum(losses)/self.nb_models
        jgaploss = avg_sm_loss - mloss

        loss = (
              mloss + self.jgap_weight * self.gamma * jgaploss)  ## with gamma equal to 1, this is the same as the standard ensemble training loss (independent).
        accuracy = self.accuracy(logoutput.max(1)[1], labels)

        lr = self.trainer.lr_schedulers[0]["scheduler"].get_last_lr()[-1]
        self.log("lr/lr", lr)
        self.log("loss/train", loss)
        self.log("acc/train", accuracy * 100)
        self.log("reg/mloss", mloss)
        self.log("reg/jgap", jgaploss)
        self.log("reg/avg_sm_loss", avg_sm_loss)
        return loss


class ClassasRegressionEnsembleModel(Regression_Models):
    def __init__(self,hparams):
        super().__init__(hparams)
        self.nb_models = hparams.nb_models

        self.criterion = torch.nn.MSELoss()
        self.acc = Accuracy()

        self.models = torch.nn.ModuleList([all_regressors[self.hparams.classifier]() for i in range(self.nb_models)]) ## now we add several different instances of the model. 
        #del self.model
    
    def forward(self,batch):
        """for forward, we want to take the softmax, aggregate the ensemble output, and then take the logit.  

        """
        images, labels = batch

        all_predictions = []
        for m in self.models:
            predictions = m(torch.flatten(images,start_dim=1))
            all_predictions.append(predictions)
        mean = torch.mean(torch.stack(all_predictions),dim = 0) 
        ## we can pass this  through directly to the accuracy function. 
        tloss = self.criterion(mean,labels)## beware: this is a transformed input, don't evaluate on test loss of ensembles. 
        accuracy = self.acc(mean,labels)
        return tloss,accuracy

    def calibration(self,batch):
        """Like forward, but just exit with the predictions and labels. . 
        """
        images, labels = batch
        softmax = torch.nn.Softmax(dim = 1)

        all_predictions = []
        for m in self.models:
            predictions = m(torch.flatten(images,start_dim=1))
            all_predictions.append(predictions)
        #gmean = torch.exp(torch.mean(torch.log(torch.stack(softmaxes)),dim = 0)) ## implementation from https://stackoverflow.com/questions/59722983/how-to-calculate-geometric-mean-in-a-differentiable-way   
        mean = torch.mean(torch.stack(all_predictions),dim = 0) 
        return mean,labels

    def training_step(self, batch, batch_nb):
        """When we train, we want to train independently. 
        """
        
        images, labels = batch
        losses = []
        accs = []
        for m in self.models:
            predictions = m(torch.flatten(images,start_dim=1)) ## this just a bunch of unnormalized scores? 
            mloss = self.criterion(predictions, labels)
            accuracy = self.acc(predictions,labels)
            losses.append(mloss)
            accs.append(accuracy) 
        loss = sum(losses)/self.nb_models ## calculate the sum with pure python functions.    
        avg_accuracy = sum(accs)/self.nb_models

        self.log("loss/train", loss)
        self.log("acc/train", avg_accuracy)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.models.parameters(),
            #lr=self.hparams.learning_rate*len(self.models), ## when jointly training, we need to multiply the learning rate times the number of ensembles to make sure that the effective learning rate for each model stays the same. 
            lr=self.hparams.learning_rate, ## when jointly training, we need to multiply the learning rate times the number of ensembles to make sure that the effective learning rate for each model stays the same. 
            weight_decay=self.hparams.weight_decay,
            momentum=0.9,
            nesterov=True,
        )
        total_steps = self.hparams.max_epochs * len(self.train_dataloader())
        scheduler = self.setup_scheduler(optimizer,total_steps)
        return [optimizer], [scheduler]

    def setup_scheduler(self,optimizer,total_steps):
        """Chooses between the cosine learning rate scheduler that came with the repo, or step scheduler based on wideresnet training.For the ensemble, we need to manually set the warmup and eta_min parameters to maintain the right scaling for individual models.  
        """
        if self.hparams.scheduler in [None,"cosine"]: 
            scheduler = {
                "scheduler": WarmupCosineLR(
                    optimizer,
                    warmup_epochs=total_steps*0.3,
                    max_epochs=total_steps,
                    warmup_start_lr = 1e-8,
                    eta_min = 1e-8
                    #warmup_start_lr = 1e-8*len(self.models),
                    #eta_min = 1e-8*len(self.models)
                ),
                "interval": "step",
                "name": "learning_rate",
            }
        elif self.hparams.scheduler == "step":    
            scheduler = {
                "scheduler": torch.optim.lr_scheduler.MultiStepLR(
                    optimizer, milestones = [60,120,160], gamma = 0.2, last_epoch=-1
                ),
                "interval": "epoch",
                "frequency":1,
                "name": "learning_rate",
                }
        return scheduler    

class ClassasRegressionEnsemble_JGModel(ClassasRegressionEnsembleModel):
    def __init__(self,hparams):
        super().__init__(hparams)
        self.traincriterion = torch.nn.MSELoss()
        self.jg = Regression_Var("torch")
        self.gamma = hparams.gamma

    def training_step(self, batch, batch_nb):
        """When we train, we want to train independently. 
        """
        
        images, labels = batch
        all_predictions = []
        for m in self.models:
            predictions = m(torch.flatten(images,start_dim=1)) ## this just a bunch of unnormalized scores? 
            all_predictions.append(predictions)
        output = torch.mean(torch.stack(all_predictions),dim = 0)
        mloss = self.traincriterion(output, labels)
        varloss = torch.mean(torch.sum(self.jg.var(all_predictions),axis = -1))
        loss = (mloss + self.gamma*varloss) ## with gamma equal to 1, this is the same as the standard ensemble training loss (independent). 
        accuracy = self.acc(output,labels)

        lr = self.trainer.lr_schedulers[0]["scheduler"].get_last_lr()[-1]
        self.log("lr/lr",lr)
        self.log("loss/train", loss)
        self.log("acc/train", accuracy)
        self.log("reg/dkl",varloss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.models.parameters(),
            lr=self.hparams.learning_rate, ## when jointly training, we need to multiply the learning rate times the number of ensembles to make sure that the effective learning rate for each model stays the same. 
            #lr=self.hparams.learning_rate*len(self.models), ## when jointly training, we need to multiply the learning rate times the number of ensembles to make sure that the effective learning rate for each model stays the same. 
            weight_decay=self.hparams.weight_decay,
            momentum=0.9,
            nesterov=True,
        )
        total_steps = self.hparams.max_epochs * len(self.train_dataloader())
        scheduler = self.setup_scheduler(optimizer,total_steps)
        return [optimizer], [scheduler]

    def setup_scheduler(self,optimizer,total_steps):
        """Chooses between the cosine learning rate scheduler that came with the repo, or step scheduler based on wideresnet training.For the ensemble, we need to manually set the warmup and eta_min parameters to maintain the right scaling for individual models.  
        """
        if self.hparams.scheduler in [None,"cosine"]: 
            scheduler = {
                "scheduler": WarmupCosineLR(
                    optimizer,
                    warmup_epochs=total_steps*0.3,
                    max_epochs=total_steps,
                    #warmup_start_lr = 1e-8*len(self.models),
                    #eta_min = 1e-8*len(self.models)
                    warmup_start_lr = 1e-8,
                    eta_min = 1e-8
                    ),
                "interval": "step",
                "name": "learning_rate",
            }
        elif self.hparams.scheduler == "step":    
            scheduler = {
                "scheduler": torch.optim.lr_scheduler.MultiStepLR(
                    optimizer, milestones = [60,120,160], gamma = 0.2, last_epoch=-1
                ),
                "interval": "epoch",
                "frequency":1,
                "name": "learning_rate",
                }
        return scheduler    

#####

class CIFAR100_Models(pl.LightningModule):
    """Abstract base class for CIFAR100 Models

    """
    def __init__(self,hparams):
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

    def setup_scheduler(self,optimizer,total_steps):
        """Chooses between the cosine learning rate scheduler that came with the repo, or step scheduler based on wideresnet training. 

        """
        if self.hparams.scheduler in [None,"cosine"]: 
            scheduler = {
                "scheduler": WarmupCosineLR(
                    optimizer, warmup_epochs=total_steps*0.3, max_epochs=total_steps
                ),
                "interval": "step",
                "name": "learning_rate",
            }
        elif self.hparams.scheduler == "step":    
            scheduler = {
                "scheduler": torch.optim.lr_scheduler.MultiStepLR(
                    optimizer, milestones = [60,120,160], gamma = 0.2, last_epoch=-1
                ),
                "interval": "epoch",
                "frequency":1,
                "name": "learning_rate",
                }
        return scheduler    

################

class CIFAR10_Models(pl.LightningModule):
    """Abstract base class for CIFAR10 Models

    """
    def __init__(self,hparams):
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

    def setup_scheduler(self,optimizer,total_steps):
        """Chooses between the cosine learning rate scheduler that came with the repo, or step scheduler based on wideresnet training. 

        """
        if self.hparams.scheduler in [None,"cosine"]: 
            scheduler = {
                "scheduler": WarmupCosineLR(
                    optimizer, warmup_epochs=total_steps*0.3, max_epochs=total_steps
                ),
                "interval": "step",
                "name": "learning_rate",
            }
        elif self.hparams.scheduler == "step":    
            scheduler = {
                "scheduler": torch.optim.lr_scheduler.MultiStepLR(
                    optimizer, milestones = [60,120,160], gamma = 0.2, last_epoch=-1
                ),
                "interval": "epoch",
                "frequency":1,
                "name": "learning_rate",
                }
        elif self.hparams.scheduler == "lambdalr":
            scheduler = {
                "scheduler": torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                          lambda epoch: 0.1 ** (epoch // 30)
                ),
                "interval": "epoch",
                "frequency": 1,
                "name": "learning_rate",
                }

        return scheduler

class CIFAR10LinearGroupModule(CIFAR10_Models):
    """Replaces the final layer with a LogSoftmaxGroupLinear layer, and correspondingly changes the loss. 

    """
    def __init__(self, hparams):
        super().__init__(hparams)

        self.criterion = torch.nn.NLLLoss()
        self.accuracy = Accuracy()
        assert self.hparams.classifier.endswith("grouplinear"), "will give strange results for those that don't have softmax grouplinear output"
        self.model = all_classifiers[self.hparams.classifier]()

    def forward(self, batch):
        images, labels = batch
        predictions = self.model(images)
        loss = self.criterion(predictions, labels)
        accuracy = self.accuracy(predictions, labels)
        return loss, accuracy * 100

    def calibration(self, batch):
        images, labels = batch
        predictions = self.model(images)
        return predictions, labels

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

class CIFAR10Module(CIFAR10_Models):
    def __init__(self, hparams):
        super().__init__(hparams)
        print(hparams)
        print(self.hparams)
        self.label_smoothing = self.hparams.get('label_smoothing', 0.0)
        self.criterion = torch.nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        self.accuracy = Accuracy()

        self.num_classes = hparams.get('num_classes', 10)
        self.model = all_classifiers[self.hparams.classifier](
            num_classes=self.num_classes)

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

        lr = self.trainer.lr_schedulers[0]["scheduler"].get_last_lr()[-1]
        self.log("lr/lr",lr)

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

class CIFAR10EnsembleModule(CIFAR10_Models):   
    """Customized module to train an ensemble of models independently

    """
    def __init__(self,hparams):
        super().__init__(hparams)
        self.nb_models = hparams.nb_models

        self.label_smoothing = self.hparams.get('label_smoothing', 0.0)
        self.criterion = torch.nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        self.fwd_criterion = torch.nn.NLLLoss()
        self.accuracy = Accuracy()
        self.num_classes = hparams.get('num_classes', 10)

        self.models = torch.nn.ModuleList([all_classifiers[self.hparams.classifier](num_classes=self.num_classes) for i in range(self.nb_models)]) ## now we add several different instances of the model.
        #del self.model
    
    def forward(self,batch):
        """for forward pass, we want to take the softmax,
        aggregate the ensemble output, take log(\bar{f}) and apply NNLoss.
        prediction  = \bar{f}
        """
        images, labels = batch
        softmax = torch.nn.Softmax(dim = 1)

        losses = []
        accs = []
        softmaxes = []
        for m in self.models:
            predictions = m(images)
            normed = softmax(predictions)
            softmaxes.append(normed)
        mean = torch.mean(torch.stack(softmaxes),dim = 0) 
        ## we can pass this  through directly to the accuracy function.
        logoutput = torch.log(mean)
        tloss = self.fwd_criterion(logoutput,labels) ## beware: this is a transformed input, don't evaluate on test loss of ensembles.
        accuracy = self.accuracy(mean,labels)
        return tloss,accuracy*100

    def calibration(self,batch, store_split=False):
        """Like forward, but just exit with the predictions and labels. . 
        """
        images, labels = batch
        softmax = torch.nn.Softmax(dim = 1)

        losses = []
        accs = []
        softmaxes = []
        for m in self.models:
            predictions = m(images)
            normed = softmax(predictions)
            softmaxes.append(normed)
        #gmean = torch.exp(torch.mean(torch.log(torch.stack(softmaxes)),dim = 0)) ## implementation from https://stackoverflow.com/questions/59722983/how-to-calculate-geometric-mean-in-a-differentiable-way   
        if store_split:
            mean = torch.stack(softmaxes, 1)
        else:
            mean = torch.mean(torch.stack(softmaxes),dim = 0)
        return mean,labels

    def training_step(self, batch, batch_nb):
        """When we train, we want to train independently.
        Loss is the  average single model loss
        Loss = 1/M sum_i L( f_i, y), where f_i is the model output for the ith model.
        """
        
        images, labels = batch
        losses = []
        accs = []
        for m in self.models:
            predictions = m(images) ## this just a bunch of unnormalized scores? 
            mloss = self.criterion(predictions, labels)
            accuracy = self.accuracy(predictions,labels)
            losses.append(mloss)
            accs.append(accuracy) 
        loss = sum(losses)/self.nb_models ## calculate the sum with pure python functions.    
        avg_accuracy = sum(accs)/self.nb_models

        self.log("loss/train", loss)
        self.log("acc/train", avg_accuracy*100)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.models.parameters(),
            #lr=self.hparams.learning_rate*len(self.models), ## when jointly training, we need to multiply the learning rate times the number of ensembles to make sure that the effective learning rate for each model stays the same. 
            lr=self.hparams.learning_rate, ## when jointly training, we need to multiply the learning rate times the number of ensembles to make sure that the effective learning rate for each model stays the same. 
            weight_decay=self.hparams.weight_decay,
            momentum=0.9,
            nesterov=True,
        )
        total_steps = self.hparams.max_epochs * len(self.train_dataloader())
        scheduler = self.setup_scheduler(optimizer,total_steps)
        return [optimizer], [scheduler]

    def setup_scheduler(self,optimizer,total_steps):
        """Chooses between the cosine learning rate scheduler that came with the repo, or step scheduler based on wideresnet training.For the ensemble, we need to manually set the warmup and eta_min parameters to maintain the right scaling for individual models.  
        """
        if self.hparams.scheduler in [None,"cosine"]: 
            scheduler = {
                "scheduler": WarmupCosineLR(
                    optimizer,
                    warmup_epochs=total_steps*0.3,
                    max_epochs=total_steps,
                    warmup_start_lr = 1e-8,
                    eta_min = 1e-8
                    #warmup_start_lr = 1e-8*len(self.models),
                    #eta_min = 1e-8*len(self.models)
                ),
                "interval": "step",
                "name": "learning_rate",
            }
        elif self.hparams.scheduler == "step":    
            scheduler = {
                "scheduler": torch.optim.lr_scheduler.MultiStepLR(
                    optimizer, milestones = [60,120,160], gamma = 0.2, last_epoch=-1
                ),
                "interval": "epoch",
                "frequency":1,
                "name": "learning_rate",
                }
        elif self.hparams.scheduler == "lambdalr":
            scheduler = {
                "scheduler": torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                          lambda epoch: 0.1 ** (epoch // 30)
                                                              ),
                "interval": "epoch",
                "frequency": 1,
                "name": "learning_rate",
                }

        return scheduler    

class CIFAR10EnsembleDKLModule(CIFAR10EnsembleModule):
    """Formulation of the ensemble as a regularized single model with variable weight on regularization. 
    Note: label_smoothing does not act on regularizer.
    For full label smoothing use CIFAR10EnsembleJGAPModule
    """
    def __init__(self,hparams):
        super().__init__(hparams)
        self.traincriterion = NLLLoss_label_smooth(self.num_classes, self.label_smoothing)
        self.kl = Model_D_KL("torch")
        self.gamma = hparams.gamma

    def training_step(self, batch, batch_nb):
        """When we train, we want to train independently. 
        Loss = NLL(log \bar{f}, y ) + gamma*DKL(softmaxes, label)
        where DKL= 1/M sum_i^M [log (1/M) - log ((f_i^{(y)})/(\sum_i^M f_i^{(y)}))]
        """
        softmax = torch.nn.Softmax(dim = 1)
        
        images, labels = batch
        losses = []
        accs = []
        softmaxes = []
        for m in self.models:
            # get logits
            predictions = m(images)
            normed = softmax(predictions)
            softmaxes.append(normed)
            #mloss = self.criterion(predictions, labels)
            #accuracy = self.accuracy(predictions,labels)
            #losses.append(mloss)
            #accs.append(accuracy)
        logoutput = torch.log(torch.mean(torch.stack(softmaxes),dim = 0))
        mloss = self.traincriterion(logoutput, labels)
        dklloss = torch.mean(self.kl.kl(softmaxes,labels))
        loss = (mloss + self.gamma*dklloss) ## with gamma equal to 1, this is the same as the standard ensemble training loss (independent). 
        accuracy = self.accuracy(logoutput,labels)

        lr = self.trainer.lr_schedulers[0]["scheduler"].get_last_lr()[-1]
        self.log("lr/lr",lr)
        self.log("loss/train", loss)
        self.log("acc/train", accuracy*100)
        self.log("reg/dkl",dklloss)
        self.log("reg/mloss", mloss)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.models.parameters(),
            lr=self.hparams.learning_rate, ## when jointly training, we need to multiply the learning rate times the number of ensembles to make sure that the effective learning rate for each model stays the same. 
            #lr=self.hparams.learning_rate*len(self.models), ## when jointly training, we need to multiply the learning rate times the number of ensembles to make sure that the effective learning rate for each model stays the same. 
            weight_decay=self.hparams.weight_decay,
            momentum=0.9,
            nesterov=True,
        )
        total_steps = self.hparams.max_epochs * len(self.train_dataloader())
        scheduler = self.setup_scheduler(optimizer,total_steps)
        return [optimizer], [scheduler]

    def setup_scheduler(self,optimizer,total_steps):
        """Chooses between the cosine learning rate scheduler that came with the repo, or step scheduler based on wideresnet training.For the ensemble, we need to manually set the warmup and eta_min parameters to maintain the right scaling for individual models.  
        """
        if self.hparams.scheduler in [None,"cosine"]: 
            scheduler = {
                "scheduler": WarmupCosineLR(
                    optimizer,
                    warmup_epochs=total_steps*0.3,
                    max_epochs=total_steps,
                    #warmup_start_lr = 1e-8*len(self.models),
                    #eta_min = 1e-8*len(self.models)
                    warmup_start_lr = 1e-8,
                    eta_min = 1e-8
                    ),
                "interval": "step",
                "name": "learning_rate",
            }
        elif self.hparams.scheduler == "step":    
            scheduler = {
                "scheduler": torch.optim.lr_scheduler.MultiStepLR(
                    optimizer, milestones = [60,120,160], gamma = 0.2, last_epoch=-1
                ),
                "interval": "epoch",
                "frequency":1,
                "name": "learning_rate",
                }
        return scheduler


class CIFAR10EnsembleJGAPModule(CIFAR10EnsembleModule):
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
        avg_sm_loss = sum(losses)/self.nb_models
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

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.models.parameters(),
            lr=self.hparams.learning_rate,
            ## when jointly training, we need to multiply the learning rate times the number of ensembles to make sure that the effective learning rate for each model stays the same.
            # lr=self.hparams.learning_rate*len(self.models), ## when jointly training, we need to multiply the learning rate times the number of ensembles to make sure that the effective learning rate for each model stays the same.
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
                    warmup_epochs=total_steps * 0.3,
                    max_epochs=total_steps,
                    # warmup_start_lr = 1e-8*len(self.models),
                    # eta_min = 1e-8*len(self.models)
                    warmup_start_lr=1e-8,
                    eta_min=1e-8
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
        elif self.hparams.scheduler == "lambdalr":
            scheduler = {
                "scheduler": torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                          lambda epoch: 0.1 ** (epoch // 30)
                                                              ),
                "interval": "epoch",
                "frequency": 1,
                "name": "learning_rate",
                }

        return scheduler


class CIFAR10EnsembleJGAPLModule(CIFAR10EnsembleModule):
    """Formulation of the ensemble as a regularized single model with variable weight on regularization.

    """

    def __init__(self, hparams):
        super().__init__(hparams)
        self.gamma = hparams.gamma

    def training_step(self, batch, batch_nb):
        """When we train, we want to train independently.
        Loss =  CE(\bar{f}, y) + gamma*JGAP(logits, label)
        JGAP = 1/M sum_i^M CE(f_i,y) - CE(\bar{f}, y)
        where f are logits.
        """
        softmax = torch.nn.Softmax(dim=1)

        images, labels = batch
        losses = []
        accs = []
        softmaxes = []
        for m in self.models:
            # get logits
            predictions = m(images)
            softmaxes.append(predictions)
            mloss = self.criterion(predictions, labels)
            # accuracy = self.accuracy(predictions,labels)
            losses.append(mloss)
            # accs.append(accuracy)
        outputs = torch.mean(torch.stack(softmaxes), dim=0)
        mloss = self.criterion(outputs, labels)

        # jensen gap
        avg_sm_loss = sum(losses)/self.nb_models
        jgaploss = avg_sm_loss - mloss

        loss = (
              mloss + self.gamma * jgaploss)  ## with gamma equal to 1, this is the same as the standard ensemble training loss (independent).
        accuracy = self.accuracy(outputs, labels)

        lr = self.trainer.lr_schedulers[0]["scheduler"].get_last_lr()[-1]
        self.log("lr/lr", lr)
        self.log("loss/train", loss)
        self.log("acc/train", accuracy * 100)
        self.log("reg/mloss", mloss)
        self.log("reg/jgap", jgaploss)
        self.log("reg/avg_sm_loss", avg_sm_loss)
        return loss

    def calibration(self,batch):
        """Like forward, but just exit with the predictions and labels. .
        """
        images, labels = batch
        softmax = torch.nn.Softmax(dim = 1)

        losses = []
        accs = []
        softmaxes = []
        for m in self.models:
            predictions = m(images)
            softmaxes.append(predictions)
        #gmean = torch.exp(torch.mean(torch.log(torch.stack(softmaxes)),dim = 0)) ## implementation from https://stackoverflow.com/questions/59722983/how-to-calculate-geometric-mean-in-a-differentiable-way
        mean = torch.mean(torch.stack(softmaxes),dim = 0)
        mean = softmax(mean)
        return mean, labels

    def forward(self, batch):
        """for forward pass, we want to take the average the predictions

        """
        images, labels = batch
        softmax = torch.nn.Softmax(dim=1)

        losses = []
        accs = []
        softmaxes = []
        for m in self.models:
            predictions = m(images)
            softmaxes.append(predictions)
        mean = torch.mean(torch.stack(softmaxes), dim=0)
        ## we can pass this  through directly to the accuracy function.
        tloss = self.criterion(mean, labels)
        accuracy = self.accuracy(mean, labels)
        return tloss, accuracy * 100

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.models.parameters(),
            lr=self.hparams.learning_rate,
            ## when jointly training, we need to multiply the learning rate times the number of ensembles to make sure that the effective learning rate for each model stays the same.
            # lr=self.hparams.learning_rate*len(self.models), ## when jointly training, we need to multiply the learning rate times the number of ensembles to make sure that the effective learning rate for each model stays the same.
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
                    warmup_epochs=total_steps * 0.3,
                    max_epochs=total_steps,
                    # warmup_start_lr = 1e-8*len(self.models),
                    # eta_min = 1e-8*len(self.models)
                    warmup_start_lr=1e-8,
                    eta_min=1e-8
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


class CIFAR10EnsemblePAC2BModule(CIFAR10EnsembleModule):
    """Customized module to train with PAC2B loss from ortega et al. 
    NOTE: Unlike all other losses below and above, here we SUBTRACT diversity, so gammas will flip sign.  

    """
    def __init__(self,hparams):
        super().__init__(hparams)
        self.traincriterion = torch.nn.NLLLoss()
        self.var = Model_Ortega_Variance("torch")
        self.gamma = hparams.gamma

    def training_step(self, batch, batch_nb):
        """When we train, we want to train independently. 
        """
        softmax = torch.nn.Softmax(dim = 1)
        
        images, labels = batch
        losses = []
        accs = []
        softmaxes = []
        for m in self.models:
            predictions = m(images) ## this just a bunch of unnormalized scores? 
            normed = softmax(predictions)
            softmaxes.append(normed)
            mloss = self.criterion(predictions, labels)
            accuracy = self.accuracy(predictions,labels)
            losses.append(mloss)
            accs.append(accuracy) 
        ## standard loss: 
        llloss = sum(losses)/self.nb_models ## calculate the sum with pure python functions.    
        avg_accuracy = sum(accs)/self.nb_models

        ## diversity term:
        varloss = torch.mean(self.var.var([torch.log(s) for s in softmaxes],labels))

        loss = (llloss - self.gamma*varloss) ## with gama = 1, this is equal to the PAC2B loss. 

        self.log("loss/train_ll", llloss)
        self.log("reg/var",varloss)
        self.log("loss/train", loss)
        self.log("acc/train", avg_accuracy*100)

        lr = self.trainer.lr_schedulers[0]["scheduler"].get_last_lr()[-1]
        self.log("lr/lr",lr)
        return loss

class CIFAR10EnsembleJS_Unif_Module(CIFAR10EnsembleModule):
    """Customized module to train with  

    """
    def __init__(self,hparams):
        super().__init__(hparams)
        self.traincriterion = torch.nn.NLLLoss()
        self.js = Model_JS_Unif("torch")
        self.gamma = hparams.gamma

    def training_step(self, batch, batch_nb):
        """When we train, we want to train independently. 
        """
        softmax = torch.nn.Softmax(dim = 1)
        
        images, labels = batch
        losses = []
        accs = []
        softmaxes = []
        for m in self.models:
            predictions = m(images) ## this just a bunch of unnormalized scores? 
            normed = softmax(predictions)
            softmaxes.append(normed)
            mloss = self.criterion(predictions, labels)
            accuracy = self.accuracy(predictions,labels)
            losses.append(mloss)
            accs.append(accuracy) 
        ## standard loss: 
        llloss = sum(losses)/self.nb_models ## calculate the sum with pure python functions.    
        avg_accuracy = sum(accs)/self.nb_models

        ## diversity term:
        divloss = torch.mean(self.js.js_unif([s for s in softmaxes]))

        loss = (llloss + self.gamma*divloss) ## with gama = 0, this is equal to normal training. 

        self.log("loss/train_ll", llloss)
        self.log("reg/var",divloss)
        self.log("loss/train", loss)
        self.log("acc/train", avg_accuracy*100)

        lr = self.trainer.lr_schedulers[0]["scheduler"].get_last_lr()[-1]
        self.log("lr/lr",lr)
        return loss

class CIFAR10EnsembleJS_Avg_Module(CIFAR10EnsembleModule):
    """Customized module to train with  

    """
    def __init__(self,hparams):
        super().__init__(hparams)
        self.traincriterion = torch.nn.NLLLoss()
        self.js = Model_JS_Avg("torch")
        self.gamma = hparams.gamma

    def training_step(self, batch, batch_nb):
        """When we train, we want to train independently. 
        """
        softmax = torch.nn.Softmax(dim = 1)
        
        images, labels = batch
        losses = []
        accs = []
        softmaxes = []
        for m in self.models:
            predictions = m(images) ## this just a bunch of unnormalized scores? 
            normed = softmax(predictions)
            softmaxes.append(normed)
            mloss = self.criterion(predictions, labels)
            accuracy = self.accuracy(predictions,labels)
            losses.append(mloss)
            accs.append(accuracy) 
        ## standard loss: 
        llloss = sum(losses)/self.nb_models ## calculate the sum with pure python functions.    
        avg_accuracy = sum(accs)/self.nb_models

        ## diversity term:
        divloss = torch.mean(self.js.js_avg([s for s in softmaxes]))

        loss = (llloss + self.gamma*divloss) ## with gama = 0, this is equal to normal training 

        self.log("loss/train_ll", llloss)
        self.log("reg/var",divloss)
        self.log("loss/train", loss)
        self.log("acc/train", avg_accuracy*100)

        lr = self.trainer.lr_schedulers[0]["scheduler"].get_last_lr()[-1]
        self.log("lr/lr",lr)
        return loss

class CIFAR10EnsembleDKL_Avg_Module(CIFAR10EnsembleModule):
    """Customized module to train with  

    """
    def __init__(self,hparams):
        super().__init__(hparams)
        self.traincriterion = torch.nn.NLLLoss()
        self.dkl = Model_DKL_Avg("torch")
        self.gamma = hparams.gamma

    def training_step(self, batch, batch_nb):
        """When we train, we want to train independently.
        Loss = 1/M sum_i CE(f_i, y) + gamma * DKL
        where
        DKL = DKL(\bar{f}, f_i) = 1/M sum_j KL(\bar{f}, f_j)
        implemented as 1/M \sum_{m=1}^M \sum_c=1^{C} \bar{f}_{b,c} * (\log(\bar{f}_{b,c}) - \log(f_{m, b, c}))
        where \bar{f} is B x C, and f is M x B x C
        and f_i are the probabilities (not logits).
        """
        softmax = torch.nn.Softmax(dim = 1)
        
        images, labels = batch
        losses = []
        accs = []
        softmaxes = []
        for m in self.models:
            predictions = m(images) ## this just a bunch of unnormalized scores? 
            normed = softmax(predictions)
            softmaxes.append(normed)
            mloss = self.criterion(predictions, labels)
            accuracy = self.accuracy(predictions,labels)
            losses.append(mloss)
            accs.append(accuracy) 
        ## standard loss: 
        llloss = sum(losses)/self.nb_models ## calculate the sum with pure python functions.    
        avg_accuracy = sum(accs)/self.nb_models

        ## diversity term:
        divloss = torch.mean(self.dkl.dkl_avg([s for s in softmaxes]))

        loss = (llloss + self.gamma*divloss) ## with gama = 1, this is equal to the PAC2B loss. 

        self.log("loss/train_ll", llloss)
        self.log("reg/var",divloss)
        self.log("loss/train", loss)
        self.log("acc/train", avg_accuracy*100)

        lr = self.trainer.lr_schedulers[0]["scheduler"].get_last_lr()[-1]
        self.log("lr/lr",lr)
        return loss

class CIFAR10AttentionEnsembleModule(CIFAR10_Models):
    """Customized module to train with attention. Initialized the same way as standard ensembles.  

    """
    def __init__(self,hparams):
        super().__init__(hparams)
        self.nb_models = hparams.nb_models

        #self.criterion = torch.nn.CrossEntropyLoss()
        self.criterion = torch.nn.NLLLoss(reduction = "none")

        self.accuracy = Accuracy()

        self.models = torch.nn.ModuleList([all_classifiers[self.hparams.classifier]() for i in range(self.nb_models)]) ## now we add several different instances of the model. 
        ## applied at logit layer... 
        self.posenc = PosEncodingsSq(10,0.1,10)
        self.attnlayer = self.get_attnlayer(10,hparams.embedding_dim) ## project from 10 dimensional output (CIFAR10 logits) to embedding dimension.
        self.model = torch.nn.ModuleList([self.models,self.attnlayer])
        
    def get_attnlayer(self,in_dim,out_dim):
        """get the attention layer we will use; changed to MLP for right now. 
        """
        return AttnMLPComparison(in_dim,out_dim)


    def forward(self,batch): 
        """First pass forward: try treating like a standard ensemble? 

        """
        images, labels = batch
        softmax = torch.nn.Softmax(dim = 2)

        losses = []
        accs = []
        logits = []
        for m in self.models: ## take these logits, and build up another set of outputs on them. 
            predictions = m(images) ## these are just the pre-softmax outputs. 
            logits.append(predictions)
        logittensor = torch.stack(logits,axis =1)
        #logittensor = self.posenc(logittensor) ## shape [batch,models,predictions]    

        ## Split into two branches: one to calculate attention weights and another to calculate output. 
        # branch 1: calculate attention weights.  
        weights = self.attnlayer(logittensor,logittensor) ## gives attention weights with shape [batch,queries, models]
        #self.log("attn/normq",torch.linalg.matrix_norm(self.attnlayer.linear_q.weight))
        #self.log("attn/normk",torch.linalg.matrix_norm(self.attnlayer.linear_k.weight))
        self.log("attn/weightvar",torch.mean(torch.var(weights,axis = 0))) ## add logging for weights. 
        self.log("attn/weight0",weights[0,0,0]) ## add logging for weights. 
        self.log("attn/weight1",weights[0,0,1]) ## add logging for weights. 
        self.log("attn/weight2",weights[0,0,2]) ## add logging for weights. 
        self.log("attn/weight3",weights[0,0,3]) ## add logging for weights. 

        ## branch 2: calculate probabilities before taking expectation, then return log probabilities. 
        softmax_probs = torch.nn.Softmax(dim=2)
        probs = softmax_probs(logittensor)
        weighted_outs = torch.matmul(weights,probs) ## shape [batch,queries,predictions]
        chosen = weighted_outs[:,0,:]
        acc = self.accuracy(chosen,labels)
        return torch.log(chosen), acc*100

    def training_step(self, batch, batch_nb):
        """When we train, we want to train independently. 
        """
        
        images, labels = batch
        softmax = torch.nn.Softmax(dim = 2)

        losses = []
        accs = []
        logits = []
        for m in self.models: ## take these logits, and build up another set of outputs on them. 
            predictions = m(images) ## these are just the pre-softmax outputs. 
            logits.append(predictions)
        logittensor = torch.stack(logits,axis =1) ## shape [batch,models,predictions]    
        #logittensor = self.posenc(logittensor) ## shape [batch,models,predictions]    

        ## Split into two branches: 1 to calculate weights, and another to get individual losses. 

        ## Branch 1: weights. 
        if self.current_epoch > self.hparams.start_attention: ## 
            weights = self.attnlayer(logittensor,logittensor) ## gives attention weights with shape [batch,queries, models]
            loss_weights = weights[:,0,:] ## batch, models
        else:    
            select = torch.randint(0,len(self.models),(images.shape[0],),device = self.device)
            loss_weights = torch.zeros((images.shape[0],len(self.models)),device = self.device)
            loss_weights[range(images.shape[0]),select] = 1


        ## Branch 2: losses: 
        logprobssoftmax = torch.nn.LogSoftmax(dim=2)
        logprobs = logprobssoftmax(logittensor)

        losses = torch.stack([self.criterion(logprobs[:,i,:], labels) for i in range(len(self.models))],axis = 1) ## this is a list of elements each of size (batch,models)
        weighted_loss = torch.mean(torch.sum(losses*loss_weights,axis = 1))
        accuracies = torch.stack([self.accuracy(logprobs[:,i,:],labels) for i in range(len(self.models))]) ## (models,)
        bulk_attn = torch.mean(loss_weights,dim = 0) # (models,)
        weighted_acc = torch.sum(accuracies*bulk_attn)

        ## weighted average: 

        self.log("loss/train", weighted_loss)
        self.log("acc/train", weighted_acc*100) ## this is an approximation of the true accuracy. 
        return weighted_loss

    def validation_step(self, batch, batch_nb):
        images, labels = batch
        predictions,weights = self.forward(batch)
        loss = torch.mean(self.criterion(predictions, labels))
        accuracy = self.accuracy(predictions,labels)
        self.log("loss/val", loss)
        self.log("acc/val", accuracy*100)

    def test_step(self, batch, batch_nb):
        images, labels = batch
        predictions,weights = self.forward(batch)
        loss = torch.mean(self.criterion(predictions, labels))
        accuracy = self.accuracy(predictions,labels)
        self.log("loss/test", loss)
        self.log("acc/test", accuracy*100)

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

class CIFAR10AttentionEnsembleMLPSkipModule(CIFAR10AttentionEnsembleModule):
    """There's actually no skip here- only mlp 

    """
    def __init__(self,hparams):
        super().__init__(hparams)
        pre_fc_dim = self.models[0].fc.weight.shape[1]
        self.posenc = PosEncodingsSin(pre_fc_dim,0.1,10)
        self.attnlayer = self.get_attnlayer(pre_fc_dim,hparams.embedding_dim) ## project from 10 dimensional output (CIFAR10 logits) to embedding dimension.
        self.model = torch.nn.ModuleList([self.models,self.attnlayer])

    def forward(self,batch):
        """Move this all before the MLP layer; worth checking later.  

        """
        images, labels = batch
        softmax = torch.nn.Softmax(dim = 2)

        pre_logits = []

        for m in self.models: ## take these logits, and build up another set of outputs on them. 
            pre_logit = m.before_fc(images) ## these are just the pre-softmax outputs. 
            pre_logits.append(pre_logit)
        prelogittensor = torch.stack(pre_logits,axis =1)

        ## branch 1: generate weights
        #prelogittensor = self.posenc(prelogittensor) ## shape [batch,models,predictions]    
        weights = self.attnlayer(prelogittensor,prelogittensor) ## gives attention weights with shape [batch,queries, models]
        self.log("attn/weightvar",torch.mean(torch.var(weights,axis = 0))) ## add logging for weights. 
        self.log("attn/weight0",weights[0,0,0]) ## add logging for weights. 
        self.log("attn/weight1",weights[0,0,1]) ## add logging for weights. 
        self.log("attn/weight2",weights[0,0,2]) ## add logging for weights. 
        self.log("attn/weight3",weights[0,0,3]) ## add logging for weights. 

        ## branch 2: generate logits
        logittensor = torch.stack([submodel.fc(prelogittensor[:,i,:]) for i,submodel in enumerate(self.models)],axis = 1)
        softmax_probs = torch.nn.Softmax(dim=2)
        probs = softmax_probs(logittensor)

        ## now weight the outputs. 
        weighted_outs = torch.matmul(weights,probs) ## shape [batch,queries,predictions]
        chosen = weighted_outs[:,0,:]
        acc = self.accuracy(chosen,labels)
        return torch.log(chosen), acc*100

    def training_step(self, batch, batch_nb):
        """When we train, we want to train independently. 
        """
        
        images, labels = batch
        softmax = torch.nn.Softmax(dim = 2)

        losses = []
        accs = []
        pre_logits = []
        for m in self.models: ## take these logits, and build up another set of outputs on them. 
            pre_logit = m.before_fc(images) ## these are just the pre-softmax outputs. 
            pre_logits.append(pre_logit)
        prelogittensor = torch.stack(pre_logits,axis =1)

        ## branch 1: generate weights
        #prelogittensor = self.posenc(prelogittensor) ## shape [batch,models,predictions]    
        if self.current_epoch > self.hparams.start_attention: ##
            weights = self.attnlayer(prelogittensor,prelogittensor) ## gives attention weights with shape [batch,queries, models]
            loss_weights = weights[:,0,:] ## batch, models
        else:
            select = torch.randint(0,len(self.models),(images.shape[0],),device = self.device)
            loss_weights = torch.zeros((images.shape[0],len(self.models)),device = self.device)
            loss_weights[range(images.shape[0]),select] = 1



        ## Branch 2: losses: 
        logittensor = torch.stack([submodel.fc(prelogittensor[:,i,:]) for i,submodel in enumerate(self.models)],axis = 1)
        logprobssoftmax = torch.nn.LogSoftmax(dim=2)
        logprobs = logprobssoftmax(logittensor)

        losses = torch.stack([self.criterion(logprobs[:,i,:], labels) for i in range(len(self.models))],axis = 1) ## this is a list of elements each of size (batch,models)
        weighted_loss = torch.mean(torch.sum(losses*loss_weights,axis = 1))
        accuracies = torch.stack([self.accuracy(logprobs[:,i,:],labels) for i in range(len(self.models))]) ## (models,)
        bulk_attn = torch.mean(loss_weights,dim = 0) # (models,)
        weighted_acc = torch.sum(accuracies*bulk_attn)

        ## weighted average: 

        self.log("loss/train", weighted_loss)
        self.log("acc/train", weighted_acc*100) ## this is an approximation of the true accuracy. 
        return weighted_loss

class CIFAR10AttentionEnsembleSkipModule(CIFAR10AttentionEnsembleModule):
    def forward(self,batch):
        """Use skip connections as well

        """
        images, labels = batch
        softmax = torch.nn.Softmax(dim = 2)

        losses = []
        accs = []
        logits = []
        for m in self.models: ## take these logits, and build up another set of outputs on them. 
            predictions = m(images) ## these are just the pre-softmax outputs. 
            logits.append(predictions)
        logittensor = self.posenc(torch.stack(logits,axis =1)) ## shape [batch,models,predictions]    
        weights = self.attnlayer(logittensor,logittensor) ## gives attention weights with shape [batch,queries, models]
        self.log("attn/weightvar",torch.mean(torch.var(weights,axis = 0))) ## add logging for weights. 
        self.log("attn/weight0",weights[0,0,0]) ## add logging for weights. 
        self.log("attn/weight1",weights[0,0,1]) ## add logging for weights. 
        self.log("attn/weight2",weights[0,0,2]) ## add logging for weights. 
        self.log("attn/weight3",weights[0,0,3]) ## add logging for weights. 
        weighted_outs = logittensor + torch.matmul(weights,logittensor) ## shape [batch,queries,predictions]
        chosen = weighted_outs[:,0,:]
        acc = self.accuracy(chosen,labels)
        return weighted_outs[:,0,:], acc*100

class CIFAR10InterEnsembleModule(CIFAR10_Models):
    """Customized module to train a convex combination of a wide model and smaller models. 

    """
    def __init__(self,lamb,hparams):

        super().__init__(hparams)
        self.lamb = lamb
        self.hparams = hparams

        self.criterion = torch.nn.CrossEntropyLoss()
        self.accuracy = Accuracy()

        #self.interpmodel = # define this##  
        self.model = all_classifiers[self.hparams.classifier]()
        self.basemodel = wideresnet18()
        self.submodels = torch.nn.ModuleList([widesubresnet18(self.basemodel,i) for i in range(4)])
     
    def forward(self,batch): 
        """First pass forward: try treating like a standard ensemble? 

        """
        images, labels = batch
        softmax = torch.nn.Softmax(dim = 1)

        losses = []
        accs = []
        softmaxes = []
        for m in self.submodels:
            predictions = m(images)
            normed = softmax(predictions)
            softmaxes.append(normed)
        ## todo: should we add the main model predictions too?

        gmean = torch.exp(torch.mean(torch.log(torch.stack(softmaxes)),dim = 0)) ## implementation from https://stackoverflow.com/questions/59722983/how-to-calculate-geometric-mean-in-a-differentiable-way   
        bigpred = self.basemodel(images)
        bignormed = softmax(bigpred)
        grand_mean = torch.sum(torch.stack([self.lamb*bignormed,(1-self.lamb)*gmean]),dim = 0)
        ## we can pass this  through directly to the accuracy function. 
        tloss = self.criterion(grand_mean,labels)## beware: this is a transformed input, don't evaluate on test loss of ensembles. 
        accuracy = self.accuracy(grand_mean,labels)
        return tloss,accuracy*100

    def calibration(self,batch):
        """Like forward, but just exit with the predictions and labels. . 
        """
        images, labels = batch
        softmax = torch.nn.Softmax(dim = 1)

        losses = []
        accs = []
        softmaxes = []
        for m in self.submodels:
            predictions = m(images)
            normed = softmax(predictions)
            softmaxes.append(normed)
        ## todo: should we add the main model predictions too?

        gmean = torch.exp(torch.mean(torch.log(torch.stack(softmaxes)),dim = 0)) ## implementation from https://stackoverflow.com/questions/59722983/how-to-calculate-geometric-mean-in-a-differentiable-way   
        bigpred = self.basemodel(images)
        bignormed = softmax(bigpred)
        grand_mean = torch.sum(torch.stack([self.lamb*bignormed,(1-self.lamb)*gmean]),dim = 0)
        return grand_mean,labels

    def training_step(self,batch,batch_nb):
        """This training_step function takes a convex combination of the original model and subnet models. 

        """
        images,labels = batch
        losses = []
        accs = []

        main_preds = self.basemodel(images)
        main_loss = self.criterion(main_preds,labels)
        losses.append(self.lamb*main_loss)
        accs.append(self.lamb*self.accuracy(main_preds,labels))
        nb_subnets = self.submodels[0].nb_subnets

        for m in self.submodels:
            subnet_preds = m(images)
            subnet_loss = self.criterion(subnet_preds,labels)
            losses.append((1-self.lamb)*(1/nb_subnets)*subnet_loss)
            accs.append((1-self.lamb)*(1/nb_subnets)*self.accuracy(subnet_preds,labels))
        loss = sum(losses)    
        avg_accuracy = sum(accs) 
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            list(self.submodels.parameters())+list(self.basemodel.parameters()),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
            momentum=0.9,
            nesterov=True,
        )
        total_steps = self.hparams.max_epochs * len(self.train_dataloader())
        scheduler = self.setup_scheduler(optimizer,total_steps)
        return [optimizer], [scheduler]

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


def mse_loss_classification(predictions, labels_onehot):
    """
    MSE loss for classification as (1) in https://arxiv.org/pdf/2006.07322.pdf.
    :return: loss, float
    """
    loss = torch.pow(predictions - labels_onehot, 2).mean(1).mean(0)
    return loss


class MSELoss_classification(torch.nn.Module):
    """ Class to compute MSE loss for classification

    """
    def __init__(self, num_classes):
        self.num_classes = num_classes
        super(MSELoss_classification, self).__init__()

    def forward(self, predictions, labels):
        labels_onehot = torch.nn.functional.one_hot(labels, self.num_classes).to(torch.float32)
        loss = mse_loss_classification(predictions, labels_onehot)
        return loss
