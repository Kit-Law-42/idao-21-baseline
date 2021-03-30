import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
import torchvision.models as models
import torch.optim

class Print(nn.Module):
    """Debugging only"""

    def forward(self, x):
        print(x.size())
        return x


class Clamp(nn.Module):
    """Clamp energy output"""

    def forward(self, x):
        x = torch.clamp(x, min=0, max=30)
        return x


class SimpleConv(pl.LightningModule): # pl.LightningModule replaces nn.Module in pyTorch
    #def __init__(self, mode: ["classification", "regression"] = "classification"):
    def __init__(self, mode = "classification"):
        super().__init__()
        self.mode = mode
        # image size = 120*120, cropped from 576*576.
        
        self.layer1 = nn.Sequential(
                    nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2), # 120-5+2*2/1 +1 = 120 => (Input-kernel + 2*padding) /Stride +1
                    nn.BatchNorm2d(16),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=19, stride=7), # 120-2*0 -1*(19-1)-1 / 7 +1= 14.4 +1= 15(rounddown)
                    nn.Flatten(),
                )

        self.drop_out = nn.Dropout()

        self.fc1 = nn.Linear(3600, 500) # 15*15 *16filter.
        self.fc2 = nn.Linear(500, 2)  # for classification
        self.fc3 = nn.Linear(500, 1)  # for regression


        self.stem = nn.Sequential(
            self.layer1, self.drop_out, self.fc1,
            )
        if self.mode == "classification":
            self.classification = nn.Sequential(self.stem, self.fc2)
        else:
            self.regression = nn.Sequential(self.stem, self.fc3)

        self.train_acc = pl.metrics.Accuracy()
        self.valid_acc = pl.metrics.Accuracy()
        self.test_acc = pl.metrics.Accuracy()

    def training_step(self, batch, batch_idx):
        # --------------------------
        #batch = sample, target, self.name_to_energy(path), self.name_to_index(path)
        x_target, class_target, reg_target, _ = batch
        if self.mode == "classification":
            class_pred = self.classification(x_target.float())
            class_loss = F.binary_cross_entropy_with_logits(
                class_pred, class_target.float()
            )
            self.train_acc(torch.sigmoid(class_pred), class_target)
            self.log("train_acc", self.train_acc, on_step=True, on_epoch=False)
            self.log("classification_loss", class_loss)

            return class_loss

        else:
            reg_pred = self.regression(x_target.float())
            #             reg_loss = F.l1_loss(reg_pred, reg_target.float().view(-1, 1))
            reg_loss = F.mse_loss(reg_pred, reg_target.float().view(-1, 1))

            #             reg_loss = torch.sum(torch.abs(reg_pred - reg_target.float().view(-1, 1)) / reg_target.float().view(-1, 1))
            self.log("regression_loss", reg_loss)
            return reg_loss

    def training_epoch_end(self, outs):
        # log epoch metric
        if self.mode == "classification":
            self.log("train_acc_epoch", self.train_acc.compute())
        else:
            pass

    def validation_step(self, batch, batch_idx):
        #batch = sample, target, self.name_to_energy(path), self.name_to_index(path) from IDAODataset.__getitem__
        x_target, class_target, reg_target, _ = batch
        if self.mode == "classification":
            class_pred = self.classification(x_target.float())
            class_loss = F.binary_cross_entropy_with_logits(
                class_pred, class_target.float()
            )
            self.valid_acc(torch.sigmoid(class_pred), class_target)
            self.log("valid_acc", self.valid_acc.compute())
            self.log("classification_loss", class_loss)
            return class_loss

        else:
            reg_pred = self.regression(x_target.float())
            #             reg_loss = F.l1_loss(reg_pred, reg_target.float().view(-1, 1))
            # .view() works like reshape. It forces reg_target to be N rows * 1 column
            reg_loss = F.mse_loss(reg_pred, reg_target.float().view(-1, 1))

            #             reg_loss = torch.sum(torch.abs(reg_pred - reg_target.float().view(-1, 1)) / reg_target.float().view(-1, 1))
            self.log("regression_loss", reg_loss)
            return reg_loss

#    def test_step(self, batch, batch_idx):
#        # --------------------------
#        x_target, class_target, _, reg_target = batch
#        if self.mode == "classification":
#            class_pred = self.classification(x_target.float())
#            class_loss = F.binary_cross_entropy_with_logits(
#                class_pred, class_target.float()
#            )
#            self.test_acc(torch.sigmoid(class_pred), class_target)
#            self.log("test_acc", self.train_acc, on_step=True, on_epoch=False)
#            self.log("classification_loss", class_loss)
#            return class_loss
#
#        else:
#            reg_pred = self.regression(x_target.float())
#            #             reg_loss = F.l1_loss(reg_pred, reg_target.float().view(-1, 1))
#            reg_loss = F.mse_loss(reg_pred, reg_target.float().view(-1, 1))
#
#            #             reg_loss = torch.sum(torch.abs(reg_pred - reg_target.float().view(-1, 1)) /reg_target.float().view(-1, 1))
#            self.log("regression_loss", reg_loss)
#            return reg_loss

    #         return exp_predicted, class_target

    # --------------------------

#    def test_epoch_end(self, test_step_outputs):
#        print(self.test_acc.compute())
#
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def forward(self, x):
        if self.mode == "classification":
            class_pred = self.classification(x.float())
            return {"class": torch.sigmoid(class_pred)}
        else:
            reg_pred = self.regression(x.float())
            return {"energy": reg_pred}

class ResNetModel(pl.LightningModule):
    def __init__(self, mode="classification"):
        super().__init__()
        
        # log hyperparameters
        self.save_hyperparameters()
        self.mode = mode
        
        # transfer learning if pretrained=True
        self.feature_extractor = models.resnet50(pretrained=True)
        # replace first layer from 3 channels to 1 channel.
        # Source from: https://stackoverflow.com/questions/51995977/how-can-i-use-a-pre-trained-neural-network-with-grayscale-images
        self.feature_extractor.conv1.weight.data = models.resnet50(pretrained=True).conv1.weight.data.sum(axis=1).reshape(64, 1, 7, 7)
        # conv1_weight = models.resnet18(pretrained=True).conv1.weight
        # print(conv1_weight.shape)
        # self.feature_extractor.conv1.weight = nn.Parameter(conv1_weight.sum(dim=1, keepdim=True))

        # layers are frozen by using eval()
        self.feature_extractor.eval()
        # freeze params
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        for param in self.feature_extractor.layer4.parameters(): # unfreeze layer 4.
            param.requires_grad = True
        #n_sizes = 1000

        #self.fc2 = nn.Linear(n_sizes, 2)
        #self.fc3 = nn.Linear(n_sizes, 1)
        if self.mode == "classification":
            self.feature_extractor.fc = nn.Sequential(nn.Dropout(0.2),nn.Linear(2048, 256), nn.Linear(256, 2))
            self.classification =self.feature_extractor
                 
        else:
            self.feature_extractor.fc = nn.Sequential(nn.Dropout(0.2),nn.Linear(2048, 256), nn.Linear(256, 1))
            self.regression =self.feature_extractor         

        self.train_acc = pl.metrics.Accuracy()
        self.valid_acc = pl.metrics.Accuracy()
        self.test_acc = pl.metrics.Accuracy()

    def training_step(self, batch, batch_idx):
        # --------------------------
        #batch = sample, target, self.name_to_energy(path), self.name_to_index(path)
        x_target, class_target, reg_target, _ = batch
        if self.mode == "classification":
            class_pred = self.classification(x_target.float())
            class_loss = F.binary_cross_entropy_with_logits(
                class_pred, class_target.float()
            )
            self.train_acc(torch.sigmoid(class_pred), class_target)
            self.log("train_acc", self.train_acc, on_step=True, on_epoch=False)
            self.log("classification_loss", class_loss)

            return class_loss

        else:
            reg_pred = self.regression(x_target.float())
            #             reg_loss = F.l1_loss(reg_pred, reg_target.float().view(-1, 1))
            # map regression level to 6 evergy level values.
            #reg_pred = reg_pred.map(lambda x : ResNetModel._regression_to_level(x))
            reg_loss = F.mse_loss(reg_pred, reg_target.float().view(-1, 1))

            #             reg_loss = torch.sum(torch.abs(reg_pred - reg_target.float().view(-1, 1)) / reg_target.float().view(-1, 1))
            self.log("regression_loss", reg_loss)
            return reg_loss
        
    def training_epoch_end(self, outs):
        # log epoch metric
        if self.mode == "classification":
            self.log("train_acc_epoch", self.train_acc.compute())
        else:
            pass

    # def _regression_to_level(reg_pred_single_val):
    #     energyInterval = [2,4.5,8,15,25]
    #     evergyLv = [1,3,6,10,20,30]
    #     for idx,i in enumerate(energyInterval):
    #             if (reg_pred_single_val < i):
    #                 reg_pred_single_val = evergyLv[idx]
    #                 return reg_pred_single_val
    #     return 30

    def validation_step(self, batch, batch_idx):
        #batch = sample, target, self.name_to_energy(path), self.name_to_index(path) from IDAODataset.__getitem__
        x_target, class_target, reg_target, _ = batch
        if self.mode == "classification":
            class_pred = self.classification(x_target.float())
            class_loss = F.binary_cross_entropy_with_logits(
                class_pred, class_target.float()
            )
            self.valid_acc(torch.sigmoid(class_pred), class_target)
            self.log("valid_acc", self.valid_acc.compute())
            self.log("classification_loss", class_loss)
            return class_loss

        else:
            reg_pred = self.regression(x_target.float())
            #             reg_loss = F.l1_loss(reg_pred, reg_target.float().view(-1, 1))
            # .view() works like reshape. It forces reg_target to be N rows * 1 column
            reg_loss = F.mse_loss(reg_pred, reg_target.float().view(-1, 1))

            #             reg_loss = torch.sum(torch.abs(reg_pred - reg_target.float().view(-1, 1)) / reg_target.float().view(-1, 1))
            self.log("regression_loss", reg_loss)
            return reg_loss
        
    
    def configure_optimizers(self):
        params_to_update = []
        if self.mode == "classification":
            for name,param in self.classification.named_parameters():
                if param.requires_grad == True:
                    params_to_update.append((name,param))
                    print("Classification parameters: \t",name)
            optimizer = torch.optim.Adam(params_to_update, lr=2e-2)
            #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.2)
            return optimizer#, scheduler
        else:
            for name,param in self.regression.named_parameters():
                if param.requires_grad == True:
                    params_to_update.append((name,param))
                    print("Regression parameters: \t",name)
            optimizer = torch.optim.Adam(params_to_update, lr=2e-2)
            #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.2)
            return optimizer#, scheduler

    # will be used during inference
    def forward(self, x):
      if self.mode == "classification":
            class_pred = self.classification(x.float())
            return {"class": torch.sigmoid(class_pred)}
      else:
            reg_pred = self.regression(x.float())
            return {"energy": reg_pred}
       
      return x