
import torch
import torch.nn as nn
import torch.functional as F
import pytorch_lightning as pl
from torchmetrics.functional import accuracy
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import wandb
from pytorch_lightning.loggers.wandb import WandbLogger
import pickle
from generic_models import BinaryClassifierHead, ClassifierHead, MLP
from data_loader import Loader

def default_config():
    config = {
        "epochs": 100,
        "learning_rate": 0.01,
        "batch_size": 64,
        "optimizer": "adam",
        "nworkers": 7,
        "pin_memory": True,
        "emb_dim": 2,
        "dropout": 0.5,
        "criterion": "CrossEntropyLoss",
        "name": "default",
        "initialize": False,
        "devices": 1,
        "gradient_clip": 1,
        "nclasses": 2,
    }
    return config

class Model(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.save_hyperparameters(dict(config))
        self.model = self.get_model()

        if self.config["initialize"] == True:
            self.initialize_model()

        self.learning_rate = self.config.get("learning_rate")
        self.criterion = self.get_criterion()

    def get_model(self):
        """
        Get the model to train.

        Returns:
            Any: torch model
        """

        # TODO
        model = MLP(
            ((500,500), (500,500)),
            act="gelu",
            dropout=self.config.get("dropout"),
        )

        # if self.config.get("dropout") > 0.0:
        #     model.add_module("dropout", nn.Dropout(self.config.get("dropout")))

        model.add_module("act", nn.GELU())
        emb_dim = self.config.get("emb_dim")

        if self.config.get("criterion") == "CrossEntropyLoss":
            classifier = ClassifierHead(emb_dim, self.config.get("nclasses"))
            model.add_module("classifier_head", classifier)
        else:
            classifier = BinaryClassifierHead(emb_dim)
            model.add_module("binary_classifier_head", classifier)
        model.layers.append(classifier)

        return model

    def get_criterion(self):
        """
        Get Loss function. Supports:
            "BCELogits": Binary Cross Entropy with Logits
            "CrossEntropyLoss": Cross Entropy
            "MSE": Mean Squared Error

        Returns:
            any: the appropriate torch.nn loss class
        """
        criterion = self.config.get("criterion")
        if criterion == "BCELogits":
            return nn.BCEWithLogitsLoss()
        elif criterion == "CrossEntropyLoss":
            return nn.CrossEntropyLoss()
        elif criterion == "MSE":
            return nn.MSELoss()

    def _recursive_initialize(self, mod):
        """
        Recursively initialize modules
        uses kaiming uniform for convolutional nets
        and xaviar uniform for sequential/linear layers

        Args:
            mod (any): torch.nn module to initialize
        """
        if isinstance(mod, nn.Conv2d):
            nn.init.kaiming_uniform_(mod.weight.data, nonlinearity="relu")
        elif isinstance(mod, nn.Linear):
            nn.init.xavier_uniform_(mod.weight.data)
            nn.init.constant_(mod, 0.01)
        elif isinstance(mod, nn.Sequential):
            for submod in [x for x in list(mod.children())]:
                self._initialize(submod)
        else:
            try:
                _ = [self._initialize(x) for x in list(mod.children())]
            except Exception as e:
                pass

    def initialize_model(self):
        """
        recursively initialize modules
        """
        if isinstance(self.model, nn.Sequential):
            self._recursive_initialize(self.model)
        else:
            for mod in list(list(self.model.children())[0].children()):
                try:
                    temp = [x for x in mod]
                    for submod in temp:
                        self._recursive_initialize(submod)
                except Exception as e:
                    self.log({"error": e})

    def forward(self, x):
        out = self.model(x)
        return out

    def eval_batch(self, batch, batch_idx):
        x, y = batch
        y_pred = self.model(x)

        if self.config.get("criterion") == "CrossEntropyLoss":
            loss = self.criterion(y_pred, y.long())
            y_pred = torch.argmax(F.softmax(y_pred, dim=1), dim=1)
            acc = accuracy(y_pred, y, "multiclass")

        elif self.config.get("criterion") == "BCELogits":
            loss = self.criterion(y_pred, y.unsqueeze(1).float())
            y_pred = torch.where(torch.sigmoid(y_pred) > 0.5, 1, 0)
            acc = accuracy(y_pred.flaten(), y, "binary")
        elif self.config.get("criterion") == "MSE":
            loss = self.criterion(y_pred.flatten(), y)
            acc = accuracy(y_pred.flatten(), y, "binary")
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self.eval_batch(batch, batch_idx)
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self.eval_batch(batch, batch_idx)
        self.log("validation_loss", loss)
        self.log("validation_acc", acc)
        return loss

    def test_step(self, batch, batch_idx):
        loss, acc = self.eval_batch(batch, batch_idx)
        self.log("test_loss", loss)
        self.log("test_acc", acc)
        return loss

    def configure_optimizers(self):
        """
        Configure optimizers.
        Supported:
            adadelta
            adagrad
            rmsprop
            SGD
            adam [default]

        Returns:
            torch.optim class: appropriate torch optimizer
        """
        learning_rate = self.config.get("learning_rate")
        opt = self.config.get("optimizer")
        if opt == "adadelta":
            return torch.optim.Adadelta(self.parameters(), learning_rate)
        elif opt == "adagrad":
            return torch.optim.Adagrad(self.parameters(), learning_rate)
        elif opt == "sgd":
            return torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        elif opt == "rmsprop":
            return torch.optim.RMSprop(self.parameters(), learning_rate)
        else:
            return torch.optim.Adam(self.parameters(), learning_rate)


def train():
    config = default_config()
    model = Model(config)
    data = Loader(config)
    # wandb.init(config=config)
    # wandb_logger = WandbLogger(project="analysis", log_model="all")
    # config = wandb.config
    # wandb_logger.watch(model, log="all")

    trainer_callbacks = [EarlyStopping(
        "validation_loss", min_delta=0.001, mode="min")]

    trainer = pl.Trainer(
        max_epochs=config.get("epochs"),
        accelerator="auto",
        devices=config.get("devices") if torch.cuda.is_available() else None,
        gradient_clip_val=config.get("gradient_clip"),
        callbacks=trainer_callbacks,
        # logger=wandb_logger,
    )

    trainer.fit(model, data)

    trainer.save_checkpoint("training.ckpt")
    torch.save(model.model.state_dict, "model.pt")
    # with open("config.pickle", "wb") as f:
    #     # config['model_name'] = model.model._get_name 
    #     pickle.dump(config, f, protocol=pickle.HIGHEST_PROTOCOL)
    return trainer, model, data
