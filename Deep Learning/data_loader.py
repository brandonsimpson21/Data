
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from typing import Optional

class Data(Dataset):

    def __init__(self, data_path):
        pass

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass


class Loader(pl.LightningDataModule):
    def __init__(self, config, data_path="data/"):
        super().__init__()
        self.config = config
        self.save_hyperparameters(dict(config))
        self.nworkers = config.get("nworkers")
        self.batch_size = config.get("batch_size")

        #TODO
        self.train_root = data_path + "train/"
        self.val_root = data_path + "validate/"
        self.test_root = data_path + "test/"

    def setup(self, stage: Optional[str] = None):
        """
        sets up the Datasets used for training, validation, and testing.

        Args:
            stage (Optional[str], optional): currently unused but allows isolating logic between stages Defaults to None.
        """

        self.train_data = Data(self.train_root + "data.csv")  # TODO

        self.val_data = Data(self.val_root + "data.csv")  # TODO

        self.test_data = Data(self.test_root + "data.csv")  # TODO

    def train_dataloader(self):
        data = DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            num_workers=self.nworkers,
            shuffle=True,
        )
        return data

    def val_dataloader(self):
        data = DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            num_workers=self.nworkers,
            shuffle=False,
        )
        return data

    def test_dataloader(self):
        data = DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            num_workers=self.nworkers,
            shuffle=False,
        )
        return data

    def predict_dataloader(self):
        return
