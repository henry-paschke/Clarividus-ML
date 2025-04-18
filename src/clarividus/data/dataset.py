from enum import Enum
import torch
import torch.utils.data.dataloader
from ..utils import Config, DatasetName


class Dataset:
    def __init__(self, name: DatasetName, config: Config):
        self.name = name
        self.data_split = config.dataset.data_split
        self.config = config
        self.trainloader = None
        self.testloader = None
        self.valloader = None

    def initialize_loaders(self, trainset, testset, valset):
        self.trainloader = torch.utils.data.DataLoader(
            trainset,
            batch_size=self.config.dataset.batch_size,
            shuffle=True,
            num_workers=self.config.dataset.num_workers,
        )

        self.testloader = torch.utils.data.DataLoader(
            testset,
            batch_size=self.config.dataset.batch_size,
            shuffle=False,
            num_workers=self.config.dataset.num_workers,
        )

        self.valloader = torch.utils.data.DataLoader(
            valset,
            batch_size=self.config.dataset.batch_size,
            shuffle=False,
            num_workers=self.config.dataset.num_workers,
        )

    def get_trainloader(self) -> torch.utils.data.dataloader.DataLoader:
        """
        Get the trainloader.

        :return: Train loader.
        :raises ValueError: If the train loader is not initialized.
        """
        if self.trainloader is None:
            raise ValueError("Trainloader is not initialized.")
        return self.trainloader

    def get_testloader(self) -> torch.utils.data.dataloader.DataLoader:
        """
        Get the test loader.

        :return: Test loader.
        :raises ValueError: If the test loader is not initialized.
        """
        if self.testloader is None:
            raise ValueError("Testloader is not initialized.")
        return self.testloader

    def get_valloader(self) -> torch.utils.data.dataloader.DataLoader:
        """
        Get the validation loader.

        :return: Validation loader.
        :raises ValueError: If the validation loader is not initialized.
        """
        if self.valloader is None:
            raise ValueError("Valloader is not initialized.")
        return self.valloader
