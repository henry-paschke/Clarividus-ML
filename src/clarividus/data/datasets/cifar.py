import torchvision, torch
from ...utils import Config
from ..dataset import Dataset, DatasetName
from ..transforms import get_trainset_transforms, get_testset_transforms
from ..dataset_registry import register_dataset


@register_dataset(DatasetName.CIFAR10)
class CIFAR10(Dataset):
    def __init__(self, config: Config):
        super().__init__(DatasetName.CIFAR10, config)

        trainset = torchvision.datasets.CIFAR10(
            root="./data",
            train=True,
            download=True,
            transform=get_trainset_transforms(config),
        )

        testset = torchvision.datasets.CIFAR10(
            root="./data",
            train=False,
            download=True,
            transform=get_testset_transforms(),
        )

        test_size = int(config.dataset.data_split[1] * len(testset))
        val_size = len(testset) - test_size
        valset, testset = torch.utils.data.random_split(
            testset, [val_size, test_size], generator=torch.Generator().manual_seed(42)
        )

        self.initialize_loaders(trainset, testset, valset)
