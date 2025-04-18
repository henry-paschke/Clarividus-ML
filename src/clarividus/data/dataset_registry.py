from ..utils.enums import DatasetName
from ..utils.config import Config
from .dataset import Dataset

DATASET_REGISTRY = {}


def register_dataset(name: DatasetName):
    """
    Decorator to register a dataset function in the dataset registry.

    :param name: Name of the dataset to register.
    :return: Decorator function.
    """

    def decorator(obj: callable):
        DATASET_REGISTRY[name] = obj
        return obj

    return decorator


def get_dataset_class(name: DatasetName) -> Dataset:
    """
    Get a dataset function from the dataset registry.

    :param name: Name of the dataset to retrieve.
    :return: Dataset function.
    """
    if name not in DATASET_REGISTRY:
        raise ValueError(
            f"Dataset {name} is not registered. Available datasets: {list(DATASET_REGISTRY.keys())}"
        )
    return DATASET_REGISTRY[name]


def get_dataset(config: Config) -> Dataset:
    """
    Initialize a dataset with the given name and data split.

    :param name: Name of the dataset to initialize.
    :param data_split: List of proportions for train, validation, and test splits.
    :return: Initialized dataset.
    """
    dataset_class = get_dataset_class(config.dataset.name)
    return dataset_class(config)
