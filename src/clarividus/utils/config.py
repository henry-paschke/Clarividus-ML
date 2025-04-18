import warnings
from .enums import DatasetName
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
import yaml


class AugmentationConfig(BaseModel):
    """Configuration for augmentations."""

    flip: bool = Field(
        default=False, description="Whether to apply random horizontal flip."
    )
    padding: int = Field(
        default=0.0,
        description="How much padding to apply to random crops. 0.0 means no cropping.",
    )
    brightness: float = Field(
        default=0.0, description="Brightness factor for color jitter."
    )
    contrast: float = Field(
        default=0.0, description="Contrast factor for color jitter."
    )
    saturation: float = Field(
        default=0.0, description="Saturation factor for color jitter."
    )
    hue: float = Field(default=0.0, description="Hue factor for color jitter.")


class DatasetConfig(BaseModel):
    """Configuration for datasets."""

    name: DatasetName = Field(
        default=DatasetName.CIFAR10,
        description="Name of the dataset. Supported datasets: CIFAR10, CIFAR100.",
    )
    data_split: list[float] = Field(
        default=[0.8, 0.1, 0.1],
        description="Proportions for train, validation, and test splits.",
    )
    batch_size: int = Field(default=32, description="Batch size for data loading.")
    num_workers: int = Field(
        default=4, description="Number of workers for data loading."
    )


class Config(BaseModel):
    """Main configuration class."""

    dataset: DatasetConfig
    augmentation: AugmentationConfig


def load_config(config_path: str) -> Config:
    """
    Load configuration from a YAML file.

    :param config_path: Path to the YAML configuration file.
    :return: Config object containing the loaded configuration.
    """
    if not config_path.endswith(".yaml") and not config_path.endswith(".yml"):
        warnings.warn(
            f"Config file {config_path} does not have a .yaml extension. "
            "This may lead to unexpected behavior."
        )

    with open(config_path, "r") as file:
        config_dict = yaml.safe_load(file)

    print(config_dict)
    if config_dict is None:
        raise ValueError(f"Config file {config_path} is empty or invalid.")

    return Config(**config_dict)
