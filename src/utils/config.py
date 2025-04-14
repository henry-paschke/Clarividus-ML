import warnings
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


class Config(BaseModel):
    augmentation: AugmentationConfig


def load_config(config_path: str) -> Config:
    """
    Load configuration from a YAML file.

    :param config_path: Path to the YAML configuration file.
    :return: Config object containing the loaded configuration.
    """
    with open(config_path, "r") as file:
        config_dict = yaml.safe_load(file)

    return Config(**config_dict)
