import torchvision.transforms as transforms
from utils import Config


def get_base_transforms():
    """
    Returns the base transforms for the dataset to be run during loading, but after augmentation

    :return: A list of transforms to be applied to the dataset
    """
    return [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]


def get_augmentation_transforms(config: Config):
    """
    Returns the augmentation transforms for the dataset to be run during loading

    :return: A list of transforms to be applied to the dataset
    """
    augmentations = []

    if config.augmentation.flip:
        augmentations.append(transforms.RandomHorizontalFlip())
        augmentations.append(transforms.RandomVerticalFlip())
    if config.augmentation.padding > 0:
        augmentations.append(
            transforms.RandomCrop(32, padding=config.augmentation.padding)
        )
    if (
        config.augmentation.brightness > 0
        and config.augmentation.brightness > 0
        and config.augmentation.saturation > 0
        and config.augmentation.hue > 0
    ):
        augmentations.append(
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2
            )
        )

    return augmentations


def get_trainset_transforms():
    """
    Returns the train transforms for the dataset to be run during loading

    :return: A list of transforms to be applied to the dataset
    """
    return transforms.Compose(get_augmentation_transforms() + get_base_transforms())


def get_testset_transforms():
    """
    Returns the test transforms for the dataset to be run during loading

    :return: A list of transforms to be applied to the dataset
    """
    return transforms.Compose(get_base_transforms())
