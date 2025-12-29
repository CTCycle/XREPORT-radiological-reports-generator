from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset

from XREPORT.server.utils.services.training.processing import TokenizerHandler


###############################################################################
class DataLoaderProcessor:
    def __init__(self, configuration: dict[str, Any]) -> None:
        self.img_shape = (224, 224)
        self.num_channels = 3
        self.image_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.image_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        self.augmentation = configuration.get("use_img_augmentation", False)
        self.batch_size = configuration.get("batch_size", 32)
        self.inference_batch_size = configuration.get("inference_batch_size", 32)
        self.color_encoding = (
            cv2.COLOR_BGR2RGB if self.num_channels == 3 else cv2.COLOR_BGR2GRAY
        )

        handler = TokenizerHandler(configuration)
        self.pad_token = handler.pad_token
        self.configuration = configuration

    # -------------------------------------------------------------------------
    def load_image(self, path: str, as_array: bool = False) -> np.ndarray:
        image = cv2.imread(path)
        image = cv2.cvtColor(image, self.color_encoding)
        image = cv2.resize(image, self.img_shape)
        if as_array:
            image = np.asarray(image, dtype=np.float32)
        else:
            image = image.astype(np.float32)

        return image

    # -------------------------------------------------------------------------
    def load_data_for_training(
        self, path: str, text: list[int] | np.ndarray
    ) -> tuple[tuple[np.ndarray, np.ndarray], np.ndarray]:
        rgb_image = self.load_image(path)
        rgb_image = (
            self.image_augmentation(rgb_image) if self.augmentation else rgb_image
        )
        rgb_image = self.image_normalization(rgb_image)

        token_array = np.asarray(text, dtype=np.int32)
        input_text = token_array[:-1]
        output_text = token_array[1:]

        return (rgb_image, input_text), output_text

    # -------------------------------------------------------------------------
    def load_data_for_inference(self, path: str) -> np.ndarray:
        rgb_image = self.load_image(path)
        rgb_image = self.image_normalization(rgb_image)

        return rgb_image

    # -------------------------------------------------------------------------
    def image_normalization(self, image: np.ndarray) -> np.ndarray:
        normalized_image = image / 255.0
        normalized_image = (normalized_image - self.image_mean) / self.image_std

        return normalized_image

    # -------------------------------------------------------------------------
    def image_augmentation(self, image: np.ndarray) -> np.ndarray:
        if np.random.rand() <= 0.5:
            image = np.fliplr(image)

        if np.random.rand() <= 0.5:
            image = np.flipud(image)

        if np.random.rand() <= 0.25:
            brightness_delta = np.random.uniform(-0.2, 0.2)
            image = np.clip(image + brightness_delta, 0.0, 255.0)

        if np.random.rand() <= 0.35:
            contrast_factor = np.random.uniform(0.7, 1.3)
            mean = np.mean(image, axis=(0, 1), keepdims=True)
            image = np.clip((image - mean) * contrast_factor + mean, 0.0, 255.0)

        return image


###############################################################################
class XRAYDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        processor: DataLoaderProcessor,
        training: bool,
    ) -> None:
        self.paths = data["path"].to_list()
        self.tokens = data["tokens"].to_list()
        self.processor = processor
        self.training = training

    # -------------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.paths)

    # -------------------------------------------------------------------------
    def __getitem__(
        self, index: int
    ) -> tuple[tuple[np.ndarray, np.ndarray], np.ndarray] | np.ndarray:
        path = self.paths[index]
        tokens = self.tokens[index]
        if self.training:
            return self.processor.load_data_for_training(path, tokens)

        return self.processor.load_data_for_inference(path)


###############################################################################
class XRAYDataLoader:
    def __init__(self, configuration: dict[str, Any], shuffle: bool = True) -> None:
        self.processor = DataLoaderProcessor(configuration)
        self.batch_size = configuration.get("batch_size", 32)
        self.inference_batch_size = configuration.get("inference_batch_size", 32)
        self.num_workers = configuration.get("dataloader_workers", 0)
        self.prefetch_factor = configuration.get("prefetch_factor", 2)
        self.pin_memory = configuration.get(
            "pin_memory", configuration.get("use_device_GPU", False)
        )
        self.persistent_workers = configuration.get("persistent_workers", True)
        self.configuration = configuration
        self.shuffle = shuffle

    # -------------------------------------------------------------------------
    def build_training_dataloader(
        self, data: pd.DataFrame, batch_size: int | None = None
    ) -> DataLoader:
        batch_size = self.batch_size if batch_size is None else batch_size
        dataset = XRAYDataset(data, self.processor, training=True)
        loader_settings: dict[str, Any] = {
            "batch_size": batch_size,
            "shuffle": self.shuffle,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "drop_last": False,
        }
        if self.num_workers > 0:
            loader_settings["prefetch_factor"] = self.prefetch_factor
            loader_settings["persistent_workers"] = self.persistent_workers
        dataset = DataLoader(dataset, **loader_settings)

        return dataset

    # -------------------------------------------------------------------------
    def build_inference_dataloader(
        self,
        data: pd.DataFrame,
        batch_size: int | None = None,
        buffer_size: int | None = None,
    ) -> DataLoader:
        batch_size = self.inference_batch_size if batch_size is None else batch_size
        num_workers = self.num_workers if buffer_size is None else buffer_size
        dataset = XRAYDataset(data, self.processor, training=False)
        loader_settings: dict[str, Any] = {
            "batch_size": batch_size,
            "shuffle": False,
            "num_workers": num_workers,
            "pin_memory": self.pin_memory,
            "drop_last": False,
        }
        if num_workers > 0:
            loader_settings["prefetch_factor"] = self.prefetch_factor
            loader_settings["persistent_workers"] = self.persistent_workers
        dataset = DataLoader(dataset, **loader_settings)

        return dataset
