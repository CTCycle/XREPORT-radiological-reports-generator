from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from XREPORT.server.utils.learning.processing import TokenizerHandler


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
        self.color_encoding = cv2.COLOR_BGR2RGB
        self.rng = np.random.default_rng(seed=42)

        handler = TokenizerHandler(configuration)
        self.pad_token = handler.pad_token
        self.configuration = configuration

    # -------------------------------------------------------------------------
    def load_image(self, path: str) -> np.ndarray:
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, self.color_encoding)
        image = cv2.resize(image, self.img_shape, interpolation=cv2.INTER_AREA)
        # Always return float32 for processing
        return image.astype(np.float32, copy=False)

    # -------------------------------------------------------------------------
    def load_data_for_training(
        self, path: str, text: list[int] | np.ndarray
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        rgb_image = self.load_image(path)
        rgb_image = (
            self.image_augmentation(rgb_image) if self.augmentation else rgb_image
        )
        rgb_image = self.image_normalization(rgb_image)
        rgb_image = np.ascontiguousarray(rgb_image)

        token_array = np.asarray(text, dtype=np.int64)
        input_text = token_array[:-1]
        output_text = token_array[1:]

        return (
            torch.from_numpy(rgb_image),
            torch.from_numpy(input_text),
        ), torch.from_numpy(output_text)

    # -------------------------------------------------------------------------
    def load_data_for_inference(self, path: str) -> torch.Tensor:
        rgb_image = self.load_image(path)
        rgb_image = self.image_normalization(rgb_image)
        rgb_image = np.ascontiguousarray(rgb_image)
        return torch.from_numpy(rgb_image)

    # -------------------------------------------------------------------------
    def image_normalization(self, image: np.ndarray) -> np.ndarray:
        # In-place normalization
        image *= 1.0 / 255.0
        image -= self.image_mean
        image /= self.image_std
        return image

    # -------------------------------------------------------------------------
    def image_augmentation(self, image: np.ndarray) -> np.ndarray:
        if self.rng.random() <= 0.5:
            image = np.fliplr(image)

        if self.rng.random() <= 0.5:
            image = np.flipud(image)

        if self.rng.random() <= 0.25:
            brightness_delta = self.rng.uniform(-0.2, 0.2)
            image = np.clip(image + brightness_delta, 0.0, 255.0)

        if self.rng.random() <= 0.35:
            contrast_factor = self.rng.uniform(0.7, 1.3)
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
        self.paths = data["path"].to_numpy(dtype="object", copy=False)
        self.tokens = data["tokens"].to_numpy(dtype="object", copy=False)
        self.processor = processor
        self.training = training

    # -------------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.paths)

    # -------------------------------------------------------------------------
    def __getitem__(
        self, index: int
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor] | torch.Tensor:
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
        self.prefetch_factor = configuration.get("prefetch_factor", 1)
        self.pin_memory = configuration.get("pin_memory", False)
        self.persistent_workers = configuration.get("persistent_workers", False)
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
        
        # Uses default_collate automatically
        return DataLoader(dataset, **loader_settings)

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
            
        return DataLoader(dataset, **loader_settings)
