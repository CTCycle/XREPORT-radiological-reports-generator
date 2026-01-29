from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from XREPORT.server.utils.learning.processing import TokenizerHandler


###############################################################################
class XRAYDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        training: bool,
        image_transform: transforms.Compose | None = None,
        target_transform: Any = None,
    ) -> None:
        self.paths = data["path"].to_numpy(dtype="object", copy=False)
        self.tokens = data["tokens"].to_numpy(dtype="object", copy=False)
        self.training = training
        self.transform = image_transform
        self.target_transform = target_transform

    # -------------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.paths)

    # -------------------------------------------------------------------------
    def __getitem__(
        self, index: int
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor] | torch.Tensor:
        path = self.paths[index]
        
        with Image.open(path) as img:
            image = img.convert("RGB")
            
            if self.transform:
                image = self.transform(image)
        
        if not self.training:
            return image # type: ignore

        token_array = np.asarray(self.tokens[index], dtype=np.int64)
        input_text = torch.from_numpy(token_array[:-1])
        output_text = torch.from_numpy(token_array[1:])

        return (image, input_text), output_text # type: ignore


###############################################################################
class XRAYDataLoader:
    def __init__(self, configuration: dict[str, Any], shuffle: bool = True) -> None:
        self.img_shape = (224, 224)
        self.image_mean = [0.485, 0.456, 0.406]
        self.image_std = [0.229, 0.224, 0.225]
        
        self.augmentation = configuration.get("use_img_augmentation", False)
        self.batch_size = configuration.get("batch_size", 32)
        self.inference_batch_size = configuration.get("inference_batch_size", 32)
        
        self.num_workers = configuration.get("dataloader_workers", 0)
        self.prefetch_factor = configuration.get("prefetch_factor", 1)
        self.pin_memory = configuration.get("pin_memory", False)
        self.persistent_workers = configuration.get("persistent_workers", False)
        
        self.shuffle = shuffle
        
        self._tokenizer_handler = TokenizerHandler(configuration)

    # -------------------------------------------------------------------------
    def _get_transforms(self, training: bool) -> transforms.Compose:
        transform_list = []

        if training and self.augmentation:
            transform_list.extend([
                transforms.Resize(self.img_shape),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomApply([
                    transforms.ColorJitter(brightness=0.2, contrast=0.2)
                ], p=0.25),
            ])
        else:
            transform_list.append(transforms.Resize(self.img_shape))

        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.image_mean, std=self.image_std),
            # Permute to (H, W, C) for Keras compatibility as model.py expects channels last
            transforms.Lambda(lambda x: x.permute(1, 2, 0)), 
        ])

        return transforms.Compose(transform_list)

    # -------------------------------------------------------------------------
    def build_training_dataloader(
        self, data: pd.DataFrame, batch_size: int | None = None
    ) -> DataLoader:
        batch_size = self.batch_size if batch_size is None else batch_size
        
        training_transforms = self._get_transforms(training=True)
        dataset = XRAYDataset(
            data, 
            training=True, 
            image_transform=training_transforms
        )

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
        
        inference_transforms = self._get_transforms(training=False)
        dataset = XRAYDataset(
            data, 
            training=False, 
            image_transform=inference_transforms
        )

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
