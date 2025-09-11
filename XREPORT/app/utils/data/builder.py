from __future__ import annotations
from typing import Any
import pandas as pd


# [DATA SPLITTING]
###############################################################################
class BuildRadiographyDataset:
    def __init__(self, configuration: dict[str, Any], dataframe: pd.DataFrame) -> None:
        # Set the sizes for the train and validation datasets
        self.validation_size = configuration.get("validation_size", 1.0)
        self.seed = configuration.get("split_seed", 42)
        self.train_size = 1.0 - self.validation_size
        self.dataframe = dataframe

        # Compute the sizes of each split
        total_samples = len(dataframe)
        self.train_size = int(total_samples * self.train_size)
        self.val_size = int(total_samples * self.validation_size)

    # -------------------------------------------------------------------------
    def split_train_and_validation(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        dataframe = self.dataframe.sample(frac=1.0, random_state=self.seed).reset_index(
            drop=True
        )
        train_data = dataframe.iloc[: self.train_size]
        validation_data = dataframe.iloc[
            self.train_size : self.train_size + self.val_size
        ]

        return train_data, validation_data
