from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from xreport_studio.domain.validation import (
    ImageStatistics,
    PixelDistribution,
    TextStatistics,
)
from xreport_shared.common.utils.logger import logger

###############################################################################
@dataclass(frozen=True)
class ImageMeasurement:
    height: int
    width: int
    mean: float
    median: float
    std: float
    minimum: float
    maximum: float
    noise_std: float
    noise_ratio: float

###############################################################################
class DatasetValidator:
    """Service class for dataset validation analytics."""

    # -------------------------------------------------------------------------
    def __init__(self, dataset: pd.DataFrame, dataset_name: str) -> None:
        self.dataset = dataset
        self.dataset_name = dataset_name

    # -------------------------------------------------------------------------
    def calculate_text_statistics(self) -> tuple[TextStatistics, pd.DataFrame]:
        """Calculate statistics for the text corpus.

        Returns tuple of (aggregate stats for dashboard, DB DataFrame payload).
        """
        empty_df = pd.DataFrame(
            columns=["record_id", "word_count", "char_count", "unique_words"]
        )

        if "text" not in self.dataset.columns or self.dataset.empty:
            return TextStatistics(
                count=0,
                total_words=0,
                unique_words=0,
                avg_words_per_report=0.0,
                min_words_per_report=0,
                max_words_per_report=0,
            ), empty_df

        texts = self.dataset["text"].astype(str).tolist()
        word_counts = []
        all_words: set[str] = set()
        total_word_count = 0

        for text in texts:
            words = text.split()
            count = len(words)
            word_counts.append(count)
            total_word_count += count
            all_words.update(words)

        word_counts_np = np.array(word_counts)
        record_ids = (
            pd.to_numeric(self.dataset["record_id"], errors="coerce").tolist()
            if "record_id" in self.dataset.columns
            else [None] * len(texts)
        )
        per_record_df = pd.DataFrame(
            {
                "record_id": record_ids,
                "word_count": word_counts,
                "char_count": [len(text) for text in texts],
                "unique_words": [len(set(text.split())) for text in texts],
            }
        )

        aggregate_stats = TextStatistics(
            count=len(texts),
            total_words=total_word_count,
            unique_words=len(all_words),
            avg_words_per_report=float(np.mean(word_counts_np))
            if len(texts) > 0
            else 0.0,
            min_words_per_report=int(np.min(word_counts_np)) if len(texts) > 0 else 0,
            max_words_per_report=int(np.max(word_counts_np)) if len(texts) > 0 else 0,
        )

        return aggregate_stats, per_record_df

    # -------------------------------------------------------------------------
    def calculate_image_statistics(
        self,
        progress_callback: Callable[[float], None] | None = None,
    ) -> tuple[ImageStatistics, pd.DataFrame]:
        """Calculate statistics for the images.

        Returns tuple of (aggregate stats for dashboard, per-record DataFrame for DB).
        """
        if "path" not in self.dataset.columns or self.dataset.empty:
            return self._empty_image_statistics(), self._empty_image_records()

        names, record_ids, image_paths = self._image_identifiers()
        measurements: list[ImageMeasurement] = []
        records: list[dict[str, Any]] = []
        total_images = len(image_paths)
        for idx, path in enumerate(image_paths):
            measurement = self._measure_image(path)
            if measurement is None:
                continue
            measurements.append(measurement)
            records.append(
                {
                    "record_id": record_ids[idx],
                    "dataset_name": self.dataset_name,
                    "name": names[idx],
                    "height": measurement.height,
                    "width": measurement.width,
                    "mean": measurement.mean,
                    "median": measurement.median,
                    "std": measurement.std,
                    "min": measurement.minimum,
                    "max": measurement.maximum,
                    "pixel_range": measurement.maximum - measurement.minimum,
                    "noise_std": measurement.noise_std,
                    "noise_ratio": measurement.noise_ratio,
                }
            )
            self._report_image_progress(
                idx,
                total_images,
                progress_callback,
            )

        if not measurements:
            return self._empty_image_statistics(), self._empty_image_records()
        return self._aggregate_image_statistics(measurements), pd.DataFrame(records)

    # -------------------------------------------------------------------------
    def _image_identifiers(
        self,
    ) -> tuple[list[str], list[Any], list[Any]]:
        if "image" in self.dataset.columns:
            image_series = self.dataset["image"]
            if not isinstance(image_series, pd.Series):
                raise ValueError("Dataset contains duplicate image columns")
            names = image_series.astype(str).tolist()
        else:
            names = [f"record_{i}" for i in range(len(self.dataset))]
        if "record_id" in self.dataset.columns:
            record_id_series = self.dataset["record_id"]
            if not isinstance(record_id_series, pd.Series):
                raise ValueError("Dataset contains duplicate record_id columns")
            numeric_record_ids = pd.to_numeric(record_id_series, errors="coerce")
            if not isinstance(numeric_record_ids, pd.Series):
                raise TypeError("Record identifier conversion did not return a Series")
            record_ids: list[Any] = numeric_record_ids.tolist()
        else:
            record_ids = [None for _ in range(len(self.dataset))]
        path_series = self.dataset["path"]
        if not isinstance(path_series, pd.Series):
            raise ValueError("Dataset contains duplicate path columns")
        return names, record_ids, path_series.tolist()

    # -------------------------------------------------------------------------
    @staticmethod
    def _empty_image_records() -> pd.DataFrame:
        return pd.DataFrame(
            columns=[
                "record_id",
                "dataset_name",
                "name",
                "height",
                "width",
                "mean",
                "median",
                "std",
                "min",
                "max",
                "pixel_range",
                "noise_std",
                "noise_ratio",
            ]
        )

    # -------------------------------------------------------------------------
    @staticmethod
    def _empty_image_statistics() -> ImageStatistics:
        return ImageStatistics(
            count=0,
            mean_height=0.0,
            mean_width=0.0,
            mean_pixel_value=0.0,
            std_pixel_value=0.0,
            mean_noise_std=0.0,
            mean_noise_ratio=0.0,
        )

    # -------------------------------------------------------------------------
    @staticmethod
    def _measure_image(path: Any) -> ImageMeasurement | None:
        if not Path(path).exists():
            return None
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            return None
        height, width = image.shape
        mean = float(np.mean(image))
        std = float(np.std(image))
        minimum = float(np.min(image))
        maximum = float(np.max(image))
        blurred = cv2.GaussianBlur(image, (3, 3), 0)
        noise = image.astype(np.float32) - blurred.astype(np.float32)
        noise_std = float(np.std(noise))
        return ImageMeasurement(
            height=height,
            width=width,
            mean=mean,
            median=float(np.median(image)),
            std=std,
            minimum=minimum,
            maximum=maximum,
            noise_std=noise_std,
            noise_ratio=noise_std / (std + 1e-9),
        )

    # -------------------------------------------------------------------------
    @staticmethod
    def _report_image_progress(
        index: int,
        total_images: int,
        progress_callback: Callable[[float], None] | None,
    ) -> None:
        log_interval = max(1, total_images // 10)
        if (index + 1) % log_interval != 0 and index + 1 != total_images:
            return
        progress = (index + 1) / total_images
        logger.info(
            "  Image statistics progress: %s/%s (%0.f%%)",
            index + 1,
            total_images,
            progress * 100,
        )
        if progress_callback:
            progress_callback(progress)

    # -------------------------------------------------------------------------
    @staticmethod
    def _aggregate_image_statistics(
        measurements: list[ImageMeasurement],
    ) -> ImageStatistics:
        return ImageStatistics(
            count=len(measurements),
            mean_height=float(np.mean([item.height for item in measurements])),
            mean_width=float(np.mean([item.width for item in measurements])),
            mean_pixel_value=float(np.mean([item.mean for item in measurements])),
            std_pixel_value=float(np.mean([item.std for item in measurements])),
            mean_noise_std=float(np.mean([item.noise_std for item in measurements])),
            mean_noise_ratio=float(
                np.mean([item.noise_ratio for item in measurements])
            ),
        )

    # -------------------------------------------------------------------------
    def calculate_pixel_distribution(
        self,
        progress_callback: Callable[[float], None] | None = None,
    ) -> tuple[PixelDistribution, int]:
        """Calculate the pixel intensity histogram and the number of images read."""
        if "path" not in self.dataset.columns or self.dataset.empty:
            return PixelDistribution(bins=[], counts=[]), 0

        image_paths = self.dataset["path"].tolist()
        combined_hist = np.zeros(256, dtype=np.int64)
        total_images = len(image_paths)
        log_interval = max(1, total_images // 10)  # Log every 10%
        processed = 0

        for idx, path in enumerate(image_paths):
            if not Path(path).exists():
                continue

            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            # Calculate histogram
            hist = cv2.calcHist([img], [0], None, [256], [0, 256]).flatten()
            combined_hist += hist.astype(np.int64)
            processed += 1

            # Log progress every 10%
            if (idx + 1) % log_interval == 0 or (idx + 1) == total_images:
                progress = ((idx + 1) / total_images) * 100
                logger.info(
                    f"  Pixel distribution progress: {idx + 1}/{total_images} ({progress:.0f}%)"
                )
                if progress_callback:
                    progress_callback((idx + 1) / total_images)

        return (
            PixelDistribution(
                bins=list(range(256)),
                counts=combined_hist.tolist(),
            ),
            processed,
        )

###############################################################################
def summarize_distribution(values: Sequence[float | int | None]) -> dict[str, Any] | None:
    """Summarize a measure; missing values are excluded, never counted as zero.

    Returns `None` when no value is available, so callers store no row at all.
    """
    array = np.array([value for value in values if value is not None], dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return None
    p05, p25, p75, p95 = np.percentile(array, [5, 25, 75, 95])
    return {
        "count": int(array.size),
        "total": float(np.sum(array)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "std": float(np.std(array)),
        "p05": float(p05),
        "p25": float(p25),
        "p75": float(p75),
        "p95": float(p95),
    }

###############################################################################
def histogram_bins(
    values: Sequence[float | int], bin_count: int = 20
) -> list[tuple[float, float, int]]:
    """Equal-width histogram as `(lower, upper, count)` rows."""
    array = np.array(values, dtype=float)
    if array.size == 0:
        return []
    counts, edges = np.histogram(array, bins=bin_count)
    return [
        (float(edges[index]), float(edges[index + 1]), int(count))
        for index, count in enumerate(counts)
    ]
