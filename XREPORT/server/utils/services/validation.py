from __future__ import annotations

import os

import cv2
import numpy as np
import pandas as pd

from XREPORT.server.schemas.validation import (
    ImageStatistics,
    PixelDistribution,
    TextStatistics,
)
from XREPORT.server.utils.logger import logger


###############################################################################
class DatasetValidator:
    """Service class for dataset validation analytics."""
    
    def __init__(self, dataset: pd.DataFrame, dataset_name: str = "default") -> None:
        self.dataset = dataset
        self.dataset_name = dataset_name
    
    # -------------------------------------------------------------------------
    def calculate_text_statistics(self) -> tuple[TextStatistics, pd.DataFrame]:
        """Calculate statistics for the text corpus.
        
        Returns tuple of (aggregate stats for dashboard, per-record DataFrame for DB).
        """
        empty_df = pd.DataFrame(columns=["dataset_name", "name", "words_count"])
        
        if "text" not in self.dataset.columns or self.dataset.empty:
            return TextStatistics(
                count=0,
                total_words=0,
                unique_words=0,
                avg_words_per_report=0.0,
                min_words_per_report=0,
                max_words_per_report=0,
            ), empty_df
        
        # Get record identifiers (use 'image' column as name if available)
        if "image" in self.dataset.columns:
            names = self.dataset["image"].astype(str).tolist()
        else:
            names = [f"record_{i}" for i in range(len(self.dataset))]
            
        texts = self.dataset["text"].astype(str).tolist()
        word_counts = []
        all_words: set[str] = set()
        total_word_count = 0
        
        # Per-record data for database
        records: list[dict] = []
        
        for idx, text in enumerate(texts):
            words = text.split()
            count = len(words)
            word_counts.append(count)
            total_word_count += count
            all_words.update(words)
            
            # Build per-record entry
            records.append({
                "dataset_name": self.dataset_name,
                "name": names[idx],
                "words_count": count,
            })
            
        word_counts_np = np.array(word_counts)
        per_record_df = pd.DataFrame(records)
        
        aggregate_stats = TextStatistics(
            count=len(texts),
            total_words=total_word_count,
            unique_words=len(all_words),
            avg_words_per_report=float(np.mean(word_counts_np)) if len(texts) > 0 else 0.0,
            min_words_per_report=int(np.min(word_counts_np)) if len(texts) > 0 else 0,
            max_words_per_report=int(np.max(word_counts_np)) if len(texts) > 0 else 0,
        )
        
        return aggregate_stats, per_record_df

    # -------------------------------------------------------------------------
    def calculate_image_statistics(self) -> tuple[ImageStatistics, pd.DataFrame]:
        """Calculate statistics for the images.
        
        Returns tuple of (aggregate stats for dashboard, per-record DataFrame for DB).
        """
        empty_df = pd.DataFrame(columns=[
            "dataset_name", "name", "height", "width", "mean", "median",
            "std", "min", "max", "pixel_range", "noise_std", "noise_ratio"
        ])
        
        if "path" not in self.dataset.columns or self.dataset.empty:
            return ImageStatistics(
                count=0,
                mean_height=0.0,
                mean_width=0.0,
                mean_pixel_value=0.0,
                std_pixel_value=0.0,
                mean_noise_std=0.0,
                mean_noise_ratio=0.0,
            ), empty_df
        
        # Get record identifiers
        if "image" in self.dataset.columns:
            names = self.dataset["image"].astype(str).tolist()
        else:
            names = [f"record_{i}" for i in range(len(self.dataset))]
            
        image_paths = self.dataset["path"].tolist()
        
        # Aggregate collectors
        heights: list[int] = []
        widths: list[int] = []
        means: list[float] = []
        stds: list[float] = []
        noise_stds: list[float] = []
        noise_ratios: list[float] = []
        
        # Per-record data for database
        records: list[dict] = []
        
        valid_count = 0
        total_images = len(image_paths)
        log_interval = max(1, total_images // 10)  # Log every 10%
        
        for idx, path in enumerate(image_paths):
            if not os.path.exists(path):
                continue
                
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
                
            h, w = img.shape
            heights.append(h)
            widths.append(w)
            
            mean_val = float(np.mean(img))
            std_val = float(np.std(img))
            min_val = float(np.min(img))
            max_val = float(np.max(img))
            median_val = float(np.median(img))
            pixel_range = max_val - min_val
            
            means.append(mean_val)
            stds.append(std_val)
            
            # Estimate noise
            blurred = cv2.GaussianBlur(img, (3, 3), 0)
            noise = img.astype(np.float32) - blurred.astype(np.float32)
            noise_std = float(np.std(noise))
            noise_ratio = noise_std / (std_val + 1e-9)
            
            noise_stds.append(noise_std)
            noise_ratios.append(noise_ratio)
            
            # Build per-record entry
            records.append({
                "dataset_name": self.dataset_name,
                "name": names[idx],
                "height": h,
                "width": w,
                "mean": mean_val,
                "median": median_val,
                "std": std_val,
                "min": min_val,
                "max": max_val,
                "pixel_range": pixel_range,
                "noise_std": noise_std,
                "noise_ratio": noise_ratio,
            })
            
            valid_count += 1
            
            # Log progress every 10%
            if (idx + 1) % log_interval == 0 or (idx + 1) == total_images:
                progress = ((idx + 1) / total_images) * 100
                logger.info(f"  Image statistics progress: {idx + 1}/{total_images} ({progress:.0f}%)")
        
        per_record_df = pd.DataFrame(records)
            
        if valid_count == 0:
            return ImageStatistics(
                count=0,
                mean_height=0.0,
                mean_width=0.0,
                mean_pixel_value=0.0,
                std_pixel_value=0.0,
                mean_noise_std=0.0,
                mean_noise_ratio=0.0,
            ), empty_df
            
        aggregate_stats = ImageStatistics(
            count=valid_count,
            mean_height=float(np.mean(heights)),
            mean_width=float(np.mean(widths)),
            mean_pixel_value=float(np.mean(means)),
            std_pixel_value=float(np.mean(stds)),
            mean_noise_std=float(np.mean(noise_stds)),
            mean_noise_ratio=float(np.mean(noise_ratios)),
        )
        
        return aggregate_stats, per_record_df

    # -------------------------------------------------------------------------
    def calculate_pixel_distribution(self) -> PixelDistribution:
        """Calculate pixel intensity distribution (histogram)."""
        if "path" not in self.dataset.columns or self.dataset.empty:
            return PixelDistribution(bins=[], counts=[])
            
        image_paths = self.dataset["path"].tolist()
        combined_hist = np.zeros(256, dtype=np.int64)
        total_images = len(image_paths)
        log_interval = max(1, total_images // 10)  # Log every 10%
        processed = 0
        
        for idx, path in enumerate(image_paths):
            if not os.path.exists(path):
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
                logger.info(f"  Pixel distribution progress: {idx + 1}/{total_images} ({progress:.0f}%)")
            
        return PixelDistribution(
            bins=list(range(256)),
            counts=combined_hist.tolist(),
        )
