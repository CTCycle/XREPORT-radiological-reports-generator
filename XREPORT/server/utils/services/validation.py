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


###############################################################################
class DatasetValidator:
    """Service class for dataset validation analytics."""
    
    def __init__(self, dataset: pd.DataFrame) -> None:
        self.dataset = dataset
    
    # -------------------------------------------------------------------------
    def calculate_text_statistics(self) -> TextStatistics:
        """Calculate statistics for the text corpus."""
        if "text" not in self.dataset.columns or self.dataset.empty:
            return TextStatistics(
                count=0,
                total_words=0,
                unique_words=0,
                avg_words_per_report=0.0,
                min_words_per_report=0,
                max_words_per_report=0,
            )
            
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
        
        return TextStatistics(
            count=len(texts),
            total_words=total_word_count,
            unique_words=len(all_words),
            avg_words_per_report=float(np.mean(word_counts_np)) if len(texts) > 0 else 0.0,
            min_words_per_report=int(np.min(word_counts_np)) if len(texts) > 0 else 0,
            max_words_per_report=int(np.max(word_counts_np)) if len(texts) > 0 else 0,
        )

    # -------------------------------------------------------------------------
    def calculate_image_statistics(self) -> ImageStatistics:
        """Calculate statistics for the images."""
        if "path" not in self.dataset.columns or self.dataset.empty:
            return ImageStatistics(
                count=0,
                mean_height=0.0,
                mean_width=0.0,
                mean_pixel_value=0.0,
                std_pixel_value=0.0,
                mean_noise_std=0.0,
                mean_noise_ratio=0.0,
            )
            
        image_paths = self.dataset["path"].tolist()
        
        heights: list[int] = []
        widths: list[int] = []
        means: list[float] = []
        stds: list[float] = []
        noise_stds: list[float] = []
        noise_ratios: list[float] = []
        
        valid_count = 0
        
        for path in image_paths:
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
            
            means.append(mean_val)
            stds.append(std_val)
            
            # Estimate noise
            blurred = cv2.GaussianBlur(img, (3, 3), 0)
            noise = img.astype(np.float32) - blurred.astype(np.float32)
            noise_std = float(np.std(noise))
            noise_ratio = noise_std / (std_val + 1e-9)
            
            noise_stds.append(noise_std)
            noise_ratios.append(noise_ratio)
            
            valid_count += 1
            
        if valid_count == 0:
            return ImageStatistics(
                count=0,
                mean_height=0.0,
                mean_width=0.0,
                mean_pixel_value=0.0,
                std_pixel_value=0.0,
                mean_noise_std=0.0,
                mean_noise_ratio=0.0,
            )
            
        return ImageStatistics(
            count=valid_count,
            mean_height=float(np.mean(heights)),
            mean_width=float(np.mean(widths)),
            mean_pixel_value=float(np.mean(means)),
            std_pixel_value=float(np.mean(stds)),
            mean_noise_std=float(np.mean(noise_stds)),
            mean_noise_ratio=float(np.mean(noise_ratios)),
        )

    # -------------------------------------------------------------------------
    def calculate_pixel_distribution(self) -> PixelDistribution:
        """Calculate pixel intensity distribution (histogram)."""
        if "path" not in self.dataset.columns or self.dataset.empty:
            return PixelDistribution(bins=[], counts=[])
            
        image_paths = self.dataset["path"].tolist()
        combined_hist = np.zeros(256, dtype=np.int64)
        
        for path in image_paths:
            if not os.path.exists(path):
                continue
                
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
                
            # Calculate histogram
            hist = cv2.calcHist([img], [0], None, [256], [0, 256]).flatten()
            combined_hist += hist.astype(np.int64)
            
        return PixelDistribution(
            bins=list(range(256)),
            counts=combined_hist.tolist(),
        )
