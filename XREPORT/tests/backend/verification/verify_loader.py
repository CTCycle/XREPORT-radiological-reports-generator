import os
import sys
import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import DataLoader

# Add project root to path
sys.path.append(r"g:\Projects\Repository\XREPORT Radiological Reports")

from XREPORT.server.learning.training.dataloader import XRAYDataLoader


def verify_loader():
    print("Verifying DataLoader...")

    # Create dummy images
    os.makedirs("temp_images", exist_ok=True)
    img_path1 = "temp_images/img1.png"
    img_path2 = "temp_images/img2.png"
    cv2.imwrite(img_path1, np.zeros((100, 100, 3), dtype=np.uint8))
    cv2.imwrite(img_path2, np.zeros((100, 100, 3), dtype=np.uint8))

    # Create dummy DataFrame
    data = pd.DataFrame(
        {"path": [img_path1, img_path2], "tokens": [[1, 2, 3], [4, 5, 6]]}
    )

    config = {
        "batch_size": 2,
        "inference_batch_size": 2,
        "use_img_augmentation": False,
        "dataloader_workers": 0,
        "tokenizer_path": "dummy_tokenizer",  # Mock or not used directly if only pad_token needed
    }

    # Mock TokenizerHandler to avoid needing real files
    # The dataloader imports TokenizerHandler. We can monkeypatch it or just provide a mocked config that makes it happy?
    # Actually it tries to instantiate TokenizerHandler(configuration).
    # Let's mock the class in sys.modules if needed, or better, let's just mock the import or ensure it runs.
    # Analyzing TokenizerHandler usage: only 'pad_token' is accessed in __init__.

    # Let's mock the imported module before importing dataloader? Too late, already imported.
    # We can patch the class in the module.
    from XREPORT.server.learning.validation import validation_wizard
    from XREPORT.server.services.processing import TokenizerHandler

    # Simple mock for TokenizerHandler
    class MockHandler:
        def __init__(self, config):
            self.pad_token = 0

    import XREPORT.server.learning.training.dataloader as dl_module

    # We need to monkeypatch TokenizerHandler inside the module
    dl_module.TokenizerHandler = MockHandler

    # Initialize DataLoader
    loader_cls = dl_module.XRAYDataLoader(config, shuffle=False)

    # Test Training Loader
    print("Testing Training Loader...")
    train_loader = loader_cls.build_training_dataloader(data)
    batch = next(iter(train_loader))

    (images, inputs), outputs = batch

    assert isinstance(images, torch.Tensor), "Images should be Tensor"
    assert isinstance(inputs, torch.Tensor), "Inputs should be Tensor"
    assert isinstance(outputs, torch.Tensor), "Outputs should be Tensor"

    # Check shapes. Processor keeps images as HWC.
    # Batch size 2. Image (224, 224, 3).
    # Tokens: [1, 2, 3] -> input [1, 2], output [2, 3] (length 2)
    print(f"Image shape: {images.shape}")
    print(f"Input shape: {inputs.shape}")
    print(f"Output shape: {outputs.shape}")

    assert images.shape == (2, 224, 224, 3), f"Incorrect image shape: {images.shape}"
    assert inputs.shape == (2, 2), f"Incorrect input shape: {inputs.shape}"
    assert outputs.shape == (2, 2), f"Incorrect output shape: {outputs.shape}"

    # Check Types
    assert images.dtype == torch.float32, (
        f"Image dtype should be float32, got {images.dtype}"
    )
    assert inputs.dtype == torch.int64, (
        f"Input dtype should be int64, got {inputs.dtype}"
    )
    assert outputs.dtype == torch.int64, (
        f"Output dtype should be int64, got {outputs.dtype}"
    )

    # Test Inference Loader
    print("Testing Inference Loader...")
    inf_loader = loader_cls.build_inference_dataloader(data)
    batch_inf = next(iter(inf_loader))

    assert isinstance(batch_inf, torch.Tensor), "Inference batch should be Tensor"
    assert batch_inf.shape == (2, 224, 224, 3), (
        f"Incorrect inference shape: {batch_inf.shape}"
    )
    assert batch_inf.dtype == torch.float32, (
        f"Inference dtype should be float32, got {batch_inf.dtype}"
    )

    print("Verification Successful!")

    # Cleanup
    import shutil

    shutil.rmtree("temp_images")


if __name__ == "__main__":
    verify_loader()
