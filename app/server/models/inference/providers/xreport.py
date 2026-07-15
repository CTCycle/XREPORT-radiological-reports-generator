from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from fastapi import HTTPException, status

from server.common.utils.security import resolve_checkpoint_path
from server.domain.inference import InferenceImage
from server.models.inference import TextGenerator
from server.models.training.dataloader import XRAYDataLoader
from server.repositories.serialization.model import ModelSerializer


class XReportCheckpointProvider:
    """Runs existing XREPORT checkpoints without changing their decoding behavior."""

    def validate_checkpoint(self, checkpoint: str) -> str:
        try:
            checkpoint_dir = resolve_checkpoint_path(checkpoint)
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        if not (Path(checkpoint_dir) / "saved_model.keras").is_file():
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Checkpoint not found: {Path(checkpoint_dir).name}")
        return Path(checkpoint_dir).name

    def generate(self, checkpoint: str, generation_mode: str, images: list[InferenceImage], should_stop: Callable[[], bool], report_progress: Callable[[int, int, dict[str, str]], None]) -> dict[str, str]:
        serializer = ModelSerializer()
        try:
            model, _, model_metadata, _, _ = serializer.load_checkpoint(checkpoint)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Checkpoint not found: {checkpoint}") from exc
        model.summary(expand_nested=True)
        generator = TextGenerator(model, model_metadata, model_metadata.get("max_report_size", 200))
        tokenizers_info = generator.load_tokenizer_and_configuration()
        if tokenizers_info is None:
            raise RuntimeError("Failed to load tokenizer")
        tokenizer, tokenizer_config = tokenizers_info
        generator_fn = generator.generator_image_methods.get(generation_mode)
        if generator_fn is None:
            raise RuntimeError(f"Unknown generation mode: {generation_mode}")
        reports: dict[str, str] = {}
        vocabulary = tokenizer.get_vocab()
        dataloader = XRAYDataLoader(model_metadata, shuffle=False)
        for image_index, stored_image in enumerate(images, start=1):
            if should_stop():
                break
            try:
                image = dataloader.prepare_inference_image_bytes(stored_image.data)
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError("Failed to decode inference image") from exc
            reports[stored_image.filename] = generator_fn(tokenizer_config, vocabulary, image, stream_callback=None)
            report_progress(image_index, len(images), reports)
        return reports

    def unload(self) -> None:
        """Keras checkpoint models are scoped to a single generation request."""
