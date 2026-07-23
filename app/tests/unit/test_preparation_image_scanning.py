from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from server.services.preparation import (
    count_image_files,
    count_images_in_folder,
    scan_image_folder,
)

###############################################################################
def test_image_scanning_counts_recursively_and_filters_required_stems() -> None:
    with TemporaryDirectory(dir=Path.cwd()) as temp_dir:
        root = Path(temp_dir)
        image_folder = root / "images"
        nested_folder = image_folder / "nested"
        nested_folder.mkdir(parents=True)
        (image_folder / "one.jpg").write_bytes(b"image")
        (image_folder / "ignore.txt").write_text("not an image", encoding="utf-8")
        (nested_folder / "two.PNG").write_bytes(b"image")
        (nested_folder / "three.bmp").write_bytes(b"image")

        assert count_images_in_folder(str(image_folder)) == 1
        assert count_image_files(str(image_folder)) == 3

        matched = scan_image_folder(
            str(image_folder),
            required_stems={"two"},
        )

        assert [Path(path).name for path in matched] == ["two.PNG"]
