import sys
import types
from pathlib import Path

import pytest

# Create minimal stub modules to satisfy imports in visionAPI
for name in ["httpx", "pytesseract", "toml"]:
    sys.modules[name] = types.ModuleType(name)

pil_module = types.ModuleType("PIL")
pil_image_module = types.ModuleType("Image")
pil_module.Image = pil_image_module
sys.modules["PIL"] = pil_module
sys.modules["PIL.Image"] = pil_image_module

sys.path.append(str(Path(__file__).resolve().parent.parent))
from visionAPI import ImageProcessor


def test_clean_ocr_text_removes_empty_lines():
    input_text = "line1\n\nline2\n  \nline3"
    expected = "line1\nline2\nline3"
    assert ImageProcessor.clean_ocr_text(input_text) == expected


def test_parse_response_with_valid_json():
    response = "Here is data {\"interpret\": \"Artist\", \"album_title\": \"Album\", \"release_year\": 1990} end"
    expected = {
        "interpret": "Artist",
        "album_title": "Album",
        "release_year": 1990,
    }
    assert ImageProcessor.parse_response(response) == expected


def test_parse_response_without_json_returns_none():
    response = "No JSON here"
    assert ImageProcessor.parse_response(response) is None

