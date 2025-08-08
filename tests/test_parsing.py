import sys
import types
from pathlib import Path

import pytest

# Create minimal stub modules to satisfy imports in visionAPI
for name in ["httpx", "pytesseract", "toml"]:
    sys.modules[name] = types.ModuleType(name)

# Provide a minimal implementation for toml.load so configuration loading works.
sys.modules["toml"].load = lambda f: {"api": {"key": "dummy"}}

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


def test_process_lookup_price_uses_configured_token(monkeypatch, tmp_path):
    """process() should pass the Discogs token without raising errors."""
    # Create temporary files required by ImageProcessor
    config_file = tmp_path / "config.toml"
    config_file.write_text("[api]\nkey='dummy'\n")
    image_file = tmp_path / "image.jpg"
    image_file.write_text("fake")

    processor = ImageProcessor(config_file, image_file)
    processor.parsed_response = {"album_title": "Dummy"}

    # Avoid heavy processing or network calls
    monkeypatch.setattr(ImageProcessor, "preprocess_image", lambda self: None)
    monkeypatch.setattr(ImageProcessor, "ask_question_to_vision_api", lambda self, send_image: None)

    called = {"token": None}

    def fake_fetch(info, token):
        called["token"] = token
        return 1.23

    monkeypatch.setattr("visionAPI.fetch_discogs_price", fake_fetch)

    processor.process(send_image=False, lookup_price=True)

    assert called["token"] is None
    assert processor.parsed_response["discogs_price_eur"] == 1.23

