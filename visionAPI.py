import argparse
import base64
import io
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

import httpx
import pytesseract
import toml
from PIL import Image


class ConfigLoader:
    """Load configuration values from a TOML file."""

    def __init__(self, config_file: Path):
        self.config_file = config_file
        self.config = self.load_config()

    def load_config(self) -> Dict[str, Any]:
        with open(self.config_file, "r", encoding="utf-8") as f:
            config = toml.load(f)
        return config


class ImageProcessor:
    """Process images and query the OpenAI Vision API."""

    def __init__(self, config_file: Path, image_path: Path):
        self.config_loader = ConfigLoader(config_file)
        self.config = self.config_loader.config
        self.image_path = image_path
        self.api_key = self.config["api"].get("key") or os.getenv("OPENAI_API_KEY")
        self.text: Optional[str] = None
        self.image_base64: Optional[str] = None
        self.response: Optional[str] = None
        self.original_image: Optional[Image.Image] = None
        self.parsed_response: Optional[Dict[str, Any]] = None

    def process(self, send_image: bool) -> None:
        """Run preprocessing and send question to the Vision API."""
        self.preprocess_image()
        self.ask_question_to_vision_api(send_image)

    def show_image(self) -> None:
        """Display the original image."""
        if self.original_image:
            self.original_image.show()
        else:
            print("process first")

    def preprocess_image(self) -> None:
        """Run OCR and prepare the image for API submission."""
        image = Image.open(self.image_path)
        self.original_image = image
        try:
            text = pytesseract.image_to_string(image)
            self.text = self.clean_ocr_text(text)
        except Exception as e:
            print(f"Error during OCR: {e}")
            self.text = ""

        max_size = 1000
        if max(image.size) > max_size:
            scaling_factor = max_size / max(image.size)
            new_size = (
                int(image.size[0] * scaling_factor),
                int(image.size[1] * scaling_factor),
            )
            image = image.resize(new_size)

        image_bytes = io.BytesIO()
        image.save(image_bytes, format="JPEG")
        image_bytes = image_bytes.getvalue()
        self.image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    @staticmethod
    def clean_ocr_text(text: str) -> str:
        """Remove empty lines from OCR text."""
        lines = text.split("\n")
        cleaned_lines = [line for line in lines if line.strip()]
        return "\n".join(cleaned_lines)

    def ask_question_to_vision_api(self, send_image: bool = False) -> None:
        """Query the Vision API and parse the JSON response."""
        question = f"""
        please fill the following json strings with the informations from the image and the ocr data without further informations and comments without mardown notation.
            "interpret": null,
            "album_title": null,
            "release_year": null,
            "country printed": null,
            "catalog_number": null
        Here is also the result of an OCR:\n{self.text}"""

        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        content = [{"type": "text", "text": question}]
        if send_image and self.image_base64:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{self.image_base64}"},
                }
            )
        data = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 300,
        }
        try:
            with httpx.Client() as client:
                response = client.post(url, headers=headers, json=data, timeout=10)
            response.raise_for_status()
            json_response = response.json()
            message_content = json_response["choices"][0]["message"]["content"]
            print(message_content)
            self.parsed_response = self.parse_response(message_content)
        except httpx.HTTPError as exc:
            print(f"HTTP error: {exc}")

    @staticmethod
    def parse_response(response: str) -> Optional[Dict[str, Any]]:
        """Extract a JSON string from the API response and return it as a dictionary."""
        json_match = re.search(r"{.*}", response, re.DOTALL)
        if json_match:
            json_string = json_match.group(0)
            try:
                return json.loads(json_string)
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON: {e}")
        else:
            print("No JSON string found.")
        return None

    def save_response(self, output_path: Path) -> None:
        """Save the parsed response as a JSON file."""
        if self.parsed_response is None:
            print("Nothing to save.")
            return
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.parsed_response, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Process an image and query the OpenAI Vision API."
    )
    parser.add_argument("image", type=Path, help="Path to the image file.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.toml"),
        help="Path to the TOML configuration file.",
    )
    parser.add_argument(
        "--send-image",
        action="store_true",
        help="Include the image in the request to the API.",
    )
    parser.add_argument(
        "--output", type=Path, help="Optional path to save the JSON response."
    )
    args = parser.parse_args()

    processor = ImageProcessor(args.config, args.image)
    processor.process(send_image=args.send_image)
    if args.output:
        processor.save_response(args.output)


if __name__ == "__main__":
    main()
