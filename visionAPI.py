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

    @property
    def discogs_token(self) -> Optional[str]:
        """Return the Discogs personal access token if configured."""
        return self.config.get("discogs", {}).get("token")


class ImageProcessor:
    """Process images and query the OpenAI Vision API."""

    def __init__(
        self,
        config_file: Path,
        image_path: Path,
        model: Optional[str] = None,
        prompt_template: Optional[str] = None,
        discogs_token: Optional[str] = None,
    ):
        self.config_loader = ConfigLoader(config_file)
        self.config = self.config_loader.config
        self.image_path = image_path
        self.api_key = self.config["api"].get("key") or os.getenv("OPENAI_API_KEY")

        self.model = model or self.config.get("api", {}).get("model")
        self.prompt_template = (
            prompt_template or self.config.get("api", {}).get("prompt_template")
        )
        self.discogs_token = discogs_token or self.config_loader.discogs_token

        self.text: Optional[str] = None
        self.image_base64: Optional[str] = None
        self.response: Optional[str] = None
        self.original_image: Optional[Image.Image] = None
        self.parsed_response: Optional[Dict[str, Any]] = None

    def process(self, send_image: bool, lookup_price: bool = False) -> None:
        """Run preprocessing, query the Vision API and optionally look up Discogs prices."""
        self.preprocess_image()
        self.ask_question_to_vision_api(send_image)
        if lookup_price and self.parsed_response:
            price = fetch_discogs_price(self.parsed_response, self.discogs_token)
            if price is not None:
                self.parsed_response["discogs_price_eur"] = price

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
        question = self.prompt_template.format(ocr_text=self.text)

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
            "model": self.model,
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


def fetch_discogs_price(info: Dict[str, Any], token: Optional[str]) -> Optional[float]:
    """Fetch price suggestions from Discogs and return the lowest value."""
    if not token:
        return None
    headers = {
        "Authorization": f"Discogs token={token}",
        "User-Agent": "plattenkiste/1.0",
    }
    params = {"type": "release"}
    if info.get("album_title"):
        params["release_title"] = info.get("album_title")
    if info.get("catalog_number"):
        params["catno"] = info.get("catalog_number")
    if info.get("release_year"):
        params["year"] = info.get("release_year")
    try:
        with httpx.Client() as client:
            resp = client.get(
                "https://api.discogs.com/database/search",
                headers=headers,
                params=params,
                timeout=10,
            )
        resp.raise_for_status()
        results = resp.json().get("results") or []
        if not results:
            return None
        release_id = results[0].get("id")
        if not release_id:
            return None
        with httpx.Client() as client:
            resp = client.get(
                f"https://api.discogs.com/marketplace/price_suggestions/{release_id}",
                headers=headers,
                timeout=10,
            )
        resp.raise_for_status()
        price_data = resp.json()
        values = [v["value"] for v in price_data.values() if isinstance(v, dict) and "value" in v]
        if not values:
            return None
        return min(values)
    except httpx.HTTPError as exc:
        print(f"Discogs lookup failed: {exc}")
        return None


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
        "--lookup-price",
        action="store_true",
        help="Look up Discogs price suggestions for the parsed release.",
    )
    parser.add_argument(
        "--output", type=Path, help="Optional path to save the JSON response."
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Override the model specified in the config file.",
    )
    parser.add_argument(
        "--prompt-template",
        type=str,
        help="Override the prompt template from the config file.",
    )
    args = parser.parse_args()

    if args.output:
        processor.save_response(args.output)


if __name__ == "__main__":
    main()
