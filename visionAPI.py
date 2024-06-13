import httpx
from PIL import Image
import io
from pathlib import Path
import base64
import pytesseract
import toml
import json
import re


class ConfigLoader:
    def __init__(self, config_file):
        self.config_file = config_file
        self.config = self.load_config()

    def load_config(self):
        with open(self.config_file, "r") as f:
            config = toml.load(f)
        return config


class ImageProcessor:
    def __init__(self, config_file, image_path):
        self.config_loader = ConfigLoader(config_file)
        self.config = self.config_loader.config
        self.image_path = image_path
        self.api_key = self.config["api"]["key"]
        self.text = None
        self.image_base64 = None
        self.response = None
        self.original_image = None
        self.parsed_response = None

    def process(self):
        self.preprocess_image()
        self.ask_question_to_vision_api()

    def show_image(self):
        if self.original_image:
            self.original_image.show()
        else:
            print("process first")

    def preprocess_image(self):
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
    def clean_ocr_text(text):
        # Entferne leere Zeilen
        lines = text.split("\n")
        cleaned_lines = [line for line in lines if line.strip()]
        return "\n".join(cleaned_lines)

    def ask_question_to_vision_api(self):
        question = f"""
        Give me interpret, title, label, release year, and country printed of these vinyl json string.
        Here is also the result of an OCR:\n{self.text}"""

        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        data = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{self.image_base64}"
                            },
                        },
                    ],
                }
            ],
            "max_tokens": 300,
        }
        with httpx.Client() as client:
            response = client.post(url, headers=headers, json=data, timeout=10)
        response = response.json()
        response = response["choices"][0]["message"]["content"]
        self.parsed_response = self.parse_response(response)

    @staticmethod
    def parse_response(response):
        json_match = re.search(r"{.*}", response, re.DOTALL)
        if json_match:
            json_string = json_match.group(0)
            try:
                # Parse the JSON string into a dictionary
                parsed_dict = json.loads(json_string)
                # Print the parsed dictionary
                return parsed_dict
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON: {e}")
        else:
            print("No JSON string found.")


if __name__ == "__main__":
    image_processor = ImageProcessor(
        "config.toml", Path(r"C:\Users\simon.schulte\Downloads\Bild2.jpg")
    )
    image_processor.process()
    print(image_processor.parsed_response)
    image_processor.show_image()
