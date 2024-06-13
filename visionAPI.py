import httpx
from PIL import Image
import io
from pathlib import Path
import base64
import pytesseract
import toml


def load_config(config_file):
    with open(config_file, "r") as f:
        config = toml.load(f)
    return config


def preprocess_image(image_path):
    try:
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image)
        text = clean_ocr_text(text)
    except Exception as e:
        print(f"Error during OCR: {e}")
        text = ""

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
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    return image_base64, text


def clean_ocr_text(text):
    # Entferne leere Zeilen
    lines = text.split("\n")
    cleaned_lines = [line for line in lines if line.strip()]
    return "\n".join(cleaned_lines)


def ask_question_to_vision_api(image_bytes, question, api_key):
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    data = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_bytes}"},
                    },
                ],
            }
        ],
        "max_tokens": 300,
    }
    with httpx.Client() as client:
        response = client.post(url, headers=headers, json=data)
    return response.json()


# Beispielverwendung
if __name__ == "__main__":
    config = load_config("config.toml")
    image_path = Path(config["paths"]["image_path"])
    api_key = config["api"]["key"]

    image_base64, ocr_result = preprocess_image(image_path)
    question = f"Give me interpret, title, label, release year, and country printed of these vinyl as JSON structured data without any other informations. Here is also the result of an OCR:\n{ocr_result}"

    response = ask_question_to_vision_api(image_base64, question, api_key)
    if response:
        print(response["choices"][0]["message"]["content"])
