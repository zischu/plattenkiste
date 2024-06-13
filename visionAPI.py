import httpx
from PIL import Image
import io
from pathlib import Path
import base64
import pytesseract


def preprocess_image(image_path):
    image = Image.open(image_path)
    try:
        text = pytesseract.image_to_string(image)
    except:
        text = ""

    max_size = 1000
    if max(image.size) > max_size:
        scaling_factor = max_size / max(image.size)
        new_size = (int(image.size[0] * scaling_factor), int(image.size[1] * scaling_factor))
        image = image.resize(new_size)


    image_bytes = io.BytesIO()
    image.save(image_bytes, format='JPEG')
    image_bytes = image_bytes.getvalue()
    image_base64 = base64.b64encode(image_bytes).decode('utf-8')



    return image_base64, text


def ask_question_to_vision_api(image_bytes, question, api_key):
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": question
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_bytes}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 300
    }
    with httpx.Client() as client:
        response = client.post(url, headers=headers, json=data)
    return response.json()


# Beispielverwendung
if __name__ == "__main__":
    image_path = Path(r"C:\Users\simon.schulte\Downloads\Bild2.jpg")

    image_bytes, ocr_result = preprocess_image(image_path)
    question = f"give me interpret, title, label, release year, and country printed of these vinyl as json structured data without any other informations. Here is also the result of an ocr:\n{ocr_result}"
    api_key = "your API Key"

    response = ask_question_to_vision_api(image_bytes, question, api_key)
    print(response["choices"][0]["message"]["content"])
