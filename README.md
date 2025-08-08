# plattenkiste
This repository contains scripts designed to process images and interact with the OpenAI Vision API. The main functionalities include loading and resizing images, extracting text using OCR, and sending queries to the Vision API for image analysis.

## Usage

```bash
python visionAPI.py path/to/image.jpg --config config.toml --send-image --output result.json
```

Provide an `OPENAI_API_KEY` environment variable or store the key in a TOML file:

```toml
[api]
key = "your key"
model = "gpt-4o"
prompt_template = """
please fill the following json strings with the informations from the image and the ocr data without further informations and comments without mardown notation.
    "interpret": null,
    "album_title": null,
    "release_year": null,
    "country printed": null,
    "catalog_number": null
Here is also the result of an OCR:
{ocr_text}
"""
```

The `--send-image` flag includes the processed image in the request. Using `--output` writes the structured response to the specified JSON file.
Use `--model` or `--prompt-template` to override the respective values from the configuration file.
