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
```

The `--send-image` flag includes the processed image in the request. Using `--output` writes the structured response to the specified JSON file.

## Testing

Run the test suite with:

```bash
pytest
```

