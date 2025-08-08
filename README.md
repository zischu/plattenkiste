# plattenkiste
This repository contains scripts designed to process images and interact with the OpenAI Vision API. The main functionalities include loading and resizing images, extracting text using OCR, and sending queries to the Vision API for image analysis.

## Usage

```bash
python visionAPI.py path/to/image.jpg --config config.toml --send-image --lookup-price --output result.json
```

Provide an `OPENAI_API_KEY` environment variable or store the key in a TOML file. The configuration can also include a Discogs token for price lookups:

```toml
[api]
key = "your key"
