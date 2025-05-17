import base64
import io
from openai import OpenAI
from PIL import Image


class OpenAIImageGenerator:

    def __init__(self, base_url: str, api_key: str, model: str, quality: str, moderation: str, size: str):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.quality = quality
        self.moderation = moderation
        self.size = size
    
    def generate_image(self, prompt: str) -> Image.Image:
        response = self.client.images.generate(
            model=self.model,
            prompt=prompt,
            quality=self.quality,
            moderation=self.moderation,
            size=self.size
        )
        image_data = response.data[0].b64_json
        image_bytes = base64.b64decode(image_data)
        return Image.open(io.BytesIO(image_bytes))
        