from app import create_app
from captioner import BlipCaptioner

app = create_app(BlipCaptioner())
