# a3_client.py -- synthetic image + chat Q&A end-to-end probe for the
# vision path. Draws two colored shapes; the answer should name them.
import base64
import io
import json
import sys
import time
import urllib.request

from PIL import Image, ImageDraw

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8101
URL = "http://127.0.0.1:%d/v1/chat/completions" % PORT

img = Image.new("RGB", (448, 448), (30, 30, 40))
d = ImageDraw.Draw(img)
d.ellipse([80, 140, 240, 300], fill=(220, 40, 40))       # red circle
d.rectangle([280, 180, 400, 300], fill=(40, 90, 220))    # blue square
d.line([60, 60, 400, 60], fill=(240, 240, 240), width=10)  # white bar
buf = io.BytesIO()
img.save(buf, format="PNG")
b64 = base64.b64encode(buf.getvalue()).decode()

payload = {
    "model": "qwen38",
    "messages": [{
        "role": "user",
        "content": [
            {"type": "image_url",
             "image_url": {"url": "data:image/png;base64," + b64}},
            {"type": "text",
             "text": "Name each colored object in this image with its shape "
                     "and color. Answer in short English."},
        ],
    }],
    "max_tokens": 96,
    "temperature": 0.0,
}

req = urllib.request.Request(URL, data=json.dumps(payload).encode(),
                             headers={"Content-Type": "application/json"})
t0 = time.time()
with urllib.request.urlopen(req, timeout=600) as r:
    resp = json.load(r)
dt = time.time() - t0
text = resp["choices"][0]["message"]["content"]
print("SECONDS: %.1f" % dt)
print("RESPONSE:", repr(text))
low = text.lower()
score = sum(k in low for k in ("red", "circle", "blue", "square", "white", "line"))
print("KEYWORD_SCORE: %d/6" % score)
sys.exit(0 if score >= 3 else 2)
