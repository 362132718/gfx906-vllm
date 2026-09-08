# a3_matrix_client.py -- multi-modal acceptance matrix against a running
# gfx906-serve: single small / single large / 2 images / 4 images / image+long
# text. Every image is synthetic with a known answer, so scoring is keyword
# based and deterministic (temperature=0).
import base64, io, json, sys, time, urllib.error, urllib.request
from PIL import Image, ImageDraw

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8170
MODEL = sys.argv[2] if len(sys.argv) > 2 else "qwen38"
URL = "http://127.0.0.1:%d/v1/chat/completions" % PORT


def png(w, h, draw):
    img = Image.new("RGB", (w, h), (245, 245, 245))
    d = ImageDraw.Draw(img)
    draw(d, w, h)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def circle(d, w, h):
    d.ellipse([w * .18, h * .3, w * .5, h * .8], fill=(220, 40, 40))


def square(d, w, h):
    d.rectangle([w * .55, h * .3, w * .88, h * .8], fill=(40, 90, 220))


def triangle(d, w, h):
    d.polygon([(w * .2, h * .8), (w * .45, h * .25), (w * .7, h * .8)], fill=(30, 170, 80))


def diamond(d, w, h):
    d.polygon([(w * .8, h * .25), (w * .95, h * .55), (w * .8, h * .85), (w * .65, h * .55)],
              fill=(240, 200, 30))


def ask(content, max_tokens=90):
    body = json.dumps({"model": MODEL, "max_tokens": max_tokens, "temperature": 0,
                       "messages": [{"role": "user", "content": content}]}).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        d = json.load(urllib.request.urlopen(req, timeout=900))
    except urllib.error.HTTPError as e:
        return None, "HTTP_%d %s" % (e.code, e.read().decode()[:300]), time.time() - t0
    return d["choices"][0]["message"]["content"], None, d.get("usage", {})


CASES = [
    ("small_224", [circle], (224, 224), "图里是什么颜色的什么形状？", ["红", "圆"]),
    ("large_1568x700", [circle, square], (1568, 700),
     "描述这张图里的两个图形及其颜色。", ["红", "蓝"]),
    ("two_images", [circle, square], (640, 640),
     "按顺序说明每张图里的形状和颜色。", ["红", "蓝"]),
    ("four_images", [circle, square, triangle, diamond], (512, 512),
     "逐张说明每张图里是什么颜色的什么形状。", ["红", "蓝", "绿", "黄"]),
    ("image_plus_text", [circle], (768, 768),
     "先回答：这张图里有几个图形？它是什么颜色什么形状？然后把这个颜色联想到"
     "一个常见水果，用一句话说完，保持简短。", ["红"]),
]

total = len(CASES)
ok_n = 0
for name, shapes, size, q, keys in CASES:
    content = [{"type": "image_url", "image_url": {"url": png(size[0], size[1], fn)}}
               for fn in shapes]
    content.append({"type": "text", "text": q})
    txt, err, usage = ask(content)
    if err:
        print("%-15s FAIL  %s" % (name, err))
        continue
    low = txt.lower()
    hits = [k for k in keys if k in txt or k.lower() in low]
    ok = len(hits) == len(keys)
    ok_n += ok
    pt = usage.get("prompt_tokens", "?")
    print("%-15s %s  %d/%d keys  prompt=%s  %r" % (
        name, "OK " if ok else "MISS", len(hits), len(keys), pt, txt[:72].replace("\n", " ")))
    time.sleep(1)

print("MATRIX_SCORE: %d/%d" % (ok_n, total))
sys.exit(0 if ok_n == total else 1)
