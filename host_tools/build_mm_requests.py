import json, os

base = r'D:\gfx906-vllm\host_tools'
b64 = open(os.path.join(base, 'mm_test_img.b64')).read().strip()
data_url = "data:image/png;base64," + b64

text_req = {
    "model": "Qwen3.8-27B-OBLITERATED-AWQ-W4A16",
    "messages": [{"role": "user", "content": "用一句话说明你在正常运行。"}],
    "max_tokens": 48,
    "temperature": 0,
}

image_req = {
    "model": "Qwen3.8-27B-OBLITERATED-AWQ-W4A16",
    "messages": [{
        "role": "user",
        "content": [
            {"type": "text", "text": "这张图片的主要颜色和渐变方向是什么？用一句话回答。"},
            {"type": "image_url", "image_url": {"url": data_url}},
        ],
    }],
    "max_tokens": 64,
    "temperature": 0,
}

for name, obj in (("mm_text.json", text_req), ("mm_image.json", image_req)):
    p = os.path.join(base, name)
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False)
    print(name, os.path.getsize(p), "bytes")
