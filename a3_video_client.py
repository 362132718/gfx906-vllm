# a3_video_client.py -- video probe over the OpenAI chat API.
# vLLM's serving schema needs {"type":"video_url","video_url":{"url":...}} with
# a real video (a frame list is rejected), so the clip is synthesised with cv2
# and sent as a base64 data URL. The clip shows a red square sliding
# left->right over a blue field; a correct answer names colour, shape, direction.
import base64, json, subprocess, sys, time, urllib.error, urllib.request

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8160
MODEL = sys.argv[2] if len(sys.argv) > 2 else "qwen38"
NFRAMES = int(sys.argv[3]) if len(sys.argv) > 3 else 8
URL = "http://127.0.0.1:%d/v1/chat/completions" % PORT

path = "/tmp/probe_%d.mp4" % NFRAMES
if subprocess.call([sys.executable, "/tmp/mkvideo.py", path, str(NFRAMES), "2.0"],
                   stdout=subprocess.DEVNULL) != 0:
    print("VIDEO_GEN_FAILED")
    sys.exit(3)
b64 = base64.b64encode(open(path, "rb").read()).decode()

payload = {
    "model": MODEL,
    "max_tokens": 96,
    "temperature": 0,
    "messages": [{"role": "user", "content": [
        {"type": "video_url", "video_url": {"url": "data:video/mp4;base64," + b64}},
        {"type": "text", "text": "这段视频里什么颜色的什么形状在移动？往哪个方向？简短回答。"},
    ]}],
}
req = urllib.request.Request(URL, data=json.dumps(payload).encode(),
                             headers={"Content-Type": "application/json"})
t0 = time.time()
try:
    d = json.load(urllib.request.urlopen(req, timeout=900))
except urllib.error.HTTPError as e:
    print("HTTP_%d_BODY: %s" % (e.code, e.read().decode()[:700]))
    sys.exit(1)
el = time.time() - t0
txt = d["choices"][0]["message"]["content"]
print("SECONDS: %.1f" % el)
print("RESPONSE: %r" % txt)
score = (("红" in txt) + ("方" in txt or "square" in txt.lower())
         + ("右" in txt or "right" in txt.lower()))
print("VIDEO_KEYWORD_SCORE: %d/3" % score)
print("USAGE: %s" % d.get("usage"))
sys.exit(0 if score >= 2 else 2)
