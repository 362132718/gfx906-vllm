import zlib, struct, base64, sys

W = H = 448

def chunk(t, d):
    return struct.pack('>I', len(d)) + t + d + struct.pack('>I', zlib.crc32(t + d) & 0xffffffff)

raw = bytearray()
for y in range(H):
    raw.append(0)
    for x in range(W):
        raw += bytes(((x * 255) // W, (y * 255) // H, 128))

png = b'\x89PNG\r\n\x1a\n'
png += chunk(b'IHDR', struct.pack('>IIBBBBB', W, H, 8, 2, 0, 0, 0))
png += chunk(b'IDAT', zlib.compress(bytes(raw), 9))
png += chunk(b'IEND', b'')

b64 = base64.b64encode(png).decode()
with open(r'D:\gfx906-vllm\host_tools\mm_test_img.b64', 'w') as f:
    f.write(b64)
print("png bytes:", len(png), "b64 chars:", len(b64))
