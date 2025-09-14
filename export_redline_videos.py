import os, json, re, cv2

video_root = "data"
image_manifest = os.path.join("image", "manifest.json")
coords_path = os.path.join("redlines", "redline_coords.json")
out_root = "redline_videos"
os.makedirs(out_root, exist_ok=True)

DURATION_SEC = 120
TARGET_W, TARGET_H = 130, 40

# 载入 manifest
if not os.path.exists(image_manifest):
    raise FileNotFoundError("缺少 image/manifest.json，请先完成第1步")
with open(image_manifest, "r", encoding="utf-8") as f:
    manifest = json.load(f)
print(f"🗂️ 载入 manifest：{image_manifest}")

# 载入 redline_coords
with open(coords_path, "r", encoding="utf-8") as f:
    redline_coords = json.load(f)
print(f"✅ 载入红线坐标：{coords_path}")

def resolve_video_path_by_id(video_id: str) -> tuple[str, str]:
    """返回 (abs_video_path, group)；优先使用 manifest"""
    m = manifest.get(video_id)
    if m:
        p = m.get("video_path")
        g = m.get("group", "")
        if p and os.path.exists(p):
            return p, g
    # 回退
    rel_noext = video_id.replace("__", os.sep)
    exts = [".mp4",".MP4",".mts",".MTS",".mov",".MOV",".m4v",".M4V"]
    for ext in exts:
        cand = os.path.join(video_root, rel_noext + ext)
        if os.path.exists(cand):
            group = rel_noext.split(os.sep)[0]
            return cand, group
    raise FileNotFoundError(f"无法解析视频路径：{video_id}")

def parse_strip(fn: str):
    m = re.search(r"_strip_(\d{3})\.", fn)
    return m.group(1) if m else None

for img_name, rec in list(redline_coords.items()):
    parent_meta = rec.get("parent_meta", {})
    video_id = parent_meta.get("video_id") or rec.get("video_id")
    group = parent_meta.get("group") or rec.get("group")
    abs_box = rec.get("absolute")
    if not video_id or not abs_box:
        print(f"⚠️ 跳过：{img_name} 缺少 video_id 或 absolute")
        continue

    try:
        video_path, group2 = resolve_video_path_by_id(video_id)
        if not group: group = group2
    except FileNotFoundError as e:
        print(f"❌ {e}")
        continue

    x, y, w, h = int(abs_box["x"]), int(abs_box["y"]), int(abs_box["w"]), int(abs_box["h"])
    if w <= 0 or h <= 0:
        print(f"⚠️ 跳过：{img_name} 裁剪尺寸非法")
        continue

    # 输出目录：redline_videos/<group>/
    out_dir = os.path.join(out_root, str(group))
    os.makedirs(out_dir, exist_ok=True)
    strip_suffix = parse_strip(img_name)
    out_name = f"{video_id}_strip_{strip_suffix}_redline_0to{DURATION_SEC}s_{TARGET_W}x{TARGET_H}.mp4" if strip_suffix \
               else f"{video_id}_redline_0to{DURATION_SEC}s_{TARGET_W}x{TARGET_H}.mp4"
    out_path = os.path.join(out_dir, out_name)

    print(f"\n🎬 导出：{os.path.basename(video_path)} → {os.path.join(str(group), out_name)}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ 无法打开视频：{video_path}")
        continue

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if fps <= 0: fps = 30.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    cx, cy = x + w/2.0, y + h/2.0
    new_x = int(round(cx - TARGET_W/2.0))
    new_y = int(round(cy - TARGET_H/2.0))
    new_x = max(0, min(new_x, frame_w - TARGET_W))
    new_y = max(0, min(new_y, frame_h - TARGET_H))
    out_w, out_h = TARGET_W, TARGET_H

    end_frame = int(DURATION_SEC * fps)
    end_frame = min(end_frame, total_frames) if total_frames > 0 else end_frame

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (out_w, out_h))
    if not writer.isOpened():
        print(f"❌ 无法打开写出流：{out_path}")
        cap.release()
        continue

    idx = 0; wrote = 0
    while True:
        if total_frames > 0 and idx >= end_frame: break
        ret, frame = cap.read()
        if not ret: break
        crop = frame[new_y:new_y+out_h, new_x:new_x+out_w]
        if crop.shape[0] != out_h or crop.shape[1] != out_w:
            crop = cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
        writer.write(crop)
        wrote += 1; idx += 1
        if end_frame > 0 and idx % max(end_frame//10,1) == 0:
            print(f"  进度 {idx}/{end_frame} 帧")

    cap.release(); writer.release()
    print(f"✅ 完成：{out_path}（{out_w}x{out_h} @ {fps:.2f}fps，{wrote}帧）")

print("\n🎉 全部分组红线视频导出完成（按 20 个顶层文件夹归档）。")
