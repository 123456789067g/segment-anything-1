import os, json, cv2

# ===== 基本路径 =====
data_root = "data"     # 现在是 data/<group>/<subdir>/*.mp4
image_root = "image"   # 输出基准帧（镜像 data 的层级）
os.makedirs(image_root, exist_ok=True)

target_time_sec = 120  # 抓取第 120 秒
manifest_path = os.path.join(image_root, "manifest.json")
manifest = {}

def to_video_id(relpath_noext: str) -> str:
    # 把相对路径里的分隔符 / -> __，统一成一个字符串作为 video_id
    return relpath_noext.replace(os.sep, "__")

def safe_makedirs(path: str):
    os.makedirs(path, exist_ok=True)

def iter_videos(root: str):
    for dirpath, _, files in os.walk(root):
        for fn in files:
            if fn.lower().endswith((".mp4", ".mov", ".m4v", ".mts")):
                abs_path = os.path.join(dirpath, fn)
                rel_path = os.path.relpath(abs_path, root)               # e.g. 0/11-20/2.mp4
                rel_noext = os.path.splitext(rel_path)[0]                # e.g. 0/11-20/2
                group = rel_path.split(os.sep)[0]                        # e.g. "0"
                rel_dir = os.path.dirname(rel_path)                      # e.g. "0/11-20"
                yield abs_path, rel_path, rel_noext, group, rel_dir, fn

count = 0
for abs_path, rel_path, rel_noext, group, rel_dir, fn in iter_videos(data_root):
    print(f"🎬 处理：{rel_path}")

    cap = cv2.VideoCapture(abs_path)
    if not cap.isOpened():
        print(f"❌ 无法打开视频：{abs_path}")
        continue

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration = (total_frames / fps) if fps > 0 else 0.0

    if duration < target_time_sec:
        print(f"⚠️ 时长仅 {duration:.1f}s，跳过")
        cap.release()
        continue

    target_msec = int(target_time_sec * 1000)
    target_frame = int(round(fps * target_time_sec)) if fps > 0 else 0
    if target_frame >= max(total_frames - 1, 0):
        target_frame = max(total_frames - 1, 0)
        target_msec = int(1000.0 * target_frame / fps) if fps > 0 else target_msec

    got = False
    used_mode = None
    for mode in ("msec", "frame"):
        cap.set(cv2.CAP_PROP_POS_MSEC if mode=="msec" else cv2.CAP_PROP_POS_FRAMES,
                target_msec if mode=="msec" else target_frame)
        ret, frame = cap.read()
        if ret and frame is not None:
            got = True
            used_mode = mode
            break

    if not got:
        print(f"❌ 无法读取目标帧：{abs_path}")
        cap.release()
        continue

    # 输出路径：image/<group>/<subdirs>/
    out_dir = os.path.join(image_root, os.path.dirname(rel_path))
    safe_makedirs(out_dir)

    base_noext = os.path.basename(rel_noext)        # e.g. "2"
    out_png = os.path.join(out_dir, f"{base_noext}.png")
    cv2.imwrite(out_png, frame)
    print(f"✅ 已保存基准帧：{out_png}（寻址={used_mode}）")

    vid = to_video_id(rel_noext)                    # e.g. "0__11-20__2"
    manifest[vid] = {
        "video_path": abs_path,
        "image_path": out_png,
        "fps": fps, "width": width, "height": height,
        "total_frames": total_frames, "duration_sec": duration,
        "target_time_sec": target_time_sec,
        "target_msec": target_msec, "target_frame": target_frame,
        "seek_mode_used": used_mode,
        # 结构信息
        "group": group,                     # 顶层分组（0、2、…、1000）
        "rel_dir": os.path.dirname(rel_path),  # 相对 data 的目录（含 group）
        "file": os.path.basename(rel_path)     # 文件名（含扩展名）
    }

    cap.release()
    count += 1

with open(manifest_path, "w", encoding="utf-8") as f:
    json.dump(manifest, f, ensure_ascii=False, indent=2)

print(f"\n🎉 基准帧提取完成，共 {count} 个。清单：{manifest_path}")
