# step2_detect_strips_no_rotate_fixed.py
import os
import cv2
import json
import numpy as np
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# ===== 基础路径 =====
image_root = "image"     # 第1步输出（支持子目录）
crops_root = "crops"     # 第2步输出（镜像目录）
os.makedirs(crops_root, exist_ok=True)

coords_save_path = os.path.join(crops_root, "strip_coords.json")
coords_dict = {}

# ===== SAM 模型 =====
sam_checkpoint = "sam_vit_h.pth"
model_type = "vit_h"
device = "cuda" if torch.cuda.is_available() else "cpu"

print("🔧 加载 SAM 模型中...")
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device)
mask_generator = SamAutomaticMaskGenerator(sam)
print("✅ SAM 加载完成（不做任何旋转）")

# ===== 工具函数 =====
def iter_images(root: str):
    """递归遍历所有图片"""
    for dirpath, _, files in os.walk(root):
        for fn in files:
            if fn.lower().endswith((".png", ".jpg", ".jpeg")):
                abs_path = os.path.join(dirpath, fn)
                rel_path = os.path.relpath(abs_path, root)  # 例: 50/11-20/2.png
                yield abs_path, rel_path

def rel_to_video_id(rel_png: str) -> str:
    """把相对路径转成 video_id（用 __ 替换路径分隔符）"""
    noext = os.path.splitext(rel_png)[0]      # "50/11-20/2"
    return noext.replace(os.sep, "__")        # "50__11-20__2"

# ===== 主循环 =====
kept_total = 0
for abs_img, rel_img in iter_images(image_root):
    print(f"\n🖼️ 处理：{rel_img}")
    bgr = cv2.imread(abs_img)
    if bgr is None:
        print("❌ 读图失败"); continue

    H0, W0 = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # —— 绝不旋转 —— #
    rotated = False

    # SAM 分割
    masks = mask_generator.generate(rgb)
    print(f"🧠 候选区域：{len(masks)}")

    # 筛选候选框（横条 + 颜色/对比/面积/长宽比）
    cand = []
    for m in masks:
        seg = m["segmentation"].astype(np.uint8)
        x, y, w, h = cv2.boundingRect(seg)
        if w == 0 or h == 0:
            continue

        crop = rgb[y:y+h, x:x+w]
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        gray_std = float(np.std(gray))
        aspect = float(max(w, h)) / float(max(1, min(w, h)))
        area = int(w * h)

        hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
        m1 = cv2.inRange(hsv, (0, 70, 50), (10, 255, 255))
        m2 = cv2.inRange(hsv, (160, 70, 50), (180, 255, 255))
        red_ratio = float(np.sum((m1 | m2) > 0)) / float(w * h)
        white_ratio = float(np.sum(gray > 170)) / float(w * h)

        if (11 <= aspect <= 25 and
            w > h and                       # 横条假设
            area > 3000 and
            gray_std > 10 and
            (red_ratio > 0.10 or (red_ratio > 0.03 and white_ratio > 0.01))):
            cand.append((y, x, w, h))

    # 去重（按 y 坐标，避免同一条多框）
    cand.sort(key=lambda b: b[0])
    dedup, min_y_gap = [], 30
    for box in cand:
        if all(abs(box[0]-b[0]) > min_y_gap for b in dedup):
            dedup.append(box)

    print(f"✅ 去重后保留：{len(dedup)}")

    # 输出目录：crops/<同相对路径目录>/
    out_dir = os.path.join(crops_root, os.path.dirname(rel_img))
    os.makedirs(out_dir, exist_ok=True)

    video_id = rel_to_video_id(rel_img)        # 如 "50__11-20__2"
    group = rel_img.split(os.sep)[0]           # "50"
    video_base = os.path.splitext(os.path.basename(rel_img))[0]  # "2"
    saved_boxes = []

    # 保存裁剪图（检测坐标系内缩 10%）并记录原始坐标
    for i, (yy, xx, ww, hh) in enumerate(dedup, start=1):
        crop_rgb = rgb[yy:yy+hh, xx:xx+ww]
        pad_x, pad_y = int(ww*0.1), int(hh*0.1)
        y1, y2 = pad_y, max(hh - pad_y, 0)
        x1, x2 = pad_x, max(ww - pad_x, 0)
        crop_inner = crop_rgb[y1:y2, x1:x2]

        save_path = os.path.join(out_dir, f"{video_id}_strip_{i:03}.jpg")
        cv2.imwrite(save_path, cv2.cvtColor(crop_inner, cv2.COLOR_RGB2BGR))
        print(f"📸 保存：{save_path}")

        # 原始坐标（未旋转，直接记录）
        x0, y0, w0, h0 = int(xx), int(yy), int(ww), int(hh)
        x0 = max(0, min(x0, W0-1)); y0 = max(0, min(y0, H0-1))
        w0 = max(1, min(w0, W0-x0)); h0 = max(1, min(h0, H0-y0))
        saved_boxes.append({"index": i, "x": x0, "y": y0, "w": w0, "h": h0})

    # 记录到 JSON
    coords_dict[rel_img] = {
        "video_id": video_id,
        "video_base": video_base,   # ★ 新增，供 Step3 使用
        "group": group,
        "image_rel": rel_img,
        "width": W0,
        "height": H0,
        "rotated_for_detection": rotated,
        "rotation_mode": "none",
        "boxes": saved_boxes
    }
    kept_total += len(dedup)

# 保存坐标清单
with open(coords_save_path, "w", encoding="utf-8") as f:
    json.dump(coords_dict, f, ensure_ascii=False, indent=2)

print(f"\n🎉 完成。总保留 {kept_total} 条；坐标清单：{coords_save_path}")
