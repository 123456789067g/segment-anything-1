# step3_redline_extract_fixedsize.py
import os
import re
import cv2
import json
import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor

# ======================
# 参数设置
# ======================
input_dir = "crops"           # 第2步输出的试纸条图像（已做10%内缩），按分组镜像目录
output_dir = "redlines"       # 第3步红线区域输出（将按分组镜像目录）
sam_checkpoint = "sam_vit_h.pth"
model_type = "vit_h"
device = "cuda" if torch.cuda.is_available() else "cpu"

WHITE_THRESHOLD = 0.8
MARGIN = 10
GROUP_PAD_RATIO = 0.03
BAND_COL_RATIO = 0.5

TARGET_W = 130
TARGET_H = 40

BRIGHTNESS_MIN = 70
CONTRAST_MIN = 12
DARK_PIXEL_THRESH = 40
DARK_RATIO_MAX = 0.60

os.makedirs(output_dir, exist_ok=True)

print("🔧 加载 SAM 模型中...")
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device)
predictor = SamPredictor(sam)

# ===== 读取 step2 的坐标，并建立多重索引 =====
strip_coords_path = os.path.join("crops", "strip_coords.json")
if not os.path.exists(strip_coords_path):
    raise FileNotFoundError(f"未找到 {strip_coords_path}")
with open(strip_coords_path, "r", encoding="utf-8") as f:
    strip_coords = json.load(f)
print(f"🗂️ 已加载试纸坐标：{strip_coords_path}")

# 建索引：尽量多路匹配，最后用 group 过滤
by_video_id = {}
by_ir_basename = {}    # image_rel / image_path 的基名（去扩展名）
by_video_base = {}
by_last_token = {}

def _add_idx(key, entry, d: dict):
    if not key:
        return
    d.setdefault(str(key), []).append(entry)

for _, entry in strip_coords.items():
    vid = entry.get("video_id")
    vbase = entry.get("video_base")
    irel = entry.get("image_rel") or entry.get("image_path") or ""
    ir_base = os.path.splitext(os.path.basename(irel))[0] if irel else None
    last_tok = vid.split("__")[-1] if vid and "__" in vid else None

    _add_idx(vid, entry, by_video_id)
    _add_idx(ir_base, entry, by_ir_basename)
    _add_idx(vbase, entry, by_video_base)
    _add_idx(last_tok, entry, by_last_token)

# 输出 redline 坐标
redline_coords_path = os.path.join(output_dir, "redline_coords.json")
redline_coords = {}

# ========= 小工具 =========
def parse_crops_name(name: str):
    # e.g. 20__1-10__2_strip_003.jpg  -> ("20__1-10__2", 3)
    m = re.match(r"^(?P<vid>.+?)_strip_(?P<idx>\d{3})\.(jpg|jpeg|png)$", name, flags=re.IGNORECASE)
    if not m:
        return None, None
    return m.group("vid"), int(m.group("idx"))

def is_too_dark(bgr_img) -> bool:
    if bgr_img is None or bgr_img.size == 0:
        return True
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    mean_val = float(gray.mean())
    std_val = float(gray.std())
    dark_ratio = float((gray < DARK_PIXEL_THRESH).mean())
    cond_bright = mean_val < BRIGHTNESS_MIN
    cond_contrast = std_val < CONTRAST_MIN
    cond_darkratio = dark_ratio > DARK_RATIO_MAX
    return cond_bright or (cond_bright and cond_contrast) or cond_darkratio

def get_group_from_rel(rel_path: str) -> str:
    # crops/ 下的第一层目录就是 group（如 "14"、"20"）
    parts = rel_path.replace("\\", "/").split("/")
    return parts[0] if len(parts) > 1 else ""

def pick_entry(candidates, expect_group: str):
    if not candidates:
        return None
    if expect_group:
        for e in candidates:
            if str(e.get("group", "")) == str(expect_group):
                return e
    return sorted(candidates, key=lambda e: len(str(e.get("video_id") or "")))[0]

def find_parent_entry(vb: str, expect_group: str):
    last_token = vb.split("__")[-1] if "__" in vb else vb
    cand = by_video_id.get(vb, [])
    e = pick_entry(cand, expect_group)
    if e: return e
    cand = (by_ir_basename.get(vb, []) + by_ir_basename.get(last_token, []))
    e = pick_entry(cand, expect_group)
    if e: return e
    cand = by_video_base.get(last_token, [])
    e = pick_entry(cand, expect_group)
    if e: return e
    cand = by_last_token.get(last_token, [])
    e = pick_entry(cand, expect_group)
    return e

def find_parent_box(entry: dict, index: int):
    if not entry:
        return None
    for box in entry.get("boxes", []):
        if int(box.get("index", -1)) == int(index):
            return { "x": int(box["x"]), "y": int(box["y"]), "w": int(box["w"]), "h": int(box["h"]) }
    return None

# ========= 遍历 crops =========
image_files = []
for dirpath, _, files in os.walk(input_dir):
    for fn in files:
        if fn.lower().endswith((".jpg", ".jpeg", ".png")) and "_strip_" in fn:
            rel = os.path.relpath(os.path.join(dirpath, fn), input_dir)
            image_files.append(rel)
image_files.sort()

group_bands = {}

with torch.no_grad():
    for rel_img_path in image_files:
        print(f"\n🖼️ 正在处理: {rel_img_path}")
        abs_img_path = os.path.join(input_dir, rel_img_path)
        image = cv2.imread(abs_img_path)
        if image is None:
            print(f"❌ 无法读取图像: {abs_img_path}")
            continue

        h, w = image.shape[:2]
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        predictor.set_image(image_rgb)

        img_name = os.path.basename(rel_img_path)
        vb, idx = parse_crops_name(img_name)
        if vb is None or idx is None:
            print(f"⚠️ 无法解析文件名中的 video_id/index：{img_name}")
            continue

        expect_group = get_group_from_rel(rel_img_path)

        # ===== 1) 中线打点（与你之前一致）=====
        cy = h // 2
        xi = lambda p: max(0, min(int(round(w * p)), w - 1))
        pos_points = np.array([[xi(0.44), cy],[xi(0.615), cy]], dtype=np.float32)
        neg_points = np.array([[xi(0.41), cy],[xi(0.64), cy]], dtype=np.float32)
        input_points = np.vstack([pos_points, neg_points])
        input_labels = np.array([1, 1, 0, 0], dtype=np.int32)

        # ===== 2) 分组水平带宽缓存 =====
        gkey = expect_group or vb.split("__")[0]
        cached_band = group_bands.get(gkey, None)

        def predict_with_band(band):
            if band is None:
                return predictor.predict(point_coords=input_points, point_labels=input_labels, multimask_output=False)
            gx1, gx2 = band
            bx1 = max(int(gx1), 0)
            bx2 = min(int(gx2), w - 1)
            box = np.array([bx1, 0, bx2, h], dtype=np.float32)
            return predictor.predict(point_coords=input_points, point_labels=input_labels, box=box, multimask_output=False)

        tries = []
        if cached_band is not None:
            tries.append(("cached", cached_band))
        tries.append(("nobox", None))

        masks = None
        used_mode, used_band = None, None
        for mode, band in tries:
            masks, scores, logits = predict_with_band(band)
            if masks is not None and len(masks) > 0 and masks[0].sum() > 0:
                used_mode, used_band = mode, band
                break
        if masks is None or len(masks) == 0 or masks[0].sum() == 0:
            print("⚠️ 没有生成有效 mask，跳过...")
            continue

        mask01 = masks[0].astype(np.uint8)
        mask = (mask01 * 255).astype(np.uint8)
        mask_h, mask_w = mask.shape

        # ===== 3) 首次估计并缓存水平带宽 =====
        if cached_band is None:
            col_ratio = mask.mean(axis=0) / 255.0
            cols = np.where(col_ratio > BAND_COL_RATIO)[0]
            if len(cols) > 0:
                gx1, gx2 = int(cols.min()), int(cols.max())
                pad = int(GROUP_PAD_RATIO * w)
                gx1 = max(gx1 - pad, 0)
                gx2 = min(gx2 + pad, w - 1)
                group_bands[gkey] = (gx1, gx2)
                print(f"💾 记录分组 '{gkey}' 的水平带宽: [{gx1}, {gx2}]")

        # ===== 4) 基于 WHITE_THRESHOLD 找外接边界 =====
        col_start, col_end = 0, mask_w - 1
        for i in range(mask_w):
            if np.mean(mask[:, i] == 255) >= WHITE_THRESHOLD:
                col_start = i; break
        for i in range(mask_w - 1, -1, -1):
            if np.mean(mask[:, i] == 255) >= WHITE_THRESHOLD:
                col_end = i; break
        row_start, row_end = 0, mask_h - 1
        for i in range(mask_h):
            if np.mean(mask[i, :] == 255) >= WHITE_THRESHOLD:
                row_start = i; break
        for i in range(mask_h - 1, -1, -1):
            if np.mean(mask[i, :]) >= WHITE_THRESHOLD:
                row_end = i; break

        # ===== 5) 可用区域 + 居中裁（先不强制 130x40）=====
        x1 = max(col_start + MARGIN, 0)
        x2 = min(col_end   - MARGIN, w - 1)
        y1 = max(row_start + MARGIN, 0)
        y2 = min(row_end   - MARGIN, h - 1)
        if x1 >= x2 or y1 >= y2:
            print(f"⚠️ 可用区域无效，跳过...")
            continue

        avail_w = x2 - x1 + 1
        avail_h = y2 - y1 + 1
        tw = min(TARGET_W, avail_w)
        th = min(TARGET_H, avail_h)
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        crop_x1 = max(x1, min(cx - tw // 2, x2 - tw + 1))
        crop_y1 = max(y1, min(cy - th // 2, y2 - th + 1))
        crop_x2 = crop_x1 + tw
        crop_y2 = crop_y1 + th
        if not (0 <= crop_x1 < crop_x2 <= w and 0 <= crop_y1 < crop_y2 <= h):
            print(f"⚠️ 居中裁剪区域无效，跳过...")
            continue

        cropped = image[crop_y1:crop_y2, crop_x1:crop_x2]
        if is_too_dark(cropped):
            print("🗑️ 裁剪结果整体偏暗，丢弃。")
            continue

        # 输出目录镜像到 redlines/
        out_rel_dir = os.path.dirname(rel_img_path)  # e.g. 20/11-20
        out_abs_dir = os.path.join(output_dir, out_rel_dir)
        os.makedirs(out_abs_dir, exist_ok=True)

        save_path = os.path.join(out_abs_dir, img_name)
        cv2.imwrite(save_path, cropped)
        print(f"✅ 已保存红线区域: {save_path}")

        # ===== 6) 计算原视频坐标系的绝对坐标，并强制固定为 130x40 =====
        parent_entry = find_parent_entry(vb, expect_group)
        parent_box = find_parent_box(parent_entry, idx)

        record = {
            "image_name": img_name,
            "video_id": vb,
            "group": expect_group,                 # 供 Step4 使用
            "input_path": abs_img_path,
            "output_path": save_path,
            "target_size": {"w": TARGET_W, "h": TARGET_H},
            "actual_size": {"w": int(tw), "h": int(th)},   # 在 crops 内部的尺寸（可能小于 130x40）
            "image_size": {"w": int(w), "h": int(h)},      # crops 图尺寸
            "group_key": expect_group,
            "band_used": None if (group_bands.get(gkey) is None) else {
                "x1": int(group_bands[gkey][0]), "x2": int(group_bands[gkey][1])
            },
            "local": {"x": int(crop_x1), "y": int(crop_y1), "w": int(tw), "h": int(th)},
        }

        if parent_entry and parent_box:
            parent_x = int(parent_box["x"]); parent_y = int(parent_box["y"])
            parent_w = int(parent_box["w"]); parent_h = int(parent_box["h"])

            # crops 图（第2步内缩后）相对父框的 pad 估计
            inner_w, inner_h = w, h
            pad_x = max((parent_w - inner_w) // 2, 0)
            pad_y = max((parent_h - inner_h) // 2, 0)

            # 先把 crops 内的局部框映射回“父框坐标系”
            abs_x = parent_x + pad_x + int(crop_x1)
            abs_y = parent_y + pad_y + int(crop_y1)
            abs_w = int(tw)
            abs_h = int(th)

            # —— 关键修正：以该框中心为基准，强制输出固定 130×40 并夹紧到原图大小 ——
            img_w = int(parent_entry.get("width", 0)) or parent_w
            img_h = int(parent_entry.get("height", 0)) or parent_h

            cx_abs = abs_x + abs_w / 2.0
            cy_abs = abs_y + abs_h / 2.0

            out_x = int(round(cx_abs - TARGET_W / 2.0))
            out_y = int(round(cy_abs - TARGET_H / 2.0))
            out_x = max(0, min(out_x, img_w - TARGET_W))
            out_y = max(0, min(out_y, img_h - TARGET_H))

            record["absolute"] = {"x": out_x, "y": out_y, "w": TARGET_W, "h": TARGET_H}
            record["parent_meta"] = {
                "video_id": parent_entry.get("video_id") or vb,
                "video_base": parent_entry.get("video_base"),
                "group": parent_entry.get("group") or expect_group,
                "image_rel": parent_entry.get("image_rel"),
                "image_path": parent_entry.get("image_path"),
                "img_w": int(img_w),
                "img_h": int(img_h),
                "rotated_for_detection": bool(parent_entry.get("rotated_for_detection", False)),
                "rotation_mode": parent_entry.get("rotation_mode", "none"),
            }
        else:
            print(f"⚠️ 未在 strip_coords.json 中找到父外框：video_id={vb}, index={idx}。仅写入 local 坐标。")

        redline_coords[img_name] = record

# 保存 redline 坐标
with open(redline_coords_path, "w", encoding="utf-8") as f:
    json.dump(redline_coords, f, ensure_ascii=False, indent=2)

print("\n🎉 所有图像红线提取完成！")
print(f"🗂️ 红线坐标已保存至：{redline_coords_path}")
