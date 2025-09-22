#!/usr/bin/env python3
# app_image_filters.py

import io
import math
import zipfile
from pathlib import Path
import random

import numpy as np
import cv2  # opencv-python-headless
import streamlit as st
from PIL import Image, ImageOps


APP_TITLE = "📸 Image Filter Lab"

st.set_page_config(page_title="Image Filter Lab", page_icon="📸", layout="wide")
st.title(APP_TITLE)
st.caption("Upload one or more images, choose a filter, and download results. Optimized for web deployment.")

# ---------------------------
# Helpers
# ---------------------------
def load_pil(file) -> Image.Image:
    img = Image.open(file)
    # Auto-apply EXIF orientation from cameras/phones
    try:
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")
    return img

def pil_to_cv(img: Image.Image) -> np.ndarray:
    """PIL (RGB/RGBA) -> OpenCV RGB ndarray (H,W,3)"""
    if img.mode == "RGBA":
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img).convert("RGB")
    return np.array(img)  # RGB

def cv_to_pil(arr: np.ndarray) -> Image.Image:
    """OpenCV RGB ndarray -> PIL RGB (safe for display)"""
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")

def ensure_rgb(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    if arr.shape[2] == 4:
        return cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
    return arr

def resize_max(img: Image.Image, max_dim: int) -> Image.Image:
    w, h = img.size
    m = max(w, h)
    if m <= max_dim:
        return img
    scale = max_dim / m
    nw, nh = int(w * scale), int(h * scale)
    return img.resize((nw, nh), Image.LANCZOS)

# ---------------------------
# Core filters (existing)
# ---------------------------
def f_grayscale(rgb: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2RGB)

def f_sepia(rgb: np.ndarray, strength: float = 1.0) -> np.ndarray:
    M = np.array([[0.393, 0.769, 0.189],
                  [0.349, 0.686, 0.168],
                  [0.272, 0.534, 0.131]])
    out = rgb.astype(np.float32).dot(M.T)
    out = np.clip(out, 0, 255)
    if strength < 1.0:
        out = (strength * out + (1 - strength) * rgb)
    return out

def f_threshold(rgb: np.ndarray, thresh: int = 128) -> np.ndarray:
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    _, bw = cv2.threshold(g, thresh, 255, cv2.THRESH_BINARY)
    return cv2.cvtColor(bw, cv2.COLOR_GRAY2RGB)

def f_brightness_contrast(rgb: np.ndarray, brightness: int = 0, contrast: int = 0) -> np.ndarray:
    beta = brightness * 255 / 100.0
    alpha = 1.0 + (contrast / 100.0)
    out = cv2.convertScaleAbs(rgb, alpha=alpha, beta=beta)
    return out

def f_gaussian_blur(rgb: np.ndarray, radius: int = 3) -> np.ndarray:
    k = max(1, radius * 2 + 1)
    if k % 2 == 0:
        k += 1
    return cv2.GaussianBlur(rgb, (k, k), sigmaX=0)

def f_motion_blur(rgb: np.ndarray, ksize: int = 9, angle_deg: int = 0) -> np.ndarray:
    ksize = max(3, ksize | 1)
    kernel = np.zeros((ksize, ksize), dtype=np.float32)
    kernel[ksize // 2, :] = 1.0 / ksize
    M = cv2.getRotationMatrix2D((ksize / 2 - 0.5, ksize / 2 - 0.5), angle_deg, 1.0)
    kernel = cv2.warpAffine(kernel, M, (ksize, ksize))
    s = kernel.sum()
    kernel /= s if s != 0 else 1
    return cv2.filter2D(rgb, -1, kernel)

def f_unsharp(rgb: np.ndarray, amount: float = 1.0, radius: int = 3) -> np.ndarray:
    k = max(1, radius * 2 + 1)
    if k % 2 == 0:
        k += 1
    blur = cv2.GaussianBlur(rgb, (k, k), 0)
    out = cv2.addWeighted(rgb, 1 + amount, blur, -amount, 0)
    return out

def f_canny(rgb: np.ndarray, t1: int = 100, t2: int = 200) -> np.ndarray:
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(g, t1, t2)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

def f_hsv_adjust(rgb: np.ndarray, hue_shift: int = 0, sat_scale: float = 1.0, val_scale: float = 1.0) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    h, s, v = cv2.split(hsv)
    h = (h + hue_shift) % 180
    s = np.clip(s * sat_scale, 0, 255)
    v = np.clip(v * val_scale, 0, 255)
    hsv2 = cv2.merge([h, s, v]).astype(np.uint8)
    return cv2.cvtColor(hsv2, cv2.COLOR_HSV2RGB)

def f_bilateral_denoise(rgb: np.ndarray, d: int = 9, sigma_color: int = 75, sigma_space: int = 75) -> np.ndarray:
    return cv2.bilateralFilter(rgb, d, sigma_color, sigma_space)

def f_emboss(rgb: np.ndarray, strength: float = 1.0) -> np.ndarray:
    k = np.array([[-2, -1, 0],
                  [-1,  1, 1],
                  [ 0,  1, 2]], dtype=np.float32)
    out = cv2.filter2D(rgb, -1, k)
    out = np.clip(out + 128 * strength, 0, 255)
    return out

def f_oil_paint(rgb: np.ndarray, levels: int = 8, radius: int = 3) -> np.ndarray:
    if hasattr(cv2, "xphoto") and hasattr(cv2.xphoto, "oilPainting"):  # type: ignore[attr-defined]
        return cv2.xphoto.oilPainting(rgb, size=radius, dynRatio=levels)  # type: ignore[attr-defined]
    step = max(1, int(256 / max(2, levels)))
    quant = (rgb // step) * step + step // 2
    return cv2.medianBlur(quant, max(3, radius | 1))

def f_cartoon(rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.medianBlur(gray, 7)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY, 9, 9)
    color = cv2.bilateralFilter(rgb, 9, 250, 250)
    return cv2.bitwise_and(color, color, mask=edges)

def f_warm(rgb: np.ndarray, amount: float = 0.15) -> np.ndarray:
    out = rgb.astype(np.float32)
    out[..., 0] *= (1 - amount)
    out[..., 2] *= (1 + amount)
    return np.clip(out, 0, 255)

def f_cool(rgb: np.ndarray, amount: float = 0.15) -> np.ndarray:
    out = rgb.astype(np.float32)
    out[..., 0] *= (1 + amount)
    out[..., 2] *= (1 - amount)
    return np.clip(out, 0, 255)

def f_vintage(rgb: np.ndarray, sepia_strength: float = 0.6) -> np.ndarray:
    base = f_sepia(rgb, sepia_strength)
    return f_unsharp(base, amount=0.2, radius=3)

def f_custom_kernel(rgb: np.ndarray, k: np.ndarray) -> np.ndarray:
    return cv2.filter2D(rgb, -1, k.astype(np.float32))

# ---------------------------
# NEW: Fun filters 🎉
# ---------------------------
def f_pixelate(rgb: np.ndarray, block: int = 12) -> np.ndarray:
    h, w = rgb.shape[:2]
    small = cv2.resize(rgb, (max(1, w // block), max(1, h // block)), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

def f_posterize(rgb: np.ndarray, levels: int = 6) -> np.ndarray:
    levels = int(np.clip(levels, 2, 16))
    step = 256 // levels
    return ((rgb // step) * step + step // 2).astype(np.uint8)

def f_color_splash(rgb: np.ndarray, hue_center: int = 0, hue_width: int = 10, sat_thresh: int = 30) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    diff = np.abs((h.astype(int) - hue_center + 90) % 180 - 90)  # circular distance
    mask = (diff <= hue_width) & (s >= sat_thresh)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray3 = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    out = gray3.copy()
    out[mask] = rgb[mask]
    return out

def f_vignette(rgb: np.ndarray, strength: float = 0.5) -> np.ndarray:
    h, w = rgb.shape[:2]
    y, x = np.ogrid[-1:1:h*1j, -1:1:w*1j]
    r = np.sqrt(x*x + y*y)
    mask = 1 - np.clip(strength * (r**2), 0, 0.9)
    out = rgb.astype(np.float32) * mask[..., None]
    return np.clip(out, 0, 255)

def f_glitch(rgb: np.ndarray, slices: int = 6, max_shift: int = 25, channel_shift: int = 10) -> np.ndarray:
    out = rgb.copy()
    h, w = out.shape[:2]
    ys = sorted(random.sample(range(1, h-1), k=min(slices-1, max(1, h//60)))) + [h-1]
    prev = 0
    for y in ys:
        shift = random.randint(-max_shift, max_shift)
        out[prev:y, :] = np.roll(out[prev:y, :], shift, axis=1)
        prev = y
    for c in range(3):
        out[:, :, c] = np.roll(out[:, :, c], random.randint(-channel_shift, channel_shift), axis=1)
    return np.clip(out, 0, 255)

def f_face_mosaic(rgb: np.ndarray, block: int = 15, scale_factor: float = 1.2, min_neighbors: int = 5) -> np.ndarray:
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors)
    out = rgb.copy()
    for (x, y, w, h) in faces:
        roi = out[y:y+h, x:x+w]
        roi_pix = f_pixelate(roi, block=block)
        out[y:y+h, x:x+w] = roi_pix
    return out

def face_count(rgb: np.ndarray, scale_factor: float = 1.2, min_neighbors: int = 5) -> int:
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors)
    if isinstance(faces, tuple):
        return 0
    return 0 if faces is None else len(faces)

def dominant_hue(rgb: np.ndarray, sat_thresh: int = 30) -> int:
    """Return 0..179 hue with highest histogram count (ignoring low-saturation pixels)."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, s, _ = cv2.split(hsv)
    mask = (s >= sat_thresh).astype(np.uint8)
    hist = cv2.calcHist([h], [0], mask, [180], [0, 180])
    return int(hist.argmax())

# ---------------------------
# Re-apply helper for batch
# ---------------------------
def apply_settings(arr: np.ndarray, settings: dict) -> np.ndarray:
    if not settings:
        return arr
    mode = settings.get("mode")
    kind = settings.get("kind")

    if mode == "Basic":
        if kind == "Grayscale": return f_grayscale(arr)
        if kind == "Sepia": return f_sepia(arr, settings["strength"])
        if kind == "B&W Threshold": return f_threshold(arr, settings["thresh"])
        if kind == "Brightness/Contrast": return f_brightness_contrast(arr, settings["brightness"], settings["contrast"])
        if kind == "Gaussian Blur": return f_gaussian_blur(arr, settings["radius"])
        if kind == "Motion Blur": return f_motion_blur(arr, settings["ksize"], settings["angle"])
        if kind == "Sharpen": return f_unsharp(arr, settings["amount"], settings["radius"])
        if kind == "Edge (Canny)": return f_canny(arr, settings["t1"], settings["t2"])

    elif mode == "Advanced":
        if kind == "HSV Adjust": return f_hsv_adjust(arr, settings["hue_shift"], settings["sat_scale"], settings["val_scale"])
        if kind == "Denoise (Bilateral)": return f_bilateral_denoise(arr, settings["d"], settings["sigma_color"], settings["sigma_space"])
        if kind == "Emboss": return f_emboss(arr, settings["strength"])
        if kind == "Oil Paint": return f_oil_paint(arr, settings["levels"], settings["radius"])
        if kind == "Cartoon": return f_cartoon(arr)

    elif mode == "Presets":
        if kind == "Warm": return f_warm(arr, settings["amount"])
        if kind == "Cool": return f_cool(arr, settings["amount"])
        if kind == "Vintage": return f_vintage(arr, settings["sepia_strength"])

    elif mode == "Custom 3×3":
        return f_custom_kernel(arr, np.array(settings["kernel"], dtype=np.float32))

    elif mode == "Fun":
        if kind == "Pixelate": return f_pixelate(arr, settings["block"])
        if kind == "Posterize": return f_posterize(arr, settings["levels"])
        if kind == "Color Splash": return f_color_splash(arr, settings["hue_center"], settings["hue_width"], settings["sat_thresh"])
        if kind == "Vignette": return f_vignette(arr, settings["strength"])
        if kind == "Glitch": return f_glitch(arr, settings["slices"], settings["max_shift"], settings["channel_shift"])
        if kind == "Face Mosaic": return f_face_mosaic(arr, settings["block"], settings["scale_factor"], settings["min_neighbors"])
        if kind == "Meme Caption": return f_meme_caption(arr, settings["top"], settings["bottom"])

    return arr

# ---------------------------
# Sidebar: Upload & global opts
# ---------------------------
with st.sidebar:
    st.header("Files & Options")
    files = st.file_uploader("Upload image(s)", type=["png", "jpg", "jpeg", "webp"], accept_multiple_files=True)
    max_dim = st.slider("Max display/process size (px)", 512, 3072, 1600, step=128,
                        help="Large images are automatically resized for speed & memory.")
    st.markdown("---")
    st.caption("Tip: Use batch mode at the bottom to process all uploads and download a ZIP.")

with st.sidebar:
    st.markdown("---")
    st.subheader("Orientation")
    rot = st.selectbox("Rotate (°)", [0, 90, 180, 270], index=0)
    flip_h = st.checkbox("Flip horizontally", value=False)
    flip_v = st.checkbox("Flip vertically", value=False)

# No images yet
if not files:
    st.info("Upload one or more images to get started.")
    st.stop()

# Build list of PIL images (resized)
pil_images, names = [], []
for f in files:
    try:
        img = load_pil(f)
        img = resize_max(img, max_dim)
        pil_images.append(img)
        names.append(Path(f.name).name)
    except Exception:
        st.warning(f"Could not read: {getattr(f, 'name', 'file')}. Skipping.")

if not pil_images:
    st.error("No valid images.")
    st.stop()

# Active image selector
active_idx = 0 if len(pil_images) == 1 else st.selectbox(
    "Choose image",
    list(range(len(pil_images))),
    format_func=lambda i: names[i],
    key="active_img"
)

# Apply manual orientation AFTER selection (and after EXIF auto-transpose from load)
src_pil = pil_images[active_idx]
if rot:
    src_pil = src_pil.rotate(int(rot), expand=True)
if flip_h:
    src_pil = ImageOps.mirror(src_pil)
if flip_v:
    src_pil = ImageOps.flip(src_pil)

# Convert once, after orientation
src = ensure_rgb(pil_to_cv(src_pil))

# ---------------------------
# Filter controls
# ---------------------------
st.subheader("Filter & Controls")
col_controls, col_preview = st.columns([1, 2], vertical_alignment="top")

settings = {}  # current choice for batch reuse
out = src.copy()

with col_controls:
    mode = st.radio("Mode", ["Basic", "Advanced", "Presets", "Custom 3×3", "Fun"],
                    help="Choose a filter group to reveal its controls.")
    settings["mode"] = mode

    if mode == "Basic":
        basic = st.selectbox("Filter", [
            "Grayscale", "Sepia", "B&W Threshold",
            "Brightness/Contrast", "Gaussian Blur",
            "Motion Blur", "Sharpen", "Edge (Canny)"])
        settings["kind"] = basic
        if basic == "Grayscale":
            out = f_grayscale(src)
        elif basic == "Sepia":
            strength = st.slider("Strength", 0.0, 1.0, 0.9, 0.05); settings.update(strength=strength); out = f_sepia(src, strength)
        elif basic == "B&W Threshold":
            thr = st.slider("Threshold", 0, 255, 128); settings.update(thresh=thr); out = f_threshold(src, thr)
        elif basic == "Brightness/Contrast":
            b = st.slider("Brightness", -100, 100, 0); c = st.slider("Contrast", -100, 100, 0)
            settings.update(brightness=b, contrast=c); out = f_brightness_contrast(src, b, c)
        elif basic == "Gaussian Blur":
            r = st.slider("Radius", 1, 25, 3); settings.update(radius=r); out = f_gaussian_blur(src, r)
        elif basic == "Motion Blur":
            k = st.slider("Kernel size", 3, 51, 15, step=2); ang = st.slider("Angle (deg)", -90, 90, 0)
            settings.update(ksize=k, angle=ang); out = f_motion_blur(src, k, ang)
        elif basic == "Sharpen":
            amt = st.slider("Amount", 0.0, 2.0, 0.8, 0.05); rad = st.slider("Radius", 1, 20, 3)
            settings.update(amount=amt, radius=rad); out = f_unsharp(src, amt, rad)
        else:
            t1 = st.slider("Threshold 1", 0, 500, 100); t2 = st.slider("Threshold 2", 0, 500, 200)
            settings.update(t1=t1, t2=t2); out = f_canny(src, t1, t2)

    elif mode == "Advanced":
        adv = st.selectbox("Filter", ["HSV Adjust", "Denoise (Bilateral)", "Emboss", "Oil Paint", "Cartoon"])
        settings["kind"] = adv
        if adv == "HSV Adjust":
            hue = st.slider("Hue shift (°)", -90, 90, 0); sat = st.slider("Saturation ×", 0.0, 3.0, 1.0, 0.05); val = st.slider("Value ×", 0.0, 3.0, 1.0, 0.05)
            settings.update(hue_shift=hue // 2, sat_scale=sat, val_scale=val); out = f_hsv_adjust(src, hue_shift=hue // 2, sat_scale=sat, val_scale=val)
        elif adv == "Denoise (Bilateral)":
            d = st.slider("Diameter", 3, 21, 9, step=2); sc = st.slider("Sigma Color", 10, 250, 75); ss = st.slider("Sigma Space", 10, 250, 75)
            settings.update(d=d, sigma_color=sc, sigma_space=ss); out = f_bilateral_denoise(src, d=d, sigma_color=sc, sigma_space=ss)
        elif adv == "Emboss":
            s = st.slider("Strength", 0.0, 2.0, 1.0, 0.05); settings.update(strength=s); out = f_emboss(src, s)
        elif adv == "Oil Paint":
            levels = st.slider("Levels", 2, 32, 8); radius = st.slider("Radius", 1, 15, 3)
            settings.update(levels=levels, radius=radius); out = f_oil_paint(src, levels=levels, radius=radius)
        else:
            settings["kind"] = "Cartoon"; out = f_cartoon(src)

    elif mode == "Presets":
        preset = st.selectbox("Style", ["Warm", "Cool", "Vintage"]); amt = st.slider("Intensity", 0.0, 1.0, 0.5, 0.05)
        settings["kind"] = preset
        if preset == "Warm":
            settings.update(amount=amt * 0.3); out = f_warm(src, amount=amt * 0.3)
        elif preset == "Cool":
            settings.update(amount=amt * 0.3); out = f_cool(src, amount=amt * 0.3)
        else:
            settings.update(sepia_strength=amt); out = f_vintage(src, sepia_strength=amt)

    elif mode == "Custom 3×3":
        st.caption("Enter a 3×3 kernel. Enable normalization to auto-scale by sum.")
        k_vals, cols = [], st.columns(3)
        for r in range(3):
            row = []
            for c in range(3):
                row.append(cols[c].number_input(f"k{r}{c}", value=0.0 if (r, c) != (1, 1) else 1.0, step=0.1, key=f"k_{r}_{c}"))
            k_vals.append(row)
        k = np.array(k_vals, dtype=np.float32)
        norm = st.checkbox("Normalize by sum (if nonzero)", value=True)
        if norm and abs(k.sum()) > 1e-6:
            k = k / k.sum()
        out = f_custom_kernel(src, k)
        settings["kind"] = "Custom 3×3"; settings["kernel"] = k.tolist()

    else:  # Fun
        fun = st.selectbox("Effect", ["Pixelate", "Posterize", "Color Splash", "Vignette", "Glitch", "Face Mosaic", "Meme Caption"])
        settings["kind"] = fun
        if fun == "Pixelate":
            block = st.slider("Block size", 8, 60, 20); settings.update(block=block); out = f_pixelate(src, block)
        elif fun == "Posterize":
            levels = st.slider("Levels", 2, 16, 6); settings.update(levels=levels); out = f_posterize(src, levels)
        elif fun == "Color Splash":
            guess = dominant_hue(src)
            hue = st.slider("Target hue (0–179)", 0, 179, guess)
            width = st.slider("Hue width", 4, 40, 18)
            sat = st.slider("Saturation threshold", 0, 255, 25)
            settings.update(hue_center=hue, hue_width=width, sat_thresh=sat)
            out = f_color_splash(src, hue_center=hue, hue_width=width, sat_thresh=sat)
            st.caption(f"Auto-picked hue ≈ **{guess}°** (you can adjust).")
        elif fun == "Vignette":
            strength = st.slider("Strength", 0.0, 1.0, 0.7, 0.05); settings.update(strength=strength); out = f_vignette(src, strength)
        elif fun == "Glitch":
            slices = st.slider("Slices", 3, 20, 8)
            max_shift = st.slider("Max shift (px)", 5, 120, 40)
            chs = st.slider("Channel shift (px)", 0, 60, 18)
            settings.update(slices=slices, max_shift=max_shift, channel_shift=chs)
            out = f_glitch(src, slices, max_shift, chs)
        elif fun == "Face Mosaic":
            block = st.slider("Block size", 8, 40, 15)
            sf = st.slider("Scale factor", 1.05, 1.5, 1.2, 0.01)
            mn = st.slider("Min neighbors", 3, 10, 5)
            settings.update(block=block, scale_factor=float(sf), min_neighbors=int(mn))
            out = f_face_mosaic(src, block, float(sf), int(mn))
            cnt = face_count(src, float(sf), int(mn))
            st.caption(f"Detected **{cnt}** face(s).")
        else:  # Meme Caption
            top = st.text_input("Top text", value="")
            bottom = st.text_input("Bottom text", value="")
            settings.update(top=top, bottom=bottom)
            out = f_meme_caption(src, top, bottom)

# persist current settings for batch use
st.session_state["last_settings"] = settings

with col_preview:
    st.markdown("**Preview**")
    c1, c2 = st.columns(2)
    c1.image(cv_to_pil(src), caption="Original", use_container_width=True)
    c2.image(cv_to_pil(out), caption="Processed", use_container_width=True)

    processed_pil = cv_to_pil(out)
    buf = io.BytesIO()
    processed_pil.save(buf, format="PNG")
    buf.seek(0)
    st.download_button(
        "⬇️ Download processed PNG",
        data=buf,
        file_name=f"{Path(names[active_idx]).stem}_filtered.png",
        mime="image/png",
        use_container_width=True
    )

st.divider()

# ---------------------------
# Batch processing
# ---------------------------
st.subheader("Batch process all uploads (same settings)")
st.caption("Applies the current filter settings to every uploaded image and returns a ZIP.")

if st.button("Process all & download ZIP", type="primary"):
    current = st.session_state.get("last_settings", {})
    zbuf = io.BytesIO()
    with zipfile.ZipFile(zbuf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for i, pimg in enumerate(pil_images):
            arr = ensure_rgb(pil_to_cv(pimg))
            try:
                out_i = apply_settings(arr, current)
            except Exception:
                out_i = arr
            pil_i = cv_to_pil(out_i)
            out_name = f"{Path(names[i]).stem}_filtered.png"
            b = io.BytesIO()
            pil_i.save(b, format="PNG")
            b.seek(0)
            zf.writestr(out_name, b.read())

    zbuf.seek(0)
    st.download_button("⬇️ Download ZIP", data=zbuf, file_name="filtered_images.zip",
                       mime="application/zip", use_container_width=True)
