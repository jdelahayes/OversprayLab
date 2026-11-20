"""Serveur web local OversprayLab.  Lancer : python app.py  puis ouvrir http://127.0.0.1:5000"""
from __future__ import annotations

import base64
import json
import re
import threading
import time
import uuid
import zipfile
from dataclasses import fields
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_file, send_from_directory
import io

from overspraylab import pipeline as pl
from overspraylab import effects as fxmod
from overspraylab import semantic
from overspraylab import teed
from overspraylab import projects
from overspraylab.svg_export import _LANG, export_zip, project_files

ROOT = Path(__file__).parent
MATERIALS_FILE = ROOT / "overspraylab" / "materials.json"

app = Flask(__name__, static_folder=str(ROOT / "static"))
# a project folder opened from the disk is sent whole (snapshots with their session dumps)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024 * 1024

# In-memory sessions (local single-user application)
SESSIONS: dict[str, dict] = {}
LOCK = threading.Lock()

PREP_KEYS = ("support_w_mm", "support_h_mm", "px_per_mm", "detail", "n_layers", "color_mode", "skip_support",
             "pre_auto_levels", "pre_clahe", "pre_gamma", "pre_contrast", "pre_shadows", "pre_highlights",
             "pre_black", "pre_white", "pre_curve", "manual_thresholds", "thresholds", "pre_saturation", "pre_sharpen", "pre_remove_bg", "protect_lines", "hue_weight", "color_vividness",
             "flat_auto", "flat_tolerance", "flat_tolerances")


def materials() -> dict:
    return json.loads(MATERIALS_FILE.read_text(encoding="utf-8"))


# ---------------- User-saved presets ----------------
USER_PRESETS_FILE = ROOT / "presets_utilisateur.json"
PRESET_KINDS = ("support", "material", "palette", "sheet", "machine", "mat")


def load_user_presets() -> dict:
    try:
        data = json.loads(USER_PRESETS_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        data = {}
    return {k: dict(data.get(k, {})) for k in PRESET_KINDS}


def save_user_presets(data: dict) -> None:
    tmp = USER_PRESETS_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(USER_PRESETS_FILE)  # atomic write


@app.get("/api/presets")
def api_presets():
    return jsonify(load_user_presets())


@app.post("/api/presets/<kind>")
def api_preset_save(kind):
    if kind not in PRESET_KINDS:
        return jsonify(error="Type de réglage inconnu"), 404
    body = request.get_json(force=True)
    name = str(body.get("name", "")).strip()[:60]
    values = body.get("values")
    if not name or not isinstance(values, dict):
        return jsonify(error="Nom ou valeurs manquants"), 400
    with LOCK:
        data = load_user_presets()
        data[kind][name] = values
        save_user_presets(data)
    return jsonify(data)


@app.post("/api/presets/<kind>/rename")
def api_preset_rename(kind):
    if kind not in PRESET_KINDS:
        return jsonify(error="Type de réglage inconnu"), 404
    body = request.get_json(force=True)
    old = str(body.get("old", "")).strip()
    new = str(body.get("new", "")).strip()[:60]
    overwrite = bool(body.get("overwrite", False))
    with LOCK:
        data = load_user_presets()
        if old not in data[kind]:
            return jsonify(error=f"« {old} » n'existe pas"), 404
        if not new:
            return jsonify(error="Nouveau nom manquant"), 400
        if new != old and new in data[kind] and not overwrite:
            return jsonify(error=f"« {new} » existe déjà", exists=True), 409
        data[kind][new] = data[kind].pop(old)
        save_user_presets(data)
    return jsonify(data)


@app.delete("/api/presets/<kind>/<path:name>")
def api_preset_delete(kind, name):
    if kind not in PRESET_KINDS:
        return jsonify(error="Type de réglage inconnu"), 404
    with LOCK:
        data = load_user_presets()
        data[kind].pop(name, None)
        save_user_presets(data)
    return jsonify(data)


def png_b64(img: np.ndarray, max_side: int = 1100) -> str:
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    if s < 1:
        img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".png", img)
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()


PREVIEW_MAX = 1100  # longest side of the preview images (px)


def rgba_b64(color_bgr, alpha: np.ndarray) -> str:
    """RGBA PNG: plain layer colour, transparency = mask (0..1)."""
    h, w = alpha.shape
    img = np.empty((h, w, 4), np.uint8)
    img[..., :3] = np.asarray(color_bgr, np.uint8)
    img[..., 3] = np.clip(alpha * 255 + 0.5, 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".png", img, [cv2.IMWRITE_PNG_COMPRESSION, 6])
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()


def hex_to_bgr(h: str):
    h = h.lstrip("#")
    return (int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16))


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/api/materials")
def api_materials():
    return jsonify(materials())


@app.get("/api/effects")
def api_effects():
    """Catalogue of the layer effects (labels, parameters, limits, defaults)."""
    return jsonify([{"type": k, **d} for k, d in fxmod.catalogue().items()])


@app.post("/api/upload")
def api_upload():
    f = request.files.get("image")
    if not f:
        return jsonify(error="Aucune image reçue"), 400
    raw = f.read()
    try:
        img = pl.decode_image(raw)
    except ValueError as e:
        return jsonify(error=str(e)), 400
    return jsonify(new_session(img, raw, f.filename or "image"))


def new_session(img: np.ndarray, raw: bytes, filename: str, original: tuple[bytes, str] | None = None) -> dict:
    """Creates a session; the original file is kept as is for the project export.
    `original`: (bytes, extension) of the source image when `raw` is a reduced version of it."""
    sid = uuid.uuid4().hex
    ext = Path(filename).suffix.lower() or ".png"
    with LOCK:
        SESSIONS[sid] = {"img": img, "name": Path(filename).stem, "raw": raw,
                         "filename": Path(filename).name, "ext": ext,
                         # source image as loaded, never reduced (project folder)
                         "orig_raw": original[0] if original else raw, "orig_ext": original[1] if original else ext,
                         "reduced": original is not None}
    h, w = img.shape[:2]
    return dict(id=sid, width=w, height=h, gray=pl.is_grayscale(img), original=png_b64(img),
                filename=Path(filename).name, project_name=Path(filename).stem)


def project_name(v, default: str) -> str:
    """Project name usable as a file name (zip): no path or reserved characters."""
    v = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', " ", str(v or "")).strip().strip(".")
    return re.sub(r"\s+", " ", v)[:80] or default


@app.post("/api/downscale")
def api_downscale():
    """Reduces the session image (too large for its size on the support) to width x height:
    painted masks and subject cut-out follow, caches are dropped, and the reduced image replaces
    the original in the project export."""
    data = request.get_json(force=True)
    sess = SESSIONS.get(data.get("id", ""))
    if not sess:
        return jsonify(error="Session inconnue, rechargez l'image"), 404
    h0, w0 = sess["img"].shape[:2]
    try:
        w, h = int(data.get("width")), int(data.get("height"))
    except (TypeError, ValueError):
        return jsonify(error="Taille invalide"), 400
    if not (1 <= w < w0 and 1 <= h < h0):
        return jsonify(error="La nouvelle taille doit être plus petite que l'image"), 400
    with LOCK:
        img = cv2.resize(sess["img"], (w, h), interpolation=cv2.INTER_AREA)
        sess["img"] = img
        def fit(m):
            return cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        for k in ("mask", "silmask", "subject", "flatmask"):
            if sess.get(k) is not None:
                sess[k] = fit(sess[k])
        for zid, m in (sess.get("zmasks") or {}).items():
            sess["zmasks"][zid] = fit(m)
        for zid, m in (sess.get("flatforce") or {}).items():
            sess["flatforce"][zid] = fit(m)
        if sess.get("fg") is not None:
            sess["fg"] = fit(sess["fg"].astype(np.uint8)) > 0
            sess["fg_ver"] = sess.get("fg_ver", 0) + 1
        for k in ("prep", "prep_key", "reco", "sem"):
            sess.pop(k, None)
        jpeg = sess.get("ext") in (".jpg", ".jpeg")
        ok, buf = cv2.imencode(".jpg" if jpeg else ".png", img, [cv2.IMWRITE_JPEG_QUALITY, 95] if jpeg else [])
        sess["raw"], sess["ext"] = buf.tobytes(), (sess["ext"] if jpeg else ".png")
        sess["reduced"] = True           # the source image itself stays in sess["orig_raw"]
    return jsonify(width=w, height=h, original=png_b64(img))


@app.post("/api/recommend")
def api_recommend():
    data = request.get_json(force=True)
    sess = SESSIONS.get(data.get("id", ""))
    if not sess:
        return jsonify(error="Session inconnue, rechargez l'image"), 404
    mode = data.get("color_mode", "auto")
    skip = bool(data.get("skip_support", True))
    p = pl.Params.from_dict(data.get("crop", {}))
    x0, y0, x1, y1 = pl.crop_rect(p, sess["img"].shape)
    src = pl.preprocess_image(sess["img"][y0:y1, x0:x1], p)   # recommendation on the preprocessed image
    gray = pl.is_grayscale(src) if mode == "auto" else mode == "gray"
    key = (gray, skip, x0, y0, x1, y1, p.pre_auto_levels, p.pre_clahe, p.pre_gamma, p.pre_contrast, p.pre_shadows,
           p.pre_highlights, p.pre_black, p.pre_white, p.pre_curve, p.pre_saturation, p.pre_sharpen)
    cache = sess.setdefault("reco", {})
    if key not in cache:
        cache[key] = pl.recommend_layers(src, gray, skip)
    return jsonify(cache[key])


MASK_KEYS = {"outline": "mask", "silhouette": "silmask", "flat": "flatmask"}


@app.post("/api/mask")
def api_mask():
    """Receives a painted mask (transparent PNG, at the size of the source image preview).
    kind "outline" (default): areas where key details are searched; kind "silhouette": area whose
    outline the outline stencil draws; kind "flat": flat zones (each painted blob takes its main tone;
    zone "auto", or "t<n>": zones forced into the layer of tone n)."""
    data = request.get_json(force=True)
    sess = SESSIONS.get(data.get("id", ""))
    if not sess:
        return jsonify(error="Session inconnue, rechargez l'image"), 404
    key = MASK_KEYS.get(data.get("kind") or "outline", "mask")
    zone = str(data.get("zone") or "z1")
    if key == "mask" and zone != "z1":       # key-detail zone other than the first: own mask
        if not ZONE_ID.match(zone):
            return jsonify(error="Zone inconnue"), 400
        store = sess.setdefault("zmasks", {})
    elif key == "flatmask" and zone not in ("z1", "auto"):   # flat zones forced into a layer
        if not FLAT_ID.match(zone):
            return jsonify(error="Zone inconnue"), 400
        store = sess.setdefault("flatforce", {})
    else:
        store, zone = sess, key
    png = data.get("png")
    if key == "flatmask":                    # flat zones change the tones: new preparation
        sess["flat_ver"] = sess.get("flat_ver", 0) + 1
    if not png:
        store.pop(zone, None)
        return jsonify(coverage=0.0)
    try:
        raw = base64.b64decode(png.split(",", 1)[-1])
        m = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
    except (ValueError, cv2.error):
        m = None
    if m is None:
        return jsonify(error="Masque illisible"), 400
    alpha = m[..., 3] if (m.ndim == 3 and m.shape[2] == 4) else (m if m.ndim == 2 else m[..., 2])
    if not (alpha > 0).any():
        store.pop(zone, None)
        return jsonify(coverage=0.0)
    h, w = sess["img"].shape[:2]
    full = cv2.resize(alpha, (w, h), interpolation=cv2.INTER_LINEAR)
    store[zone] = np.where(full > 127, 255, 0).astype(np.uint8)
    return jsonify(coverage=round(100 * float((store[zone] > 0).mean()), 1))


ZONE_ID = re.compile(r"^z\d{1,3}$")
FLAT_ID = re.compile(r"^t\d{1,2}$")       # flat zones forced into the layer of tone n


def flat_forced(sess) -> dict:
    """Flat zones forced into a layer: {tone index: mask (full source image)}."""
    return {int(k[1:]): m for k, m in (sess.get("flatforce") or {}).items() if m is not None}


def zone_masks(sess) -> dict:
    """Painted masks of the key-detail zones (full source image): the first zone's is `mask`."""
    out = dict(sess.get("zmasks") or {})
    if sess.get("mask") is not None:
        out["z1"] = sess["mask"]
    return out


def zone_regions(sess, lay, shape) -> dict:
    """Zone masks brought to the computed image area (zones without a mask are absent)."""
    x0, y0, x1, y1 = lay.src_rect
    H, W = shape
    return {zid: cv2.resize(m[y0:y1, x0:x1], (W, H), interpolation=cv2.INTER_NEAREST) > 0
            for zid, m in zone_masks(sess).items()}


def subject_overlay(sess) -> str | None:
    """Cut-out preview, at the size of the source image preview: darkened background (transparent elsewhere)."""
    fg = sess.get("fg")
    if fg is None:
        return None
    h, w = fg.shape
    s = min(1.0, 1100 / max(h, w))
    small = cv2.resize(fg.astype(np.uint8) * 255, (max(1, round(w * s)), max(1, round(h * s))),
                       interpolation=cv2.INTER_AREA)
    return rgba_b64((15, 15, 15), (255 - small).astype(np.float32) / 255.0 * 0.72)


def set_subject(sess, painted, exact: bool = False) -> dict:
    """Stores the painted subject area and computes the cut-out: GrabCut around a roughly painted
    area, or the mask as is when it comes from the object selection (exact outline)."""
    if painted is None or not (painted > 0).any():
        sess.pop("subject", None)
        sess.pop("fg", None)
        exact = False
    else:
        sess["subject"] = painted
        sess["fg"] = (painted > 0) if exact else pl.grabcut_subject(sess["img"], painted)
    sess["subject_exact"] = exact
    sess["fg_ver"] = sess.get("fg_ver", 0) + 1
    return subject_info(sess)


def subject_info(sess) -> dict:
    """State of the subject mask and its cut-out, for the browser."""
    painted, fg, exact = sess.get("subject"), sess.get("fg"), bool(sess.get("subject_exact"))
    return dict(coverage=round(100 * float((painted > 0).mean()), 1) if sess.get("subject") is not None else 0.0,
                subject_pct=round(100 * float(fg.mean()), 1) if fg is not None else None, exact=exact,
                overlay=subject_overlay(sess))


@app.post("/api/subject")
def api_subject():
    """Receives the area painted around the subject (transparent PNG, source image preview size)
    and returns the computed cut-out (removed background = bare support)."""
    data = request.get_json(force=True)
    sess = SESSIONS.get(data.get("id", ""))
    if not sess:
        return jsonify(error="Session inconnue, rechargez l'image"), 404
    png = data.get("png")
    painted = None
    if png:
        try:
            m = cv2.imdecode(np.frombuffer(base64.b64decode(png.split(",", 1)[-1]), np.uint8), cv2.IMREAD_UNCHANGED)
        except (ValueError, cv2.error):
            m = None
        if m is None:
            return jsonify(error="Masque illisible"), 400
        alpha = m[..., 3] if (m.ndim == 3 and m.shape[2] == 4) else (m if m.ndim == 2 else m[..., 2])
        h, w = sess["img"].shape[:2]
        painted = np.where(cv2.resize(alpha, (w, h), interpolation=cv2.INTER_LINEAR) > 127, 255, 0).astype(np.uint8)
    with LOCK:
        out = set_subject(sess, painted, bool(data.get("exact")))
    return jsonify(out)


def semantic_state(sess) -> dict:
    """Image analysed by the selection model (computed once per session; concurrent calls wait)."""
    with LOCK:
        lock = sess.setdefault("sem_lock", threading.Lock())
    with lock:
        if sess.get("sem") is None:
            sess["sem"] = semantic.embed(sess["img"])
        return sess["sem"]


@app.post("/api/semantic/prepare")
def api_semantic_prepare():
    """Analyses the image in advance (and downloads the model the first time), so that the first
    click on an object is immediate."""
    data = request.get_json(force=True)
    sess = SESSIONS.get(data.get("id", ""))
    if not sess:
        return jsonify(error="Session inconnue, rechargez l'image"), 404
    t = time.perf_counter()
    try:
        semantic_state(sess)
    except semantic.SemanticError as e:
        return jsonify(error=str(e)), 503
    return jsonify(ok=True, seconds=round(time.perf_counter() - t, 2))


@app.post("/api/semantic/select")
def api_semantic_select():
    """Mask of the object under the click: grey PNG at the size of the mask canvas (w, h).
    points = [[x, y, label], …] in fractions of the image, label 1 = object, 0 = not the object."""
    data = request.get_json(force=True)
    sess = SESSIONS.get(data.get("id", ""))
    if not sess:
        return jsonify(error="Session inconnue, rechargez l'image"), 404
    try:
        w, h = int(data["w"]), int(data["h"])
        points = [(float(x), float(y), int(l)) for x, y, l in data["points"]][:16]
        level = int(data.get("level", 0))
        sens = float(data.get("sensitivity", 50))
        grow = grow_args(data.get("grow"))
    except (KeyError, TypeError, ValueError):
        return jsonify(error="Requête de sélection invalide"), 400
    if not points or not (0 < w <= 4096 and 0 < h <= 4096):
        return jsonify(error="Requête de sélection invalide"), 400
    try:
        m = semantic.select(semantic_state(sess), points, (w, h), level, sens)
    except semantic.SemanticError as e:
        return jsonify(error=str(e)), 503
    if grow:
        m = grow_on_canvas(sess, m, grow)
    ok, buf = cv2.imencode(".png", m.astype(np.uint8) * 255)
    return jsonify(png="data:image/png;base64," + base64.b64encode(buf.tobytes()).decode(),
                   pct=round(100 * float(m.mean()), 1))


def grow_args(g) -> dict | None:
    """Mask enlargement settings from the browser (distances in pixels of the source image)."""
    if not g:
        return None
    out = dict(rim_px=float(g.get("rim_px", 0)) if g.get("rim") else 0.0,
               dark_l=min(100.0, max(0.0, float(g.get("dark", 35)))),
               margin_px=min(200.0, max(0.0, float(g.get("margin_px", 0)))))
    out["rim_px"] = min(200.0, max(0.0, out["rim_px"]))
    return out if out["rim_px"] > 0 or out["margin_px"] > 0 else None


def grow_on_canvas(sess, m: np.ndarray, g: dict) -> np.ndarray:
    """Enlarges a mask at the size of the mask canvas (preview of the source image)."""
    h, w = m.shape
    k = w / sess["img"].shape[1]                     # canvas pixels per source image pixel
    img = cv2.resize(sess["img"], (w, h), interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_LINEAR)
    return pl.grow_mask(m, img, g["rim_px"] * k, g["dark_l"], g["margin_px"] * k)


@app.post("/api/mask/grow")
def api_mask_grow():
    """Enlarges the mask shown in the Masks tab (transparent PNG, canvas size): dark outline around
    the shape and / or margin. Returns the grey mask at the same size."""
    data = request.get_json(force=True)
    sess = SESSIONS.get(data.get("id", ""))
    if not sess:
        return jsonify(error="Session inconnue, rechargez l'image"), 404
    try:
        g = grow_args(data.get("grow"))
        raw = base64.b64decode(str(data["png"]).split(",", 1)[-1])
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
    except (KeyError, TypeError, ValueError, cv2.error):
        return jsonify(error="Requête d'agrandissement invalide"), 400
    if img is None:
        return jsonify(error="Masque illisible"), 400
    if not g:
        return jsonify(error="Rien à agrandir : cochez « Inclure le liseré sombre » ou choisissez une marge"), 400
    alpha = img[..., 3] if (img.ndim == 3 and img.shape[2] == 4) else (img if img.ndim == 2 else img[..., 2])
    m0 = alpha > 127
    m = grow_on_canvas(sess, m0, g)
    ok, buf = cv2.imencode(".png", m.astype(np.uint8) * 255)
    return jsonify(png="data:image/png;base64," + base64.b64encode(buf.tobytes()).decode(),
                   pct=round(100 * float(m.mean()), 1), added=round(100 * float((m & ~m0).mean()), 2))


@app.post("/api/prepreview")
def api_prepreview():
    """Instant preprocessing preview (without computing the stencils): visible part of the image,
    preprocessed at preview size and placed on the support, + luminance histogram."""
    data = request.get_json(force=True)
    sess = SESSIONS.get(data.get("id", ""))
    if not sess:
        return jsonify(error="Session inconnue, rechargez l'image"), 404
    p = pl.Params.from_dict(data.get("params", {}))
    lay, _ = pl.resolve_layout(p, sess["img"].shape)
    ppm = p.px_per_mm
    SW, SH = round(p.support_w_mm * ppm), round(p.support_h_mm * ppm)
    k = min(1.0, PREVIEW_MAX / max(SW, SH))
    size = (max(1, round(SW * k)), max(1, round(SH * k)))
    vx0, vy0, vx1, vy1 = lay.vis_mm
    r = (round(vx0 * ppm * k), round(vy0 * ppm * k), round(vx1 * ppm * k), round(vy1 * ppm * k))
    r = (r[0], r[1], min(size[0], max(r[0] + 1, r[2])), min(size[1], max(r[1] + 1, r[3])))
    x0, y0, x1, y1 = lay.src_rect
    source = cv2.resize(sess["img"][y0:y1, x0:x1], (r[2] - r[0], r[3] - r[1]), interpolation=cv2.INTER_AREA)
    part = pl.preprocess_image(source, p, ppm * k)        # same sharpening radius in mm as in the full computation
    if part is source:
        part = source.copy()
    fg = None
    if p.pre_remove_bg and sess.get("fg") is not None:
        fg = cv2.resize(sess["fg"][y0:y1, x0:x1].astype(np.uint8) * 255, (part.shape[1], part.shape[0]),
                        interpolation=cv2.INTER_LINEAR) > 127
    curve_hist = pl.tone_histogram(source, p, fg)
    support = pl.hex_to_bgr(p.support_color).astype(np.uint8)
    if fg is not None:
        part[~fg] = support
    out = np.empty((size[1], size[0], 3), np.uint8)
    out[:] = support
    out[r[1]:r[3], r[0]:r[2]] = part
    L = cv2.cvtColor(part, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32) * (100.0 / 255.0)
    Lh = L[fg] if fg is not None and fg.any() else L.ravel()
    hist = np.bincount(np.clip(Lh, 0, 99.999).astype(np.int32), minlength=100).astype(np.float64)
    hist = (hist / max(1.0, hist.max())).round(4).tolist()
    return jsonify(image=png_b64(out), size=list(size), histogram=hist, curve_histogram=curve_hist)


SILHOUETTE_MASK_NAMES = {"subject": "sujet", "outline": "contours (détails clés)", "silhouette": "silhouette"}


def subject_area(sess, lay, shape, source: str = "subject"):
    """Area whose outline is drawn (option "silhouette"), brought to the computed image area, or None:
    the subject cut-out, the key-detail mask or the dedicated silhouette mask."""
    if source == "outline":            # key-detail mask: every zone painted (union of their masks)
        zm = [m for m in zone_masks(sess).values() if m is not None]
        src = np.maximum.reduce(zm) if zm else None
    else:
        src = {"subject": sess.get("fg"), "silhouette": sess.get("silmask")}.get(source)
    if src is None:
        return None
    x0, y0, x1, y1 = lay.src_rect
    H, W = shape
    m = cv2.resize((src[y0:y1, x0:x1] > 0).astype(np.uint8) * 255, (W, H), interpolation=cv2.INTER_LINEAR) > 127
    return m if m.any() and not m.all() else None


@app.post("/api/process")
def api_process():
    """Computes the stencils. `run` identifies the computation for /api/process/cancel; a cancelled
    one answers {"cancelled": true} and leaves the last complete result of the session in place."""
    data = request.get_json(force=True)
    sess = SESSIONS.get(data.get("id", ""))
    if not sess:
        return jsonify(error="Session inconnue, rechargez l'image"), 404
    run = data.get("run")
    try:
        with pl.cancellable(None if run is None else lambda: sess.get("calc_cancel") == run):
            return compute(sess, data)
    except pl.Cancelled:
        return jsonify(cancelled=True)


@app.post("/api/process/cancel")
def api_process_cancel():
    """Stops a computation replaced by a newer one (it ends at its next check)."""
    data = request.get_json(force=True)
    sess = SESSIONS.get(data.get("id", ""))
    if not sess:
        return jsonify(error="Session inconnue, rechargez l'image"), 404
    sess["calc_cancel"] = data.get("run")
    return jsonify(ok=True)


def compute(sess, data):
    p = pl.Params.from_dict(data.get("params", {}))
    lay, warnings = pl.resolve_layout(p, sess["img"].shape)
    t0 = time.perf_counter()
    key = (tuple(getattr(p, k) for k in PREP_KEYS) + (lay.src_rect, tuple(round(v, 3) for v in lay.vis_mm))
           + ((sess.get("fg_ver", 0),) if p.pre_remove_bg else ()) + (sess.get("flat_ver", 0),))
    if sess.get("prep_key") != key:
        sess["prep"] = pl.prepare(sess["img"], p, lay, sess.get("fg") if p.pre_remove_bg else None,
                                  sess.get("flatmask"), flat_forced(sess))
        sess["prep_key"] = key
    # Key-detail mask (painted in the Masks tab, outline mask), brought to the computed image area
    regions = zone_regions(sess, lay, sess["prep"].labels.shape) if p.outline else {}
    sess["prep"].subject = subject_area(sess, lay, sess["prep"].labels.shape, p.outline_subject_mask)
    if p.outline_subject and sess["prep"].subject is None:
        warnings.append(f"Silhouette : le masque {SILHOUETTE_MASK_NAMES[p.outline_subject_mask]} est vide "
                        f"(onglet Masques)")
    # Only the paints changed (and they do not change the cut): same geometry, new colours
    geom = ({f.name: getattr(p, f.name) for f in fields(p) if f.name != "layer_colors"},
            (sess["prep"], sess.get("mask"), sess.get("silmask"), sess.get("fg"),
             *[m for _, m in sorted((sess.get("zmasks") or {}).items())]))
    old = sess.get("geom")
    recolored = (sess.get("result") is not None and old is not None and pl.colors_only(p)
                 and old[0] == geom[0] and all(a is b for a, b in zip(old[1], geom[1])))
    try:
        src_map = (lay.src_rect, (sess["img"].shape[1], sess["img"].shape[0]))
        res = (pl.recolor(sess["result"], p) if recolored
               else pl.build(sess["prep"], p, zone_regions=regions, src_map=src_map))
    except teed.TeedError as e:          # learned edges: model missing or unreadable
        return jsonify(error=str(e)), 503
    if p.stencil_style == "line" and not res.layers:
        warnings.append("Trait seul : aucun trait (cochez les détails clés, la silhouette ou le coloriage)")
    sheet = hex_to_bgr((data.get("material") or {}).get("sheet_color", "#c8d3dc"))
    # Preview layers (support area): layer colour + transparency = openings.
    # The browser stacks them itself -> hiding a layer or toggling the blur is instant.
    # Everything is projected straight to the preview size (a large support costs no more).
    SW, SH = res.support_px
    k = min(1.0, PREVIEW_MAX / max(SW, SH))
    size = (max(1, round(SW * k)), max(1, round(SH * k)))
    sig = max(0.3, p.spray_mm * p.px_per_mm)
    (sx0, sy0, sx1, sy1), (ox, oy) = pl.support_overlap(res)
    dst = (round(sx0 * k), round(sy0 * k), round(sx1 * k), round(sy1 * k))

    def to_preview(arr):
        out = np.zeros((size[1], size[0]), np.float32)
        if dst[2] > dst[0] and dst[3] > dst[1]:
            part = arr[sy0 + oy:sy1 + oy, sx0 + ox:sx1 + ox]
            out[dst[1]:dst[3], dst[0]:dst[2]] = cv2.resize(part, (dst[2] - dst[0], dst[3] - dst[1]),
                                                           interpolation=cv2.INTER_AREA)
        return out

    # Images of the previous answer reused when only paints changed: overlays and blurred masks
    # do not depend on the colour, a layer view only on its own colour (kept: entries used now)
    old_cache = sess.get("png_cache", {}) if recolored else {}
    cache = {}

    def cached(key, make):
        pl.check_cancel()
        cache[key] = old_cache[key] if key in old_cache else make()
        return cache[key]

    layers = []
    for L in res.layers:
        hexc = pl.bgr_hex(L.color)
        mask = L.opening.astype(np.float32) / 255.0
        flat = cached((L.index, "flat"), lambda: to_preview(mask))
        soft = cached((L.index, "soft"), lambda: to_preview(cv2.GaussianBlur(mask, (0, 0), sig) * 0.97))
        # same without the bridges (painted over, as after retouching), for the preview option
        if L.bridges:
            nb = pl.opening_without_bridges(res, L).astype(np.float32) / 255.0
            flat_nb = cached((L.index, "flat_nb"), lambda: to_preview(nb))
            soft_nb = cached((L.index, "soft_nb"), lambda: to_preview(cv2.GaussianBlur(nb, (0, 0), sig) * 0.97))
        layers.append({
            "index": L.index,
            "color": pl.bgr_hex(L.color),
            "tone": L.tone,     # tone (index in `palette`) painted by this layer; None for the outline layer
            "sources": list(L.sources),   # line style: where the lines of this stencil come from
            "group": L.group,             # outline stencil: its outline layer (A, B...)
            # automatic bridges (sheet mm), so that the manual bridge tool can pick one to remove
            "auto_bridges": [[round(v / p.px_per_mm, 2) for v in b]
                             for b in L.bridges[getattr(L, "manual_bridges", 0):]],
            # layer view without highlights + bridges and removed details as separate overlays
            # (shown / hidden in the browser)
            "image": cached((L.index, "view", hexc, tuple(sheet)),
                            lambda: png_b64(pl.layer_view(res, L, sheet, bridges=False, lost=False, marks=False), 900)),
            # registration marks + layer-number notches (sheet coordinates, mm), drawn by the browser
            "reg": pl.reg_shapes_json(pl.registration_shapes(res, L.index)[0]),
            **cached((L.index, "overlays"), lambda: {
                k: (png_b64(o, 900) if o is not None and (k != "img_merged" or L.n_merged) else None)
                for k, o in zip(("img_bridges", "img_lost", "img_merged", "img_flat"), pl.layer_overlays(res, L))}),
            # how well the material is held (structural bridges), shown on demand
            "img_support": cached((L.index, "support"),
                                  lambda: (lambda o: png_b64(o, 900) if o is not None else None)(pl.support_map(res, L))),
            "mask_flat": cached((L.index, "mflat", hexc), lambda: rgba_b64(L.color, flat)),
            "mask_sim": cached((L.index, "msim", hexc), lambda: rgba_b64(L.color, soft)),
            "mask_flat_nb": cached((L.index, "mflat_nb", hexc), lambda: rgba_b64(L.color, flat_nb)) if L.bridges else None,
            "mask_sim_nb": cached((L.index, "msim_nb", hexc), lambda: rgba_b64(L.color, soft_nb)) if L.bridges else None,
            **pl.layer_stats(L),
        })
    reg_shapes, reg_warn = pl.registration_shapes(res)
    # Original placed on the support, to compare at scale (raw and preprocessed)
    H, W = res.prepared.labels.shape
    ix, iy = res.prepared.off_px
    r = (round(ix * k), round(iy * k), max(round(ix * k) + 1, round((ix + W) * k)), max(round(iy * k) + 1, round((iy + H) * k)))
    r = (r[0], r[1], min(size[0], r[2]), min(size[1], r[3]))

    def on_support(im):
        out = np.empty((size[1], size[0], 3), np.uint8)
        out[:] = res.support_color.astype(np.uint8)
        out[r[1]:r[3], r[0]:r[2]] = cv2.resize(im, (r[2] - r[0], r[3] - r[1]), interpolation=cv2.INTER_AREA)
        return out
    ref = on_support(res.prepared.raw if res.prepared.raw is not None else res.prepared.resized)
    pre_on = pl.preprocess_active(p) or res.prepared.fg is not None
    sheet_sim = pl.simulate_sheet(res)
    mask_src = [lay.src_rect[0] / sess["img"].shape[1], lay.src_rect[1] / sess["img"].shape[0],
                lay.src_rect[2] / sess["img"].shape[1], lay.src_rect[3] / sess["img"].shape[0]]
    out = dict(
        reference=png_b64(ref),
        # where the computed image area is in the preview images (fractions of the preview) and
        # which part of the source image it shows (fractions of the source): the Masks tab draws
        # the layers under a mask painted on the source image
        mask_map={"img": [ix * k / size[0], iy * k / size[1], (ix + W) * k / size[0], (iy + H) * k / size[1]],
                  "src": mask_src},
        # manual bridges: the computed image area in sheet mm and the part of the source it shows
        # (fractions), to convert a segment drawn on a stencil into source fractions and back;
        # state of each manual bridge (see pl.manual_report)
        bridge_map={"mm": [round(v, 4) for v in (res.img_off_px[0] / p.px_per_mm, res.img_off_px[1] / p.px_per_mm,
                                                 (res.img_off_px[0] + W) / p.px_per_mm,
                                                 (res.img_off_px[1] + H) / p.px_per_mm)],
                    "src": mask_src},
        manual_bridges=list(res.manual),
        reference_pre=png_b64(on_support(pre_view(res))) if pre_on else None,
        subject_used=res.prepared.fg is not None,
        histogram=res.prepared.hist,
        curve_histogram=pl.tone_histogram(res.prepared.raw if res.prepared.raw is not None else res.prepared.resized,
                                          p, res.prepared.fg),
        auto_thresholds=res.prepared.auto_thresholds,
        thresholds_used=res.prepared.thresholds_used,
        palette=[pl.bgr_hex(c) for c in res.prepared.palette],
        preview_size=list(size),
        layers=layers,
        base=pl.bgr_hex(res.support_color),
        support_tone=(pl.bgr_hex(res.prepared.palette[res.support_label])
                      if res.support_label is not None else None),
        support_label=res.support_label,
        warnings=warnings + reg_warn,
        gray=res.prepared.gray,
        sheet_mm=[round(v, 1) for v in res.sheet_mm],
        sheet_rect_mm=[round(v, 2) for v in res.sheet_rect_mm],
        positioning=pl.positioning(res),
        reg=pl.reg_shapes_json(reg_shapes),
        support_mm=[round(v, 1) for v in res.support_mm],
        image_mm=[round(v, 1) for v in lay.img_mm],
        visible_mm=[round(lay.vis_mm[2] - lay.vis_mm[0], 1), round(lay.vis_mm[3] - lay.vis_mm[1], 1)],
        delta_e=round(pl.delta_e(res, sheet_sim), 1),
        lines_added=res.lines_added,
        zone_lines={k: list(v) for k, v in res.zone_lines.items()},   # whole lines: kept, detected, steps
        recolored=recolored,     # colours only: the cut of the previous computation was kept
        seconds=round(time.perf_counter() - t0, 2),
    )
    pl.check_cancel()
    # stored only now: a cancelled computation leaves the previous result whole
    sess["geom"] = geom
    sess["result"] = res
    sess["png_cache"] = cache
    sess["material"] = data.get("material")
    sess["ui"] = data.get("ui")  # full interface state, for the project file
    sess["last_out"] = out       # answer kept for the snapshots (shown again without computing)
    return jsonify(out)


@app.post("/api/outline_preview")
def api_outline_preview():
    """Outline stencil detection preview, over the computed image area of the source image preview
    (Masks tab): detected lines not kept (blue), lines too short (orange), final stroke (red).
    Uses the colour layers of the last computation and the current outline settings."""
    data = request.get_json(force=True)
    sess = SESSIONS.get(data.get("id", ""))
    if not sess:
        return jsonify(error="Session inconnue, rechargez l'image"), 404
    res = sess.get("result")
    if res is None:
        return jsonify(error="Aperçu de la détection : lancez d'abord un calcul"), 409
    p = pl.Params.from_dict(data.get("params", {}))
    lay, _ = pl.resolve_layout(res.params, sess["img"].shape)
    x0, y0, x1, y1 = lay.src_rect
    H, W = res.prepared.labels.shape
    region = zone_regions(sess, lay, (H, W)).get(str(data.get("zone") or "z1"))
    res.prepared.subject = subject_area(sess, lay, (H, W), p.outline_subject_mask)
    try:
        st = pl.outline_detection(res, p, region)
    except teed.TeedError as e:
        return jsonify(error=str(e)), 503
    # size of the computed area in the source image preview (same scale as the mask canvas)
    h, w = sess["img"].shape[:2]
    s = min(1.0, 1100 / max(h, w))
    size = (max(1, round((x1 - x0) * s)), max(1, round((y1 - y0) * s)))
    k = size[0] / W                                   # preview px per computed px
    grow = np.ones((max(1, round(1.2 / k)),) * 2, np.uint8)   # thin lines stay visible once reduced
    rgba = np.zeros((H, W, 4), np.uint8)
    layers = [(st["detected"] & ~st["selected"], (235, 150, 40, 200), True),
              (st["short"], (0, 150, 255, 235), True)]
    if st.get("present") is not None:      # lines merged into the layers: those already cut there
        layers.append((cv2.dilate(st["present"].astype(np.uint8), np.ones((3, 3), np.uint8)) > 0,
                       (200, 60, 190, 225), True))
    layers.append((st["stroke"], (40, 30, 225, 215), False))
    for m, col, thick in layers:
        if thick:
            m = cv2.dilate(m.astype(np.uint8), grow) > 0
        rgba[m] = col
    small = cv2.resize(rgba, size, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".png", small, [cv2.IMWRITE_PNG_COMPRESSION, 6])
    n_lines = cv2.connectedComponents(st["kept"].astype(np.uint8), connectivity=8)[0] - 1
    return jsonify(
        image="data:image/png;base64," + base64.b64encode(buf.tobytes()).decode(),
        rect=[x0 / w, y0 / h, (x1 - x0) / w, (y1 - y0) / h],
        lines=int(n_lines),
        present=int(cv2.connectedComponents(st["present"].astype(np.uint8), connectivity=8)[0] - 1)
        if st.get("present") is not None else None,
        stroke_pct=round(100 * float(st["stroke"].mean()), 1),
    )


def pre_view(res):
    """Preprocessed image; removed background shown in the support colour."""
    im = res.prepared.resized
    if res.prepared.fg is not None:
        im = im.copy()
        im[~res.prepared.fg] = res.support_color.astype(np.uint8)
    return im


@app.post("/api/palette")
def api_palette():
    """Palette applied in the interface (name + colours), written into the project file by the
    export: it can change after the last computation, whose interface state is kept otherwise."""
    data = request.get_json(force=True)
    sess = SESSIONS.get(data.get("id", ""))
    if not sess:
        return jsonify(error="Session inconnue, rechargez l'image"), 404
    pal = data.get("palette")
    if pal is not None:
        cols = [str(c).lower() for c in (pal.get("colors") or [])] if isinstance(pal, dict) else []
        cols = [c for c in cols if re.fullmatch(r"#[0-9a-f]{6}", c)][:16]
        pal = {"key": str(pal.get("key", ""))[:80], "name": str(pal.get("name", ""))[:60], "colors": cols} if cols else None
    if isinstance(sess.get("ui"), dict):
        sess["ui"] = {**sess["ui"], "palette": pal}
    return jsonify(ok=True)


def png_bytes(m: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", m)
    return buf.tobytes()


def project_args(sess) -> dict:
    """Painted masks and interface state of the session, as `project_files` arguments."""
    def one(k):
        return png_bytes(sess[k]) if sess.get(k) is not None else None
    return dict(mask_png=one("mask"), subject_png=one("subject"), sil_png=one("silmask"), flat_png=one("flatmask"),
                flat_force_pngs={zid: png_bytes(m) for zid, m in (sess.get("flatforce") or {}).items()},
                zone_pngs={zid: png_bytes(m) for zid, m in (sess.get("zmasks") or {}).items()},
                ui=sess.get("ui"), subject_exact=bool(sess.get("subject_exact")), image_name=sess.get("filename"))


def req_lang(v) -> str:
    return "fr" if v == "fr" else "en"


@app.get("/api/export/<sid>")
def api_export(sid):
    sess = SESSIONS.get(sid)
    if not sess or "result" not in sess:
        return jsonify(error="Rien à exporter"), 404
    name = project_name(request.args.get("name"), sess["name"])
    d = sess.get("proj_dir")
    folder = projects.folder_files(d) if d and d.is_dir() else None
    data = export_zip(sess["result"], sess.get("material"), name, lang=req_lang(request.args.get("lang")),
                      original=(sess.get("raw"), sess.get("ext", ".png")), folder=folder,
                      extra={"snapshot": sess.get("snap")} if folder else None, **project_args(sess))
    return send_file(io.BytesIO(data), mimetype="application/zip", as_attachment=True,
                     download_name=name.replace(" ", "_") + ".zip")


def restore_session(info: dict, read, image: tuple[bytes, str], original: tuple[bytes, str] | None = None,
                    dump: dict | None = None):
    """New session from a saved project (export zip or snapshot folder).
    read(name) -> bytes | None reads a file of the project; image = (bytes, file name) of the working
    image; original = (bytes, extension) of the source image when the working image is a reduced one;
    dump = session dump of a snapshot (computed result, caches), used as is when given.
    Returns the answer for the browser, or (error message, status)."""
    proj = info.get("projet") or {}
    try:
        img = pl.decode_image(image[0])
    except ValueError as e:
        return str(e), 400
    out = new_session(img, image[0], image[1], original)
    sess = SESSIONS[out["id"]]
    out["project_name"] = project_name(proj.get("nom"), sess["name"])
    h, w = img.shape[:2]

    def load(name):
        """Painted mask (full image size, 0 / 255), or None."""
        data = read(name) if isinstance(name, str) and name else None
        if not data:
            return None
        m = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
        if m is None or not m.any():
            return None
        return np.where(cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST) > 127, 255, 0).astype(np.uint8)

    def b64(m):
        return "data:image/png;base64," + base64.b64encode(png_bytes(m)).decode()

    def cov(m):
        return round(100 * float((m > 0).mean()), 1)

    def single(entry, key):
        m = load(proj.get(entry))
        if m is None:
            return None, 0.0
        sess[key] = m
        return b64(m), cov(m)

    mask_b64, coverage = single("masque", "mask")
    sil_b64, sil_coverage = single("masque_silhouette", "silmask")
    flat_b64, flat_coverage = single("masque_aplats", "flatmask")
    zones, flat_layers = {}, {}
    for entry, pattern, key, dest in (("masques_zones", ZONE_ID, "zmasks", zones),
                                      ("masques_aplats_couches", FLAT_ID, "flatforce", flat_layers)):
        for zid, z_name in (proj.get(entry) or {}).items():
            if not (isinstance(zid, str) and pattern.match(zid)):
                continue
            m = load(z_name)
            if m is not None:
                sess.setdefault(key, {})[zid] = m
                dest[zid] = {"png": b64(m), "coverage": cov(m)}
    subject = None
    m = load(proj.get("masque_sujet"))
    if dump:
        # computed state of the snapshot: masks, cut-out, preparation, result, caches. Its masks
        # are the same as the files' (same snapshot) and replace them so that the "same geometry"
        # test of the next computation (identity of the masks) still holds.
        sess.update(dump)
    if m is not None:
        subject = (subject_info(sess) if dump and sess.get("subject") is not None
                   else set_subject(sess, m, bool(proj.get("masque_sujet_exact"))))
        subject["png"] = b64(m)
    return {**out, "info": info, "mask": mask_b64, "mask_coverage": coverage, "subject": subject,
            "zone_masks": zones,
            "silmask": sil_b64, "silmask_coverage": sil_coverage,
            "flatmask": flat_b64, "flatmask_coverage": flat_coverage, "flat_layer_masks": flat_layers}


def answer(out):
    """JSON answer of restore_session / open_snapshot (or its error)."""
    if isinstance(out, tuple):
        return jsonify(error=out[0]), out[1]
    return jsonify(out)


@app.post("/api/project")
def api_project():
    """Opens an exported project (.zip): original image, masks and full settings. A zip that carries
    its project folder (snapshots) is put into the projects folder and opened at its current
    snapshot."""
    f = request.files.get("project")
    if not f:
        return jsonify(error="Aucun fichier reçu"), 400
    try:
        z = zipfile.ZipFile(io.BytesIO(f.read()))
    except zipfile.BadZipFile:
        return jsonify(error="Ce fichier n'est pas un zip de projet OversprayLab"), 400
    names = z.namelist()
    # infos.json (files without prefix), or …_infos.json (exports of earlier versions); not those
    # of the snapshots in the project folder ("projet/")
    top = [n for n in names if not n.startswith("projet/")]
    info_name = next((n for n in top if n.rsplit("/", 1)[-1] == "infos.json"), None) \
        or next((n for n in top if n.endswith("_infos.json")), None)
    if not info_name:
        return jsonify(error="Fichier infos.json introuvable dans le zip"), 400
    info = json.loads(z.read(info_name).decode("utf-8"))
    proj = info.get("projet") or {}
    if "projet/" + projects.PROJECT_FILE in names:
        folder = {n[len("projet/"):]: z.read(n) for n in names if n.startswith("projet/") and not n.endswith("/")}
        try:
            with LOCK:
                d, mapping = projects.import_folder(folder)
        except (projects.ProjectError, ValueError, KeyError, OSError) as e:
            return jsonify(error=f"Dossier de projet illisible : {e}"), 400
        snap = mapping.get(proj.get("snapshot")) or (projects.snap_ids(d) or [None])[-1]
        if snap:
            return answer(open_snapshot(d, snap))
    img_name = proj.get("image")
    if not img_name or img_name not in names:
        return jsonify(error="Ce zip ne contient pas l'image d'origine (export d'une version "
                             "antérieure) : chargez l'image puis ouvrez le fichier infos.json"), 400
    # earlier versions: no "nom_image", the image is "<image name>_original.<ext>"
    image_name = proj.get("nom_image") or re.sub(r"_original(?=\.[^.]+$)", "", img_name)
    out = restore_session(info, lambda n: z.read(n) if n in names else None, (z.read(img_name), image_name))
    if not isinstance(out, tuple):
        out["project"] = project_state(None, None)
    return answer(out)


@app.post("/api/project/settings")
def api_project_settings():
    """infos.json of an export zip ("Apply the settings of a project"): the browser applies its
    settings to the loaded image."""
    f = request.files.get("project")
    if not f:
        return jsonify(error="Aucun fichier reçu"), 400
    try:
        z = zipfile.ZipFile(io.BytesIO(f.read()))
        top = [n for n in z.namelist() if not n.startswith("projet/")]
        name = next((n for n in top if n.rsplit("/", 1)[-1] == "infos.json"), None) \
            or next((n for n in top if n.endswith("_infos.json")), None)
        if not name:
            return jsonify(error="Fichier infos.json introuvable dans le zip"), 400
        return jsonify(info=json.loads(z.read(name).decode("utf-8")))
    except (zipfile.BadZipFile, ValueError):
        return jsonify(error="Ce fichier n'est pas un zip de projet OversprayLab"), 400


# ---------------- Project folders and snapshots ----------------
def project_state(d, snap) -> dict:
    """Project folder of the session for the browser: snapshots, current one, next number."""
    if d is None:
        return {"dir": None, "snapshots": [], "current": snap, "next": "v0"}
    return {"dir": d.name, "path": str(d.resolve()), "name": projects.project_info(d).get("nom") or d.name,
            "snapshots": projects.snapshots(d), "current": snap, "next": projects.next_id(d)}


# session entries kept in a snapshot's dump (the image comes from the files)
SESSION_DUMP = ("mask", "silmask", "subject", "subject_exact", "fg", "fg_ver", "flat_ver", "flatmask", "zmasks",
                "flatforce", "prep", "prep_key", "result", "geom", "png_cache", "reco", "sem", "material", "ui",
                "last_out")


def open_snapshot(d, snap):
    """New session on a snapshot: its image, masks and settings, and its computed result when its
    session dump can be read (shown again without computing)."""
    try:
        s = projects.snap_dir(d, snap)
    except projects.ProjectError as e:
        return str(e), 404
    pinfo = projects.project_info(d)
    try:
        info = json.loads((s / "infos.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return "Snapshot illisible", 400
    proj = info.get("projet") or {}
    orig_name = pinfo.get("image")
    if not (isinstance(orig_name, str) and "/" not in orig_name and (d / orig_name).is_file()):
        return "Image d'origine du projet introuvable", 400
    original = (d / orig_name).read_bytes()
    image_name = pinfo.get("nom_image") or proj.get("nom_image") or orig_name

    def read(name):
        p = s / name
        return p.read_bytes() if "/" not in name and "\\" not in name and p.is_file() else None

    red = proj.get("image")
    red = read(red) if isinstance(red, str) and red else None      # working image: reduced one
    dump = projects.read_session(s)
    if dump is not None:
        dump = {k: v for k, v in dump.items() if k in SESSION_DUMP}
    out = restore_session(info, read, (red or original, image_name),
                          (original, Path(orig_name).suffix.lower()) if red else None, dump)
    if isinstance(out, tuple):
        return out
    sess = SESSIONS[out["id"]]
    # the name belongs to the project (projet.json, renamed without a new snapshot), not to the snapshot
    out["project_name"] = project_name(pinfo.get("nom"), out["project_name"])
    sess["proj_dir"], sess["snap"] = d, snap
    projects.touch(d)
    out["computed"] = sess.get("last_out") if dump is not None and sess.get("result") is not None else None
    out["project"] = project_state(d, snap)
    return out


@app.get("/api/projects")
def api_projects():
    try:
        limit = max(0, int(request.args.get("limit", 0)))
    except ValueError:
        limit = 0
    return jsonify(projects=projects.list_projects(limit or None), folder=str(projects.ROOT))


@app.post("/api/projects/open_folder")
def api_project_folder():
    """Opens a project folder chosen on the disk ("Open a folder…"): its files, sent by the browser
    with their paths relative to the folder, are put into the projects folder like the project
    folder of an export zip (only the snapshots it lacks if the project is already there); opened
    at its last snapshot."""
    files = {}
    for f in request.files.getlist("files"):
        rel = (f.filename or "").replace("\\", "/").strip("/")
        if rel and ".." not in rel.split("/"):
            files[rel] = f.read()
    if projects.PROJECT_FILE not in files:
        return jsonify(error="Ce dossier n'est pas un dossier de projet OversprayLab (pas de projet.json)"), 400
    try:
        with LOCK:
            d, _ = projects.import_folder(files)
    except (projects.ProjectError, ValueError, KeyError, OSError) as e:
        return jsonify(error=f"Dossier de projet illisible : {e}"), 400
    ids = projects.snap_ids(d)
    if not ids:
        return jsonify(error="Ce dossier de projet ne contient aucune snapshot"), 400
    return answer(open_snapshot(d, ids[-1]))


@app.get("/api/projects/<name>")
def api_project_snapshots(name):
    try:
        d = projects.project_dir(name)
    except projects.ProjectError as e:
        return jsonify(error=str(e)), 404
    return jsonify(project_state(d, None))


@app.get("/api/projects/<name>/<snap>/thumb.jpg")
def api_snapshot_thumb(name, snap):
    try:
        s = projects.snap_dir(projects.project_dir(name), snap)
    except projects.ProjectError as e:
        return jsonify(error=str(e)), 404
    if not (s / "thumb.jpg").is_file():
        return jsonify(error="Pas de vignette"), 404
    return send_file(s / "thumb.jpg", mimetype="image/jpeg", max_age=3600)


# rendered layers of snapshots (Compare tab), keyed by (folder, mtime of the dump): a few, as
# reading a dump decompresses the whole session
_SNAP_RENDERS: dict = {}
RENDER_KEYS = ("preview_size", "support_mm", "base")
RENDER_LAYER_KEYS = ("index", "color", "kind", "mask_flat", "mask_sim", "mask_flat_nb", "mask_sim_nb")


@app.get("/api/projects/<name>/<snap>/render")
def api_snapshot_render(name, snap):
    """Computed preview of a snapshot (its layers as RGBA images, support size and colour),
    composed by the browser like the current preview to compare versions."""
    try:
        s = projects.snap_dir(projects.project_dir(name), snap)
        key = (str(s), (s / projects.SESSION_FILE).stat().st_mtime)
    except projects.ProjectError as e:
        return jsonify(error=str(e)), 404
    except OSError:
        return jsonify(error="Pas de résultat enregistré dans cette snapshot"), 404
    out = _SNAP_RENDERS.get(key)
    if out is None:
        last = (projects.read_session(s) or {}).get("last_out")
        if not isinstance(last, dict) or not last.get("layers"):
            return jsonify(error="Pas de résultat enregistré dans cette snapshot"), 404
        out = {k: last.get(k) for k in RENDER_KEYS}
        out["layers"] = [{k: L.get(k) for k in RENDER_LAYER_KEYS} for L in last["layers"]]
        with LOCK:
            while len(_SNAP_RENDERS) >= 6:
                _SNAP_RENDERS.pop(next(iter(_SNAP_RENDERS)))
            _SNAP_RENDERS[key] = out
    return jsonify(out)


def snapshot_of(name, snap):
    """(project folder, snapshot id) of a URL, or raises ProjectError."""
    d = projects.project_dir(name)
    projects.snap_dir(d, snap)
    return d


@app.post("/api/projects/<name>/<snap>/label")
def api_snapshot_label(name, snap):
    """Renames a snapshot (its label: the only thing of a snapshot that can change)."""
    try:
        d = snapshot_of(name, snap)
    except projects.ProjectError as e:
        return jsonify(error=str(e)), 404
    label = re.sub(r"[\x00-\x1f]+", " ", str(request.get_json(force=True).get("label") or "")).strip()[:80]
    projects.set_label(d, snap, label)
    return jsonify(project_state(d, None))


@app.post("/api/projects/<name>/rename")
def api_project_rename(name):
    """Project renamed in the interface: saved at once in projet.json (not a new version)."""
    try:
        d = projects.project_dir(name)
    except projects.ProjectError as e:
        return jsonify(error=str(e)), 404
    pname = project_name(request.get_json(force=True).get("name"), "")
    if not pname:
        return jsonify(error="Nom de projet vide"), 400
    projects.rename_project(d, pname)
    return jsonify(project_state(d, None))


@app.delete("/api/projects/<name>/<snap>")
def api_snapshot_delete(name, snap):
    """Deletes a snapshot (its folder). Sessions opened on it keep their state, now unsaved."""
    try:
        d = snapshot_of(name, snap)
        with LOCK:
            projects.delete_snapshot(d, snap)
            for sess in SESSIONS.values():
                if sess.get("proj_dir") == d and sess.get("snap") == snap:
                    sess["snap"] = None
    except projects.ProjectError as e:
        return jsonify(error=str(e)), 404
    except OSError as e:
        return jsonify(error=f"Suppression impossible : {e}"), 500
    return jsonify(project_state(d, None))


@app.get("/api/projects/<name>/<snap>/export")
def api_snapshot_export(name, snap):
    """Export zip of one snapshot (stencils + reloadable project holding only this snapshot), made
    from its saved result (computed again if its session dump cannot be read), without touching
    the current session."""
    try:
        d = snapshot_of(name, snap)
    except projects.ProjectError as e:
        return jsonify(error=str(e)), 404
    out = open_snapshot(d, snap)
    if isinstance(out, tuple):
        return answer(out)
    sid = out["id"]
    try:
        sess = SESSIONS[sid]
        if sess.get("result") is None:
            info = out["info"]
            with app.test_request_context(json={"id": sid, "params": info.get("parametres") or {},
                                                "material": info.get("materiau"),
                                                "ui": (info.get("projet") or {}).get("interface")}):
                r = api_process()
            if sess.get("result") is None:
                return r
        pname = project_name(out.get("project_name"), sess["name"])
        data = export_zip(sess["result"], sess.get("material"), pname, lang=req_lang(request.args.get("lang")),
                          original=(sess.get("raw"), sess.get("ext", ".png")),
                          folder=projects.folder_files(d, only=snap), extra={"snapshot": snap},
                          **project_args(sess))
    finally:
        with LOCK:
            SESSIONS.pop(sid, None)
    return send_file(io.BytesIO(data), mimetype="application/zip", as_attachment=True,
                     download_name=f"{pname.replace(' ', '_')}_{snap}.zip")


@app.post("/api/snapshot/open")
def api_snapshot_open():
    data = request.get_json(force=True)
    try:
        d = projects.project_dir(str(data.get("project", "")))
    except projects.ProjectError as e:
        return jsonify(error=str(e)), 404
    return answer(open_snapshot(d, str(data.get("snap", ""))))


@app.post("/api/snapshot")
def api_snapshot():
    """Saves the current version of the session as a new snapshot of its project folder (created
    with the first snapshot). The session must hold an up-to-date computation."""
    data = request.get_json(force=True)
    sess = SESSIONS.get(data.get("id", ""))
    if not sess:
        return jsonify(error="Session inconnue, rechargez l'image"), 404
    res = sess.get("result")
    if res is None:
        return jsonify(error="Snapshot : lancez d'abord un calcul"), 409
    if isinstance(data.get("ui"), dict):
        sess["ui"] = data["ui"]          # view options and palette may have changed since the computation
    name = project_name(data.get("name"), sess["name"])
    label = re.sub(r"[\x00-\x1f]+", " ", str(data.get("label") or "")).strip()[:80]
    with LOCK:
        d = sess.get("proj_dir")
        if d is None or not d.is_dir():
            d = projects.create_project(name, sess["orig_raw"], sess["orig_ext"], sess.get("filename"))
            sess["proj_dir"] = d
        else:
            projects.rename_project(d, name)
    token = _LANG.set(req_lang(data.get("lang")))
    try:
        files = project_files(res, sess.get("material"), name, image_file="reduite",
                              original=(sess["raw"], sess["ext"]) if sess.get("reduced") else None,
                              **project_args(sess))
    finally:
        _LANG.reset(token)
    sim = pl.simulate(res)
    k = min(1.0, 360 / max(sim.shape[:2]))
    small = cv2.resize(sim, (max(1, round(sim.shape[1] * k)), max(1, round(sim.shape[0] * k))),
                       interpolation=cv2.INTER_AREA)
    ok, thumb = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 85])
    h, w = sess["img"].shape[:2]
    last = sess.get("last_out") or {}
    meta = {"libelle": label, "date": projects.now(), "delta_e": last.get("delta_e"), "pochoirs": len(res.layers),
            "image_reduite": bool(sess.get("reduced")), "taille_image": [w, h]}
    dump = {k: sess[k] for k in SESSION_DUMP if k in sess}
    snap = projects.write_snapshot(d, files, meta, thumb.tobytes(), dump)
    sess["snap"] = snap
    return jsonify({**project_state(d, snap), "snap": snap})


if __name__ == "__main__":
    import os
    # OVERSPRAYLAB_PORT: run a second instance (e.g. for tests) next to the usual one on 5000
    port = int(os.environ.get("OVERSPRAYLAB_PORT", "5000"))
    print(f"OversprayLab : http://127.0.0.1:{port}")
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)
