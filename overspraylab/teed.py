"""Learned edge detection (TEED, "Tiny and Efficient Edge Detector", Soria et al. 2023) for the
"teed" key-detail method.

The network (BIPED checkpoint 7 of github.com/xavysp/TEED, MIT licence, see
`edge_model/LICENSE_TEED.txt`) was exported once to ONNX by `tools/export_teed_onnx.py`
(dynamic image size, fused output, raw logits). The 250 kB model ships with the code in
`edge_model/`: nothing to download. It runs with onnxruntime when installed (about twice as fast),
otherwise with OpenCV's DNN module (no other dependency).
"""
from __future__ import annotations

import threading
from collections import OrderedDict
from hashlib import blake2b
from pathlib import Path

import cv2
import numpy as np

MODEL = Path(__file__).resolve().parent / "edge_model" / "teed_biped7.onnx"
MEAN_BGR = np.array([103.939, 116.779, 123.68], np.float32)     # training mean (BIPED), BGR order

_lock = threading.Lock()                 # an OpenCV DNN net is not thread-safe
_run = None                              # input (1x3xHxW float32) -> logits (1x1xHxW)
_cache: OrderedDict = OrderedDict()      # logits of the last images (the preview recomputes often)


class TeedError(RuntimeError):
    pass


def _load():
    """Inference function of the model: onnxruntime if available, else OpenCV DNN."""
    if not MODEL.is_file():
        raise TeedError("Modèle de contours appris (TEED) introuvable : overspraylab/edge_model/teed_biped7.onnx")
    try:
        import onnxruntime as ort
    except ImportError:
        ort = None
    if ort is not None:
        try:
            sess = ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])
            return lambda x: sess.run(None, {"input": x})[0]
        except Exception:                  # unusable onnxruntime: OpenCV below
            pass
    try:
        net = cv2.dnn.readNetFromONNX(str(MODEL))
    except cv2.error as e:
        raise TeedError(f"Modèle de contours appris (TEED) illisible ({e})") from e

    def run(x):
        net.setInput(x)
        return net.forward()
    return run


def logits(bgr: np.ndarray) -> np.ndarray:
    """Edge logits of a BGR uint8 image, same size (float32; sigmoid = edge probability; the
    background is around -0.2, edges above ~0.5)."""
    global _run
    key = (blake2b(np.ascontiguousarray(bgr).tobytes(), digest_size=16).hexdigest(), bgr.shape)
    with _lock:
        if key in _cache:
            _cache.move_to_end(key)
            return _cache[key]
        if _run is None:
            _run = _load()
        h, w = bgr.shape[:2]
        H, W = (h + 7) // 8 * 8, (w + 7) // 8 * 8          # the network needs multiples of 8
        x = cv2.copyMakeBorder(bgr, 0, H - h, 0, W - w, cv2.BORDER_REFLECT).astype(np.float32) - MEAN_BGR
        out = np.ascontiguousarray(_run(np.ascontiguousarray(x.transpose(2, 0, 1)[None]))[0, 0, :h, :w], np.float32)
        _cache[key] = out
        while len(_cache) > 4:
            _cache.popitem(last=False)
        return out
