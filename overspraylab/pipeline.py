"""Processing pipeline: image -> cuttable stencil layers.

All physical quantities are in millimetres; the computation runs on a raster
grid of `px_per_mm` pixels per millimetre, then the outlines are vectorised
in mm.
"""
from __future__ import annotations

import contextvars
import io
import json
import math
import re
import os
import threading
from collections import OrderedDict
from concurrent.futures import Future, wait as wait_futures
from hashlib import blake2b
from dataclasses import dataclass, field, fields, replace

import cv2
import numpy as np

from . import effects as fxmod
from . import teed
from . import timings
from .cancel import Cancelled, cancellable, check_cancel  # noqa: F401  (used through this module by app.py)


OUTLINE_METHODS = ("canny", "xdog", "fdog", "ridge", "teed")
OUTLINE_TARGETS = ("stencil", "darkest", "nearest")   # + "L<n>": into the colour layer at position n
_LAYER_DEST = re.compile(r"^L[1-9]\d?$")


def into_layers(d) -> bool:
    """Destination in the colour layers: darkest, nearest tone, or a given layer ("L<n>")."""
    return d in ("darkest", "nearest") or (isinstance(d, str) and bool(_LAYER_DEST.match(d)))
OUTLINE_REFS = ("layers", "effects", "image")
CLOSE_LINES = ("", "median", "outer", "inner")
OUTLINE_CHANNELS = ("light", "colour", "both")
CURVE_MAX_POINTS = 16
TONE_RANGE_STRENGTH = 1.5          # shadows / highlights at 100; above ~1.9 the tones would fold back


@dataclass
class Params:
    # Support (painted surface)
    support_w_mm: float = 210.0
    support_h_mm: float = 297.0
    support_color: str = "#ffffff"
    inner_margin_mm: float = 10.0  # reference margin for "fit to support"
    # Crop of the source image (fractions 0..1)
    crop_x: float = 0.0
    crop_y: float = 0.0
    crop_w: float = 1.0
    crop_h: float = 1.0
    # Layout on the support
    image_scale: float = 100.0     # % of the "fit to support" size
    image_height_factor: float = 1.0  # vertical stretch (1 = original proportions)
    image_pos_x: float = 50.0      # image centre, as % of the support width
    image_pos_y: float = 50.0      # image centre, as % of the support height
    px_per_mm: float = 6.0
    # "Auto" button of the interface: px_per_mm follows the material limits (computed by the
    # client, autoPpm in index.html); kept here so that it is saved with the project. On by
    # default; settings saved before it existed are read as off (setVals in index.html)
    px_per_mm_auto: bool = True
    # Image preprocessing (before simplification)
    pre_auto_levels: bool = False  # automatic levels stretch (luminance)
    pre_clahe: float = 0.0         # local contrast (CLAHE), 0..100
    pre_gamma: float = 1.0         # midtones: > 1 lightens, < 1 darkens
    pre_contrast: float = 0.0      # global contrast (S curve around the midtones), -100..100
    pre_shadows: float = 0.0       # dark tones: > 0 lightens, < 0 darkens, -100..100
    pre_highlights: float = 0.0    # light tones: > 0 lightens, < 0 darkens, -100..100
    pre_black: float = 0.0         # L* (0..100) that becomes black
    pre_white: float = 100.0       # L* (0..100) that becomes white
    pre_curve: str = ""            # tone curve, L* in -> out: "x,y;x,y;..." ("" = none)
    pre_saturation: float = 100.0  # colour saturation, % (100 = unchanged)
    pre_sharpen: float = 0.0       # sharpness (unsharp mask), 0..100
    pre_remove_bg: bool = False    # removed background ("subject" mask + GrabCut): left as bare support
    manual_thresholds: bool = False  # luminance thresholds chosen by hand between tones
    thresholds: str = ""           # L* thresholds (0..100) separated by commas
    # Simplification
    n_layers: int = 3              # number of stencils
    color_mode: str = "auto"       # auto | color | gray
    detail: float = 50.0           # 0 (very simplified) .. 100 (very detailed)
    # "auto": the four settings below follow `detail` (auto_detail); "expert": each one set by hand
    detail_mode: str = "auto"
    smooth_passes: int = 3         # edge-preserving smoothing passes before the tones are computed
    smooth_strength: float = 50.0  # its strength, 0..100 (bilateral filter: colour 15..70, distance 3..9 px)
    simplify_mm: float = 0.95      # radius of the majority filter that simplifies the tone zones, mm
    detail_area_mm2: float = 7.5   # supplement of the minimum area of a piece or hole, mm²
    # weight of the hue (a*, b*) against the luminance when computing the tones (colour mode), %:
    # 0 = luminance only, ~100 = perceptual balance (ΔE76), 40 = historical weighting (default)
    hue_weight: float = 40.0
    # computed tone colours, %: 0 = mean of the pixels (dull: hues and shadows average out),
    # 100 = mean hue and lightness with the saturation of the vivid pixels (75th percentile)
    color_vividness: float = 0.0
    layering: str = "exclusive"    # exclusive | cumulative
    skip_support: bool = True      # the tone closest to the support gets no stencil
    # exclusive layers: what the material limits remove goes to the layers around it
    fill_gaps: bool = True
    # exclusive layers: each spot is painted by one layer only (see exclusive_layers); with this on, a
    # lighter layer may still paint under a darker one, which repaints it (its merges and the material
    # limits may then fill areas of the darker layer, hidden: fewer bridges)
    under_darker: bool = False
    protect_lines: bool = False    # thin strokes (eyes, eyebrows…) escape the majority filter
    # Flat areas (poster look: full flat layers, no interleaving): 0 = off .. 100 = strongest.
    # Islands of a tone whose border with the surrounding area has a low contrast are absorbed by it
    flat_auto: float = 0.0
    # Painted flat zones (Masks tab, "flat" mask): each painted blob takes its majority tone where
    # the image differs from it by less than this tolerance (0 = nothing .. 100 = the whole blob)
    flat_tolerance: float = 50.0
    # tolerance of each flat mask, JSON {"auto": v, "t<tone>": v} (the fields of the Simplification
    # section edit the mask shown); a mask without its own value takes "auto", which takes
    # `flat_tolerance` (settings saved before per-mask tolerances)
    flat_tolerances: str = ""
    # flat masks forced into a layer ("t<tone>", comma-separated) whose pieces only need the minimum
    # area of the material, without the supplement of the level of detail (small painted zones kept)
    flat_exempt: str = ""
    # Outline stencil (key details faded by the airbrush blur)
    outline: bool = False
    outline_method: str = "canny"       # canny (sharp edges) | xdog (drawn line) | ridge (thin lines, centre line)
    # where the lines go: stencil = separate outline stencil painted last;
    # darkest = into the darkest layer; nearest = into the layer of the nearest (darker) tone
    outline_target: str = "stencil"
    # exclusive layers: the key-detail lines cut into a colour layer are removed from the other
    # layers (only their layer paints them); off = they are painted over the lighter layers.
    # Default of the zones (zone key "excl": "" = this setting, "on" / "off"), of the silhouette
    # (key "sil_excl" of outline_zones) and of the outline stencils (group key "excl": the colour
    # layers are then not painted under the stencil's lines)
    outline_exclusive: bool = False
    outline_sensitivity: float = 50.0   # 0 = only the strongest details .. 100 = many details (progressive)
    outline_width_mm: float = 0.8       # line width (at least the material's minimum opening)
    outline_min_length_mm: float = 4.0  # minimum line length
    # line style: key-detail lines interrupted every … mm (0 = continuous), over … mm, distance
    # varying at random by … % (per zone in outline_zones: keys gap_every_mm, gap_mm, gap_jitter)
    outline_gap_every_mm: float = 0.0
    outline_gap_mm: float = 1.5
    outline_gap_jitter: float = 0.0
    # what the key details are compared with: layers = render of the colour layers before their
    # effects (the lines stay put when effects change); effects = render after the effects (the lines
    # make up for what the effects lose); image = the image alone, before the layers are computed
    # (all the lines of the image, chosen by the sensitivity only)
    outline_ref: str = "layers"
    # the sensitivity keeps whole lines (each candidate line kept or left out as a whole) instead of
    # the most important points (a line whose strength varies along it is then cut into pieces)
    outline_whole_lines: bool = False
    # whole lines: detection sensitivity (0..100), how faint a line may be and still count as detected
    outline_line_sens: float = 50.0
    # two lines closer than the minimum material strip: "" = left to the material limits (the strip
    # between them is opened: a black flat area); median = one line in the middle; outer / inner =
    # only the line farther from / closer to the centre of the zone
    outline_close_lines: str = ""
    # what the lines are looked for in: light = lightness (L*); colour = colourfulness (chroma),
    # where a shadow hardly shows while a black / grey line stays sharp; both = either
    outline_channel: str = "light"
    # drawn lines (DoG, FDoG) and TEED: how far, in mm, a kept line is followed along itself where it stays nearly
    # as marked (hysteresis); 0 = point by point. Longer = lines in one piece, but also faint ones
    # extended from a strong start (default = HYST_MM_DOG)
    outline_follow_mm: float = 12.0
    # local contrast (CLAHE, 0..100) of the image the zone's lines are looked for in, on top of the
    # preprocessing: brings out lines in shadows and flat areas without touching the layers
    outline_clahe: float = 0.0
    outline_subject: bool = False       # also draw the outline of a mask (silhouette of the subject)
    outline_subject_mask: str = "subject"   # mask whose outline is drawn: subject | outline | silhouette
    outline_subject_weak: bool = False      # silhouette only where the edge stands out poorly in the render
    outline_subject_de: float = 20.0        # ... i.e. colour difference across the edge (ΔE) below this
    outline_subject_gap_every_mm: float = 0.0   # silhouette interrupted every … mm (0 = continuous)
    outline_subject_gap_mm: float = 1.5     # length of an interruption (material left: a built-in bridge)
    outline_subject_gap_jitter: float = 0.0     # % of random variation of the distance between interruptions
    outline_subject_pos: str = "center"     # stroke against the outline: center | outside | inside
    outline_subject_width_mm: float = 0.8   # stroke width (0, earlier settings: the key-detail width)
    outline_subject_min_length_mm: float = 4.0  # shorter pieces of the silhouette are left out
    outline_auto_color: bool = True     # auto colour = tone with the most contrast against the support
    # Stencil style: fill = flat colour layers (+ optional lines); line = line stencils only, from the
    # key details (`outline`), the silhouette (`outline_subject`) and/or the limits between the
    # tone zones (`line_zones`, colouring-book style); one stencil per line colour.
    stencil_style: str = "fill"
    line_zones: bool = False
    # colouring book: its own settings, independent of the key details and of the silhouette
    line_zones_color: str = ""          # "" = automatic (tone with the most contrast against the support) | "tones" = darker tone of the two zones | #hex
    line_zones_width_mm: float = 0.8    # width of the limits (at least the minimum opening)
    line_zones_min_length_mm: float = 4.0   # shorter limits are left out
    line_zones_gap_every_mm: float = 0.0    # limits interrupted every … mm (0 = continuous)
    line_zones_gap_mm: float = 1.5      # length of an interruption (material left: a built-in bridge)
    line_zones_gap_jitter: float = 0.0  # % of random variation of the distance between interruptions
    line_zones_width_max_mm: float = 0.0    # > width: the width grows with the contrast across the limit up to this (0 = constant)
    line_zones_min_de: float = 0.0      # limits between tones closer than this (ΔE) are not drawn
    line_zones_min_area_mm2: float = 0.0    # smaller zones are absorbed by their neighbours before tracing
    line_zones_smooth_mm: float = 0.0   # smoothing of the zones (Gaussian sigma) before tracing their limits
    line_zones_pos: str = "center"      # stroke against the limit: center | dark (in the darker zone) | light
    line_zones_mask: str = ""           # "" = whole image | subject | silhouette | outline: only inside that mask
    line_zones_frame: bool = False      # also draw the edge of the area coloured (image, mask, subject)
    line_sil_color: str = ""            # "" = colour of the key-detail lines | #hex
    outline_color: str = "#1e1e1e"
    # Paint colours chosen for the layers: "1=#rrggbb;3=#rrggbb" (layer number -> colour).
    # Layers not listed keep their computed tone.
    layer_colors: str = ""
    # Effects stacks of the colour layers (JSON, see overspraylab/effects.py); "" = none
    layer_effects: str = ""
    # Primer stencil (layer 0, sprayed first): what the layers paint, e.g. in white so that the
    # colours stay bright on a dark support (see primer_opening). Shrunk by `primer_inset_mm` (an edge
    # of the primer showing past a colour is far more visible than a colour edge on the support);
    # `primer_lines`: also under the outline stencils (usually dark: left out)
    primer: bool = False
    primer_color: str = "#ffffff"
    primer_inset_mm: float = 0.5
    primer_lines: bool = False
    # Key-detail zones (JSON, see outline_plan); "" = one zone from the outline_* settings above
    outline_zones: str = ""
    # Material constraints
    min_opening_mm: float = 0.8    # minimum width of an opening
    min_strip_mm: float = 1.0      # minimum width of a material strip
    bridge_width_mm: float = 1.5
    min_area_mm2: float = 2.5
    double_bridge_area_mm2: float = 400.0
    min_bridges: int = 1           # minimum number of bridges per island (1 to 4)
    bridges: bool = True
    # Structural bridges (0 = off): largest spacing between two supports along the material, in mm
    # (a part held from one side only, or a long thin strip); see add_struct_bridges
    bridge_spacing_mm: float = 0.0
    # Bridges drawn by hand (JSON list, see parse_manual_bridges); applied before the merges and the
    # automatic bridges, even when the automatic bridges are off
    manual_bridges: str = ""
    # Merging zones instead of bridges (0 = off): see merge_zones
    merge_amount: float = 0.0      # 0..150, progressive scale of the largest visible change accepted
    merge_channel_mm: float = 2.5  # A: channels of opening narrower than this may be closed
    merge_channels: bool = True    # A: close narrow channels
    merge_islands: bool = True     # B: remove small islands
    merge_line_islands: bool = False  # B also for islands touching key-detail lines cut into the layer
    # Vectorisation
    tolerance_mm: float = 0.12
    smooth_curves: bool = True
    # Stencil sheet:
    #   around = support + margin all around (registered on the support corners)
    #   fit    = inside the support, fitted to the design + border (registered on crosses drawn on the support)
    #   (the former "fixed" mode - fixed size centred on the design - is read as "fit")
    sheet_mode: str = "fit"
    margin_mm: float = 15.0
    sheet_border_mm: float = 20.0
    reg_dims: bool = True          # ruler dimensions of the crosses (sheet inside the support)
    sheet_outline: bool = False
    machine_max_mm: float = 305.0  # usable cutting width of the machine
    mat_w_mm: float = 304.8        # cutting mat (0 = no mat: only the width matters)
    mat_h_mm: float = 304.8
    mat_orient: str = "portrait"   # portrait = feed direction into the machine; landscape = mat turned a quarter turn
    # Registration
    reg_corners: int = 4           # 0, 3 (keyed) or 4
    reg_shape: str = "croix"       # equerre | croix | rond
    reg_size_mm: float = 8.0
    layer_notches: bool = True     # notches giving the layer number
    # Preview
    spray_mm: float = 0.4

    def flat_tol(self, zone: str) -> float:
        """Tolerance of a flat mask ("auto" or "t<tone>"): its own value, else the "auto" one."""
        try:
            tols = json.loads(self.flat_tolerances) if self.flat_tolerances else {}
        except ValueError:
            tols = {}
        v = tols.get(zone, tols.get("auto", self.flat_tolerance)) if isinstance(tols, dict) else self.flat_tolerance
        return float(v)

    def detail_settings(self) -> tuple[int, float, float, float]:
        """(smoothing passes, smoothing strength 0..1, majority radius in mm, minimum area supplement
        in mm²): derived from `detail` in auto mode, the expert settings otherwise."""
        if self.detail_mode == "expert":
            return self.smooth_passes, self.smooth_strength / 100.0, self.simplify_mm, self.detail_area_mm2
        return auto_detail(self.detail)

    def flat_exempt_tones(self) -> set:
        """Tones whose forced flat zones ignore the detail supplement of the minimum area."""
        return {int(z[1:]) for z in self.flat_exempt.split(",") if z}

    @classmethod
    def from_dict(cls, d: dict) -> "Params":
        p = cls()
        for f in fields(cls):
            if f.name in d and d[f.name] is not None:
                default = getattr(p, f.name)
                v = d[f.name]
                if isinstance(default, bool):
                    v = v if isinstance(v, bool) else str(v).lower() in ("1", "true", "on", "yes")
                else:
                    v = type(default)(v)
                setattr(p, f.name, v)
        p.n_layers = int(np.clip(p.n_layers, 1, 10))
        p.detail = float(np.clip(p.detail, 0, 100))
        if p.detail_mode not in ("auto", "expert"):
            p.detail_mode = "auto"
        p.smooth_passes = int(np.clip(p.smooth_passes, 0, 6))
        p.smooth_strength = float(np.clip(p.smooth_strength, 0, 100))
        p.simplify_mm = float(np.clip(p.simplify_mm, 0, 5))
        p.detail_area_mm2 = float(np.clip(p.detail_area_mm2, 0, 50))
        p.hue_weight = float(np.clip(p.hue_weight, 0, 300))
        p.color_vividness = float(np.clip(p.color_vividness, 0, 100))
        p.flat_auto = float(np.clip(p.flat_auto, 0, 100))
        p.flat_tolerance = float(np.clip(p.flat_tolerance, 0, 100))
        # canonical form (also the cache key): "auto" always present, values clipped
        try:
            tols = json.loads(p.flat_tolerances) if p.flat_tolerances else {}
        except ValueError:
            tols = {}
        tols = {str(k): float(np.clip(float(v), 0, 100)) for k, v in (tols if isinstance(tols, dict) else {}).items()
                if (k == "auto" or re.fullmatch(r"t\d{1,2}", str(k))) and isinstance(v, (int, float))}
        tols.setdefault("auto", p.flat_tolerance)
        p.flat_tolerance = tols["auto"]
        p.flat_tolerances = json.dumps(dict(sorted(tols.items())))
        p.flat_exempt = ",".join(sorted({z.strip() for z in p.flat_exempt.split(",")
                                         if re.fullmatch(r"t\d{1,2}", z.strip())}, key=lambda z: int(z[1:])))
        if p.outline_subject_mask not in ("subject", "outline", "silhouette"):
            p.outline_subject_mask = "subject"
        if p.stencil_style not in ("fill", "line"):
            p.stencil_style = "fill"
        for k in ("line_zones_color", "line_sil_color"):
            v = str(getattr(p, k) or "").strip().lower()
            ok = v in ("", "tones") if k == "line_zones_color" else v == ""
            if not ok and not (len(v) == 7 and v.startswith("#") and all(c in "0123456789abcdef" for c in v[1:])):
                v = ""
            setattr(p, k, v)
        p.outline_subject_de = float(np.clip(p.outline_subject_de, 1, 100))
        p.outline_subject_gap_every_mm = float(np.clip(p.outline_subject_gap_every_mm, 0, 1000))
        if 0 < p.outline_subject_gap_every_mm < 5:
            p.outline_subject_gap_every_mm = 5.0
        p.outline_subject_gap_mm = float(np.clip(p.outline_subject_gap_mm, 0.3, 20))
        p.outline_subject_gap_jitter = float(np.clip(p.outline_subject_gap_jitter, 0, GAP_JITTER_MAX))
        p.line_zones_width_mm = float(np.clip(p.line_zones_width_mm, 0.2, 10))
        p.line_zones_min_length_mm = float(np.clip(p.line_zones_min_length_mm, 0, 200))
        p.line_zones_gap_every_mm = float(np.clip(p.line_zones_gap_every_mm, 0, 1000))
        if 0 < p.line_zones_gap_every_mm < 5:
            p.line_zones_gap_every_mm = 5.0
        p.line_zones_gap_mm = float(np.clip(p.line_zones_gap_mm, 0.3, 20))
        p.line_zones_gap_jitter = float(np.clip(p.line_zones_gap_jitter, 0, GAP_JITTER_MAX))
        p.line_zones_width_max_mm = float(np.clip(p.line_zones_width_max_mm, 0, 10))
        p.line_zones_min_de = float(np.clip(p.line_zones_min_de, 0, 100))
        p.line_zones_min_area_mm2 = float(np.clip(p.line_zones_min_area_mm2, 0, 10000))
        p.line_zones_smooth_mm = float(np.clip(p.line_zones_smooth_mm, 0, 10))
        if p.line_zones_pos not in ("center", "dark", "light"):
            p.line_zones_pos = "center"
        if p.line_zones_mask not in ("", "subject", "silhouette", "outline"):
            p.line_zones_mask = ""
        if p.outline_subject_pos not in ("center", "outside", "inside"):
            p.outline_subject_pos = "center"
        p.outline_subject_width_mm = float(np.clip(p.outline_subject_width_mm, 0, 20))
        p.outline_subject_min_length_mm = float(np.clip(p.outline_subject_min_length_mm, 0, 200))
        p.outline_gap_every_mm = float(np.clip(p.outline_gap_every_mm, 0, 1000))
        if 0 < p.outline_gap_every_mm < 5:
            p.outline_gap_every_mm = 5.0
        p.outline_gap_mm = float(np.clip(p.outline_gap_mm, 0.3, 20))
        p.outline_gap_jitter = float(np.clip(p.outline_gap_jitter, 0, GAP_JITTER_MAX))
        p.outline_follow_mm = float(np.clip(p.outline_follow_mm, 0, 50))
        p.outline_clahe = float(np.clip(p.outline_clahe, 0, 100))
        p.px_per_mm = float(np.clip(p.px_per_mm, 2, 16))
        p.support_w_mm = float(np.clip(p.support_w_mm, 20, 3000))
        p.support_h_mm = float(np.clip(p.support_h_mm, 20, 3000))
        p.inner_margin_mm = float(np.clip(p.inner_margin_mm, 0, min(p.support_w_mm, p.support_h_mm) / 2 - 5))
        p.crop_x = float(np.clip(p.crop_x, 0, 0.98))
        p.crop_y = float(np.clip(p.crop_y, 0, 0.98))
        p.crop_w = float(np.clip(p.crop_w, 0.02, 1 - p.crop_x))
        p.crop_h = float(np.clip(p.crop_h, 0.02, 1 - p.crop_y))
        p.image_scale = float(np.clip(p.image_scale, 5, 1000))
        p.image_height_factor = float(np.clip(p.image_height_factor, 0.1, 10))
        p.image_pos_x = float(np.clip(p.image_pos_x, -100, 200))
        p.image_pos_y = float(np.clip(p.image_pos_y, -100, 200))
        p.min_bridges = int(np.clip(p.min_bridges, 1, 4))
        p.bridge_spacing_mm = float(np.clip(p.bridge_spacing_mm, 0, 2000))
        if 0 < p.bridge_spacing_mm < p.bridge_width_mm:    # at least the width of a bridge
            p.bridge_spacing_mm = p.bridge_width_mm
        mb = parse_manual_bridges(p.manual_bridges)        # canonical form (also the cache key)
        p.manual_bridges = json.dumps(mb, separators=(",", ":")) if mb else ""
        p.reg_corners = p.reg_corners if p.reg_corners in (0, 3, 4) else 4
        p.sheet_mode = p.sheet_mode if p.sheet_mode in ("around", "fit") else "fit"
        p.mat_orient = p.mat_orient if p.mat_orient in ("portrait", "landscape") else "portrait"
        p.margin_mm = float(np.clip(p.margin_mm, 0, 200))
        p.sheet_border_mm = float(np.clip(p.sheet_border_mm, 3, 200))
        if not p.support_color.startswith("#") or len(p.support_color) != 7:
            p.support_color = "#ffffff"
        if not p.outline_color.startswith("#") or len(p.outline_color) != 7:
            p.outline_color = "#1e1e1e"
        p.outline_sensitivity = float(np.clip(p.outline_sensitivity, 0, 100))
        p.outline_method = p.outline_method if p.outline_method in OUTLINE_METHODS else "canny"
        p.outline_target = p.outline_target if (p.outline_target in OUTLINE_TARGETS or into_layers(p.outline_target)) else "stencil"
        p.outline_ref = p.outline_ref if p.outline_ref in OUTLINE_REFS else "layers"
        p.outline_close_lines = p.outline_close_lines if p.outline_close_lines in CLOSE_LINES else ""
        p.outline_channel = p.outline_channel if p.outline_channel in OUTLINE_CHANNELS else "light"
        p.outline_line_sens = float(np.clip(p.outline_line_sens, 0, 100))
        p.pre_clahe = float(np.clip(p.pre_clahe, 0, 100))
        p.pre_gamma = float(np.clip(p.pre_gamma, 0.3, 3.0))
        p.pre_contrast = float(np.clip(p.pre_contrast, -100, 100))
        p.pre_shadows = float(np.clip(p.pre_shadows, -100, 100))
        p.pre_highlights = float(np.clip(p.pre_highlights, -100, 100))
        p.pre_black = float(np.clip(p.pre_black, 0, 95))
        p.pre_white = float(np.clip(p.pre_white, p.pre_black + 5, 100))
        p.pre_curve = format_curve(parse_curve(p.pre_curve))
        p.pre_saturation = float(np.clip(p.pre_saturation, 0, 300))
        p.pre_sharpen = float(np.clip(p.pre_sharpen, 0, 100))
        p.outline_width_mm = float(np.clip(p.outline_width_mm, 0.2, 10))
        p.merge_amount = float(np.clip(p.merge_amount, 0, 150))
        p.merge_channel_mm = float(np.clip(p.merge_channel_mm, 0.3, 30))
        p.outline_min_length_mm = float(np.clip(p.outline_min_length_mm, 0, 200))
        p.primer_color = _hex(p.primer_color, "#ffffff")
        p.primer_inset_mm = float(np.clip(p.primer_inset_mm, 0, 5))
        return p


@dataclass
class Layout:
    src_rect: tuple    # (x0, y0, x1, y1) source image pixels actually used
    img_mm: tuple      # (width, height) of the whole image (after crop) on the support
    pos_mm: tuple      # (x, y) top-left corner of the whole image on the support
    vis_mm: tuple      # (x0, y0, x1, y1) visible part (inside the support)
    sheet_mm: tuple = (0.0, 0.0, 0.0, 0.0)  # (x0, y0, x1, y1) stencil sheet, support coordinates


def sheet_rect(p: Params, vis_mm: tuple) -> tuple[float, float, float, float]:
    """Stencil sheet in support coordinates (mm), according to the chosen mode."""
    SW, SH = p.support_w_mm, p.support_h_mm
    if p.sheet_mode == "around":
        m = p.margin_mm
        return (-m, -m, SW + m, SH + m)
    vx0, vy0, vx1, vy1 = vis_mm
    b = p.sheet_border_mm
    return (vx0 - b, vy0 - b, vx1 + b, vy1 + b)


def crop_rect(p: Params, img_shape) -> tuple[int, int, int, int]:
    h, w = img_shape[:2]
    x0 = int(round(p.crop_x * w))
    y0 = int(round(p.crop_y * h))
    x1 = max(x0 + 2, min(w, int(round((p.crop_x + p.crop_w) * w))))
    y1 = max(y0 + 2, min(h, int(round((p.crop_y + p.crop_h) * h))))
    return x0, y0, x1, y1


def resolve_layout(p: Params, img_shape) -> tuple[Layout, list[str]]:
    """Places the cropped image on the support.

    Reference size (100 %) = image fitted in the support minus the inner margin.
    Parts overflowing the support are cropped.
    """
    cx0, cy0, cx1, cy1 = crop_rect(p, img_shape)
    ar = (cy1 - cy0) / (cx1 - cx0)
    SW, SH = p.support_w_mm, p.support_h_mm
    avail_w = max(5.0, SW - 2 * p.inner_margin_mm)
    avail_h = max(5.0, SH - 2 * p.inner_margin_mm)
    warnings = []
    fit_w = min(avail_w, avail_h / ar)
    img_w = fit_w * p.image_scale / 100
    img_h = img_w * ar * p.image_height_factor
    x0 = p.image_pos_x / 100 * SW - img_w / 2
    y0 = p.image_pos_y / 100 * SH - img_h / 2
    vx0, vy0 = max(0.0, x0), max(0.0, y0)
    vx1, vy1 = min(SW, x0 + img_w), min(SH, y0 + img_h)
    if vx1 - vx0 < 2 or vy1 - vy0 < 2:
        warnings.append("L'image est entièrement hors du support : elle a été recentrée.")
        x0, y0 = (SW - img_w) / 2, (SH - img_h) / 2
        vx0, vy0 = max(0.0, x0), max(0.0, y0)
        vx1, vy1 = min(SW, x0 + img_w), min(SH, y0 + img_h)
    if (vx0 > x0 + 0.05) or (vy0 > y0 + 0.05) or (vx1 < x0 + img_w - 0.05) or (vy1 < y0 + img_h - 0.05):
        warnings.append("L'image dépasse du support : elle est rognée à ses bords (impression à fond perdu).")
    # Stencil sheet
    sh_rect = sheet_rect(p, (vx0, vy0, vx1, vy1))
    # part of the source matching the visible area
    sx = (cx1 - cx0) / img_w
    sy = (cy1 - cy0) / img_h
    src = (int(round(cx0 + (vx0 - x0) * sx)), int(round(cy0 + (vy0 - y0) * sy)),
           int(round(cx0 + (vx1 - x0) * sx)), int(round(cy0 + (vy1 - y0) * sy)))
    src = (src[0], src[1], max(src[0] + 2, src[2]), max(src[1] + 2, src[3]))
    lay = Layout(src_rect=src, img_mm=(img_w, img_h), pos_mm=(x0, y0), vis_mm=(vx0, vy0, vx1, vy1),
                 sheet_mm=sh_rect)
    sheet_w = sh_rect[2] - sh_rect[0]
    sheet_h = sh_rect[3] - sh_rect[1]
    if p.sheet_mode != "around":
        out = [n for n, d in (("gauche", -sh_rect[0]), ("haut", -sh_rect[1]),
                              ("droit", sh_rect[2] - SW), ("bas", sh_rect[3] - SH)) if d > 0.5]
        if out:
            warnings.append("La feuille déborde du support (bord " + ", ".join(out) + ") : elle sera posée "
                            "à cheval sur le bord ; les croix hors du support ne pourront pas y être tracées.")
    if p.mat_w_mm > 0 and p.mat_h_mm > 0:
        land = p.mat_orient == "landscape"
        # physical dimensions (width = machine direction), then oriented as on the support
        mw, mh = (p.mat_h_mm, p.mat_w_mm) if land else (p.mat_w_mm, p.mat_h_mm)
        mat = (f"tapis de {p.mat_w_mm:g} × {p.mat_h_mm:g} mm".replace(".", ",")
               + (" (en paysage)" if land else ""))
        if p.mat_w_mm > p.machine_max_mm + 1:
            warnings.append(f"Le {mat} est plus large que la largeur de coupe de la machine "
                            f"({p.machine_max_mm:.0f} mm) : il ne passera pas dans la machine.")
        tol = 0.05   # mm: rounding of the scale (0.01 %) and of the crop, nothing physical
        if sheet_w <= mw + tol and sheet_h <= mh + tol:
            pass
        elif sheet_h <= mw + tol and sheet_w <= mh + tol:
            warnings.append(f"La feuille ({sheet_w:.0f} × {sheet_h:.0f} mm) ne tient sur le {mat} qu'en la "
                            f"tournant d'un quart de tour : passez le tapis en "
                            f"{'portrait' if land else 'paysage'} (section Machine de découpe).")
        else:
            fix = ("réduisez la marge autour du support, passez en « feuille dans le support »"
                   if p.sheet_mode == "around" else "réduisez la bordure ou la taille du motif")
            warnings.append(f"La feuille ({sheet_w:.0f} × {sheet_h:.0f} mm) dépasse le {mat} : {fix}, "
                            f"choisissez un tapis plus grand ou découpez en plusieurs morceaux.")
    elif min(sheet_w, sheet_h) > p.machine_max_mm:
        warnings.append(f"La feuille ({sheet_w:.0f} × {sheet_h:.0f} mm) est plus large que la largeur de coupe "
                        f"de la machine ({p.machine_max_mm:.0f} mm), même tournée : il faudra la découper en plusieurs morceaux.")
    elif sheet_w > p.machine_max_mm:
        warnings.append(f"La feuille fait {sheet_w:.0f} mm de large, plus que la largeur de coupe "
                        f"({p.machine_max_mm:.0f} mm) : faites-la passer tournée d'un quart de tour dans la machine.")
    return lay, warnings


def hex_to_bgr(h: str) -> np.ndarray:
    h = h.lstrip("#")
    return np.array((int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16)), np.float32)


# --------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------

def _decode_pillow(data: bytes) -> np.ndarray | None:
    """Fallback decoder (Pillow): AVIF and other formats missing from OpenCV's wheels.
    Returns BGR or BGRA (uint8), or None if the data cannot be read."""
    try:
        from PIL import Image, ImageOps
        im = ImageOps.exif_transpose(Image.open(io.BytesIO(data)))
        alpha = im.mode in ("RGBA", "LA", "PA") or (im.mode == "P" and "transparency" in im.info)
        a = np.asarray(im.convert("RGBA" if alpha else "RGB"))
    except Exception:                  # Pillow missing, unknown or damaged file
        return None
    return cv2.cvtColor(a, cv2.COLOR_RGBA2BGRA if alpha else cv2.COLOR_RGB2BGR)


def decode_image(data: bytes) -> np.ndarray:
    """Decodes an image (transparency handled -> white background)."""
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        img = _decode_pillow(data)          # formats OpenCV does not read (AVIF…)
    if img is None:
        raise ValueError("Image illisible")
    if img.dtype != np.uint8:
        img = cv2.convertScaleAbs(img, alpha=255.0 / max(1, img.max()))
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        a = img[..., 3:4].astype(np.float32) / 255.0
        img = (img[..., :3].astype(np.float32) * a + 255 * (1 - a)).astype(np.uint8)
    # reasonable limit for memory
    h, w = img.shape[:2]
    s = 4000 / max(h, w)
    if s < 1:
        img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    return img


def is_grayscale(img: np.ndarray) -> bool:
    small = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA).astype(np.int16)
    spread = small.max(axis=2) - small.min(axis=2)
    return float(np.percentile(spread, 95)) < 12


def disk(r: float) -> np.ndarray:
    r = max(1, int(round(r)))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def remove_small(mask: np.ndarray, min_px: int, keep_border: bool = False,
                 exempt: np.ndarray | None = None, exempt_px: int = 0) -> np.ndarray:
    """Removes (white) components with an area < min_px. `exempt` (bool): the components that
    overlap it only need `exempt_px`."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return mask
    keep = stats[:, cv2.CC_STAT_AREA] >= min_px
    if exempt is not None and exempt_px < min_px:
        hit = np.zeros(n, bool)
        hit[np.unique(lab[exempt & (mask > 0)])] = True
        keep |= hit & (stats[:, cv2.CC_STAT_AREA] >= exempt_px)
    keep[0] = False
    if keep_border:
        h, w = mask.shape
        x, y, bw, bh = (stats[:, i] for i in range(4))
        touches = (x == 0) | (y == 0) | (x + bw >= w) | (y + bh >= h)
        keep |= touches
        keep[0] = False
    return np.where(keep[lab], 255, 0).astype(np.uint8)


def lab_to_bgr(lab_colors: np.ndarray) -> np.ndarray:
    arr = np.clip(lab_colors, 0, 255).astype(np.uint8).reshape(1, -1, 3)
    return cv2.cvtColor(arr, cv2.COLOR_LAB2BGR).reshape(-1, 3)


def original(prep: "Prepared") -> np.ndarray:
    """Original photo at the working size (without preprocessing): reference of the paint colours."""
    return prep.raw if prep.raw is not None else prep.resized


def vivid_tone(mean: np.ndarray, pixels: np.ndarray, amount: float) -> np.ndarray:
    """Tone colour with livelier saturation: same lightness and hue as `mean` (OpenCV 8-bit Lab),
    chroma raised towards the 75th percentile of the pixels' chroma (`amount` 0..1), kept in gamut."""
    a0, b0 = float(mean[1]) - 128, float(mean[2]) - 128
    c0 = float(np.hypot(a0, b0))
    if amount <= 0 or c0 < 1 or len(pixels) == 0:
        return mean
    sub = pixels[:: max(1, len(pixels) // 50000)]
    c_vivid = float(np.percentile(np.hypot(sub[:, 1] - 128, sub[:, 2] - 128), 75))
    target = c0 + amount * max(0.0, c_vivid - c0)
    for c in np.linspace(target, c0, 9)[:-1]:      # largest chroma that stays inside sRGB
        cand = np.array([mean[0], 128 + a0 * c / c0, 128 + b0 * c / c0], np.float32)
        back = cv2.cvtColor(lab_to_bgr(cand[None]).reshape(1, 1, 3), cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)
        if np.linalg.norm(back - cand) < 3:
            return cand
    return mean


def bgr_hex(c) -> str:
    b, g, r = (int(v) for v in c)
    return f"#{r:02x}{g:02x}{b:02x}"


# --------------------------------------------------------------------------
# 1-2. Preprocessing + quantisation
# --------------------------------------------------------------------------

@dataclass
class Prepared:
    resized: np.ndarray      # resized and preprocessed image (simplification reference)
    labels: np.ndarray       # label map, sorted from lightest (0) to darkest
    palette: np.ndarray      # BGR colours sorted light -> dark
    gray: bool
    off_px: tuple = (0, 0)   # position of the visible part on the support (px)
    raw: np.ndarray | None = None   # same area, original image without preprocessing
    hist: list = None        # L* luminance histogram (100 bins, 0..1) of the smoothed image
    auto_thresholds: list = None    # L* thresholds the automatic computation would have chosen
    thresholds_used: list = None    # thresholds actually applied (manual) or None
    fg: np.ndarray | None = None    # subject (True) / removed background (False), computed image size
    # area whose outline is drawn by the outline (option `outline_subject`): selected subject or
    # another mask, computed image size (set by the server for each computation; None = prep.fg)
    subject: np.ndarray | None = None
    # area the colouring-book limits are limited to (option `line_zones_mask`), computed image size
    # (set by the server for each computation; None = the whole image)
    zones_area: np.ndarray | None = None
    # painted flat zones, computed image size (shown on the layer views): "auto" zones (main tone)
    # and the zones forced into the layer of a tone {tone: mask}
    flat_zones: np.ndarray | None = None
    flat_forced_zones: dict | None = None


def parse_curve(txt: str) -> list[tuple[float, float]]:
    """Points of a tone curve "x,y;x,y;..." (L* 0..100), sorted; [] when invalid or identity."""
    pts = []
    for part in str(txt or "").split(";")[:CURVE_MAX_POINTS]:
        try:
            x, y = (float(v) for v in part.split(","))
        except ValueError:
            return []
        if not (math.isfinite(x) and math.isfinite(y)):
            return []
        pts.append((round(min(100.0, max(0.0, x)), 1), round(min(100.0, max(0.0, y)), 1)))
    pts.sort()
    out = []
    for pt in pts:                     # two points need distinct inputs
        if not out or pt[0] - out[-1][0] >= 0.5:
            out.append(pt)
    if len(out) < 2 or all(abs(x - y) < 0.05 for x, y in out) and out[0][0] == 0 and out[-1][0] == 100:
        return []
    return out


def format_curve(pts: list) -> str:
    return ";".join(f"{x:g},{y:g}" for x, y in pts)


def curve_values(pts: list, v: np.ndarray) -> np.ndarray:
    """Tone curve at `v` (0..1): monotone cubic through the points (no overshoot between them), flat
    outside them. Same interpolation as `curveFn` in the browser, which draws it."""
    x, y = np.array(pts, np.float64).T / 100.0
    h, d = np.diff(x), np.diff(y) / np.diff(x)
    m = np.empty(len(x))
    m[0], m[-1] = d[0], d[-1]
    for i in range(1, len(x) - 1):
        if d[i - 1] * d[i] <= 0:
            m[i] = 0.0
        else:
            w1, w2 = 2 * h[i] + h[i - 1], h[i] + 2 * h[i - 1]
            m[i] = (w1 + w2) / (w1 / d[i - 1] + w2 / d[i])
    u = np.clip(v, x[0], x[-1])
    i = np.clip(np.searchsorted(x, u, side="right") - 1, 0, len(h) - 1)
    t = (u - x[i]) / h[i]
    return ((2 * t ** 3 - 3 * t ** 2 + 1) * y[i] + (t ** 3 - 2 * t ** 2 + t) * h[i] * m[i]
            + (-2 * t ** 3 + 3 * t ** 2) * y[i + 1] + (t ** 3 - t ** 2) * h[i] * m[i + 1])


def tone_active(p: Params) -> bool:
    return bool(abs(p.pre_gamma - 1) > 1e-3 or p.pre_contrast or p.pre_shadows or p.pre_highlights
                or p.pre_black > 0 or p.pre_white < 100 or p.pre_curve)


def tone_lut(p: Params) -> np.ndarray:
    """Luminance table (uint8 -> uint8) of the tone settings, in this order: black and white points,
    shadows, highlights, global contrast, gamma, curve. Computed as a whole, so that several
    settings do not add up rounding steps."""
    v = np.arange(256, dtype=np.float64) / 255.0
    if p.pre_black > 0 or p.pre_white < 100:
        b, w = p.pre_black / 100.0, p.pre_white / 100.0
        v = np.clip((v - b) / (w - b), 0, 1)
    if p.pre_shadows:                  # at most about 16 L* around L* 25; black can only be reached
        v = np.clip(v + p.pre_shadows / 100.0 * TONE_RANGE_STRENGTH * v * (1 - v) ** 3, 0, 1)
    if p.pre_highlights:               # same around L* 75
        v = np.clip(v + p.pre_highlights / 100.0 * TONE_RANGE_STRENGTH * v ** 3 * (1 - v), 0, 1)
    if p.pre_contrast:                 # slope 1 +- c at the middle: stays monotone
        v = np.clip(v - p.pre_contrast / 100.0 * np.sin(2 * np.pi * v) / (2 * np.pi), 0, 1)
    if abs(p.pre_gamma - 1) > 1e-3:
        v = v ** (1.0 / p.pre_gamma)
    pts = parse_curve(p.pre_curve)
    if pts:
        v = curve_values(pts, v)
    return np.clip(v * 255 + 0.5, 0, 255).astype(np.uint8)


def preprocess_active(p: Params) -> bool:
    return bool(p.pre_auto_levels or p.pre_clahe > 0 or tone_active(p)
                or abs(p.pre_saturation - 100) > 1e-3 or p.pre_sharpen > 0)


def tone_histogram(img: np.ndarray, p: Params, fg: np.ndarray | None = None) -> list:
    """L* histogram (100 bins, highest = 1) of the image as it enters the tone curve: what the curve
    editor shows behind the curve. Measured on a reduced copy (`fg`: subject only)."""
    h, w = img.shape[:2]
    s = min(1.0, 400 / max(h, w))
    if s < 1:
        size = (max(1, round(w * s)), max(1, round(h * s)))
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        if fg is not None:
            fg = cv2.resize(fg.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0
    im = preprocess_image(img, replace(p, pre_curve="", pre_sharpen=0.0))
    L = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32) * (100.0 / 255.0)
    L = L[fg] if fg is not None and fg.any() else L.ravel()
    hist = np.bincount(np.clip(L, 0, 99.999).astype(np.int32), minlength=100).astype(np.float64)
    return (hist / max(1.0, hist.max())).round(4).tolist()


def preprocess_image(img: np.ndarray, p: Params, px_per_mm: float | None = None) -> np.ndarray:
    """Auto levels, local contrast (CLAHE), tone settings (`tone_lut`) and sharpening on the
    luminance (Lab); saturation on the a, b colour channels. `px_per_mm`: image scale (sharpening
    radius)."""
    if not preprocess_active(p):
        return img
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L = lab[..., 0]
    if p.pre_auto_levels:
        lo, hi = np.percentile(L, (0.5, 99.5))
        if hi - lo > 5:
            L = np.clip((L.astype(np.float32) - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)
    if p.pre_clahe > 0:
        clahe = cv2.createCLAHE(clipLimit=1.0 + p.pre_clahe / 100.0 * 4.0, tileGridSize=(8, 8))
        L = clahe.apply(L)
    if tone_active(p):
        L = tone_lut(p)[L]
    if p.pre_sharpen > 0:
        # unsharp mask: radius ~0.3 mm at stencil scale (1.5 px on the source image)
        sig = max(0.8, 0.3 * px_per_mm) if px_per_mm else 1.5
        Lf = L.astype(np.float32)
        L = np.clip(Lf + (Lf - cv2.GaussianBlur(Lf, (0, 0), sig)) * (p.pre_sharpen / 100.0 * 1.5),
                    0, 255).astype(np.uint8)
    lab[..., 0] = L
    if abs(p.pre_saturation - 100) > 1e-3:
        ab = lab[..., 1:].astype(np.float32) - 128.0
        lab[..., 1:] = np.clip(ab * (p.pre_saturation / 100.0) + 128.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def grow_mask(mask: np.ndarray, img: np.ndarray, rim_px: float = 0.0, dark_l: float = 35.0,
              margin_px: float = 0.0) -> np.ndarray:
    """Enlarges a mask (bool, same size as `img`, BGR).
    rim_px > 0: the dark outline around the shape (e.g. the black line around a drawn subject, left
    out by the object selection) is added: pixels with L* <= dark_l reached from the mask through
    dark pixels only, at most rim_px away. Then margin_px: plain enlargement all around."""
    m = mask.astype(bool)
    if rim_px >= 0.5 and m.any():
        L = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32) * (100.0 / 255.0)
        reach = cv2.dilate(m.astype(np.uint8), disk(rim_px)) > 0
        allowed = (reach & (L <= dark_l)) | m
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cur = m.astype(np.uint8)
        for _ in range(int(math.ceil(rim_px)) + 1):        # geodesic growth through the dark pixels
            nxt = cv2.dilate(cur, k) & allowed.astype(np.uint8)
            if np.array_equal(nxt, cur):
                break
            cur = nxt
        # pinholes left in the line (light pixels of anti-aliasing or texture) are filled
        added = cv2.morphologyEx(cur & ~m.astype(np.uint8) & 1, cv2.MORPH_CLOSE, k)
        m = m | (added > 0)
    if margin_px >= 0.5 and m.any():
        m = cv2.dilate(m.astype(np.uint8), disk(margin_px)) > 0
    return m


def grabcut_subject(img: np.ndarray, painted: np.ndarray, max_side: int = 700) -> np.ndarray:
    """Cuts out the subject: `painted` (0/255, image size) = area roughly painted around the subject.

    Outside the painted area = sure background; inside = probable subject, refined by GrabCut on a
    downscaled copy, then brought back to the image size. Returns a boolean mask (True = subject).
    """
    h, w = img.shape[:2]
    s = min(1.0, max_side / max(h, w))
    sw, sh = max(1, round(w * s)), max(1, round(h * s))
    small = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
    pm = cv2.resize(painted, (sw, sh), interpolation=cv2.INTER_NEAREST) > 0
    if not pm.any():
        return np.ones((h, w), bool)
    if pm.all():                      # no background designated: nothing to remove
        return np.ones((h, w), bool)
    mask = np.where(pm, cv2.GC_PR_FGD, cv2.GC_BGD).astype(np.uint8)
    # the core of the painted area (more than ~5 % of the image size from its edge) is surely the
    # subject: GrabCut cannot remove it, it only adjusts the band near the stroke
    r = max(3, int(round(0.05 * min(sw, sh))))
    core = cv2.erode(pm.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))) > 0
    mask[core] = cv2.GC_FGD
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(small, mask, None, bgd, fgd, 6, cv2.GC_INIT_WITH_MASK)
        fg = (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)
    except cv2.error:
        fg = pm
    # cleanup: small crumbs removed, small holes filled
    k = np.ones((3, 3), np.uint8)
    fg8 = cv2.morphologyEx(fg.astype(np.uint8) * 255, cv2.MORPH_OPEN, k)
    fg8 = cv2.morphologyEx(fg8, cv2.MORPH_CLOSE, k)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(fg8, connectivity=8)
    if n > 2:                          # keep the non-negligible pieces (>= 1 % of the largest)
        big = stats[1:, cv2.CC_STAT_AREA].max()
        keep = np.zeros(n, bool)
        keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= 0.01 * big
        fg8 = np.where(keep[lab], 255, 0).astype(np.uint8)
    full = cv2.resize(fg8, (w, h), interpolation=cv2.INTER_LINEAR)
    return full > 127


def parse_layer_colors(txt: str) -> dict[int, np.ndarray]:
    out = {}
    for part in str(txt or "").replace(",", ";").split(";"):
        k, _, v = part.partition("=")
        v = v.strip()
        try:
            if len(v) == 7 and v.startswith("#"):
                out[int(k)] = hex_to_bgr(v).astype(np.uint8)
        except ValueError:
            pass
    return out


def parse_tone_colors(txt: str) -> dict[int, np.ndarray]:
    """Paints chosen for the tones of the "line" style (tone<n> stencils): "t3=#rrggbb" entries of
    `layer_colors` (keyed by tone, so they follow the tone whatever the order of the stencils)."""
    out = {}
    for part in str(txt or "").replace(",", ";").split(";"):
        k, _, v = part.partition("=")
        k, v = k.strip(), v.strip()
        if k[:1] == "t" and k[1:].isdigit() and len(v) == 7 and v.startswith("#"):
            try:
                out[int(k[1:])] = hex_to_bgr(v).astype(np.uint8)
            except ValueError:
                pass
    return out


def auto_detail(detail: float) -> tuple[int, float, float, float]:
    """Settings that follow the level of detail (0..100) in auto mode, see Params.detail_settings
    (the client copies them into the expert settings: same formulas in index.html, autoDetail)."""
    f = 1.0 - detail / 100.0
    return 1 + int(round(f * 3)), f, 0.15 + f * 1.6, f * 15.0


def parse_thresholds(txt: str) -> list[float]:
    out = []
    for t in str(txt or "").replace(";", ",").split(","):
        try:
            out.append(float(np.clip(float(t), 0.5, 99.5)))
        except ValueError:
            pass
    return sorted(out)


def prepare(img: np.ndarray, p: Params, lay: Layout, fg_src: np.ndarray | None = None,
            flat_src: np.ndarray | None = None, flat_forced: dict | None = None) -> Prepared:
    """`fg_src`: subject mask at the size of the source image (removed background), or None.
    `flat_src`: painted flat zones at the size of the source image (tone chosen automatically), or
    None. `flat_forced`: {tone index (0 = lightest): painted flat zones of that tone's layer}."""
    ppm = p.px_per_mm
    sx0, sy0, sx1, sy1 = lay.src_rect
    src = img[sy0:sy1, sx0:sx1]
    vx0, vy0, vx1, vy1 = lay.vis_mm
    SW = int(round(p.support_w_mm * ppm))
    SH = int(round(p.support_h_mm * ppm))
    ox, oy = int(round(vx0 * ppm)), int(round(vy0 * ppm))
    W = max(4, min(SW - ox, int(round((vx1 - vx0) * ppm))))
    H = max(4, min(SH - oy, int(round((vy1 - vy0) * ppm))))
    h0, w0 = src.shape[:2]
    interp = cv2.INTER_AREA if W < w0 else cv2.INTER_CUBIC
    raw = cv2.resize(src, (W, H), interpolation=interp)
    check_cancel()
    resized = preprocess_image(raw, p, ppm)
    check_cancel()
    fg = None
    if p.pre_remove_bg and fg_src is not None:
        fg = cv2.resize(fg_src[sy0:sy1, sx0:sx1].astype(np.uint8) * 255, (W, H),
                        interpolation=cv2.INTER_LINEAR) > 127
        if not fg.any() or fg.all():
            fg = None

    gray = is_grayscale(src) if p.color_mode == "auto" else p.color_mode == "gray"
    passes, strength, simplify_mm, _ = p.detail_settings()

    # Edge-preserving smoothing (stronger and repeated when the level of detail is low)
    sm = resized.copy()
    for _ in range(passes):
        check_cancel()
        sm = cv2.bilateralFilter(sm, 7, 15 + strength * 55, 3 + strength * 6)
    check_cancel()

    k = p.n_layers + (1 if p.skip_support else 0)
    lab = cv2.cvtColor(sm, cv2.COLOR_BGR2LAB).astype(np.float32)
    L100 = lab[..., 0] * (100.0 / 255.0)
    if gray:
        feat = lab[..., :1]
    else:
        # OpenCV 8-bit Lab: L* is stretched to 0..255 while a*, b* keep their scale, so at the
        # default weight (40 %) the luminance counts ~2.5 times more than the hue; 100 % = balance
        f = p.hue_weight / 40.0
        feat = lab if f == 1.0 else np.concatenate([lab[..., :1], 128 + (lab[..., 1:] - 128) * f], axis=-1)
    X = feat.reshape(-1, feat.shape[-1])

    # tones learnt on the subject only when the background is removed
    pool = np.flatnonzero(fg.ravel()) if fg is not None else np.arange(len(X))
    rng = np.random.default_rng(0)
    idx = pool[rng.choice(len(pool), size=min(len(pool), 60000), replace=False)]
    cv2.setRNGSeed(0)
    k_eff = int(min(k, len(np.unique(X[idx].round(), axis=0))))
    k_eff = max(1, k_eff)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 0.2)
    _, _, centers = cv2.kmeans(X[idx], k_eff, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    check_cancel()

    # L* luminance (0..100) of the smoothed image: histogram and equivalent automatic thresholds
    Lh = L100[fg] if fg is not None else L100.ravel()
    hist = np.bincount(np.clip(Lh, 0, 99.999).astype(np.int32), minlength=100).astype(np.float64)
    hist = (hist / max(1.0, hist.max())).round(4).tolist()
    cl = np.sort(centers[:, 0] * (100.0 / 255.0))
    auto_thr = [round(float((a + b) / 2), 1) for a, b in zip(cl[:-1], cl[1:])]

    thr = parse_thresholds(p.thresholds) if p.manual_thresholds else []
    if thr and len(thr) == k - 1:
        # Manual thresholds: each tone = a luminance range (controlled posterisation)
        labels = np.digitize(L100, thr).astype(np.int32)
        n_lab = k
    else:
        thr = []
        # Assign every pixel to the nearest centre
        best = np.full(len(X), np.inf, np.float32)
        labels = np.zeros(len(X), np.int32)
        chans = [np.ascontiguousarray(X[:, j]) for j in range(X.shape[1])]
        dist, part = np.empty(len(X), np.float32), np.empty(len(X), np.float32)
        for i, c in enumerate(centers):
            check_cancel()
            # squared distance, channel by channel (same float32 sums, in the same order, as
            # ((X - c) ** 2).sum(axis=1), without its large temporary arrays)
            np.subtract(chans[0], c[0], out=dist)
            np.square(dist, out=dist)
            for j in range(1, len(chans)):
                np.subtract(chans[j], c[j], out=part)
                np.square(part, out=part)
                dist += part
            upd = dist < best
            best[upd] = dist[upd]
            labels[upd] = i
        labels = labels.reshape(H, W)
        n_lab = len(centers)

    # Majority filter (shape simplification)
    r = simplify_mm * p.px_per_mm
    before = labels
    labels = majority_filter(labels, n_lab, r)
    check_cancel()
    if p.protect_lines:
        # thin strokes (dark or light) keep their own tone instead of being swallowed by the area around
        prot = thin_features(L100, p.px_per_mm)
        if fg is not None:
            prot &= fg
        labels = np.where(prot, before, labels)

    check_cancel()
    # Tones sorted light -> dark by their mean lightness on the smoothed, preprocessed image
    # (the image the cut is computed on)
    lab_full = lab.reshape(H, W, 3)
    inside = fg if fg is not None else np.ones((H, W), bool)
    used = [i for i in range(n_lab) if np.any((labels == i) & inside)]
    if not used:
        used, inside = [i for i in range(n_lab) if np.any(labels == i)], np.ones((H, W), bool)
    seg_L = np.array([lab_full[..., 0][(labels == i) & inside].mean() for i in used], np.float32)
    order = np.argsort(-seg_L)
    remap = np.zeros(n_lab, np.int32)
    for i in range(n_lab):            # labels absent from the subject: attached to the nearest one
        if i not in used:
            remap[i] = int(np.argmin(np.abs(seg_L[order] - (centers[i, 0] if i < len(centers) else 128))))
    for new, old_pos in enumerate(order):
        remap[used[old_pos]] = new
    labels = remap[labels]
    # Flat areas: automatic (low-contrast islands absorbed), then the painted zones
    n_used = len(used)
    labels0 = labels
    forced = {int(t): m for t, m in (flat_forced or {}).items() if m is not None and 0 <= int(t) < n_used}

    def zone_map(src):
        m = cv2.resize(src[sy0:sy1, sx0:sx1], (W, H), interpolation=cv2.INTER_NEAREST) > 0
        return m & fg if fg is not None else m

    check_cancel()
    if p.flat_auto > 0 or flat_src is not None or forced:
        flab = _flat_lab(lab_full, gray)
        if p.flat_auto > 0:
            labels = flatten_auto(labels, n_used, flab, p.flat_auto, inside if fg is not None else None, ppm)
            check_cancel()
        if flat_src is not None:
            zones_m = zone_map(flat_src)
            if zones_m.any():
                labels = flatten_zones(labels, n_used, flab, zones_m, p.flat_tol("auto"))
    # the colours are measured before the zones forced into a layer: a light area forced into a
    # dark layer must not change the paint computed for that layer
    labels_c = labels
    flat_zones = zone_map(flat_src) if flat_src is not None else None
    flat_forced_zones = {}
    for t, src in sorted(forced.items()):
        check_cancel()
        zones_m = zone_map(src)
        if zones_m.any():
            flat_forced_zones[t] = zones_m
            labels = flatten_zones(labels, n_used, flab, zones_m, p.flat_tol(f"t{t}"), forced=t)
    # Paint colours measured on the original photo (resized, without preprocessing) over the same
    # zones: preprocessing helps the cut but must not dull or shift the colours
    check_cancel()
    lab_raw = cv2.cvtColor(raw, cv2.COLOR_BGR2LAB).astype(np.float32)
    zones = [(labels_c == new) & inside for new in range(len(used))]
    # a tone absorbed everywhere by the flat areas keeps the colour of its former zone
    zones = [z if z.any() else (labels0 == i) & inside for i, z in enumerate(zones)]
    means = np.array([lab_raw[z].mean(axis=0) for z in zones], np.float32)
    if not gray and p.color_vividness > 0:
        means = np.array([vivid_tone(m, lab_raw[z], p.color_vividness / 100) for m, z in zip(means, zones)], np.float32)
    if gray:
        means[:, 1:] = 128
    palette = lab_to_bgr(means)
    return Prepared(resized=resized, labels=labels, palette=palette, gray=gray, off_px=(ox, oy),
                    raw=raw, hist=hist, auto_thresholds=auto_thr, thresholds_used=thr or None, fg=fg,
                    flat_zones=flat_zones if flat_zones is not None and flat_zones.any() else None,
                    flat_forced_zones=flat_forced_zones or None)


def recommend_layers(img: np.ndarray, gray: bool, skip_lightest: bool, max_colors: int = 9) -> dict:
    """Recommends a number of stencils for an image.

    The (downscaled) image is quantised with k = 1..max_colors tones and the mean difference
    (ΔE) to the image is measured. We stop at the smallest k that:
      - reaches a difference judged "good" (ΔE <= target), or
      - is hardly improved by one more tone (elbow of the curve), or
      - whose next tone would only cover a negligible area.
    """
    h, w = img.shape[:2]
    s = 240 / max(h, w)
    small = cv2.resize(img, (max(8, round(w * s)), max(8, round(h * s))), interpolation=cv2.INTER_AREA)
    small = cv2.bilateralFilter(small, 5, 30, 4)
    lab = cv2.cvtColor(small.astype(np.float32) / 255, cv2.COLOR_BGR2LAB).reshape(-1, 3)
    X = lab[:, :1].copy() if gray else lab.copy()
    target = 4.0 if gray else 7.0
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.1)
    uniq = len(np.unique(X.round(1), axis=0))

    curve = []  # (k, ΔE, share of the smallest tone)
    for k in range(1, min(max_colors, uniq) + 1):
        cv2.setRNGSeed(0)
        _, lab_k, centers = cv2.kmeans(X, k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
        lab_k = lab_k.ravel()
        err = float(np.sqrt(((X - centers[lab_k]) ** 2).sum(axis=1)).mean())
        share = float(np.bincount(lab_k, minlength=k).min() / len(lab_k))
        curve.append((k, err, share))

    # Truly distinct tones: centres of k max merged when close (ΔE < 12)
    k_max, _, _ = curve[-1]
    cv2.setRNGSeed(0)
    _, lab_k, centers = cv2.kmeans(X, k_max, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    shares = np.bincount(lab_k.ravel(), minlength=k_max) / len(lab_k)
    kept = [c for c, sh in sorted(zip(centers, shares), key=lambda t: -t[1]) if sh >= 0.01]
    distinct = []
    for c in kept:
        if all(np.linalg.norm(c - d) >= (8 if gray else 12) for d in distinct):
            distinct.append(c)

    e1 = curve[0][1] or 1.0
    chosen, reason = curve[-1][0], "la courbe ne se stabilise pas : image riche en nuances"
    for idx in range(1, len(curve)):
        k, err, _ = curve[idx]
        if err <= target:
            chosen, reason = k, f"écart à l'original jugé bon (ΔE {err:.1f} ≤ {target:g})"
            break
        if idx + 1 < len(curve):
            gain = (err - curve[idx + 1][1]) / e1
            if curve[idx + 1][2] < 0.015:
                chosen, reason = k, "une teinte de plus ne couvrirait qu'une surface négligeable"
                break
            if gain < 0.04:
                chosen, reason = k, "une teinte de plus n'améliore presque plus le rendu"
                break

    offset = 1 if skip_lightest else 0
    rec = int(np.clip(chosen - offset, 1, 8))
    options = [{"layers": k - offset, "delta_e": round(err, 1)}
               for k, err, _ in curve if 1 <= k - offset <= 8]
    return {"layers": rec, "colors": chosen, "distinct_colors": len(distinct),
            "reason": reason, "curve": options, "gray": gray}


def majority_filter(labels: np.ndarray, k: int, r: float) -> np.ndarray:
    if r < 1:
        return labels
    acc = np.full(labels.shape, -1.0, np.float32)
    out = labels.copy()
    for i in range(k):
        check_cancel()
        m = (labels == i).astype(np.float32)
        if not m.any():
            continue
        b = cv2.GaussianBlur(m, (0, 0), r / 2)
        upd = b > acc
        out[upd] = i
        acc[upd] = b[upd]
    return out


# --------------------------------------------------------------------------
# Flat areas (poster look): full flat layers instead of interleaved tones
# --------------------------------------------------------------------------

def _flat_lab(lab8: np.ndarray, gray: bool) -> np.ndarray:
    """CIE L*a*b* (L* 0..100) of an OpenCV 8-bit Lab image; hue ignored in black and white."""
    out = lab8.astype(np.float32).copy()
    out[..., 0] *= 100.0 / 255.0
    out[..., 1:] = 0.0 if gray else out[..., 1:] - 128.0
    return out


def _tone_gap(labels: np.ndarray, k: int, lab: np.ndarray, inside: np.ndarray | None = None) -> float:
    """Typical colour difference (ΔE76) between two neighbouring tones: median of the differences
    between the mean colours of tones that follow each other from light to dark."""
    flat = labels.ravel() if inside is None else labels[inside]
    pix = lab.reshape(-1, 3) if inside is None else lab[inside]
    cnt = np.bincount(flat, minlength=k)[:k]
    sums = np.stack([np.bincount(flat, weights=pix[:, c].astype(np.float64), minlength=k)[:k]
                     for c in range(3)], 1)
    present = cnt > 0
    means = sums[present] / cnt[present, None]
    if len(means) < 2:
        return 10.0
    return float(max(2.0, np.median(np.linalg.norm(np.diff(means, axis=0), axis=1))))


def _components(labels: np.ndarray, k: int) -> tuple[np.ndarray, int, np.ndarray]:
    """Connected areas of every tone (4-connectivity): id map (1..n), n, tone of each id (index 0
    unused)."""
    comp = np.zeros(labels.shape, np.int32)
    tones = [0]
    n = 0
    for t in range(k):
        m = (labels == t).view(np.uint8)
        c, cl = cv2.connectedComponents(m, connectivity=4)
        if c < 2:                       # tone absent
            continue
        if n:
            cv2.add(cl, n, dst=cl, mask=m)  # ids after those of the tones before
        cv2.add(comp, cl, dst=comp)     # the tones do not overlap: a sum places each one
        tones += [t] * (c - 1)
        n += c - 1
    return comp, n, np.array(tones, np.int32)


def _borders(comp: np.ndarray, n: int, lab: np.ndarray, reach: int):
    """Borders between neighbouring areas, seen from both sides: source area, neighbour area,
    border length in pixels, mean contrast across the border (ΔE76 between the image `reach`
    pixels on each side: low on a smooth gradient cut by a threshold, high on a real edge) and the
    mean colour of the neighbour along the border."""
    H, W = comp.shape
    P = np.pad(lab, ((reach, reach), (reach, reach), (0, 0)), mode="edge")
    WP = W + 2 * reach
    flat_c, flat_l, flat_p = comp.ravel(), np.ascontiguousarray(lab).reshape(-1, 3), P.reshape(-1, 3)
    srcs, dsts, conts, nbs = [], [], [], []
    # the pixel pairs across a border are gathered by flat index (same values and order as
    # boolean indexing of the shifted views, much faster on these 3-channel images)
    for axis in (1, 0):
        if axis == 1:       # horizontal neighbours (x, x+1)
            ys, xs = np.nonzero(comp[:, :-1] != comp[:, 1:])
            ia, ib = ys * W + xs, ys * W + xs + 1
            pa, pb = (ys + reach) * WP + xs, (ys + reach) * WP + xs + 2 * reach + 1
        else:               # vertical neighbours (y, y+1)
            ys, xs = np.nonzero(comp[:-1, :] != comp[1:, :])
            ia, ib = ys * W + xs, (ys + 1) * W + xs
            pa, pb = ys * WP + xs + reach, (ys + 2 * reach + 1) * WP + xs + reach
        if not len(ys):
            continue
        a, b = flat_c[ia], flat_c[ib]
        c = np.linalg.norm(flat_p[pa] - flat_p[pb], axis=1)
        srcs += [a, b]
        dsts += [b, a]
        conts += [c, c]
        nbs += [flat_l[ib], flat_l[ia]]
    if not srcs:
        e = np.zeros(0, np.int64)
        return e, e, np.zeros(0), np.zeros(0), np.zeros((0, 3))
    src, dst = np.concatenate(srcs).astype(np.int64), np.concatenate(dsts).astype(np.int64)
    cont, nb = np.concatenate(conts), np.concatenate(nbs)
    uk, inv = np.unique(src * (n + 1) + dst, return_inverse=True)
    length = np.bincount(inv).astype(np.float64)
    contrast = np.bincount(inv, weights=cont) / length
    nb_mean = np.stack([np.bincount(inv, weights=nb[:, ch]) for ch in range(3)], 1) / length[:, None]
    return uk // (n + 1), uk % (n + 1), length, contrast, nb_mean


def flatten_auto(labels: np.ndarray, k: int, lab: np.ndarray, strength: float,
                 inside: np.ndarray | None = None, ppm: float = 6.0) -> np.ndarray:
    """Automatic flat areas: an area of one tone is absorbed by the neighbouring area it shares the
    longest low-contrast border with (smooth luminance variation cut by a threshold, not a real
    edge), when it is smaller than that neighbour and than a size limit. `strength` 0..100 raises
    both the contrast tolerated and the size limit. `lab`: CIE Lab of the smoothed image."""
    s = float(np.clip(strength, 0, 100)) / 100.0
    if s <= 0:
        return labels
    gap = _tone_gap(labels, k, lab, inside)
    max_contrast = gap * (0.3 + 0.55 * s)
    total = labels.size if inside is None else max(1, int(inside.sum()))
    max_area = total * 0.03 * s * s
    reach = max(1, int(round(0.6 * ppm)))
    blur = cv2.GaussianBlur(lab, (0, 0), max(0.6, reach / 2))
    # channels as the weights bincount uses (float64, contiguous): converted once, not at each pass
    lab_ch = [lab[..., ch].astype(np.float64).ravel() for ch in range(3)]
    out = labels.copy()
    for _ in range(4):
        comp, n, tone = _components(out, k)
        if n < 2:
            break
        flat_c = comp.ravel()
        area = np.bincount(flat_c, minlength=n + 1).astype(np.float64)
        src, dst, ln, ct, nb_mean = _borders(comp, n, blur, reach)
        if not len(src):
            break
        # contrast of an area with a neighbour: across their border (a real edge) and between the
        # mean colour of the area and that of the neighbour along the border (a thin line is
        # thinner than the reach: only its mean colour shows it)
        mean = np.stack([np.bincount(flat_c, weights=lab_ch[ch], minlength=n + 1)
                         for ch in range(3)], 1) / np.maximum(area, 1)[:, None]
        ct = np.maximum(ct, np.linalg.norm(mean[src] - nb_mean, axis=1))
        ok = (ct < max_contrast) & (area[src] < area[dst]) & (area[src] <= max_area)
        if inside is not None:      # removed background: left as it is
            ins = np.zeros(n + 1, bool)
            ins[np.unique(comp[inside])] = True
            ok &= ins[src] & ins[dst]
        if not ok.any():
            break
        src, dst, ln = src[ok], dst[ok], ln[ok]
        # per source area, the neighbour with the longest qualifying border
        order = np.lexsort((-ln, src))
        src, dst = src[order], dst[order]
        first = np.r_[True, src[1:] != src[:-1]]
        src, dst = src[first], dst[first]
        # smallest areas first; an area that absorbs another is not absorbed in the same pass
        target = np.arange(n + 1)
        taken = np.zeros(n + 1, bool)
        changed = False
        for i in np.argsort(area[src], kind="stable"):
            sa, da = src[i], dst[i]
            if taken[sa] or target[da] != da:
                continue
            target[sa] = da
            taken[da] = True
            changed = True
        if not changed:
            break
        new_tone = tone[target]
        out = new_tone[comp]
        if inside is not None:
            out = np.where(inside, out, labels)
    return out.astype(labels.dtype)


def flatten_zones(labels: np.ndarray, k: int, lab: np.ndarray, zones: np.ndarray,
                  tolerance: float, forced: int | None = None) -> np.ndarray:
    """Painted flat zones: each connected painted blob takes its majority tone wherever the image
    is close enough to it. Every area of another tone inside the blob is compared (mean colour)
    with the majority tone of the blob: it joins it when the difference is below the tolerance
    (0 = nothing, 50 = the neighbouring tones, 100 = the whole blob). Strong details (eyes, dark
    lines) stay. `forced`: tone given to the flattened part instead of the majority tone (zone
    painted in a layer)."""
    tol = float(np.clip(tolerance, 0, 100))
    if tol <= 0 and forced is None:
        return labels
    gap = _tone_gap(labels, k, lab)
    limit = np.inf if tol >= 100 else gap * tol / 40.0
    nz, zmap = cv2.connectedComponents(zones.astype(np.uint8), connectivity=8)
    out = labels.copy()
    comp, n, tone = _components(labels, k)
    pix = lab.reshape(-1, 3)
    for z in range(1, nz):
        zm = zmap == z
        zl = labels[zm]
        cnt = np.bincount(zl, minlength=k)
        t = int(np.argmax(cnt))
        ref = lab[zm & (labels == t)].mean(axis=0)
        # mean colour of each area, over its part inside the blob
        ids = comp[zm]
        u, inv = np.unique(ids, return_inverse=True)
        c = np.bincount(inv).astype(np.float64)
        zp = pix[zm.ravel()]
        mean = np.stack([np.bincount(inv, weights=zp[:, ch]) for ch in range(3)], 1) / c[:, None]
        join = (np.linalg.norm(mean - ref, axis=1) < limit) | (tone[u] == t)
        dest = t if forced is None else forced
        if join.any():
            out[zm & np.isin(comp, u[join])] = dest
    return out


# --------------------------------------------------------------------------
# 3-4. Layers + cutting constraints
# --------------------------------------------------------------------------

def clean_mask(mask: np.ndarray, p: Params, extra_area_mm2: float = 0.0,
               exempt: tuple | None = None) -> np.ndarray:
    """Material limits of a layer. `exempt`: (pieces, holes) bool masks or None (see `area_exempt`):
    the pieces, resp. holes, that overlap them only need `min_area_mm2`, without `extra_area_mm2`."""
    check_cancel()
    ppm = p.px_per_mm
    m = mask
    r_open = p.min_opening_mm * ppm / 2
    if r_open >= 0.5:
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, disk(r_open))
    r_close = p.min_strip_mm * ppm / 2
    if r_close >= 0.5:
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, disk(r_close))
    check_cancel()
    amin = int((p.min_area_mm2 + extra_area_mm2) * ppm * ppm)
    if amin > 1:
        a0 = int(p.min_area_mm2 * ppm * ppm)
        pieces, holes = exempt if exempt is not None else (None, None)
        m = remove_small(m, amin, exempt=pieces, exempt_px=a0)
        m = 255 - remove_small(255 - m, amin, keep_border=True, exempt=holes, exempt_px=a0)
    return m


def area_exempt(prep: "Prepared", p: Params, color_ids: list) -> list:
    """Per colour layer, the (pieces, holes) of `clean_mask` exempt from the detail supplement of the
    minimum area: the forced flat zones chosen in `p.flat_exempt`, as pieces of the layers that paint
    their tone and as holes of the others (a zone forced into a light layer inside a dark one, or
    into the bare support). None for a layer without any."""
    forced = prep.flat_forced_zones or {}
    zones = {t: forced[t] for t in p.flat_exempt_tones() if t in forced}
    out = []
    for li, ci in enumerate(color_ids):
        painted = set(color_ids[li:]) if p.layering == "cumulative" else {ci}
        mine = [z for t, z in zones.items() if t in painted]
        other = [z for t, z in zones.items() if t not in painted]
        out.append(None if not zones else (np.logical_or.reduce(mine) if mine else None,
                                            np.logical_or.reduce(other) if other else None))
    return out


def trim_taken(mask: np.ndarray, taken: np.ndarray, p: Params, extra_area_mm2: float = 0.0,
               exempt: tuple | None = None) -> np.ndarray:
    """`mask` without `taken` (bool), then the material limits that only remove opening: the parts
    left thinner than the minimum opening or smaller than the minimum area are removed, only around
    what was taken (unlike clean_mask, never adds opening: what a layer gives up must stay given up)."""
    m = np.where(taken, np.uint8(0), mask)
    if not (taken & (mask > 0)).any():
        return mask
    ppm = p.px_per_mm
    t = m
    r_open = p.min_opening_mm * ppm / 2
    if r_open >= 0.5:
        t = cv2.morphologyEx(t, cv2.MORPH_OPEN, disk(r_open))
    amin = int((p.min_area_mm2 + extra_area_mm2) * ppm * ppm)
    if amin > 1:
        t = remove_small(t, amin, exempt=exempt[0] if exempt is not None else None,
                         exempt_px=int(p.min_area_mm2 * ppm * ppm))
    gone = (m > 0) & (t == 0)
    if gone.any():
        near = cv2.dilate(taken.astype(np.uint8), disk(max(r_open, 1.0) * 2 + 1)) > 0
        n, lab = cv2.connectedComponents(gone.astype(np.uint8), connectivity=8)
        hit = np.unique(lab[gone & near])
        m = np.where(np.isin(lab, hit[hit > 0]), np.uint8(0), m)
    return m


def exclusive_layers(masks: list, targets: list, p: Params, extra_area_mm2: float = 0.0,
                     keep: dict | None = None, exempts: list | None = None) -> list:
    """Exclusive stacking: each spot painted by one layer only. A spot several layers paint (the
    material limits closing a narrow gap or filling a small hole of one layer over another one)
    stays with the layer whose tone it is (`targets`), else with the darkest of them; the others
    give it up (trim_taken), except a lighter layer when `p.under_darker` (painted under, hidden by
    the darker one). `keep`: {layer: spots it keeps anyway} (the Overlap effect, on purpose).
    `exempts`: per layer, its exemptions of the minimum area (`area_exempt`)."""
    n = len(masks)
    if n < 2:
        return masks
    cnt = np.zeros(masks[0].shape, np.uint8)
    for m in masks:
        cnt += m > 0
    multi = cnt > 1
    if not multi.any():
        return masks
    owner = np.full(multi.shape, -1, np.int16)
    for k in range(n):                       # darkest painting it
        owner[multi & (masks[k] > 0)] = k
    for k in range(n):                       # the layer of its tone (targets are disjoint)
        owner[multi & (masks[k] > 0) & (targets[k] > 0)] = k
    out = list(masks)
    for k in range(n):
        take = multi & (masks[k] > 0) & (owner != k)
        if p.under_darker:
            take &= owner < k                # under a darker layer: kept
        if keep and k in keep:
            take &= ~(keep[k] > 0)
        if take.any():
            out[k] = trim_taken(masks[k], take, p, extra_area_mm2, exempts[k] if exempts else None)
    return out


def _grow_layers(grown: list, gained: list, todo: np.ndarray, refused: list, tone: np.ndarray,
                 dist: np.ndarray) -> np.ndarray:
    """Grows the layers together into the `todo` pixels, one pixel (4-neighbours) per step, until
    none can grow: a pixel reached by several layers in the same step goes to the one whose tone is
    the closest to its own (lowest layer index on a tie); a layer never takes the pixels `refused`
    to it. Updates `grown` (uint8 masks) and `gained` (bool) in place, returns what is left to fill.
    Only the front of each layer is followed (the gaps are a small part of the image): the steps
    work on pixel indices in a copy padded by one pixel, so that neighbours never wrap."""
    h, w = todo.shape
    W = w + 2
    pad = lambda a: np.pad(a, 1).ravel()                # noqa: E731
    todo_f, tone_f = pad(todo), pad(tone)
    refused_f = [pad(r) for r in refused]
    steps = np.array([-W, -1, 1, W])
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    # first step from the whole masks, then from the pixels each layer just received
    fronts = [np.flatnonzero(pad((cv2.dilate(m, kernel) > 0) & todo)) for m in grown]
    first = True
    won = [[] for _ in grown]
    while True:
        idx, dd, kk = [], [], []
        for k, f in enumerate(fronts):
            if not len(f):
                continue
            c = f if first else np.unique((f[:, None] + steps).ravel())
            c = c[todo_f[c] & ~refused_f[k][c]]
            if len(c):
                idx.append(c)
                dd.append(dist[tone_f[c], k])
                kk.append(np.full(len(c), k, np.int32))
        first = False
        if not idx:
            break
        idx, dd, kk = np.concatenate(idx), np.concatenate(dd), np.concatenate(kk)
        o = np.lexsort((kk, dd, idx))       # by pixel, then closest tone, then lowest layer
        idx, kk = idx[o], kk[o]
        keep = np.ones(len(idx), bool)
        keep[1:] = idx[1:] != idx[:-1]
        idx, kk = idx[keep], kk[keep]
        todo_f[idx] = False
        fronts = [idx[kk == k] for k in range(len(grown))]
        for k, f in enumerate(fronts):
            won[k].append(f)
    for k, parts in enumerate(won):
        if parts:
            f = np.concatenate(parts)
            r, c = f // W - 1, f % W - 1
            grown[k][r, c] = 255
            gained[k][r, c] = True
    return todo_f.reshape(h + 2, W)[1:-1, 1:-1].copy()


def fill_gaps(masks: list, targets: list, labels: np.ndarray, palette: np.ndarray,
              tones: list, p: Params) -> tuple[list, list]:
    """Exclusive layers: gives what the cleaning removed to the layers around it, so that the support
    does not show through. Each layer is cleaned alone and drops the pieces it cannot cut; they only
    stay cuttable as part of an opening they touch. The layers grow together into each gap, one pixel
    at a time: a strip between two tones is shared along its length instead of becoming a thin
    tongue of one of them. A pixel reached by several layers goes to the tone closest to its own.
    What a layer receives must respect the material limits: a part too thin for it is offered to the
    other layers. Parts no layer can take stay bare. Returns the masks and the targets (what each
    layer now has to paint, so that the pieces received are not reported as lost details)."""
    zone = np.logical_or.reduce([t > 0 for t in targets])
    todo = zone & ~np.logical_or.reduce([m > 0 for m in masks])
    if not todo.any():
        return masks, targets
    r_open, r_close = p.min_opening_mm * p.px_per_mm / 2, p.min_strip_mm * p.px_per_mm / 2
    margin = int(max(r_open, r_close)) + 3
    ys, xs = np.nonzero(todo)
    y0, y1 = max(int(ys.min()) - margin, 0), int(ys.max()) + margin + 1
    x0, x1 = max(int(xs.min()) - margin, 0), int(xs.max()) + margin + 1
    todo, area = todo[y0:y1, x0:x1], zone[y0:y1, x0:x1]
    tone = labels[y0:y1, x0:x1]
    lab = _lab(palette)
    dist = np.linalg.norm(lab[:, None] - lab[tones][None], axis=2)      # tone x layer
    grown = [m[y0:y1, x0:x1].copy() for m in masks]
    gained = [np.zeros_like(todo) for _ in masks]
    refused = [np.zeros_like(todo) for _ in masks]      # too thin for this layer
    for _ in range(4):
        todo = _grow_layers(grown, gained, todo, refused, tone, dist)
        again = False
        for k, m in enumerate(grown):
            if r_open < .5 or not gained[k].any():
                continue
            thin = gained[k] & ~(cv2.morphologyEx(m, cv2.MORPH_OPEN, disk(r_open)) > 0)
            if thin.any():
                grown[k] = np.where(thin, 0, m).astype(np.uint8)
                gained[k] &= ~thin
                refused[k] |= thin
                todo = todo | thin
                again = True
        if not again:
            break
    masks, targets = list(masks), list(targets)
    for k, m in enumerate(grown):
        if not gained[k].any():
            continue
        add = gained[k]
        if r_close >= .5:                # no material strip too narrow between the old and new parts
            add = add | ((cv2.morphologyEx(m, cv2.MORPH_CLOSE, disk(r_close)) > 0) & area)
        full = np.zeros_like(masks[k])
        full[y0:y1, x0:x1] = np.where(add, 255, 0).astype(np.uint8)
        masks[k] = masks[k] | full
        got = np.zeros_like(masks[k])
        got[y0:y1, x0:x1] = np.where(gained[k], 255, 0).astype(np.uint8)
        targets[k] = targets[k] | got
    return masks, targets


def merge_area_mm2(amount: float) -> float:
    """Largest visible area (mm²) a merge may change, from the "merge instead of bridges" setting
    (0..150, progressive scale): 0 = never merge, ~0.3 mm² at 1, ~5 mm² at 50, 90 mm² at 100,
    ~1560 mm² at 150."""
    if amount <= 0:
        return 0.0
    return 0.3 * 300.0 ** (min(amount, 150.0) / 100.0)


def merge_zones(opening: np.ndarray, p: Params, hidden: np.ndarray | None = None,
                weight: float = 1.0, protect: np.ndarray | None = None,
                others: np.ndarray | None = None) -> tuple[np.ndarray, int]:
    """Holds material islands by merging zones instead of bridges, when the change is small enough.

    A. closes a narrow channel of opening (narrower than `merge_channel_mm`) separating an island
       from the rest of the material: the island is welded along the whole narrow part;
    B. removes a small island: the openings around it merge and it gets painted.
    Cost of a merge = changed area not repainted by a later layer (`hidden`) x `weight` (visibility
    of this layer's paint over what is under it, 0..1); merges are applied from the cheapest, while
    the cost stays under `merge_area_mm2(p.merge_amount)` and the merge links separate groups.
    `protect`: areas no merge may change (key-detail lines cut into the layer); the islands touching
    them are kept too, unless `merge_line_islands` (the lines themselves are never closed).
    `others`: areas of other layers this layer may not paint (exclusive stacking): an island
    reaching them is not removed (B).
    Modifies `opening` in place; returns (mask of the changed areas, number of merges).
    """
    check_cancel()
    merged = np.zeros(opening.shape, bool)
    amax = merge_area_mm2(p.merge_amount) * p.px_per_mm ** 2
    if amax <= 0 or not (p.merge_channels or p.merge_islands):
        return merged, 0
    material = opening == 0
    mat8 = material.astype(np.uint8)
    n, lab, lab_stats, _ = cv2.connectedComponentsWithStats(mat8, connectivity=4)
    main = int(lab[0, 0])
    if n <= 2:                          # no island
        return merged, 0
    w = max(float(weight), 0.05)

    def points(m8: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(ys, xs) of the non-zero pixels of a uint8 mask (the islands and channels are small:
        working on their pixels only is much faster than on the whole sheet)."""
        pts = cv2.findNonZero(m8)
        if pts is None:
            return np.zeros(0, np.intp), np.zeros(0, np.intp)
        pts = pts.reshape(-1, 2)                  # (N, 1, 2) up to OpenCV 4, (N, 2) since 5
        return pts[:, 1].astype(np.intp), pts[:, 0].astype(np.intp)

    def vis_area(labels: np.ndarray, ys: np.ndarray, xs: np.ndarray, nl: int) -> np.ndarray:
        """Visible area (pixels not `hidden`) x w of each label, from all the pixels (ys, xs) of
        the labels needed (sums of 0 / 1: the same values as over the whole sheet)."""
        wts = None if hidden is None else (~hidden[ys, xs]).astype(np.float64)
        return np.bincount(labels[ys, xs], wts, minlength=nl).astype(np.float64) * w

    actions = []                        # (cost, kind, index, material labels linked)
    if p.merge_islands:
        iy, ix = points(cv2.bitwise_and(cv2.compare(lab, main, cv2.CMP_NE), mat8))   # every island pixel
        vis_lab = vis_area(lab, iy, ix, n)
        # islands kept (only the islands matter: the loop below skips the frame `main`)
        ring = set()
        if protect is not None and not p.merge_line_islands:   # islands bordering a protected line are kept
            near = cv2.dilate(protect.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
            ring = set(np.unique(lab[iy, ix][near[iy, ix]]).tolist())
        if others is not None:              # island over another layer's paint: kept
            ring |= set(np.unique(lab[iy, ix][others[iy, ix]]).tolist())
        for i in range(1, n):
            if i in ring:
                continue
            if i != main and vis_lab[i] <= amax:
                actions.append((float(vis_lab[i]), 1, i, (i,)))
    fills = []                          # (labels, stats) of the channels closed at each width
    # Channels are searched at several widths up to the maximum: closing with the widest disk alone
    # would also weld the narrow parts into large, costly pieces (a wider maximum must never give
    # fewer merges than a narrower one)
    wmax = p.merge_channel_mm
    widths = sorted({wmax} | {wmax * 0.6 ** k for k in range(1, 6) if wmax * 0.6 ** k >= p.min_opening_mm})
    pad = None
    open8 = np.ascontiguousarray(~material).view(np.uint8)       # where the channels may be filled
    if protect is not None:
        open8 = cv2.bitwise_and(open8, np.ascontiguousarray(~protect).view(np.uint8))
    for wd in (widths if p.merge_channels else []):
        r = wd * p.px_per_mm / 2
        if r >= 0.5:
            fill = cv2.bitwise_and(cv2.morphologyEx(mat8, cv2.MORPH_CLOSE, disk(r)), open8)
            nf, flab, fstats, _ = cv2.connectedComponentsWithStats(fill, connectivity=8)
            fi = len(fills)
            fills.append((flab, fstats))
            if nf > 1:
                fy, fx = points(fill)
                vis_f = vis_area(flab, fy, fx, nf)
                # material labels touching each filled part (4-neighbours)
                if pad is None:
                    pad = np.pad(lab, 1)
                fl = flab[fy, fx].astype(np.int64)
                codes = []
                for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                    nb = pad[fy + 1 + dy, fx + 1 + dx]
                    m = nb > 0
                    codes.append(fl[m] * n + nb[m])
                codes = np.unique(np.concatenate(codes))
                touch: dict[int, list] = {}
                for c in codes.tolist():
                    touch.setdefault(c // n, []).append(c % n)
                for j, labs in touch.items():
                    if len(labs) >= 2 and vis_f[j] <= amax:
                        actions.append((float(vis_f[j]), 0, (fi, j), tuple(labs)))
    if not actions:
        return merged, 0

    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    size = [1] * n
    # A closed channel must weld the island at least as wide as the material strip minimum: a fill
    # linking two pieces of material only by a pinch (two corners almost touching) leaves a weld
    # far thinner, which `clean_mask` (run before the merges) no longer sees. Such a merge is
    # skipped and the island keeps its bridge (or a wider channel width closes it more broadly).
    weld_r = p.min_strip_mm * p.px_per_mm / 2
    weld_k = disk(weld_r) if weld_r >= 0.5 else None

    def weld_holds(sub, x, y, roots) -> bool:
        """True when the channel fill `sub` (box x, y) links at least two of the material groups
        `roots` through material at least `weld_r` * 2 wide, around the fill."""
        if weld_k is None:
            return True
        m = int(np.ceil(weld_r)) * 4 + 2
        ya, yb = max(0, y - m), min(opening.shape[0], y + sub.shape[0] + m)
        xa, xb = max(0, x - m), min(opening.shape[1], x + sub.shape[1] + m)
        mat = (opening[ya:yb, xa:xb] == 0).astype(np.uint8)
        mat[y - ya:y - ya + sub.shape[0], x - xa:x - xa + sub.shape[1]][sub] = 1
        strong = cv2.morphologyEx(mat, cv2.MORPH_OPEN, weld_k)
        nc, clab = cv2.connectedComponents(strong, connectivity=4)
        src = lab[ya:yb, xa:xb]
        sel = (clab > 0) & (src > 0)
        pairs = np.unique(clab[sel].astype(np.int64) * n + src[sel])
        seen: dict[int, set] = {}
        for c in pairs.tolist():
            r = find(c % n)
            if r in roots and (c % n) not in removed:
                seen.setdefault(c // n, set()).add(r)
        return any(len(s) >= 2 for s in seen.values())

    removed = set()                     # islands removed (B): they no longer link anything
    count = 0
    filled_any = False
    for cost, kind, idx, labs in sorted(actions):
        if kind == 1:                   # B: remove an island still alone (not linked yet)
            i = labs[0]
            if i in removed or find(i) != i or size[i] != 1 or find(i) == find(main):
                continue
            x, y, bw, bh = lab_stats[i, :4]
            sub = lab[y:y + bh, x:x + bw] == i
            opening[y:y + bh, x:x + bw][sub] = 255
            merged[y:y + bh, x:x + bw] |= sub
            removed.add(i)
            count += 1
        else:                           # A: close a channel if it links separate groups
            roots = {find(a) for a in labs if a not in removed}
            if len(roots) < 2:
                continue
            flab, fstats = fills[idx[0]]
            x, y, bw, bh = fstats[idx[1], :4]
            sub = flab[y:y + bh, x:x + bw] == idx[1]
            if not weld_holds(sub, x, y, roots):
                continue
            opening[y:y + bh, x:x + bw][sub] = 0
            merged[y:y + bh, x:x + bw] |= sub
            rl = list(roots)
            for a in rl[1:]:
                ra, rb = find(rl[0]), find(a)
                if ra != rb:
                    parent[rb] = ra
                    size[ra] += size[rb]
            filled_any = True
            count += 1
    if filled_any:
        # slivers of opening left too thin at the ends of the closed channels: material too
        ro = p.min_opening_mm * p.px_per_mm / 2
        if ro >= 0.5:
            near = cv2.dilate(merged.astype(np.uint8), disk(ro * 2 + 1)) > 0
            kept = cv2.morphologyEx(opening, cv2.MORPH_OPEN, disk(ro)) > 0
            sliver = near & (opening > 0) & ~kept
            if protect is not None:
                sliver &= ~protect
            opening[sliver] = 0
            merged |= sliver
    return merged, count


def _crosses(forbid: np.ndarray | None, a, q) -> bool:
    """Does the segment a -> q touch the `forbid` mask (bridges removed by hand)?"""
    if forbid is None:
        return False
    n = max(2, int(math.ceil(math.hypot(q[0] - a[0], q[1] - a[1]))) + 1)
    H, W = forbid.shape
    lx = np.linspace(a[0], q[0], n).round().astype(int).clip(0, W - 1)
    ly = np.linspace(a[1], q[1], n).round().astype(int).clip(0, H - 1)
    return bool(forbid[ly, lx].any())


def add_bridges(opening: np.ndarray, p: Params, hidden: np.ndarray | None = None,
                forbid: np.ndarray | None = None) -> list[tuple]:
    """Connects every material island to the frame. Modifies `opening` in place.

    `hidden`: mask of the areas repainted by a later layer;
    a bridge running there will be invisible in the final result -> preferred.
    `forbid`: mask no bridge may touch (automatic bridges removed by hand); an island with no
    other place is left without a bridge (counted as short).
    Returns the list of bridges [(x1, y1, x2, y2)] in pixels.
    """
    ppm = p.px_per_mm
    bw = max(1, int(round(p.bridge_width_mm * ppm)))
    bridges = []
    double_px = p.double_bridge_area_mm2 * ppm * ppm
    n_islands = None
    short_islands = 0  # islands that could not get the requested number of bridges
    abandoned = []     # a pixel of each island left without a bridge (every place forbidden)

    for _ in range(2000):
        # One pass = components and distance map computed ONCE, then all pending islands
        # processed from nearest to farthest from the frame (much faster than recomputing
        # everything after each island). A bridge may stop on another island (chain): both
        # then form a group, linked to the frame in the next pass.
        check_cancel()
        material = (opening == 0).astype(np.uint8)
        n, lab, stats, cents = cv2.connectedComponentsWithStats(material, connectivity=4)
        main = lab[0, 0]
        pending = [i for i in range(1, n) if i != main]
        if n_islands is None:
            n_islands = len(pending)
        gone = {int(lab[y, x]) for x, y in abandoned}
        if gone:
            pending = [i for i in pending if i not in gone]
        if not pending:
            break
        is_main = lab == main
        src = np.where(is_main, 0, 255).astype(np.uint8)
        # distance to the frame + nearest frame point of every pixel (a single pass)
        dist, near_lbl = cv2.distanceTransformWithLabels(src, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
        zy, zx = np.nonzero(src == 0)
        near = NearestMain(is_main, near_lbl, zx, zy)
        order = []
        for i in pending:
            x, y, w, h, _a = stats[i]
            sub = np.where(lab[y:y + h, x:x + w] == i, dist[y:y + h, x:x + w], np.inf)
            j = int(np.argmin(sub))
            order.append((float(sub.flat[j]), i, x + j % w, y + j // w))
        order.sort()
        # only islands close to the frame are processed in this pass: farther ones wait
        # for their neighbours to be connected (shorter, chained bridges)
        reach = order[0][0] * 1.5 + 4 * ppm
        touched = set()              # islands already connected to another one during this pass
        drawn = False
        for dv, i, px, py in order:
            if dv > reach:
                break
            if i in touched:
                continue
            check_cancel()
            want = max(p.min_bridges, 2 if stats[i, cv2.CC_STAT_AREA] > double_px else 1)
            def allowed(cs):
                # not through a removed bridge, nor landing on an island left without a bridge
                if forbid is None:
                    return cs
                return [c for c in cs if c[2] is None or
                        (not _crosses(forbid, c[1], c[2]) and int(lab[c[2][1], c[2][0]]) not in gone)]
            cands = allowed(_candidates(lab, i, stats[i], cents[i], dist, near, hidden, dv, ppm,
                                        wide=want > 1, bw=bw))
            if not cands or min(c[0] for c in cands) >= 1e6:
                # no clean bridge near the island: search its whole outline, with no length limit
                far = allowed(_candidates(lab, i, stats[i], cents[i], dist, near, hidden, dv, ppm,
                                          wide=want > 1, bw=bw, unlimited=True))
                if far and min(c[0] for c in far) < 1e6:
                    cands = far
            if not cands:
                q = near(px, py, dv)
                if q:
                    c = _clip_bridge(lab, i, px, py, q, bw)
                    q = c[0] if c else q
                cands = allowed([(dv, (px, py), q, 0.0)] if q else [])
                if not cands:              # every place forbidden: the island keeps no bridge
                    short_islands += 1
                    abandoned.append((px, py))
                    drawn = True           # the next pass goes on with the other islands
                    continue
            chosen = _spread(cands, want)
            if len(chosen) < want:
                short_islands += 1
            for _s, (ax, ay), q, _a in chosen:
                if q is not None and (ax, ay) != q:
                    cv2.line(opening, (ax, ay), q, 0, thickness=bw, lineType=cv2.LINE_8)
                    bridges.append((ax, ay, q[0], q[1]))
                    drawn = True
                    other = int(lab[q[1], q[0]])
                    if other not in (0, main, i):   # landing on another island: group to connect next
                        touched.add(other)
        if not drawn:   # safety: nothing changed, avoid an endless loop
            break
    return bridges, n_islands or 0, short_islands


def _spread(cands: list, want: int) -> list:
    """Picks `want` bridges well spread around the island (angular gap), with the best score.

    "Clear" bridges come first (even if less well spread); those running along an edge
    (score >= 1e6) only complete the selection as a last resort.
    """
    clean = [c for c in cands if c[0] < 1e6]
    chosen = _spread_in(clean, want, [])
    # Not enough room around: a clear bridge close to another is better than one running along
    # an edge -> take the clear one with the widest angle from those already chosen.
    while chosen and len(chosen) < want:
        rest = [c for c in clean if c not in chosen and all(c[1] != d[1] for d in chosen)]
        if not rest:
            break
        gap = lambda c: min(abs((c[3] - d[3] + math.pi) % (2 * math.pi) - math.pi) for d in chosen)
        chosen.append(max(rest, key=lambda c: (round(gap(c), 2), -c[0])))
    if len(chosen) < want:
        chosen = _spread_in(cands, want, chosen)
    return chosen


def _spread_in(cands: list, want: int, chosen: list) -> list:
    chosen = list(chosen)
    if not cands:
        return chosen
    if not chosen:
        chosen = [min(cands, key=lambda c: c[0])]
    sep = 2 * math.pi / want * 0.75
    while len(chosen) < want and sep >= math.radians(25):
        ok = [c for c in cands if c not in chosen and all(
            abs((c[3] - d[3] + math.pi) % (2 * math.pi) - math.pi) >= sep for d in chosen)]
        if ok:
            chosen.append(min(ok, key=lambda c: c[0]))
        else:
            sep *= 0.75  # not enough room: accept closer bridges
    return chosen


class NearestMain:
    """Nearest frame point (main component) of a pixel, read from the labels of
    `distanceTransformWithLabels` (O(1)); falls back to a local search if the label is inconsistent."""

    def __init__(self, is_main, lbl, zx, zy):
        self.is_main, self.lbl, self.zx, self.zy = is_main, lbl, zx, zy

    def __call__(self, x, y, d):
        k = int(self.lbl[y, x]) - 1
        if 0 <= k < len(self.zx):
            qx, qy = int(self.zx[k]), int(self.zy[k])
            if self.is_main[qy, qx]:
                return qx, qy
        return _nearest(self.is_main, x, y, d)


def _nearest(is_main, x, y, d):
    R = int(math.ceil(d)) + 3
    H, W = is_main.shape
    x0, x1 = max(0, x - R), min(W, x + R + 1)
    y0, y1 = max(0, y - R), min(H, y + R + 1)
    ys, xs = np.nonzero(is_main[y0:y1, x0:x1])
    if len(xs) == 0:
        return None
    d2 = (xs + x0 - x) ** 2 + (ys + y0 - y) ** 2
    k = int(np.argmin(d2))
    return int(xs[k] + x0), int(ys[k] + y0)


def _clip_bridge(lab, i, ax, ay, q, bw=1):
    """Shortens the bridge [anchor -> q] so that it only crosses empty space.

    After leaving island `i`, the bridge stops on the first material it meets (frame or
    another island: islands are then chained). If it starts by crossing island `i` or
    crosses it again further on (anchor on a "back" face), the candidate is rejected (None).
    Otherwise returns (landing point, clear); clear = False if the bridge runs along or grazes material.
    """
    n = max(2, int(math.ceil(math.hypot(q[0] - ax, q[1] - ay))) + 1)
    lx = np.linspace(ax, q[0], n).round().astype(int)
    ly = np.linspace(ay, q[1], n).round().astype(int)
    v = lab[ly, lx]
    gap = np.flatnonzero(v == 0)
    if len(gap) == 0:          # material immediately adjacent
        return q, True
    s = gap[0]
    if s > 3:                  # the bridge would start by crossing the island (anchor on the far side)
        return None
    hit = np.flatnonzero(v[s:] != 0)
    if len(hit) == 0:
        return q, _clear_of_material(lab, i, ax, ay, q, bw)
    h = s + hit[0]
    if v[h] == i:
        return None
    q = int(lx[h]), int(ly[h])
    return q, _clear_of_material(lab, i, ax, ay, q, bw)


def _clear_of_material(lab, i, ax, ay, q, bw):
    """Does the bridge keep away from material outside its two attachments?

    The bridge is widened by a clearance on each side; outside the attachment discs around
    the anchor (radius 1.5 x widened half width: the bridge must leave the island at more than
    ~40° from its edge) and the landing point (radius 2 x, ~30°), this band must touch no
    material. This discards bridges that run along or graze an edge, even when they stay
    entirely in empty space and look "stuck" to the island.
    """
    half = bw / 2 + max(2.0, bw / 2)          # half width + clearance
    r_dep = 1.5 * half + 1
    r_att = 2 * half + 1
    H, W = lab.shape
    r = int(math.ceil(half)) + 1
    x0, x1 = max(0, min(ax, q[0]) - r), min(W, max(ax, q[0]) + r + 1)
    y0, y1 = max(0, min(ay, q[1]) - r), min(H, max(ay, q[1]) + r + 1)
    band = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.line(band, (ax - x0, ay - y0), (q[0] - x0, q[1] - y0), 1, thickness=max(1, int(round(2 * half))))
    cv2.circle(band, (ax - x0, ay - y0), int(r_dep), 0, -1)
    cv2.circle(band, (q[0] - x0, q[1] - y0), int(r_att), 0, -1)
    touch = int((band.astype(bool) & (lab[y0:y1, x0:x1] != 0)).sum())
    return touch <= 1


def _candidates(lab, i, st, cent, dist, near, hidden, dmin, ppm, max_pts=96, wide=False, bw=1,
                unlimited=False):
    """Candidate bridges from the island edge: (score, anchor, frame point, angle).

    Score = length weighted by the visible share of the bridge (not repainted afterwards).
    """
    x, y, w, h, _ = st
    m = (lab[y:y + h, x:x + w] == i).astype(np.uint8)
    border = m & ~cv2.erode(m, np.ones((3, 3), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0)
    ys, xs = np.nonzero(border)
    if len(xs) == 0:
        return []
    xs = xs + x
    ys = ys + y
    dd = dist[ys, xs]
    # Several bridges requested: accept longer bridges so they can be spread
    limit = np.inf if unlimited else max(3 * dmin, dmin + 12 * ppm) * (4 if wide else 1)
    ok = dd <= limit
    xs, ys, dd = xs[ok], ys[ok], dd[ok]
    if len(xs) > max_pts:
        step = len(xs) / max_pts
        sel = (np.arange(max_pts) * step).astype(int)
        # always keep the nearest point
        sel = np.unique(np.append(sel, int(np.argmin(dd))))
        xs, ys, dd = xs[sel], ys[sel], dd[sel]
    out = []
    for ax, ay, d in zip(xs.tolist(), ys.tolist(), dd.tolist()):
        q = near(ax, ay, d)
        if q is None:
            continue
        c = _clip_bridge(lab, i, ax, ay, q, bw)
        if c is None:   # the bridge would cross back over the island itself
            continue
        q, clean = c
        length = math.hypot(q[0] - ax, q[1] - ay)
        vis = 1.0
        if hidden is not None and length > 0:
            n = max(2, int(length))
            lx = np.linspace(ax, q[0], n).round().astype(int)
            ly = np.linspace(ay, q[1], n).round().astype(int)
            vis = 1.0 - float(hidden[ly, lx].mean())
        score = (length + 1) * (0.15 + 0.85 * vis)
        if not clean:   # runs along an edge: last resort only
            score += 1e6
        out.append((score, (ax, ay), q, math.atan2(ay - cent[1], ax - cent[0])))
    return out


# --------------------------------------------------------------------------
# 4b. Structural bridges: material that is attached, but badly held
# --------------------------------------------------------------------------
# A part of the sheet may touch the frame and still be badly held: a large piece hanging from a
# narrow strip (it hinges, a sheet stored rolled lifts it), or a long thin strip held at one end
# (or at both ends, but very long). The "solid" sheet is the material joined to the frame by paths
# at least SPAN_RIGID_MM wide; every other edge of material (wide pieces and thin strips alike) must
# be within half the spacing (`bridge_spacing_mm`) of a support, measured through the material
# (geodesic distance, on a reduced copy). The farthest edge gets a bridge first (onto material that
# is better held), plus one on the opposite edge when it is a thin strip; bridges become supports.
SPAN_PPM = 3.0             # resolution of the analysis (px/mm)
SPAN_RIGID_MM = 6.0        # material joined to the frame by paths at least this wide is solid
SPAN_BRIDGE_MAX_MM = 15.0  # longest structural bridge
SPAN_MAX_BRIDGES = 150     # per layer (safety)
_K8 = np.ones((3, 3), np.uint8)
_K4 = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))


def _box_union(a: tuple, b: tuple) -> tuple:
    """Box (x, y, w, h) around two boxes (an empty box, w = 0, is ignored)."""
    if not b[2]:
        return a
    if not a[2]:
        return b
    x0, y0 = min(a[0], b[0]), min(a[1], b[1])
    return x0, y0, max(a[0] + a[2], b[0] + b[2]) - x0, max(a[1] + a[3], b[1] + b[3]) - y0


def _geodesic(src: np.ndarray, mat: np.ndarray, limit: float) -> np.ndarray:
    """Geodesic distance (px, octagonal approximation) from `src` through `mat`, up to `limit` px;
    inf beyond it or where unreachable.
    Grows the reached area one step at a time; instead of writing the distance of the new pixels
    at each step (slow full-image numpy passes), counts in how many steps each pixel was reached
    (saturating uint8 counter flushed every 255 steps): first step = steps - count.
    Each step only works in the box of the pixels reached at the two steps before (plus 1 px): a
    pixel reached earlier has had a K8 and a K4 step since, so it can reach nothing new (from a
    small source, e.g. a bridge, the box grows with the front instead of covering the window);
    the counter only covers the box of the pixels reached so far."""
    m = (np.asarray(mat) != 0).astype(np.uint8)
    cur = cv2.bitwise_and((np.asarray(src) != 0).astype(np.uint8), m)
    H, W = m.shape
    acc = np.zeros(m.shape, np.int32)
    cnt = cur.copy()
    new = np.empty_like(cur)
    last, before = cv2.boundingRect(cur), (0, 0, 0, 0)      # boxes (x, y, w, h) of the last two steps
    reached = last
    states, in_cnt = 1, 1
    for it in range(1, int(limit / 1.08) + 2):
        x, y, w, h = _box_union(last, before)
        if not w:
            break
        x0, y0, x1, y1 = max(0, x - 1), max(0, y - 1), min(W, x + w + 1), min(H, y + h + 1)
        c, nw = cur[y0:y1, x0:x1], new[y0:y1, x0:x1]
        cv2.dilate(c, _K8 if it % 2 else _K4, dst=nw)
        cv2.bitwise_and(nw, m[y0:y1, x0:x1], dst=nw)
        cv2.subtract(nw, c, dst=nw)                          # pixels reached at this step
        bx, by, bw, bh = cv2.boundingRect(nw)
        if not bw:                          # nothing new: the area stops growing
            break
        cv2.bitwise_or(c, nw, dst=c)
        before, last = last, (bx + x0, by + y0, bw, bh)
        reached = _box_union(reached, last)
        if in_cnt == 255:
            acc += cnt
            cnt[:] = 0
            in_cnt = 0
        rx, ry, rw, rh = reached
        cv2.add(cnt[ry:ry + rh, rx:rx + rw], cur[ry:ry + rh, rx:rx + rw], dst=cnt[ry:ry + rh, rx:rx + rw])
        states += 1
        in_cnt += 1
    acc += cnt
    first = (states - acc).astype(np.float64) * 1.08
    return np.where(acc > 0, first, np.inf).astype(np.float32)


def _ray_bridge(work, xs, ys, maxlen, land_ok, hidden, bw, away=None, n_ang=48, avoid=None, focus=None,
                forbid=None):
    """Best straight bridge leaving the material at one of the edge points (xs, ys), crossing an
    opening up to the first material met, where `land_ok(ax, ay, qx, qy, length)` (arrays -> bool
    array) accepts it. `away`: unit vector, only directions within 60° of it. Score = length weighted
    by the visible share (as `_candidates`); bridges running along an edge are rejected.
    `avoid`: (existing bridges [(x1, y1, x2, y2)], radius px): a bridge whose middle is that close to
    the middle of one of them is a last resort. `focus`: (x, y) the point to hold; the distance
    from the anchor to it counts like the length of the bridge. `forbid`: mask no bridge may touch
    (bridges removed by hand). Returns ((ax, ay), (qx, qy)) or None."""
    if len(xs) == 0:
        return None
    H, W = work.shape
    ang = np.linspace(0, 2 * np.pi, n_ang, endpoint=False)
    dx, dy = np.cos(ang), np.sin(ang)
    if away is not None:
        keep = dx * away[0] + dy * away[1] >= 0.5
        dx, dy = dx[keep], dy[keep]
    steps = np.arange(1, int(maxlen) + 1, dtype=np.float32)
    px = np.rint(xs[:, None, None] + dx[None, :, None] * steps).astype(np.int32)
    py = np.rint(ys[:, None, None] + dy[None, :, None] * steps).astype(np.int32)
    inb = (px >= 0) & (px < W) & (py >= 0) & (py < H)
    flat = np.clip(py, 0, H - 1, out=np.empty_like(py))
    flat *= W
    flat += np.clip(px, 0, W - 1)
    m = np.take(np.ascontiguousarray(work).ravel(), flat) > 0    # (one flat take: faster than 2-D indexing)
    empty = ~m & inb
    solid = m & inb
    g0 = np.argmax(empty, axis=2)
    ok = empty.any(axis=2) & (g0 <= 3)             # the ray leaves the material at once
    idx = np.arange(len(steps))[None, None, :]
    after = solid & (idx > g0[..., None])
    h = np.argmax(after, axis=2)
    ok &= after.any(axis=2)
    ci, ai = np.nonzero(ok)
    if len(ci) == 0:
        return None
    hh = h[ci, ai]
    # the opening crossed must not be left through the sheet edge before the landing point
    ok_in = inb[ci, ai, hh]
    ci, ai, hh = ci[ok_in], ai[ok_in], hh[ok_in]
    ax, ay = xs[ci], ys[ci]
    qx, qy = px[ci, ai, hh], py[ci, ai, hh]
    length = (hh + 1).astype(np.float32)
    good = land_ok(ax, ay, qx, qy, length)
    if not good.any():
        return None
    ax, ay, qx, qy, length = ax[good], ay[good], qx[good], qy[good], length[good]
    off = np.zeros(len(length), np.float32) if focus is None else         np.hypot(ax - focus[0], ay - focus[1]).astype(np.float32)
    order = np.argsort(length + off)[:40]
    if avoid is not None and avoid[0]:
        near = np.zeros(len(length), bool)
        mx, my = (ax + qx) / 2, (ay + qy) / 2
        for (x1, y1, x2, y2) in avoid[0]:
            near |= np.hypot(mx - (x1 + x2) / 2, my - (y1 + y2) / 2) < avoid[1]
        order = np.argsort(length + off + near * 1e6)[:40]
    best = []
    for k in order:
        a, q, ln = (int(ax[k]), int(ay[k])), (int(qx[k]), int(qy[k])), float(length[k])
        vis = 1.0
        if hidden is not None:
            n = max(2, int(ln))
            lx = np.linspace(a[0], q[0], n).round().astype(int)
            ly = np.linspace(a[1], q[1], n).round().astype(int)
            vis = 1.0 - float(hidden[ly, lx].mean())
        crowd = 1e6 if avoid is not None and avoid[0] and min(
            math.hypot((a[0] + q[0] - x1 - x2) / 2, (a[1] + q[1] - y1 - y2) / 2) for (x1, y1, x2, y2) in avoid[0]) < avoid[1] else 0
        best.append(((ln + 1) * (0.15 + 0.85 * vis) + float(off[k]) + crowd, a, q))
    best.sort()
    for _s, a, q in best:
        if _clear_of_material(work, 1, a[0], a[1], q, bw) and not _crosses(forbid, a, q):
            return a, q
    return None


def _solid_sheet(work: np.ndarray, ppm: float, s: float) -> tuple:
    """Material reduced by `s` (bool), the solid sheet in it (material joined to the frame by paths
    at least SPAN_RIGID_MM wide; None when there is no material) and the material at least that wide."""
    H, W = work.shape
    w, h = max(1, int(round(W * s))), max(1, int(round(H * s)))
    mat = cv2.resize(work.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA) > 0.3
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(3, int(SPAN_RIGID_MM * ppm * s) | 1),) * 2)
    opened = cv2.morphologyEx(mat.astype(np.uint8), cv2.MORPH_OPEN, k)
    n, plab = cv2.connectedComponents(opened, connectivity=4)
    if n < 2:
        return mat, None, opened > 0
    frame = int(plab[0, 0])
    if not frame:
        frame = 1 + int(np.argmax(np.bincount(plab.ravel())[1:]))
    return mat, plab == frame, opened > 0


SUPPORT_MAP_REF_MM = 80.0   # spacing the support map is scaled to when structural bridges are off


def support_map(res: "Result", L: "Layer") -> np.ndarray | None:
    """How well the material of a layer is held (BGRA overlay, sheet size; None without material):
    the solid sheet is left transparent; the rest of the material is coloured by its distance to a
    support (solid sheet or bridge) through the material, compared with half the spacing between
    supports: green = next to a support, yellow, orange = at the limit (still accepted), red =
    beyond it (a bridge is missing there: only where none found room).
    Strong along the edges (what the structural bridges hold), light inside wide pieces."""
    p = res.params
    ppm = p.px_per_mm
    s = min(1.0, SPAN_PPM / ppm)
    work = (L.opening == 0).astype(np.uint8)
    mat, solid, _wide = _solid_sheet(work, ppm, s)
    if solid is None:
        return None
    half = (p.bridge_spacing_mm if p.bridge_spacing_mm > 0 else SUPPORT_MAP_REF_MM) / 2 * ppm * s
    src = solid.astype(np.uint8)
    bwl = max(1, int(round(p.bridge_width_mm * ppm * s)))
    for (x1, y1, x2, y2) in L.bridges:              # a bridge is a support (as in add_struct_bridges)
        cv2.line(src, (int(x1 * s), int(y1 * s)), (int(x2 * s), int(y2 * s)), 1, thickness=bwl)
    mat = mat | (src > 0)                           # a thin bridge may vanish from the reduced copy
    D = _geodesic(src > 0, mat, 1.5 * half)
    r = np.clip(np.where(np.isfinite(D), D / half, 1.5), 0, 1.5)
    weak = mat & ~solid
    # green (0) -> yellow (0.6) -> orange (limit) | red beyond -> dark red, BGR
    stops = np.array([[0.0, 60, 180, 40], [0.6, 0, 210, 240], [1.0, 0, 140, 255], [1.0001, 30, 30, 230],
                      [1.5, 40, 0, 120]], np.float32)
    col = np.stack([np.interp(r, stops[:, 0], stops[:, c]) for c in (1, 2, 3)], -1).astype(np.uint8)
    out = np.zeros(mat.shape + (4,), np.uint8)
    out[..., :3] = col
    # the rule holds the edges: a band along them is coloured strongly, the inside of a wide piece lightly
    depth = cv2.distanceTransform(mat.astype(np.uint8), cv2.DIST_L2, 3)
    band = depth <= SPAN_RIGID_MM / 2 * ppm * s
    out[..., 3] = np.where(weak & band, 210, np.where(weak, 70, 0)).astype(np.uint8)
    H, W = work.shape
    out = cv2.resize(out, (W, H), interpolation=cv2.INTER_NEAREST)
    out[..., 3][work == 0] = 0                     # openings stay clear (reduced copy is coarser)
    return out


def add_struct_bridges(opening: np.ndarray, p: Params, hidden: np.ndarray | None = None,
                       forbid: np.ndarray | None = None, supports: list | None = None) -> tuple[list, list]:
    """Structural bridges (see above), after the island bridges. Modifies `opening` in place.
    `forbid`: mask no bridge may touch (bridges removed by hand). `supports`: bridges already cut
    [(x1, y1, x2, y2)] (island and manual bridges): supports like the solid sheet, as on the
    support map (otherwise the edges next to them are taken as badly held).
    Returns (bridges [(x1, y1, x2, y2)] in pixels, weak spots that got no bridge [(x, y, radius)]
    in pixels: the farthest edge from a support and the size of the area left without a bridge)."""
    if not p.bridges or p.bridge_spacing_mm <= 0:
        return [], []
    ppm = p.px_per_mm
    bw = max(1, int(round(p.bridge_width_mm * ppm)))
    s = min(1.0, SPAN_PPM / ppm)          # full resolution -> analysis
    H, W = opening.shape
    w, h = max(1, int(round(W * s))), max(1, int(round(H * s)))
    span = max(1.0, p.bridge_spacing_mm / 2 * ppm * s)   # farthest an edge may be from a support
    maxlen = SPAN_BRIDGE_MAX_MM * ppm
    bwl = max(1, int(round(bw * s)))
    orig = opening.copy()
    work = (opening == 0).astype(np.uint8)
    mat, solid, wide = _solid_sheet(work, ppm, s)
    if solid is None:
        return [], []
    # thin strips (narrower than SPAN_RIGID_MM), and those long enough to need a bridge on each edge
    tn, tlab, tst, _c = cv2.connectedComponentsWithStats((mat & ~wide).astype(np.uint8), connectivity=8)
    long_strip = np.hypot(tst[:, cv2.CC_STAT_WIDTH], tst[:, cv2.CC_STAT_HEIGHT]) >= span
    long_strip[0] = False
    m8 = mat.astype(np.uint8)
    edge = mat & (cv2.erode(m8, _K8, borderType=cv2.BORDER_REPLICATE) == 0)
    measured = edge & ~solid               # edges of the material not part of the solid sheet
    # bridges already cut: material (a thin bridge may vanish from the reduced copy) and supports
    sup = np.zeros((h, w), np.uint8)
    for (x1, y1, x2, y2) in supports or ():
        cv2.line(sup, (int(x1 * s), int(y1 * s)), (int(x2 * s), int(y2 * s)), 1, thickness=bwl)
    sup = sup > 0
    mat |= sup
    measured &= ~sup
    seeds = solid | sup
    mat0 = mat.copy()                      # without the structural bridges (see _prune_struct)
    cap = 3 * span
    D = np.minimum(_geodesic(seeds, mat, cap), cap)
    D0 = D.copy()                          # distances without the structural bridges (_prune_struct)
    bridges, weak = [], []
    groups, reach = [], []                 # bridges placed together, and where they hold
    known = []                             # (material, distances) of each window, see _prune_struct
    given = np.zeros((h, w), bool)
    rr = int(span / 2) + 1                 # window of the region around the farthest edge

    def lo(x, y):
        return (np.clip((np.asarray(y) * s).astype(int), 0, h - 1),
                np.clip((np.asarray(x) * s).astype(int), 0, w - 1))

    def edge_points(region_lo, near=None, r=0.0):
        """Full-resolution material edge points inside a reduced region (at most 200)."""
        ys_, xs_ = np.nonzero(region_lo)
        if len(xs_) == 0:
            return np.zeros(0, int), np.zeros(0, int)
        x0, x1 = max(0, int(xs_.min() / s) - 2), min(W, int((xs_.max() + 1) / s) + 2)
        y0, y1 = max(0, int(ys_.min() / s) - 2), min(H, int((ys_.max() + 1) / s) + 2)
        sub = work[y0:y1, x0:x1]
        e = (sub > 0) & (cv2.erode(sub, _K8, borderType=cv2.BORDER_REPLICATE) == 0)
        ey, ex = np.nonzero(e)
        ex, ey = ex + x0, ey + y0
        keep = region_lo[lo(ex, ey)]
        if near is not None:
            keep &= np.hypot(ex - near[0], ey - near[1]) <= r
        ex, ey = ex[keep], ey[keep]
        if len(ex) > 200:
            sel = np.linspace(0, len(ex) - 1, 200).astype(int)
            ex, ey = ex[sel], ey[sel]
        return ex, ey

    def draw(a, q):
        cv2.line(opening, a, q, 0, thickness=bw)
        cv2.line(work, a, q, 1, thickness=bw)
        bridges.append((a[0], a[1], q[0], q[1]))

    def gain(need, Dref):
        return lambda ax, ay, qx, qy, ln: Dref[lo(ax, ay)] - (Dref[lo(qx, qy)] + ln * s) >= need

    while len(bridges) < SPAN_MAX_BRIDGES:
        check_cancel()
        Dm = np.where(measured & ~given, D, 0)
        far = float(Dm.max())
        if far <= span:
            break
        fy, fx = np.unravel_index(int(np.argmax(Dm)), Dm.shape)
        # region: within span / 2 of the farthest point (computed in its window only)
        ya, yb, xa, xb = max(0, fy - rr), min(h, fy + rr + 1), max(0, fx - rr), min(w, fx + rr + 1)
        wy, wx = np.ogrid[ya:yb, xa:xb]
        reg = np.zeros((h, w), bool)
        reg[ya:yb, xa:xb] = (Dm[ya:yb, xa:xb] >= far - span / 2) & ((wy - fy) ** 2 + (wx - fx) ** 2 <= (span / 2) ** 2)
        ex, ey = edge_points(reg)
        b = _ray_bridge(work, ex, ey, maxlen, gain(span / 2, D), hidden, bw,
                        avoid=(bridges, min(p.bridge_spacing_mm / 4, 10.0) * ppm), focus=(fx / s, fy / s),
                        forbid=forbid)
        if b is None:
            given |= cv2.dilate(reg.astype(np.uint8), _K8, iterations=max(1, int(span / 4))) > 0
            weak.append((float(fx / s), float(fy / s), float(span / 2 / s)))
            continue
        new = [b]
        groups.append(new)
        (ax, ay), (qx, qy) = b
        d = np.array([qx - ax, qy - ay], np.float64)
        d /= max(1e-6, float(np.hypot(*d)))
        draw(*b)
        # a long thin strip: the opposite edge of the same strip, at the same place, so the strip
        # cannot pivot on its bridge (not at a short neck, nor from the piece across the opening)
        b2 = None
        t = int(tlab[lo(ax, ay)])
        if t and long_strip[t]:
            around = np.zeros((h, w), np.uint8)
            cv2.circle(around, (int(ax * s), int(ay * s)), max(2, int(SPAN_RIGID_MM * ppm * s)), 1, -1)
            ox, oy = edge_points((around > 0) & (tlab == t), near=(ax, ay), r=SPAN_RIGID_MM * ppm)
            b2 = _ray_bridge(work, ox, oy, maxlen, gain(span / 4, D), hidden, bw, away=-d, forbid=forbid)
        if b2 is not None and len(bridges) < SPAN_MAX_BRIDGES:
            draw(*b2)
            new.append(b2)
        # the bridges are supports: distances updated around them only (paths leaving that window
        # are ignored, which can only overestimate)
        pts = np.array([c for a, q in new for c in (a, q)], np.float64) * s
        x0, y0 = (np.maximum(0, pts.min(0) - cap - 2)).astype(int)
        x1, y1 = (np.minimum((w, h), pts.max(0) + cap + 3)).astype(int)
        nb = np.zeros((y1 - y0, x1 - x0), np.uint8)
        for a, q in new:
            cv2.line(nb, (int(a[0] * s) - x0, int(a[1] * s) - y0), (int(q[0] * s) - x0, int(q[1] * s) - y0),
                     1, thickness=bwl)
        mat[y0:y1, x0:x1] |= nb > 0
        G = np.minimum(_geodesic(nb > 0, mat[y0:y1, x0:x1], cap), cap)
        D[y0:y1, x0:x1] = np.minimum(D[y0:y1, x0:x1], G)
        reach.append(((y0, y1, x0, x1), nb))
        known.append((mat[y0:y1, x0:x1].copy(), G))
    if len(groups) > 1:
        keep = _prune_struct(reach, D0, mat0, measured & ~given, span, cap, [
            sum(math.hypot(q[0] - a[0], q[1] - a[1]) for a, q in g) for g in groups], known)
        if len(keep) < len(groups):
            opening[:] = orig
            bridges = []
            for g in keep:
                for a, q in groups[g]:
                    draw(a, q)
    return bridges, weak


def _prune_struct(reach: list, base, mat0, measured, span: float, cap: float, length: list,
                  known: list | None = None) -> list:
    """Indices of the structural bridges to keep: the greedy placement (farthest edge first) never
    revisits a bridge that later ones made useless (e.g. one at the tip of a piece, then one nearer
    that also covers the tip). Removes them one at a time, the longest first, as long as every
    measured edge stays within `span` of a support (`solid`: the solid sheet and the bridges already
    cut, which stay). One entry per group of bridges placed together
    (a bridge and the one on the opposite edge of its strip): `reach` = (window, its bridges drawn in
    it), `length` = their length. Distances go through the material without the structural bridges
    (`mat0`), each group adding only its own: a removed bridge is no longer a path either.
    `base`: distances (capped) to the supports that stay, through `mat0`. `known`: (material,
    distances) of each window when its group was placed, reused when that material is the same
    (no other structural bridge in the window, e.g. the first group)."""
    dist = []
    for g, ((y0, y1, x0, x1), nb) in enumerate(reach):
        src = nb > 0
        m = mat0[y0:y1, x0:x1] | src
        if known is not None and np.array_equal(known[g][0], m):
            dist.append(known[g][1])
        else:
            dist.append(np.minimum(_geodesic(src, m, cap), cap))
    alive = list(range(len(reach)))
    while len(alive) > 0:
        check_cancel()
        # nearest and second nearest support of every pixel (the solid sheet cannot be removed)
        D1, D2 = base.copy(), np.full(base.shape, cap, np.float32)
        A = np.full(base.shape, -1, np.int32)
        for g in alive:
            (y0, y1, x0, x1), _nb = reach[g]
            G = dist[g]
            d1, d2, a = D1[y0:y1, x0:x1], D2[y0:y1, x0:x1], A[y0:y1, x0:x1]
            better = G < d1
            D2[y0:y1, x0:x1] = np.where(better, d1, np.minimum(d2, G))
            D1[y0:y1, x0:x1] = np.where(better, G, d1)
            A[y0:y1, x0:x1] = np.where(better, g, a)
        useless = []
        for g in alive:
            (y0, y1, x0, x1), _nb = reach[g]
            m = measured[y0:y1, x0:x1]
            if not m.any():
                useless.append(g)
                continue
            without = np.where(A[y0:y1, x0:x1] == g, D2[y0:y1, x0:x1], D1[y0:y1, x0:x1])
            if float(without[m].max()) <= span:
                useless.append(g)
        if not useless:
            break
        alive.remove(max(useless, key=lambda g: length[g]))
    return alive


def _weak_mm(weak: list, p: Params) -> tuple:
    """Weak spots [(x, y, radius)] from pixels to sheet mm, rounded (for the interface)."""
    return tuple(tuple(round(v / p.px_per_mm, 2) for v in w) for w in weak)


def hold_material(opening: np.ndarray, p: Params, hidden: np.ndarray | None = None,
                  forbid: np.ndarray | None = None, supports: list | None = None) -> tuple:
    """Island bridges, then structural bridges. Modifies `opening` in place. `forbid`: mask no
    bridge may touch (automatic bridges removed by hand). `supports`: manual bridges already cut.
    Returns (bridges, islands, short islands, structural bridges, weak spots without a bridge
    [(x, y, radius)] in pixels)."""
    if not p.bridges:
        return [], 0, 0, 0, []
    bridges, n_isl, n_short = add_bridges(opening, p, hidden, forbid)
    sb, weak = add_struct_bridges(opening, p, hidden, forbid, list(supports or []) + bridges)
    if weak:      # a weak spot a structural bridge placed afterwards now holds: as the support map
        weak = _still_weak(opening, list(supports or []) + bridges + sb, weak, p)
    return bridges + sb, n_isl, n_short, len(sb), weak


# --------------------------------------------------------------------------
# Manual bridges: segments drawn by hand on a stencil, stored in `Params.manual_bridges`:
#   [{"id": "m1", "layer": 2, "kind": "color", "ref": "#8a7b6c", "pts": [u1, v1, u2, v2]}]
# pts: both ends in fractions of the source image (they follow the crop, scale and position of
# the image; may lie outside 0..1, on the sheet around the image). The layer is identified by its
# number and by what it paints, `ref`: computed tone colour (kind "color"), outline layer key A, B…
# (kind "outline", separate outline stencil), stroke colour (kind "line", line style) or PRIMER_REF
# (kind "primer", the primer stencil, layer 0).
# At each computation a bridge is:
#   ok       - drawn (counted with the layer's bridges);
#   useless  - its layer was found but the segment crosses no opening: no effect, kept;
#   layer    - obsolete: no layer paints that tone any more (colour drift > MANUAL_DE);
#   outside  - obsolete: an end lies off the stencil sheet;
#   void     - obsolete: an end has no material of the layer within the snap tolerance
#              (drawing it would leave a tab hanging in the opening).
# Obsolete bridges are ignored but kept, so undoing the change makes them active again.
# An entry with "type": "block" is an automatic bridge removed by hand: its segment, widened by a
# margin (BLOCK_MARGIN_MM, at least twice the bridge width), is a zone no automatic bridge of the
# layer may touch (state ok, layer or outside only).
# A bridge with "after": true is cut after the merges and the automatic bridges, which are then
# placed as if it did not exist (they stay where they were); otherwise it is cut first and the
# automatic bridges adapt to it (an island it holds needs no other bridge). A removed automatic bridge
# with "after": true does not forbid a zone: the automatic bridges are computed as usual, then the
# ones it crosses are taken off (the opening reopened there), the others stay where they are
# (state useless when no automatic bridge is there any more).
# --------------------------------------------------------------------------
MANUAL_DE = 15.0           # largest colour drift (ΔE) of a layer's tone still taken as the same layer
_MANUAL_ID = re.compile(r"^m\d{1,6}$")
MANUAL_KINDS = ("color", "outline", "line", "primer")
PRIMER_REF = "P"           # `ref` of the manual bridges of the primer stencil
BLOCK_MARGIN_MM = 2.0      # removed bridge: margin on each side kept free of automatic bridges


def parse_manual_bridges(txt: str) -> list[dict]:
    """Valid entries of a `manual_bridges` JSON text (others dropped), in canonical form."""
    try:
        data = json.loads(txt) if txt else []
    except ValueError:
        return []
    out, seen = [], set()
    for b in data if isinstance(data, list) else []:
        if not isinstance(b, dict):
            continue
        bid, kind, ref, pts = str(b.get("id", "")), b.get("kind"), str(b.get("ref") or ""), b.get("pts")
        typ = b.get("type", "bridge")
        if not _MANUAL_ID.match(bid) or bid in seen or kind not in MANUAL_KINDS or typ not in ("bridge", "block"):
            continue
        try:
            layer = int(b.get("layer"))
            pts = [float(v) for v in pts]
        except (TypeError, ValueError):
            continue
        if kind == "primer":
            layer = 0
        if not (kind == "primer" or 1 <= layer <= 99) or len(pts) != 4 or not all(math.isfinite(v) for v in pts):
            continue
        if kind == "outline":
            if not _GROUP.match(ref):
                continue
        elif kind == "primer":
            if ref != PRIMER_REF:
                continue
        else:
            ref = _hex(ref, "")
            if not ref:
                continue
        seen.add(bid)
        extra = ({"type": "block"} if typ == "block" else {}) | ({"after": True} if b.get("after") is True else {})
        out.append({"id": bid, **extra, "layer": layer, "kind": kind,
                    "ref": ref, "pts": [round(float(np.clip(v, -2.0, 3.0)), 6) for v in pts]})
    return out


def assign_manual(bridges: list[dict], refs: list[tuple]) -> dict[int, list[dict]]:
    """Layer of each bridge: {layer index: [bridges]}. `refs`: [(index, kind, ref)] of the layers
    computed, ref = BGR tone colour (color, line), outline layer key (outline) or PRIMER_REF. A colour bridge
    keeps its layer number while that layer's tone stays within MANUAL_DE of the one it was drawn
    on, otherwise goes to the layer of the nearest tone within MANUAL_DE (layers added or removed);
    bridges of no layer are left out."""
    out: dict[int, list[dict]] = {}
    for b in bridges:
        cands = [(i, r) for i, k, r in refs if k == b["kind"]]
        if b["kind"] in ("outline", "primer"):
            hit = next((i for i, r in cands if r == b["ref"]), None)
        else:
            ref = _lab(hex_to_bgr(b["ref"]).astype(np.uint8))[0]
            d = {i: float(np.linalg.norm(_lab(np.asarray(r, np.uint8))[0] - ref)) for i, r in cands}
            ok = {i: v for i, v in d.items() if v <= MANUAL_DE}
            hit = b["layer"] if b["layer"] in ok else (min(ok, key=ok.get) if ok else None)
        if hit is not None:
            out.setdefault(hit, []).append(b)
    return out


def manual_segments(bridges: list[dict], src_map, img_off: tuple, img_size: tuple) -> list[tuple]:
    """[(id, type, x1, y1, x2, y2)] in sheet pixels (floats). `src_map`: ((x0, y0, x1, y1) part of the
    source used for the computed image area, (width, height) of the source); `img_off`, `img_size`:
    that area in the sheet (pixels)."""
    (sx0, sy0, sx1, sy1), (sw, sh) = src_map
    X0, Y0 = img_off
    W, H = img_size
    fx, fy = W / max(1e-9, sx1 - sx0), H / max(1e-9, sy1 - sy0)
    out = []
    for b in bridges:
        u1, v1, u2, v2 = b["pts"]
        typ = b.get("type", "bridge")
        if b.get("after"):
            typ = "lateblock" if typ == "block" else "late"
        out.append((b["id"], typ, X0 + (u1 * sw - sx0) * fx, Y0 + (v1 * sh - sy0) * fy,
                    X0 + (u2 * sw - sx0) * fx, Y0 + (v2 * sh - sy0) * fy))
    return out


def apply_manual_bridges(opening: np.ndarray, segs: list[tuple], p: Params, late: bool = False) -> tuple[list, list, object]:
    """Draws the manual bridges `segs` ([(id, type, x1, y1, x2, y2)], sheet pixels) as material in
    `opening` (modified in place). An end lying in an opening is moved to the nearest material
    within the snap tolerance (the bridge width, at least 1 mm), so that small moves of the edges
    keep the bridge. Returns (bridges drawn [(x1, y1, x2, y2)], states [(id, state, x1, y1, x2, y2)]
    with the ends as used (pixels), mask of the removed automatic bridges ("block" entries: no
    automatic bridge may touch it) or None). `late`: only the bridges cut after the automatic ones
    (type "late"), otherwise all the others."""
    segs = [(sid, "bridge" if t == "late" else t, *r) for sid, t, *r in segs
            if (t == "late" if late else t in ("bridge", "block"))]
    if not segs:
        return [], [], None
    Hs, Ws = opening.shape
    ppm = p.px_per_mm
    bw = max(1, int(round(p.bridge_width_mm * ppm)))
    tol = max(bw, ppm)
    r = int(math.ceil(tol))
    drawn, states, forbid = [], [], None
    for bid, typ, x1, y1, x2, y2 in segs:
        if typ == "block":
            if not all(0 <= int(round(x)) < Ws and 0 <= int(round(y)) < Hs for x, y in ((x1, y1), (x2, y2))):
                states.append((bid, "outside", x1, y1, x2, y2))
                continue
            if forbid is None:
                forbid = np.zeros((Hs, Ws), np.uint8)
            margin = max(2 * bw, BLOCK_MARGIN_MM * ppm)
            cv2.line(forbid, (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2))), 1,
                     thickness=int(round(bw + 2 * margin)))
            states.append((bid, "ok", x1, y1, x2, y2))
            continue
        ends = []
        for x, y in ((x1, y1), (x2, y2)):
            xi, yi = int(round(x)), int(round(y))
            if not (0 <= xi < Ws and 0 <= yi < Hs):
                ends = None
                break
            if opening[yi, xi] == 0:
                ends.append((xi, yi))
                continue
            wx0, wy0 = max(0, xi - r), max(0, yi - r)
            ys, xs = np.nonzero(opening[wy0:yi + r + 1, wx0:xi + r + 1] == 0)
            d2 = (xs + wx0 - xi) ** 2 + (ys + wy0 - yi) ** 2
            if len(d2) == 0 or d2.min() > tol * tol:
                ends.append(None)
            else:
                k = int(np.argmin(d2))
                ends.append((int(xs[k] + wx0), int(ys[k] + wy0)))
        if ends is None:
            states.append((bid, "outside", x1, y1, x2, y2))
            continue
        if None in ends:
            states.append((bid, "void", x1, y1, x2, y2))
            continue
        (ax, ay), (bx, by) = ends
        # does the bridge cross an opening? (window around the segment)
        m = bw + 2
        wx0, wy0 = max(0, min(ax, bx) - m), max(0, min(ay, by) - m)
        wx1, wy1 = min(Ws, max(ax, bx) + m + 1), min(Hs, max(ay, by) + m + 1)
        band = np.zeros((wy1 - wy0, wx1 - wx0), np.uint8)
        cv2.line(band, (ax - wx0, ay - wy0), (bx - wx0, by - wy0), 1, thickness=bw)
        if not (band.astype(bool) & (opening[wy0:wy1, wx0:wx1] > 0)).any():
            states.append((bid, "useless", ax, ay, bx, by))
            continue
        cv2.line(opening, (ax, ay), (bx, by), 0, thickness=bw, lineType=cv2.LINE_8)
        drawn.append((ax, ay, bx, by))
        states.append((bid, "ok", ax, ay, bx, by))
    return drawn, states, (forbid > 0 if forbid is not None else None)


def late_manual(opening: np.ndarray, pre: np.ndarray | None, held: tuple, segs: list[tuple],
                p: Params) -> tuple[tuple, list, list]:
    """After the automatic bridges (`held`, from hold_material): takes off the automatic bridges
    crossed by the removed bridges of type "lateblock" (the opening `pre`, as before the automatic
    bridges, is put back there; the other bridges are untouched), then cuts the manual bridges of
    type "late". Modifies `opening` in place. Returns (held without the bridges taken off, manual
    bridges drawn, states)."""
    bridges, n_isl, n_short, n_struct, n_weak = held
    states = []
    blocks = [sg for sg in segs if sg[1] == "lateblock"]
    if blocks and pre is not None:
        Hs, Ws = opening.shape
        bw = max(1, int(round(p.bridge_width_mm * p.px_per_mm)))
        gone = set()
        for bid, _t, x1, y1, x2, y2 in blocks:
            if not all(0 <= int(round(x)) < Ws and 0 <= int(round(y)) < Hs for x, y in ((x1, y1), (x2, y2))):
                states.append((bid, "outside", x1, y1, x2, y2))
                continue
            # the automatic bridge(s) with both ends near the removed one's (either way round): the
            # same bridge, give or take the rounding of the stored ends, not a neighbour
            tol = max(2.0, bw)

            def same(a0, a1, b0, b1):
                d1 = max(math.hypot(a0 - x1, a1 - y1), math.hypot(b0 - x2, b1 - y2))
                d2 = max(math.hypot(a0 - x2, a1 - y2), math.hypot(b0 - x1, b1 - y1))
                return min(d1, d2) <= tol
            hit = [k for k, br in enumerate(bridges) if same(*br)]
            gone.update(hit)
            states.append((bid, "ok" if hit else "useless", x1, y1, x2, y2))
        if gone:
            first_struct = len(bridges) - n_struct      # structural bridges come last
            m = np.zeros((Hs, Ws), np.uint8)
            for k in gone:
                a0, a1, b0, b1 = bridges[k]
                cv2.line(m, (int(a0), int(a1)), (int(b0), int(b1)), 1, thickness=bw)
            opening[(m > 0) & (pre > 0)] = 255
            kept = [br for k, br in enumerate(bridges) if k not in gone]
            for a0, a1, b0, b1 in kept:               # bridges sharing pixels with a removed one
                cv2.line(opening, (int(a0), int(a1)), (int(b0), int(b1)), 0, thickness=bw)
            n_struct -= sum(1 for k in gone if k >= first_struct)
            bridges = kept
    drawn, st2, _ = apply_manual_bridges(opening, segs, p, late=True)
    if drawn and n_weak:                 # weak spots a late manual bridge now holds
        n_weak = _still_weak(opening, bridges + drawn, n_weak, p)
    return (bridges, n_isl, n_short, n_struct, n_weak), drawn, states + st2


def _still_weak(opening: np.ndarray, bridges: list, weak: list, p: Params) -> list:
    """Weak spots [(x, y, radius)] still farther than half the spacing from a support (solid sheet
    or bridge, through the material, as on the support map) once `bridges` are cut."""
    ppm = p.px_per_mm
    s = min(1.0, SPAN_PPM / ppm)
    mat, solid, _wide = _solid_sheet((opening == 0).astype(np.uint8), ppm, s)
    if solid is None:
        return weak
    src = solid.astype(np.uint8)
    bwl = max(1, int(round(p.bridge_width_mm * ppm * s)))
    for (x1, y1, x2, y2) in bridges:
        cv2.line(src, (int(x1 * s), int(y1 * s)), (int(x2 * s), int(y2 * s)), 1, thickness=bwl)
    span = max(1.0, p.bridge_spacing_mm / 2 * ppm * s)
    D = _geodesic(src > 0, mat | (src > 0), 1.5 * span)
    h, w = D.shape
    return [wk for wk in weak
            if D[min(h - 1, max(0, int(wk[1] * s))), min(w - 1, max(0, int(wk[0] * s)))] > span]


def needs_pre(segs: list[tuple]) -> bool:
    """Is a copy of the opening before the automatic bridges needed (see late_manual)?"""
    return any(sg[1] == "lateblock" for sg in segs)


class ManualBridges:
    """Manual bridges of one computation: which layer gets each one, and their states."""

    def __init__(self, p: Params, src_map, img_off: tuple, img_size: tuple):
        self.p = p
        self.bridges = parse_manual_bridges(p.manual_bridges) if src_map is not None else []
        self.src_map, self.img_off, self.img_size = src_map, img_off, img_size
        self.states: dict = {}
        self.layer_of: dict = {}

    def assign(self, kind: str, refs: list[tuple]) -> dict[int, list[dict]]:
        """{layer index: [bridges]} for the layers `refs` [(index, ref)] of one kind."""
        return assign_manual([b for b in self.bridges if b["kind"] == kind], [(i, kind, r) for i, r in refs])

    def apply(self, opening: np.ndarray, index: int, bridges: list[dict], late: bool = False) -> tuple:
        """(bridges drawn, mask of the removed automatic bridges or None); `late`: the bridges cut
        after the automatic ones"""
        drawn, states, forbid = apply_manual_bridges(opening, self.segments(bridges), self.p, late)
        self.record(index, states)
        return drawn, forbid

    def late(self, opening: np.ndarray, pre, index: int, bridges: list[dict], held: tuple) -> tuple:
        """See late_manual: (held, manual bridges drawn)."""
        held, drawn, states = late_manual(opening, pre, held, self.segments(bridges), self.p)
        self.record(index, states)
        return held, drawn

    def segments(self, bridges: list[dict]) -> list[tuple]:
        if not bridges:
            return []
        return manual_segments(bridges, self.src_map, self.img_off, self.img_size)

    def record(self, index: int, states: list) -> None:
        for bid, *st in states:
            self.states[bid] = tuple(st)
            self.layer_of[bid] = index

    def report(self) -> tuple:
        return tuple(manual_report(self.bridges, self.states, self.layer_of, self.p.px_per_mm))


def manual_report(bridges: list[dict], states: dict, layer_of: dict, ppm: float) -> list[dict]:
    """State of every manual bridge for the interface: {id, layer (index or None), state,
    seg (sheet mm, ends as used / as drawn)}. `states`: {id: (state, x1, y1, x2, y2) pixels};
    `layer_of`: {id: layer index}; a bridge of no layer gets the state "layer"."""
    out = []
    for b in bridges:
        st = states.get(b["id"])
        late = {"after": True} if b.get("after") else {}
        if st is None:
            out.append({"id": b["id"], "type": b.get("type", "bridge"), **late, "layer": None, "state": "layer", "seg": None})
        else:
            out.append({"id": b["id"], "type": b.get("type", "bridge"), **late, "layer": layer_of.get(b["id"]),
                        "state": st[0], "seg": [round(v / ppm, 2) for v in st[1:]]})
    return out


# --------------------------------------------------------------------------
# 5. Vectorisation
# --------------------------------------------------------------------------

def vectorize(opening: np.ndarray, p: Params) -> list[np.ndarray]:
    """Closed outlines in mm (list of Nx2 arrays)."""
    check_cancel()
    ppm = p.px_per_mm
    cnts, _ = cv2.findContours(opening, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    sigma = max(0.8, 0.2 * ppm)
    eps = max(0.3, p.tolerance_mm * ppm)
    field = None
    out = []
    for c in cnts:
        if len(c) < 6:
            continue
        if field is None:
            field = _edge_field(opening)
        pts = _subpixel_edge(c[:, 0, :].astype(np.float32), *field)
        pts = _smooth_closed(pts, sigma)
        approx = cv2.approxPolyDP(pts.reshape(-1, 1, 2), eps, True)[:, 0, :]
        if len(approx) < 3:
            continue
        out.append((approx + 0.5) / ppm)
    return out


EDGE_SIGMA_PX = 0.7     # blur of the opening mask whose 0.5 level gives the sub-pixel outline


def _edge_field(opening: np.ndarray) -> tuple:
    """Opening mask blurred (0..1) and its gradient, for _subpixel_edge."""
    f = cv2.GaussianBlur((opening > 0).astype(np.float32), (0, 0), EDGE_SIGMA_PX)
    return f, cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)


_REMAP_ROWS = 16384        # points sampled per cv2.remap call (its maps must stay under 32767 rows)


def _subpixel_edge(pts: np.ndarray, f: np.ndarray, gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    """Outline points moved onto the edge of the pixels. `findContours` gives the centres of the
    opening's border pixels, half a pixel inside the edge (and on a staircase): the openings came out
    ~1 px narrower than computed. Each point moves along the normal (gradient of the blurred mask)
    to where the blurred mask crosses 0.5; where there is none (strand 1 px wide, no clear normal),
    it moves half a pixel, or stays when the normal is undefined."""
    def sx(img, x, y):
        # one map row per point: OpenCV refuses maps of SHRT_MAX (32767) rows or more (a long
        # outline, e.g. zone limits of the line style), so long outlines are sampled in slices
        # (each point is interpolated on its own: same values)
        return np.concatenate([cv2.remap(img, x[i:i + _REMAP_ROWS], y[i:i + _REMAP_ROWS], cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                               for i in range(0, max(1, len(x)), _REMAP_ROWS)])
    x, y = pts[:, :1].copy(), pts[:, 1:].copy()
    nx, ny = -sx(gx, x, y), -sx(gy, x, y)              # outwards: the mask decreases
    n = np.hypot(nx, ny)
    ok = n > 1e-3
    nx, ny = np.where(ok, nx / np.maximum(n, 1e-6), 0), np.where(ok, ny / np.maximum(n, 1e-6), 0)
    step = 0.125
    ts = np.arange(-1.0, 1.5 + step / 2, step, dtype=np.float32)[None, :]
    v = sx(f, (x + ts * nx).astype(np.float32), (y + ts * ny).astype(np.float32))
    a, b = v[:, :-1], v[:, 1:]
    cross = (a >= 0.5) & (b < 0.5)
    tc = ts[:, :-1] + step * np.clip((a - 0.5) / np.maximum(a - b, 1e-6), 0, 1)
    tc = np.where(cross, tc, np.inf)
    k = np.argmin(np.abs(tc - 0.5), axis=1)
    t = tc[np.arange(len(pts)), k]
    t = np.where(np.isfinite(t), t, 0.5)
    t = np.where(ok[:, 0], t, 0.0)[:, None]
    return np.concatenate([x + t * nx, y + t * ny], axis=1).astype(np.float32)


def _smooth_closed(pts: np.ndarray, sigma: float) -> np.ndarray:
    n = len(pts)
    r = int(min(3 * sigma, (n - 1) // 2))
    if r < 1:
        return pts
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    ext = np.concatenate([pts[-r:], pts, pts[:r]])
    xs = np.convolve(ext[:, 0], k, mode="valid")
    ys = np.convolve(ext[:, 1], k, mode="valid")
    return np.stack([xs, ys], axis=1).astype(np.float32)


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

@dataclass
class Layer:
    index: int               # 1..n (spraying order)
    color: np.ndarray        # BGR
    target: np.ndarray       # target mask (image area)
    opening: np.ndarray      # final opening mask (whole sheet)
    lost: np.ndarray         # details lost to the constraints (image area)
    bridges: list
    paths: list              # outlines in mm (sheet coordinates)
    islands: int = 0         # number of islands connected by bridges
    short_islands: int = 0   # islands that could not get the requested minimum of bridges
    struct_bridges: int = 0  # structural bridges (included in `bridges`), see add_struct_bridges
    weak_spots: int = 0      # parts held from one side only that got no structural bridge
    weak_points: tuple = ()  # where they are: (x, y, radius) in sheet mm
    kind: str = "color"      # "color" (flat colour), "outline" (key-detail outlines) or "primer" (index 0, see primer_layer)
    merged: np.ndarray | None = None   # areas changed by merges instead of bridges (whole sheet)
    n_merged: int = 0        # number of merges (closed channels + removed islands)
    tone: int | None = None  # index of the tone it paints in prepared.palette (colour layers)
    sources: tuple = ()      # line stencils of the "line" style: details:<group> | silhouette | zones | tone<n>
    group: str = ""          # outline stencil: key of its outline layer (A, B...)
    manual_bridges: int = 0  # manual bridges drawn (included in `bridges`), see apply_manual_bridges
    # opening pixels the bridges (manual and automatic) cut (whole sheet, bool), so that
    # opening_without_bridges puts back exactly those; None in results saved before it existed
    bridge_cut: np.ndarray | None = None


@dataclass
class Result:
    params: Params
    prepared: Prepared
    layers: list
    support_color: np.ndarray    # BGR
    support_label: int | None    # tone left to the support (no stencil)
    sheet_mm: tuple              # (width, height) of the stencil sheet
    support_mm: tuple
    image_mm: tuple
    support_px: tuple            # (width, height)
    img_off_px: tuple            # (x, y) of the image in the sheet
    sup_off_px: tuple = (0, 0)   # (x, y) of the support's top-left corner in the sheet (may be < 0)
    sheet_rect_mm: tuple = (0.0, 0.0, 0.0, 0.0)  # sheet in support coordinates (x0, y0, x1, y1)
    detect_sim: np.ndarray | None = None  # simulated render the key-detail search compared against
    lines_added: int = 0         # key-detail lines merged into the colour layers
    zone_lines: dict = field(default_factory=dict)   # whole lines: {zone id: (lines kept, lines detected)}
    detect_painted: np.ndarray | None = None  # layer painting each pixel before the lines were merged
    detect_sim_pre: np.ndarray | None = None  # same render without the layer effects (None = no effect)
    manual: tuple = ()           # state of each manual bridge (see manual_report)

    @property
    def sup_off_mm(self) -> tuple:
        return (-self.sheet_rect_mm[0], -self.sheet_rect_mm[1])


def _lab(bgr: np.ndarray) -> np.ndarray:
    arr = np.asarray(bgr, np.float32).reshape(1, -1, 3) / 255.0
    return cv2.cvtColor(arr, cv2.COLOR_BGR2LAB).reshape(-1, 3)


def merge_line_islands(opening: np.ndarray, p: Params, color, support) -> tuple[np.ndarray, int]:
    """"Merge instead of bridges" for a line stencil (lines-only style, separate outline stencil):
    small islands enclosed by lines are painted (opened) instead of being held by bridges; the
    lines themselves are never closed (no channel merge across a line). Same cost and scale as the
    colour layers. Modifies `opening` in place; returns (mask of the changed areas, number of merges)."""
    de = float(np.linalg.norm(_lab(np.asarray(color, np.uint8)) - _lab(np.asarray(support, np.uint8))))
    return merge_zones(opening, replace(p, merge_line_islands=True), None,
                       float(np.clip(de / 40.0, 0.15, 1.0)), opening > 0)


def colors_only(p: Params) -> bool:
    """True when the layer paints (`layer_colors`) do not change the cut: flat-colour style, no
    merge (weighted by the colour difference), no key-detail search (made against the simulated
    render of the paints), no silhouette limited to weak edges (same). A change of paint alone can
    then reuse the geometry of the previous result (see `recolor`)."""
    merges = p.merge_amount > 0 and (p.merge_channels or p.merge_islands)
    return (p.stencil_style == "fill" and not merges and not p.outline
            and not (p.outline_subject and p.outline_subject_weak))


def recolor(res: "Result", p: Params) -> "Result":
    """Previous result with the paints of `p.layer_colors` and `p.primer_color` (same openings, bridges and paths):
    only valid when `colors_only(p)` and nothing but `layer_colors` changed."""
    paints = parse_layer_colors(p.layer_colors)
    layers = [replace(L, color=hex_to_bgr(p.primer_color)) if L.kind == "primer"
              else L if L.kind == "outline" or L.tone is None
              else replace(L, color=paints.get(L.index, res.prepared.palette[L.tone]))
              for L in res.layers]
    return replace(res, params=p, layers=layers)


def layer_job(job):
    """Manual bridges + merges + bridges + vectorisation of one layer (module-level function: can
    run in another process). `protect`: areas no merge may change (lines added to the layer);
    `weight`: see merge_zones; `manual`: manual bridges of the layer (see apply_manual_bridges)."""
    opening, hidden, p, weight, protect, manual, others = job
    s = opening.copy()
    m_drawn, m_states, forbid = apply_manual_bridges(opening, manual, p)
    cut = _cut_by(s, opening)
    merged, n_merged = merge_zones(opening, p, hidden, weight, protect, others)
    s = opening.copy()
    pre = s if needs_pre(manual) else None
    held = hold_material(opening, p, hidden, forbid, m_drawn)
    held, l_drawn, l_states = late_manual(opening, pre, held, manual, p)   # automatic bridges kept
    cut = _bridge_cut(cut | _cut_by(s, opening), opening, merged)
    return (opening, held, vectorize(opening, p), merged, n_merged, (m_drawn + l_drawn, m_states + l_states),
            cut)


def _cut_by(before: np.ndarray, opening: np.ndarray) -> np.ndarray:
    """Opening pixels of `before` that are material in `opening` (cut by the bridges drawn since)."""
    return (before > 0) & (opening == 0)


def _bridge_cut(cut: np.ndarray, opening: np.ndarray, merged) -> np.ndarray | None:
    """Layer.bridge_cut: what the bridges cut and is still material, minus the merged areas."""
    cut &= opening == 0
    if merged is not None:
        cut &= ~(np.asarray(merged) > 0)
    return cut if cut.any() else None


def _outline_paints(cut0: np.ndarray, s: np.ndarray, opening: np.ndarray, merged) -> np.ndarray:
    """What an outline stencil paints once its bridges are retouched, its merges included: `cut0`
    cut by its manual bridges, `s` its opening before the automatic bridges, `opening` after."""
    cut = _bridge_cut(cut0 | _cut_by(s, opening), opening, merged)
    return (opening > 0) if cut is None else ((opening > 0) | cut)


# True: the first computation of the colour layers keeps its bridges when an exclusive outline
# stencil makes them computed twice (former behaviour: the outline stencil is then searched on the
# render with the bridges; about 1.5 s slower on a large sheet)
FIRST_PASS_BRIDGES = False


# Parallelism between layers: "thread" (default), "process" (separate processes), or "none" (serial)
PARALLEL = os.environ.get("OVERSPRAYLAB_PARALLEL", "thread")
_POOL = None


def _pool():
    global _POOL
    if _POOL is None:
        from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
        n = max(1, min(8, (os.cpu_count() or 2) - 1))
        _POOL = ProcessPoolExecutor(n) if PARALLEL == "process" else ThreadPoolExecutor(n)
    return _POOL


_THREADS = None


def _thread_pool():
    global _THREADS
    if _THREADS is None:
        from concurrent.futures import ThreadPoolExecutor
        _THREADS = ThreadPoolExecutor(max(2, min(8, (os.cpu_count() or 2) - 1)), thread_name_prefix="work")
    return _THREADS


def run_threads(funcs: list) -> list:
    """Runs independent functions in threads (OpenCV, numpy, onnxruntime and the PNG encoding
    release the GIL) and returns their results in order; serial with OVERSPRAYLAB_PARALLEL=none.
    Each thread sees the cancellation of the current computation. The functions must not call
    run_threads themselves (a full pool would wait on itself)."""
    if PARALLEL == "none" or len(funcs) < 2:
        return [f() for f in funcs]
    futures = [_thread_pool().submit(contextvars.copy_context().run, f) for f in funcs]
    return [f.result() for f in futures]


# Results of the last layer jobs, by content of their inputs: a change that leaves a layer as it
# was (e.g. an effect of another layer) does not recompute its merges, bridges and paths.
_JOB_CACHE: dict = {}
_JOB_CACHE_MAX = 24
# fields with no influence on layer_job (or already reflected in its masks)
_JOB_KEY_SKIP = {"layer_colors", "layer_effects", "spray_mm", "manual_bridges",
                 "primer", "primer_color", "primer_inset_mm", "primer_lines"}


def _job_key(job) -> bytes:
    import hashlib
    opening, hidden, p, weight, protect, manual, others = job
    h = hashlib.blake2b(digest_size=20)
    for a in (opening, hidden, protect, others):
        if a is None:
            h.update(b"-")
        else:
            h.update(repr((a.shape, a.dtype.str)).encode())
            h.update(np.ascontiguousarray(a).tobytes())
    h.update(repr((round(weight, 6), manual, [(f.name, getattr(p, f.name)) for f in fields(p)
                                      if f.name not in _JOB_KEY_SKIP])).encode())
    return h.digest()


def _run_jobs(jobs: list) -> list:
    if PARALLEL == "none" or len(jobs) < 2:
        return [layer_job(j) for j in jobs]
    try:
        pool = _pool()
        if PARALLEL == "process":      # a job already started in another process runs to its end
            futures = [pool.submit(layer_job, j) for j in jobs]
        else:                          # each thread sees the cancellation of this computation
            futures = [pool.submit(contextvars.copy_context().run, layer_job, j) for j in jobs]
        pending = set(futures)
        try:
            while pending:
                _, pending = wait_futures(pending, timeout=0.05)
                check_cancel()
        except Cancelled:
            for f in futures:
                f.cancel()
            raise
        return [f.result() for f in futures]
    except Exception:                  # pool unavailable (e.g. child process killed): serial
        return [layer_job(j) for j in jobs]


def run_layer_jobs(jobs: list) -> list:
    keys = [_job_key(j) for j in jobs]
    todo = [i for i, k in enumerate(keys) if k not in _JOB_CACHE]
    done = _run_jobs([jobs[i] for i in todo])
    for i, r in zip(todo, done):
        _JOB_CACHE[keys[i]] = r
    out = [_JOB_CACHE[k] for k in keys]
    for k in keys:                     # most recently used last
        _JOB_CACHE[k] = _JOB_CACHE.pop(k)
    while len(_JOB_CACHE) > _JOB_CACHE_MAX:
        _JOB_CACHE.pop(next(iter(_JOB_CACHE)))
    return out


def build(prep: Prepared, p: Params, outline_region: np.ndarray | None = None,
          zone_regions: dict | None = None, src_map: tuple | None = None, keep: dict | None = None) -> Result:
    """`zone_regions`: {zone id: boolean mask (image area) limiting the key-detail search of the
    zone, or None = whole image}; without it, `outline_region` is the first zone's.
    `src_map`: ((x0, y0, x1, y1) part of the source image computed, (width, height) of the
    source), to place the manual bridges (None: manual bridges ignored).
    `keep`: filled with what `rebridge` needs to redo only the layers whose manual bridges change
    (left empty for the line style)."""
    labels = prep.labels
    H, W = labels.shape
    ppm = p.px_per_mm
    SW = int(round(p.support_w_mm * ppm))
    SH = int(round(p.support_h_mm * ppm))
    # Sheet in support coordinates (the visible part of the image is derived from the preparation)
    vis = (prep.off_px[0] / ppm, prep.off_px[1] / ppm, (prep.off_px[0] + W) / ppm, (prep.off_px[1] + H) / ppm)
    sx0, sy0, sx1, sy1 = sheet_rect(p, vis)
    ox, oy = int(round(-sx0 * ppm)), int(round(-sy0 * ppm))       # support in the sheet
    sheet_shape = (max(int(round((sy1 - sy0) * ppm)), oy + prep.off_px[1] + H + 1),
                   max(int(round((sx1 - sx0) * ppm)), ox + prep.off_px[0] + W + 1))
    X0, Y0 = ox + prep.off_px[0], oy + prep.off_px[1]
    support = hex_to_bgr(p.support_color)
    manual = ManualBridges(p, src_map, (X0, Y0), (W, H))

    n_colors = len(prep.palette)
    sup = None
    if p.skip_support and n_colors > 1:
        d = np.linalg.norm(_lab(prep.palette) - _lab(support), axis=1)
        sup = int(np.argmin(d))
    color_ids = [i for i in range(n_colors) if i != sup]
    extra_area = p.detail_settings()[3]
    exempts = area_exempt(prep, p, color_ids)    # forced flat zones without the detail supplement

    def place(mask: np.ndarray) -> np.ndarray:
        full = np.zeros(sheet_shape, np.uint8)
        full[Y0:Y0 + H, X0:X0 + W] = mask
        return full

    # Lines: key-detail zones (`outline`) and/or silhouette (`outline_subject`), each can be on
    # without the other; each one goes to an outline layer or into the colour layers.
    entries, groups = outline_entries(p, prep, zone_regions, outline_region)
    _prefetch_lines(prep, p, entries, groups, support)

    targets, cleaned_all = [], []
    if p.stencil_style == "line":
        color_ids = []
    for li, ci in enumerate(color_ids):
        if p.layering == "cumulative":
            t = np.isin(labels, color_ids[li:])
        else:
            t = labels == ci
        if prep.fg is not None:            # removed background: left as bare support
            t &= prep.fg
        targets.append(np.where(t, 255, 0).astype(np.uint8))
    # the layers are cleaned independently of each other: in parallel
    cleaned_all = run_threads([(lambda t=t, e=e: clean_mask(t, p, extra_area, e))
                               for t, e in zip(targets, exempts)])
    if p.fill_gaps and p.layering == "exclusive" and cleaned_all:   # cumulative: the layer below shows
        cleaned_all, targets = fill_gaps(cleaned_all, targets, labels, prep.palette, color_ids, p)

    # Effects of the layers (each layer its own stack), applied before lines, merges and bridges
    fx_all = fxmod.parse(p.layer_effects)  # also the outline stencils (numbered after the colour layers)
    fx = fx_all if color_ids else {}
    pre_fx = list(cleaned_all)            # without effects: what the "removed details" are compared to
    base_masks = list(cleaned_all)        # colour layers before their effects (key details "before effects")
    fx_under = {}                         # Overlap effect: under the next layers on purpose
    if fx:
        limit = None if prep.fg is None else np.where(prep.fg, 255, 0).astype(np.uint8)
        cleaned_all = fxmod.apply_all(cleaned_all, fx, p, prep.resized, prep.off_px, limit,
                                      fxmod.options(p.layer_effects), fx_under)
    excl = p.layering == "exclusive"
    if excl:                              # each spot painted by one layer only (material limits)
        cleaned_all = exclusive_layers(cleaned_all, targets, p, extra_area, fx_under, exempts)

    paints = parse_layer_colors(p.layer_colors)
    colors = [paints.get(li + 1, prep.palette[ci]) for li, ci in enumerate(color_ids)]

    def make_result(layers, **kw):
        kw.setdefault("manual", manual.report())
        return Result(params=p, prepared=prep, layers=layers, support_color=support, support_label=sup,
                      sheet_mm=(sheet_shape[1] / ppm, sheet_shape[0] / ppm),
                      support_mm=(SW / ppm, SH / ppm), image_mm=(W / ppm, H / ppm),
                      support_px=(SW, SH), img_off_px=(X0, Y0), sup_off_px=(ox, oy),
                      sheet_rect_mm=(sx0, sy0, sx0 + sheet_shape[1] / ppm, sy0 + sheet_shape[0] / ppm), **kw)

    if p.stencil_style == "line":
        layers = line_stencils(prep, p, entries, groups, support, place, fx_all, manual)
        primer = primer_layer(layers, p, manual, None, (X0, Y0, W, H))
        return make_result(([primer] if primer is not None else []) + layers, zone_lines=_zone_lines(entries))

    # Key-detail lines merged into the colour layers (instead of a separate outline stencil):
    # searched against the render of the layers before bridges, then added to the chosen layer.
    detect_sim, detect_painted, lines_added, strokes = None, None, 0, {}
    into = [e for e in entries if into_layers(e[2])]
    # render of the colour layers before their effects, for the zones that search "before effects"
    need_pre = bool(fx) and any(e[0].outline_ref == "layers" for e in entries)
    flat = np.empty((H, W, 3), np.uint8)       # "image alone": compared with the bare support
    flat[:] = np.asarray(support, np.uint8)

    def ref_sim(pe, post, pre):
        if pe.outline_ref == "image":
            return flat
        return pre if (pe.outline_ref == "layers" and pre is not None) else post
    detect_sim_pre = None

    def render(masks):
        return image_sim(make_result([Layer(index=li + 1, color=colors[li], target=targets[li], opening=place(c),
                                            lost=c, bridges=[], paths=[]) for li, c in enumerate(masks)]))
    if into and color_ids:
        detect_sim = render(cleaned_all)
        if need_pre:
            detect_sim_pre = render(base_masks)
        painted = np.full((H, W), -1, np.int32)      # last layer painting each pixel (-1 = support)
        for li, c in enumerate(cleaned_all):
            painted[c > 0] = li
        # the zones are independent: searched in parallel, their strokes merged in their order
        found = run_threads([
            (lambda pe=pe, reg=reg: lines_into_layers(prep, ref_sim(pe, detect_sim, detect_sim_pre), pe, reg,
                                                      color_ids, painted))
            for pe, reg, _, _ in into])
        alone = {}                        # lines removed from the other layers (outline_exclusive)
        for (pe, _, _, _), (s, n_added) in zip(into, found):
            lines_added += n_added
            for li, stroke in s.items():
                strokes[li] = stroke if li not in strokes else (strokes[li] | stroke)
                if excl and pe.outline_exclusive:
                    alone[li] = stroke if li not in alone else (alone[li] | stroke)
        detect_painted = painted
        for li, stroke in strokes.items():
            line = clean_mask(stroke, p)
            targets[li] = targets[li] | stroke
            before = cleaned_all[li]
            cleaned_all[li] = clean_mask(cleaned_all[li] | line, p)
            if li + 1 in fx:
                pre_fx[li] = pre_fx[li] | line
            if excl:
                # what the cleaning added around the line over another layer: given back (over a
                # lighter one always, over a darker one unless the layer may paint under it)
                extra = (cleaned_all[li] > 0) & (before == 0) & (line == 0)
                for k, other in enumerate(cleaned_all):
                    if k != li and (k < li or not p.under_darker):
                        extra_k = extra & (other > 0)
                        if extra_k.any():
                            cleaned_all[li] = trim_taken(cleaned_all[li], extra_k, p)
        for li, stroke in alone.items():  # only their layer paints these lines
            line = (clean_mask(stroke, p) > 0) & (cleaned_all[li] > 0)
            for k in range(len(cleaned_all)):
                if k != li and (cleaned_all[k][line] > 0).any():
                    cleaned_all[k] = trim_taken(cleaned_all[k], line, p, extra_area, exempts[k])
                    targets[k] = np.where(line, np.uint8(0), targets[k])
                    pre_fx[k] = np.where(line, np.uint8(0), pre_fx[k])

    # Each layer (bridges + vectorisation) is independent of the others: computed in parallel
    by_layer = manual.assign("color", [(li + 1, prep.palette[ci]) for li, ci in enumerate(color_ids)])

    def layer_job_of(li, ci, forbid_extra=None):
        check_cancel()
        cleaned = cleaned_all[li]
        hidden = None
        if li + 1 < len(color_ids):
            later = np.zeros_like(cleaned)
            for c2 in cleaned_all[li + 1:]:
                later |= c2
            hidden = place(later) > 0
        # visibility of a merge: contrast between this paint and what shows without it
        under = colors[li - 1] if (p.layering == "cumulative" and li > 0) else support
        de = float(np.linalg.norm(_lab(colors[li]) - _lab(under)))
        protect = place(strokes[li]) > 0 if li in strokes else None
        # exclusive: the merges may not paint the areas of the lighter layers (nor of the darker
        # ones, unless the layer may paint under them)
        others = None
        if excl and len(color_ids) > 1:
            o = np.zeros_like(cleaned)
            for k, c2 in enumerate(cleaned_all):
                if k < li or (k > li and not p.under_darker):
                    o |= c2
            others = place(o) > 0 if o.any() else None
        if forbid_extra is not None:      # outline stencils excluded from the colour layers
            others = forbid_extra if others is None else (others | forbid_extra)
        return (place(cleaned), hidden, p, float(np.clip(de / 40.0, 0.15, 1.0)), protect,
                manual.segments(by_layer.get(li + 1, [])), others)

    def color_layer(li, ci, result, mb=manual):
        (opening, (bridges, n_isl, n_short, n_struct, n_weak), paths, merged, n_merged,
         (m_drawn, m_states), cut) = result
        mb.record(li + 1, m_states)
        target, cleaned = targets[li], cleaned_all[li]
        return Layer(index=li + 1, color=colors[li], target=target,
                     opening=opening, lost=cv2.bitwise_xor(pre_fx[li] if li + 1 in fx else cleaned, target),
                     bridges=list(m_drawn) + bridges, paths=paths,
                     islands=n_isl, short_islands=n_short, struct_bridges=n_struct, weak_spots=len(n_weak), weak_points=_weak_mm(n_weak, p),
                     merged=merged, n_merged=n_merged, tone=ci, manual_bridges=len(m_drawn),
                     bridge_cut=cut)

    # Outline stencils, painted last, above all layers: one per outline layer (A, B...), made of
    # the zones (and silhouette) sent to it
    order = [g for g in groups if any(e[2] == g for e in entries)]
    # An outline stencil excluded from the colour layers (exclusive stacking) makes them computed
    # twice: first for the render it is searched on, then without its lines (below). The first
    # computation skips the automatic bridges (render of the sheet once they are retouched): they
    # are computed once, in the second one, which every colour layer then goes through.
    two_pass = bool(excl and color_ids and any(group_exclusive(p, groups[g]) for g in order))
    p_first = replace(p, bridges=False) if two_pass and p.bridges and not FIRST_PASS_BRIDGES else None
    jobs1 = [layer_job_of(li, ci) for li, ci in enumerate(color_ids)]
    # layer_job changes the opening of its job: jobs1 kept as is for the second computation
    results = run_layer_jobs([(j[0].copy(), j[1], p_first or p, *j[3:]) for j in jobs1])
    layers = [color_layer(li, ci, r) for (li, ci), r in zip(enumerate(color_ids), results)]
    res = make_result(layers, detect_sim=detect_sim, detect_painted=detect_painted, lines_added=lines_added,
                      detect_sim_pre=detect_sim_pre, zone_lines=_zone_lines(entries))

    out_masks = []                         # outline stencils done (image area), for their effects
    excluded = None                        # outline stencils excluded from the colour layers (image area)
    pending = []                           # (index in layers, bridges + paths of an outline stencil)
    outline_info = []                      # what rebridge needs to redo an outline stencil

    def outline_tail(mb, o, mine):
        """Manual bridges and merges of the outline stencil `o` (dict made below), from its opening
        before them: (what it paints if exclusive, function making its bridges and paths, which
        returns (Layer, what it paints once its bridges are retouched if exclusive))."""
        opening = o["base"].copy()
        n_out, exclusive = o["n_out"], o["exclusive"]
        s = opening.copy()
        m_drawn, forbid = mb.apply(opening, n_out, mine)
        cut0 = _cut_by(s, opening)
        merged, n_merged = merge_line_islands(opening, p, o["color"], support)
        s = opening.copy()
        # what the stencil paints once its bridges are retouched (its merges included), taken
        # before they are cut: checked once they are (_outline_paints)
        part = _outline_paints(cut0, s, s, merged)[Y0:Y0 + H, X0:X0 + W] if exclusive else None

        def finish():
            pre = s if needs_pre(mb.segments(mine)) else None
            held = hold_material(opening, p, None, forbid, m_drawn)
            held, l_drawn = mb.late(opening, pre, n_out, mine, held)
            cut = _bridge_cut(cut0 | _cut_by(s, opening), opening, merged)
            bridges, n_isl, n_short, n_struct, n_weak = held
            drawn = m_drawn + l_drawn
            paints = _outline_paints(cut0, s, opening, merged)[Y0:Y0 + H, X0:X0 + W] if exclusive else None
            return Layer(index=n_out, color=o["color"], target=o["target"], opening=opening,
                         lost=o["lost"], bridges=drawn + bridges, paths=vectorize(opening, p),
                         islands=n_isl, short_islands=n_short, struct_bridges=n_struct, weak_spots=len(n_weak), weak_points=_weak_mm(n_weak, p),
                         merged=merged, n_merged=n_merged,
                         kind="outline", group=o["g"], manual_bridges=len(drawn), bridge_cut=cut), paints
        return part, finish

    sim = sim_pre = None
    for g in order:
        check_cancel()
        if sim is None:
            # the outline stencils come on top; searched on the render of the colour layers once
            # their bridges are retouched (opening_without_bridges): a bridge changes the lines only
            # through the merges it changes (see rebridge)
            sim = res.detect_sim = image_sim(_retouched(res), keep=True)
            if need_pre:           # same render (lines added to the layers included), effects left out
                pre = [base_masks[li] | strokes[li] if li in strokes else base_masks[li] for li in range(len(base_masks))]
                sim_pre = res.detect_sim_pre = render(pre)
        color = outline_color(prep, support, group_params(p, groups[g]))
        dark = outline_is_dark(color, support)
        target, gap_zones, w_sil = None, None, 0.0
        in_group = [e for e in entries if e[2] == g]       # its zones: searched in parallel
        found = run_threads([(lambda pe=pe, reg=reg: detect_key_details(prep, ref_sim(pe, sim, sim_pre), pe, reg, dark))
                             for pe, reg, _, _ in in_group])
        for (pe, reg, dest, is_sil), t in zip(in_group, found):
            sim_e = ref_sim(pe, sim, sim_pre)
            target = t if target is None else (target | t)
            if is_sil:
                _, gap_zones = subject_silhouette(prep, pe, sim_e, gaps=True)
                w_sil = max(pe.outline_width_mm, pe.min_opening_mm * 1.05)
        if gap_zones is not None:          # interruptions of the silhouette: no other line there either
            target[cv2.dilate(gap_zones.astype(np.uint8), disk(w_sil * ppm / 2)) > 0] = 0
        if target is not None and target.any():
            cleaned = clean_mask(target, p)
            lost = cv2.bitwise_xor(cleaned, target)
            n_out = len(layers) + 1
            if fx_all.get(n_out):          # effects of the outline stencil: on it alone
                cleaned = fxmod.apply_alone(cleaned, fx_all[n_out], p, prep.resized, prep.off_px,
                                            cleaned_all + out_masks)
            out_masks.append(cleaned)
            o = dict(pos=len(layers), n_out=n_out, g=g, base=place(cleaned), color=color, target=target,
                     lost=lost, exclusive=bool(excl and color_ids and group_exclusive(p, groups[g])))
            mine = manual.assign("outline", [(n_out, g)]).get(n_out, [])
            o["segs"] = manual.segments(mine)
            part, finish = outline_tail(manual, o, mine)
            if part is not None:
                excluded = part if excluded is None else (excluded | part)
            outline_info.append(o)
            pending.append((len(layers), finish))
            layers.append(None)            # filled below

    saved = (list(cleaned_all), list(targets), list(pre_fx))
    finals, last_redo = {}, []             # jobs of the second computation (not run, kept as is), redone layers

    def second_jobs(excluded):
        """(colour layers computed again, their jobs). Outline stencils `excluded` (image area): only
        they paint their lines, so the colour layers painting under them are computed without them
        (bridges, merges and paths), their merges kept off the lines. A first computation without
        the bridges: every colour layer, the others with the same masks as the first time."""
        cleaned_all[:], targets[:], pre_fx[:] = (list(v) for v in saved)
        redo, ex_sheet = [], None
        if excluded is not None:
            ex_sheet = place(excluded.astype(np.uint8) * 255) > 0
            for li in range(len(color_ids)):
                img_part = layers[li].opening[Y0:Y0 + H, X0:X0 + W] > 0
                if (img_part & excluded).any() or (cleaned_all[li][excluded] > 0).any():
                    redo.append(li)
            for li in redo:
                cleaned_all[li] = trim_taken(cleaned_all[li], excluded, p, extra_area, exempts[li])
                targets[li] = np.where(excluded, np.uint8(0), targets[li])
                pre_fx[li] = np.where(excluded, np.uint8(0), pre_fx[li])
        todo = list(range(len(color_ids))) if p_first is not None else redo
        finals.clear()
        last_redo[:] = redo
        for li in todo:                   # layer_job changes the opening of its job: run on a copy
            finals[li] = layer_job_of(li, color_ids[li], ex_sheet) if li in redo else jobs1[li]
        return todo, [(finals[li][0].copy(), *finals[li][1:]) for li in todo]

    def outlines():
        return [(i, *fin()) for i, fin in pending]

    def exact(done):
        out = None
        for _, _, part in done:
            if part is not None:
                out = part if out is None else (out | part)
        return out

    if pending and PARALLEL != "none":
        # The bridges and paths of the outline stencils are made while the colour layers are
        # computed again without them, from what they paint before their bridges. If a bridge
        # changes it (one crossing a merged area), the colour layers are computed once more.
        job = _thread_pool().submit(contextvars.copy_context().run, outlines)
        todo, jobs = second_jobs(excluded)
        again = run_layer_jobs(jobs) if jobs else []
        done = job.result()
        ex = exact(done)
        if not (ex is None and excluded is None) and (ex is None or excluded is None or not np.array_equal(ex, excluded)):
            todo, jobs = second_jobs(ex)
            again = run_layer_jobs(jobs) if jobs else []
    else:
        done = outlines()
        todo, jobs = second_jobs(exact(done))
        again = run_layer_jobs(jobs) if jobs else []
    first_layers = layers[:len(color_ids)]      # the ones the outline stencils were searched against
    for i, L, _ in done:
        layers[i] = L
    for li, r in zip(todo, again):
        layers[li] = color_layer(li, color_ids[li], r)
    res.zone_lines = _zone_lines(entries)       # counted once every zone has been searched
    # primer stencil: under what the colour layers (and the outline stencils if asked) finally paint
    painted = None
    if p.primer and color_ids:
        t = np.zeros((H, W), np.uint8)
        for tg in targets:
            t |= tg
        painted = place(t) > 0
    primer = primer_layer(layers, p, manual, painted, (X0, Y0, W, H))
    if primer is not None:
        res.layers = [primer, *layers]
    res.manual = manual.report()                # with the outline stencils and the primer
    if keep is not None and src_map is not None:
        for o, (_, _, paints) in zip(outline_info, done):
            o["paints"] = paints
        keep.update(
            manual_args=(src_map, (X0, Y0), (W, H)),
            refs=[(li + 1, prep.palette[ci]) for li, ci in enumerate(color_ids)], color_ids=list(color_ids),
            # job of the last computation of each colour layer, and of the first one (two passes)
            jobs={li: finals.get(li, jobs1[li]) for li in range(len(color_ids))},
            first={li: (jobs1[li], _retouched_opening(first_layers[li].opening, first_layers[li].bridge_cut))
                   for li in finals},
            first_no_bridges=p_first is not None,
            searched=bool(order), redo=set(last_redo), excluded=exact(done), cleaned=saved[0],
            outlines=outline_info, tail=outline_tail, color_layer=color_layer,
            states=_states_by_layer(manual), painted=painted, img_box=(X0, Y0, W, H))
    return res


def primer_opening(layers: list, p: Params, painted: np.ndarray | None) -> np.ndarray | None:
    """Openings of the primer stencil (whole sheet, uint8 0 / 255), None when nothing is painted:
    what the colour layers paint once their bridges are retouched (effects, merges and lines
    included; the outline stencils too with `p.primer_lines`), the thin gaps the material limits
    leave between two layers filled where a layer is meant to paint (`painted`, whole sheet, bool:
    their targets; never where the image leaves the support bare), then shrunk by
    `p.primer_inset_mm` (a primer edge showing past a colour is far more visible than a colour edge
    on the support) and cleaned with the material limits."""
    check_cancel()
    u = None
    for L in layers:
        if L.kind == "primer" or (L.kind == "outline" and not p.primer_lines):
            continue
        o = _retouched_opening(L.opening, L.bridge_cut) > 0
        u = o if u is None else (u | o)
    if u is None or not u.any():
        return None
    ppm = p.px_per_mm
    m = u.astype(np.uint8) * 255
    r = p.min_strip_mm * ppm / 2
    if painted is not None and r >= 0.5:
        m[(cv2.morphologyEx(m, cv2.MORPH_CLOSE, disk(r)) > 0) & painted] = 255
    r = p.primer_inset_mm * ppm
    if r >= 0.5:
        m = cv2.erode(m, disk(r))
    m = clean_mask(m, p)
    return m if m.any() else None


def primer_layer(layers: list, p: Params, mb: "ManualBridges", painted: np.ndarray | None,
                 img_box: tuple) -> "Layer | None":
    """Primer stencil (index 0, sprayed first) under the final `layers`, or None (off, or nothing
    painted): its openings (primer_opening), then its manual bridges, bridges and paths like a
    colour layer, without merges (a merge would paint primer where no colour covers it).
    `img_box`: (x, y, width, height) of the image area in the sheet."""
    if not p.primer:
        return None
    base = primer_opening(layers, p, painted)
    if base is None:
        return None
    mine = mb.assign("primer", [(0, PRIMER_REF)]).get(0, [])
    job = (base, None, replace(p, merge_amount=0.0), 1.0, None, mb.segments(mine), None)
    (opening, (bridges, n_isl, n_short, n_struct, n_weak), paths, _merged, _n,
     (m_drawn, m_states), cut) = run_layer_jobs([job])[0]
    mb.record(0, m_states)
    X0, Y0, W, H = img_box
    return Layer(index=0, color=hex_to_bgr(p.primer_color), target=base[Y0:Y0 + H, X0:X0 + W].copy(),
                 opening=opening, lost=np.zeros((H, W), np.uint8), bridges=list(m_drawn) + bridges, paths=paths,
                 islands=n_isl, short_islands=n_short, struct_bridges=n_struct, weak_spots=len(n_weak),
                 weak_points=_weak_mm(n_weak, p), kind="primer", manual_bridges=len(m_drawn), bridge_cut=cut)


def _retouched_opening(opening: np.ndarray, cut) -> np.ndarray:
    """Opening once the bridges are retouched (painted over): the pixels they cut (`cut`, see
    Layer.bridge_cut, None = none) put back."""
    return opening if cut is None else np.where(cut, np.uint8(255), opening)


def _retouched(res: Result) -> Result:
    """`res` (just built) with the openings of its layers as after their bridges are retouched."""
    return replace(res, layers=[replace(L, opening=_retouched_opening(L.opening, L.bridge_cut)) for L in res.layers])


def _states_by_layer(mb: "ManualBridges") -> dict:
    """{layer index: [(id, *state)]} of the manual bridges recorded in `mb`."""
    out: dict = {}
    for bid, st in mb.states.items():
        out.setdefault(mb.layer_of[bid], []).append((bid, *st))
    return out


def rebridge(old: Result, p: Params, ctx: dict) -> tuple | None:
    """`build` when only the manual bridges changed since `old`, made with `ctx` (its `keep`): only
    the layers whose manual bridges changed are computed again. None when the change reaches
    further: a colour layer painting something else once its bridges are retouched (a manual
    bridge changes its merges) while the key-detail lines were searched against it, or an
    exclusive outline stencil painting something else; build is needed then.
    Returns (result, ctx of that result)."""
    mb = ManualBridges(p, *ctx["manual_args"])
    body = [L for L in old.layers if L.kind != "primer"]    # positions as in build (primer added last)
    by_layer = mb.assign("color", ctx["refs"])
    jobs, first = dict(ctx["jobs"]), dict(ctx["first"])
    changed = []
    for li, j in ctx["jobs"].items():
        segs = mb.segments(by_layer.get(li + 1, []))
        if segs != j[5]:
            changed.append(li)
            jobs[li] = (j[0], j[1], p, j[3], j[4], segs, j[6])
            if li in first:
                j1, paint = first[li]
                first[li] = ((j1[0], j1[1], p, j1[3], j1[4], segs, j1[6]), paint)
    outlines, redo_out = [], []
    for o in ctx["outlines"]:
        mine = mb.assign("outline", [(o["n_out"], o["g"])]).get(o["n_out"], [])
        segs = mb.segments(mine)
        if segs != o["segs"]:
            o = dict(o, segs=segs)
            redo_out.append((o, mine))
        outlines.append(o)
    # colour layers: their last computation, and the first one where the lines were searched on it
    p1 = replace(p, bridges=False) if ctx["first_no_bridges"] else p
    again1 = [li for li in changed if li in first]
    run = ([(jobs[li][0].copy(), *jobs[li][1:]) for li in changed]
           + [(first[li][0][0].copy(), first[li][0][1], p1, *first[li][0][3:]) for li in again1])
    results = run_layer_jobs(run) if run else []
    fin, res1 = results[:len(changed)], dict(zip(again1, results[len(changed):]))
    if ctx["searched"]:
        X0, Y0 = ctx["manual_args"][1]
        W, H = ctx["manual_args"][2]
        ex = ctx["excluded"]
        for li, r in zip(changed, fin):
            if li in res1:
                r, before = res1[li], first[li][1]
            else:
                before = _retouched_opening(body[li].opening, body[li].bridge_cut)
            if not np.array_equal(_retouched_opening(r[0], r[6]), before):
                return None
            if li in res1 and ex is not None:   # still computed again without the outline lines?
                redo = bool(((r[0][Y0:Y0 + H, X0:X0 + W] > 0) & ex).any() or (ctx["cleaned"][li][ex] > 0).any())
                if redo != (li in ctx["redo"]):
                    return None
    # outline stencils: their manual bridges, merges, bridges and paths
    done = run_threads([(lambda o=o, mine=mine: ctx["tail"](mb, o, mine)[1]()) for o, mine in redo_out])
    for (o, _), (_, paints) in zip(redo_out, done):
        o["paints"] = paints
    if any(o["exclusive"] for o, _ in redo_out):
        ex = None
        for o in outlines:
            if o["paints"] is not None:
                ex = o["paints"] if ex is None else (ex | o["paints"])
        if (ex is None) != (ctx["excluded"] is None) or (ex is not None and not np.array_equal(ex, ctx["excluded"])):
            return None
    layers = list(body)
    for li, r in zip(changed, fin):
        layers[li] = ctx["color_layer"](li, ctx["color_ids"][li], r, mb)
    for (o, _), (L, _) in zip(redo_out, done):
        layers[o["pos"]] = L
    # the primer follows what the layers paint (a merge a bridge changes) and its own manual bridges
    # (its layer job is cached: unchanged, it costs only the comparison of its inputs)
    primer = primer_layer(layers, p, mb, ctx.get("painted"), ctx["img_box"])
    # states of the bridges of the layers kept as they were
    redone = {li + 1 for li in changed} | {o["n_out"] for o, _ in redo_out} | {0}
    for index, states in ctx["states"].items():
        if index not in redone:
            mb.record(index, states)
    res = replace(old, params=p, layers=([primer] if primer is not None else []) + layers, manual=mb.report())
    return res, dict(ctx, jobs=jobs, first=first, outlines=outlines, states=_states_by_layer(mb))


# --------------------------------------------------------------------------
# Key-detail zones: each zone (painted area, or the whole image) has its own detection settings and
# a destination: an outline layer (A, B... several zones sent to the same one are merged into one
# stencil) or the colour layers (darkest / nearest / "L<n>" = the layer at position n). The colour belongs to the outline layer.
#   {"zones": [{"id": "z1", "name": "Zone 1", "on": true, "method": "canny", "sensitivity": 50,
#               "width_mm": 0.8, "min_length_mm": 4, "dest": "A"}],
#    "groups": {"A": {"auto": true, "color": "#1e1e1e"}}, "sil_dest": "A"}
# --------------------------------------------------------------------------
_ZONE_ID = re.compile(r"^z\d{1,3}$")
_GROUP = re.compile(r"^[A-Z]$")


def _hex(v, default):
    v = str(v or "").strip().lower()
    return v if len(v) == 7 and v.startswith("#") and all(c in "0123456789abcdef" for c in v[1:]) else default


def outline_plan(p: Params) -> tuple[list, dict, str]:
    """(zones, groups, silhouette destination). Without `outline_zones`: one zone made of the
    outline_* settings (projects of earlier versions), its destination from `outline_target`."""
    legacy = "A" if p.outline_target == "stencil" else p.outline_target
    groups = {"A": {"auto": bool(p.outline_auto_color), "color": _hex(p.outline_color, "#1e1e1e")}}
    zones = [dict(id="z1", name="Zone 1", on=True, method=p.outline_method, sensitivity=p.outline_sensitivity,
                  width_mm=p.outline_width_mm, min_length_mm=p.outline_min_length_mm, dest=legacy,
                  ref=p.outline_ref, whole=bool(p.outline_whole_lines), line_sens=p.outline_line_sens,
                  close=p.outline_close_lines, channel=p.outline_channel, follow_mm=p.outline_follow_mm,
                  clahe=p.outline_clahe, excl="", gap_every_mm=p.outline_gap_every_mm,
                  gap_mm=p.outline_gap_mm, gap_jitter=p.outline_gap_jitter)]
    sil = legacy
    try:
        data = json.loads(p.outline_zones) if p.outline_zones else None
    except (ValueError, TypeError):
        data = None
    if not isinstance(data, dict) or not isinstance(data.get("zones"), list) or not data["zones"]:
        return zones, groups, sil
    ok_dest = lambda d: into_layers(d) or (isinstance(d, str) and bool(_GROUP.match(d)))
    groups = {}
    for k, g in (data.get("groups") or {}).items():
        if isinstance(k, str) and _GROUP.match(k) and isinstance(g, dict):
            groups[k] = {"auto": g.get("auto") is not False, "color": _hex(g.get("color"), "#1e1e1e"),
                         "excl": g.get("excl") if g.get("excl") in ("on", "off") else ""}
    zones = []
    seen = set()
    gap_every = lambda v: 5.0 if 0 < v < 5 else v
    for z in data["zones"][:20]:
        if not isinstance(z, dict) or not _ZONE_ID.match(str(z.get("id", ""))) or z["id"] in seen:
            continue
        seen.add(z["id"])

        def num(k, lo, hi, d):
            try:
                v = float(z.get(k, d))
            except (TypeError, ValueError):
                v = d
            return float(np.clip(v, lo, hi)) if math.isfinite(v) else d
        dest = z.get("dest") if ok_dest(z.get("dest")) else "A"
        zones.append(dict(id=z["id"], name=str(z.get("name") or z["id"])[:40], on=z.get("on") is not False,
                          method=z.get("method") if z.get("method") in OUTLINE_METHODS else "canny",
                          sensitivity=num("sensitivity", 0, 100, 50.0),
                          width_mm=num("width_mm", 0.2, 10, 0.8), min_length_mm=num("min_length_mm", 0, 200, 4.0),
                          dest=dest, ref=z.get("ref") if z.get("ref") in OUTLINE_REFS else "layers",
                          close=z.get("close") if z.get("close") in CLOSE_LINES else "",
                          channel=z.get("channel") if z.get("channel") in OUTLINE_CHANNELS else "light",
                          whole=z.get("whole") is True, line_sens=num("line_sens", 0, 100, 50.0),
                          follow_mm=num("follow_mm", 0, 50, HYST_MM_DOG), clahe=num("clahe", 0, 100, 0.0),
                          gap_every_mm=gap_every(num("gap_every_mm", 0, 1000, 0.0)),
                          gap_mm=num("gap_mm", 0.3, 20, 1.5),
                          gap_jitter=num("gap_jitter", 0, GAP_JITTER_MAX, 0.0),
                          excl=z.get("excl") if z.get("excl") in ("on", "off") else ""))
    if not zones:
        return outline_plan(replace(p, outline_zones=""))
    sil = data.get("sil_dest") if ok_dest(data.get("sil_dest")) else zones[0]["dest"]
    for d in [z["dest"] for z in zones] + [sil]:
        if _GROUP.match(d) and d not in groups:
            groups[d] = {"auto": True, "color": "#1e1e1e"}
    groups = dict(sorted(groups.items()))
    return zones, groups, sil


def zone_params(p: Params, z: dict) -> Params:
    """Settings of the key-detail search of one zone (the silhouette is a separate entry)."""
    return replace(p, outline=True, outline_subject=False, outline_method=z["method"],
                   outline_sensitivity=z["sensitivity"], outline_width_mm=z["width_mm"],
                   outline_min_length_mm=z["min_length_mm"], outline_ref=z["ref"], outline_whole_lines=z["whole"],
                   outline_line_sens=z["line_sens"], outline_close_lines=z["close"], outline_channel=z["channel"],
                   outline_follow_mm=z["follow_mm"], outline_clahe=z["clahe"],
                   outline_gap_every_mm=z["gap_every_mm"], outline_gap_mm=z["gap_mm"],
                   outline_gap_jitter=z["gap_jitter"],
                   outline_exclusive=p.outline_exclusive if not z.get("excl") else z["excl"] == "on",
                   outline_target="stencil" if _GROUP.match(z["dest"]) else z["dest"])


def group_exclusive(p: Params, g: dict) -> bool:
    """Outline layer excluded from the colour layers (exclusive stacking): its "excl" ("on" / "off"),
    else the general setting `outline_exclusive`."""
    e = g.get("excl", "")
    return p.outline_exclusive if e not in ("on", "off") else e == "on"


def group_params(p: Params, g: dict) -> Params:
    return replace(p, outline_auto_color=g["auto"], outline_color=g["color"])


def _sil_excl(p: Params) -> str:
    """Silhouette cut into a colour layer, excluded from the other layers: "" = the general setting
    (`outline_exclusive`), "on" / "off" (key "sil_excl" of `outline_zones`)."""
    try:
        data = json.loads(p.outline_zones) if p.outline_zones else None
    except (ValueError, TypeError):
        data = None
    v = data.get("sil_excl") if isinstance(data, dict) else None
    return v if v in ("on", "off") else ""


def outline_entries(p: Params, prep: "Prepared", zone_regions: dict | None,
                    outline_region: np.ndarray | None) -> tuple[list, dict]:
    """Searches to run: [(params, region, destination, is_silhouette)] + groups. Region of a zone:
    its painted mask (`zone_regions`; without it, `outline_region` for the first zone), limited to
    the subject when the background is removed."""
    zones, groups, sil_dest = outline_plan(p)
    entries = []
    if p.outline:
        for z in zones:
            if not z["on"]:
                continue
            if zone_regions is not None:
                r = zone_regions.get(z["id"])
            else:
                r = outline_region if z["id"] == "z1" else None
            if prep.fg is not None:
                r = prep.fg if r is None else (r.astype(bool) & prep.fg)
            pz = zone_params(p, z)
            object.__setattr__(pz, "_zone", z["id"])          # to report its line counts
            entries.append((pz, r, z["dest"], False))
    if p.outline_subject:
        excl = _sil_excl(p)
        entries.append((replace(p, outline=False, outline_subject=True, outline_ref="layers",
                                outline_exclusive=p.outline_exclusive if not excl else excl == "on",
                                outline_width_mm=silhouette_width(p),
                                outline_target="stencil" if _GROUP.match(sil_dest) else sil_dest),
                        None, sil_dest, True))
    return entries, groups


def outline_color(prep: Prepared, support: np.ndarray, p: Params) -> np.ndarray:
    """Line colour: chosen by the user, or the palette tone with the most contrast against the support."""
    if not p.outline_auto_color:
        return hex_to_bgr(p.outline_color).astype(np.uint8)
    L_sup = _lab(support)[0, 0]
    L_pal = _lab(prep.palette)[:, 0]
    return prep.palette[int(np.argmax(np.abs(L_pal - L_sup)))]


def outline_is_dark(color: np.ndarray, support: np.ndarray) -> bool:
    """True when the outline paint is darker than the support: the lines to find are dark ones."""
    lc = _lab(np.asarray(color, np.uint8).reshape(1, 3))[0, 0]
    ls = _lab(np.asarray(support, np.uint8).reshape(1, 3))[0, 0]
    return float(lc) <= float(ls)


def _crest(resp: np.ndarray, floor: float) -> np.ndarray:
    """One-pixel-wide crest lines of a line strength map: points above `floor` that are a maximum
    across the line (non-maximum suppression along the direction of strongest curvature)."""
    g = cv2.GaussianBlur(resp, (0, 0), 1.0)
    dxx = cv2.Sobel(g, cv2.CV_32F, 2, 0, ksize=3)
    dyy = cv2.Sobel(g, cv2.CV_32F, 0, 2, ksize=3)
    dxy = cv2.Sobel(g, cv2.CV_32F, 1, 1, ksize=3)
    # direction across the crest = eigenvector of the most negative curvature
    ang = 0.5 * np.arctan2(2 * dxy, dxx - dyy) + np.pi / 2
    q = np.round(ang / (np.pi / 4)).astype(np.int32) % 4       # 0, 45, 90, 135 degrees
    keep = resp > floor
    pad = np.pad(g, 1, mode="edge")
    H, W = g.shape
    for k, (dx, dy) in enumerate(((1, 0), (1, 1), (0, 1), (-1, 1))):
        a = pad[1 + dy:1 + dy + H, 1 + dx:1 + dx + W]
        b = pad[1 - dy:1 - dy + H, 1 - dx:1 - dx + W]
        keep &= (q != k) | ((g >= a) & (g >= b))
    return keep


def _line_response(L: np.ndarray, ppm: float, method: str, dark: bool,
                   scales: tuple = (0.12, 0.25, 0.45)) -> np.ndarray:
    """Line strength map (L* units, >= 0) for the "xdog" and "ridge" methods.

    xdog : difference of Gaussians, positive on the dark (or light) side of an edge and inside thin
           dark (light) lines -> a single line drawn along each contour, like a pen drawing.
    ridge: multi-scale Hessian (largest curvature across the line, scale-normalised) -> the centre
           line of thin dark (light) strokes (lashes, wrinkles, hair), where Canny finds both edges.
    """
    sign = 1.0 if dark else -1.0
    if method == "xdog":
        sig = max(0.8, 0.15 * ppm)
        d = cv2.GaussianBlur(L, (0, 0), 1.6 * sig) - cv2.GaussianBlur(L, (0, 0), sig)
        return _along_contours(np.maximum(sign * d, 0), L, ppm)
    resp = np.zeros(L.shape, np.float32)
    for smm in scales:                           # default: line widths of ~0.3 to ~1.2 mm
        sig = max(0.8, smm * ppm)
        g = cv2.GaussianBlur(L, (0, 0), sig)
        dxx = cv2.Sobel(g, cv2.CV_32F, 2, 0, ksize=3)
        dyy = cv2.Sobel(g, cv2.CV_32F, 0, 2, ksize=3)
        dxy = cv2.Sobel(g, cv2.CV_32F, 1, 1, ksize=3)
        tr, root = dxx + dyy, np.sqrt((dxx - dyy) ** 2 + 4 * dxy * dxy)
        lam = (tr + root) / 2 if dark else -(tr - root) / 2   # curvature across the line
        # a step edge also curves on one side: weaken where the slope is strong (off-centre)
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3) / 8
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3) / 8
        r = np.maximum(lam, 0) * sig * sig
        slope = np.sqrt(gx * gx + gy * gy) * sig
        resp = np.maximum(resp, r * np.exp(-(slope / (r + 1e-3)) ** 2 / 2))
    return resp


def _along_contours(resp: np.ndarray, L: np.ndarray, ppm: float, length_mm: float = 1.5,
                    n: int = 12) -> np.ndarray:
    """Line strength averaged along the local direction of the contours (structure tensor of `L`,
    Gaussian weights over ~`length_mm`): the strength of a line no longer flickers from point to
    point, so it is kept in one piece, while isolated responses (texture, grain) fade."""
    sig = max(1.0, 0.3 * ppm)
    g = cv2.GaussianBlur(L, (0, 0), sig)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    jxx = cv2.GaussianBlur(gx * gx, (0, 0), 2 * sig)
    jyy = cv2.GaussianBlur(gy * gy, (0, 0), 2 * sig)
    jxy = cv2.GaussianBlur(gx * gy, (0, 0), 2 * sig)
    ang = 0.5 * np.arctan2(2 * jxy, jxx - jyy) + np.pi / 2           # direction along the contour
    pos = (ang / (np.pi / n)) % n                                    # ... among n directions
    half = max(2, int(round(length_mm * ppm / 2)))
    out = np.zeros_like(resp)
    for k in range(n):
        th = k * np.pi / n
        ker = np.zeros((2 * half + 3, 2 * half + 3), np.float32)
        for t in np.linspace(-half, half, 8 * half + 1):             # line kernel, bilinear splat
            x, y = half + 1 + t * np.cos(th), half + 1 + t * np.sin(th)
            x0, y0 = int(np.floor(x)), int(np.floor(y))
            fx, fy, w = x - x0, y - y0, np.exp(-(t / (half / 1.5)) ** 2 / 2)
            ker[y0, x0] += w * (1 - fx) * (1 - fy); ker[y0, x0 + 1] += w * fx * (1 - fy)
            ker[y0 + 1, x0] += w * (1 - fx) * fy; ker[y0 + 1, x0 + 1] += w * fx * fy
        # each point blends its two nearest directions (a hard choice breaks curved lines where
        # the direction changes bin)
        d = np.abs(pos - k)
        wk = np.maximum(0.0, 1.0 - np.minimum(d, n - d)).astype(np.float32)
        out += wk * cv2.filter2D(resp, cv2.CV_32F, ker / ker.sum())
    return out


def _contrast_factor(L: np.ndarray, region: np.ndarray | None) -> float:
    """Factor (0.3..1) applied to the noise floor of the DoG: 1 for an image of ordinary contrast
    (5th to 95th percentile of L* 70 units apart or more), lower for a flatter one, down to 0.3.
    Measured inside the zone when one is painted."""
    v = L[region.astype(bool)] if region is not None and region.any() else L
    return float(np.clip((np.percentile(v, 95) - np.percentile(v, 5)) / 70.0, 0.3, 1.0))


FDOG_SMOOTH_MM = 2.0     # FDoG: length (sigma) of the averaging along the flow of the contours
FDOG_STEP_PX = 2.0       # ... step of the streamlines (1 px: hardly better, twice as slow)
FDOG_PPM = 6.0           # ... computed at this resolution at most (px/mm), then enlarged
_FDOG_CACHE: OrderedDict = OrderedDict()     # FDoG strength maps of the last images (slow to compute)
_FDOG_LOCK = threading.Lock()


def _fdog_strength(chans: list, ppm: float, dark: bool) -> np.ndarray:
    """Line strength of the flow-based DoG (FDoG, after Kang, Lee & Chui 2007): the DoG of each
    channel (max over the channels), averaged along the streamlines of the contours over
    ~FDOG_SMOOTH_MM on each side. Unlike `_along_contours` (straight segments), the streamlines
    bend with the contours, so curved lines (curls, rounded ends, letters) stay continuous with a
    short hysteresis. The direction comes from the smoothed structure tensor of the first channel
    (an iterative Edge Tangent Flow gave the same lines, rounded the T-junctions and was 3x
    slower). Cached: the detection preview recomputes it at each change of sensitivity."""
    check_cancel()
    h = blake2b(digest_size=16)
    for c in chans:
        h.update(np.ascontiguousarray(c, np.float32).tobytes())
    key = (h.hexdigest(), chans[0].shape, round(ppm, 3), dark)
    with _FDOG_LOCK:
        if key in _FDOG_CACHE:
            _FDOG_CACHE.move_to_end(key)
            return _FDOG_CACHE[key]
    sign = 1.0 if dark else -1.0
    sc = max(0.8, 0.15 * ppm)
    resp = None
    for c in chans:
        c = c.astype(np.float32)
        d = np.maximum(sign * (cv2.GaussianBlur(c, (0, 0), 1.6 * sc) - cv2.GaussianBlur(c, (0, 0), sc)), 0)
        resp = d if resp is None else np.maximum(resp, d)
    L = chans[0].astype(np.float32)
    # The averaging along the streamlines is a smoothing over ~FDOG_SMOOTH_MM: above FDOG_PPM it
    # runs on a reduced grid (the DoG itself stays at full resolution, its scales in mm unchanged)
    full = L.shape
    if ppm > FDOG_PPM * 1.05:
        s = FDOG_PPM / ppm
        size = (max(8, round(full[1] * s)), max(8, round(full[0] * s)))
        # Measured at 8 and 10 px/mm against the full computation: lines within 1 px of each other
        # (98-99 % on test_visage) and 2x faster. Adding back what the reduction takes from the DoG
        # (peaks of sharp lines 4-13 % lower) brought back grain and changed more of the selection.
        flow = _fdog_flow(cv2.resize(resp, size, interpolation=cv2.INTER_AREA),
                          cv2.resize(L, size, interpolation=cv2.INTER_AREA), FDOG_PPM)
        out = np.maximum(cv2.resize(flow, (full[1], full[0]), interpolation=cv2.INTER_CUBIC), 0)
    else:
        out = _fdog_flow(resp, L, ppm)
    with _FDOG_LOCK:
        _FDOG_CACHE[key] = out
        while len(_FDOG_CACHE) > 4:
            _FDOG_CACHE.popitem(last=False)
    return out


def _fdog_flow(resp: np.ndarray, L: np.ndarray, ppm: float) -> np.ndarray:
    """`resp` averaged along the streamlines of the contours of `L` (see _fdog_strength)."""
    # tangent of the contours (structure tensor of the first channel)
    sig = max(1.0, 0.3 * ppm)
    g = cv2.GaussianBlur(L, (0, 0), max(0.8, 0.15 * ppm))
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    jxx = cv2.GaussianBlur(gx * gx, (0, 0), sig)
    jyy = cv2.GaussianBlur(gy * gy, (0, 0), sig)
    jxy = cv2.GaussianBlur(gx * gy, (0, 0), sig)
    ang = 0.5 * np.arctan2(2 * jxy, jxx - jyy) + np.pi / 2
    tx, ty = np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)
    H, W = L.shape
    X, Y = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    samp = lambda img, mx, my: cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    sm = max(1.0, FDOG_SMOOTH_MM * ppm)
    acc, wsum = resp.copy(), 1.0
    for sgn in (1.0, -1.0):              # both ways along the streamline through each pixel
        px, py = X.copy(), Y.copy()
        dx, dy = sgn * tx, sgn * ty
        for i in range(1, int(np.ceil(2 * sm / FDOG_STEP_PX)) + 1):
            check_cancel()
            px += FDOG_STEP_PX * dx; py += FDOG_STEP_PX * dy
            ntx, nty = samp(tx, px, py), samp(ty, px, py)
            flip = (ntx * dx + nty * dy) < 0          # the tangent has no sign: keep going forward
            ntx[flip] *= -1; nty[flip] *= -1
            n = cv2.magnitude(ntx, nty)
            ok = n > 1e-6                            # undefined direction: keep the previous one
            dx = np.where(ok, ntx / np.maximum(n, 1e-6), dx)
            dy = np.where(ok, nty / np.maximum(n, 1e-6), dy)
            u = i * FDOG_STEP_PX
            w = float(np.exp(-u * u / (2 * sm * sm)))
            acc += w * samp(resp, px, py); wsum += w
    return (acc / np.float32(wsum)).astype(np.float32)


# methods whose hysteresis reach is the zone setting `outline_follow_mm` (others: HYST_MM)
FOLLOW_METHODS = ("xdog", "fdog", "teed")
# TEED: logit above which a pixel is on a line (background ~ -0.2), strength = logit x this
TEED_EDGE, TEED_SCALE = 0.5, 10.0
# noise floor of each method (L* units): weaker responses are never candidates
_LINE_FLOOR = {"xdog": 1.5, "fdog": 1.5, "ridge": 2.0}
# hysteresis of the key-detail selection (see detect_key_details)
HYST_MM, HYST_LOSS, HYST_SCORE = 4.0, 0.6, 0.4
# DoG: its line strength is averaged along the contours (_along_contours), so a line near the
# selection limit is above or below it over long stretches: it is followed farther (default of
# the zone setting `outline_follow_mm`; with 4 mm, long clean lines get wide gaps)
HYST_MM_DOG = 12.0
# Canny / ridges: seeds = share of the sensitivity / up to this (see hyst_seed_share). Measured on
# test images: 9 / 5 keep about as much line at a sensitivity as without the hysteresis; 3 / 2
# keep about 2 to 3 times more at sensitivity 50 (chosen: the same amount looked no different)
HYST_SEED_DIV = {"canny": 3.0, "ridge": 2.0}


def thin_features(L100: np.ndarray, ppm: float, floor: float = 6.0) -> np.ndarray:
    """Thin dark or light strokes (~0.5 to ~2.5 mm wide: eyes, eyebrows, nostrils, strands) that
    the majority filter would swallow: band of strong Hessian ridge response, small specks removed."""
    scales = (0.25, 0.5, 0.9)
    r = np.maximum(_line_response(L100, ppm, "ridge", True, scales),
                   _line_response(L100, ppm, "ridge", False, scales))
    m = (r > floor).astype(np.uint8) * 255
    return remove_small(m, max(4, int(1.0 * ppm * ppm))) > 0


def lines_into_layers(prep: Prepared, sim: np.ndarray, p: Params, region: np.ndarray | None,
                      color_ids: list, painted: np.ndarray, stages: bool = False):
    """Key-detail lines to add to the colour layers: {layer position: stroke mask (image area)}.

    `painted`: position of the last layer painting each pixel once cut (-1 = bare support).
    Lines are placed piece by piece, a piece being the part of a line over one same layer (the
    lightest one under its stroke): each piece goes to a layer painted after that one (darker
    tone, otherwise the paint would not show): the darkest layer, or (`nearest`) the one whose
    tone is closest to the piece's colour in the original, so a line may change layer on its way.
    Pieces over an area already painted by the last layer are left out (they are already cut), as
    are pieces too short to show once cut. (A whole line given to the layer most frequent under it
    lost e.g. a mouth line half over a light chin, half over the dark mask.) A given layer ("L<n>",
    beyond the last = the last): the parts of the lines over a layer painted after it are left out
    (they would be covered). With `stages`, returns the detection stages instead, the stroke
    limited to the lines added and `present` = lines already in the layers (or covered)."""
    st = detect_key_details(prep, sim, p, region, dark=True, stages=True)
    width = max(p.outline_width_mm, p.min_opening_mm * 1.05)
    k = disk(width * p.px_per_mm / 2)
    k_sil = disk(silhouette_width(p) * p.px_per_mm / 2)
    # The silhouette (one long closed line) is placed piece by piece below; the other lines whole
    # (the one of the detection: same settings and render)
    sil = st["sil"]
    if _LAYER_DEST.match(p.outline_target or ""):
        return _lines_into_layer(st, sil, k, min(int(p.outline_target[1:]), len(color_ids)) - 1,
                                 painted, stages, k_sil)
    kept = st["kept"] & ~sil if sil is not None else st["kept"]
    n, comp = cv2.connectedComponents(kept.astype(np.uint8), connectivity=8)
    if n <= 1 and sil is None:
        return dict(st, present=np.zeros_like(st["kept"])) if stages else ({}, 0)
    ys, xs = np.nonzero(comp)
    c = comp[ys, xs]
    pal = _lab(prep.palette)
    out, added = {}, 0
    present = np.zeros(comp.shape, bool)
    if len(ys):
        # pieces = (line, lightest layer under its stroke); layers are sorted light -> dark
        nl = len(color_ids)
        under = cv2.erode(painted.astype(np.int16), k)[ys, xs].astype(np.int64)
        pids, inv = np.unique(c.astype(np.int64) * (nl + 1) + under + 1, return_inverse=True)
        inv = inv.ravel()
        # mean colour of each piece in the original photo (the tone colours are measured there too)
        lab = _lab(original(prep)[ys, xs]).astype(np.float64)
        tot = np.bincount(inv, minlength=len(pids)).astype(np.float64)
        mean = np.stack([np.bincount(inv, lab[:, ch], minlength=len(pids)) for ch in range(3)], 1) \
            / np.maximum(tot, 1)[:, None]
        assign = np.full(len(pids), -1, np.int32)
        for j, pid in enumerate(pids.tolist()):
            cands = list(range(pid % (nl + 1), nl))      # layers painted after the one under the piece
            if not cands:
                continue
            if p.outline_target == "nearest":
                assign[j] = min(cands, key=lambda li: float(np.sum((pal[color_ids[li]] - mean[j]) ** 2)))
            else:
                assign[j] = cands[-1]
        a = assign[inv]
        keep = np.zeros(comp.shape, bool)
        min_size = max(2, k.shape[0])
        for li in sorted(set(assign[assign >= 0].tolist())):
            m = np.zeros(comp.shape, np.uint8)
            sel = a == li
            m[ys[sel], xs[sel]] = 1
            # pieces too short to show once cut (shorter than the stroke width): dropped
            nc, cc, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
            ok = np.maximum(stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT]) >= min_size
            ok[0] = False
            m = ok[cc]
            if m.any():
                out[li] = cv2.dilate(m.astype(np.uint8) * 255, k)
                keep |= m
        present = (comp > 0) & ~keep
        added = len(np.unique(comp[keep]))              # lines with at least one piece added
    if sil is not None:
        # Silhouette: each piece goes to a layer darker than the lightest one painted under its
        # stroke (both sides of the edge), so that it shows at least on the lighter side; only the
        # pieces running entirely over the last layer are left out (already cut there).
        under = cv2.erode(painted.astype(np.int16), k_sil)       # lightest layer under the stroke
        orig = _lab(original(prep)).astype(np.float64).reshape(*sil.shape, 3)
        for v in np.unique(under[sil]):
            piece = sil & (under == v)
            cands = list(range(int(v) + 1, len(color_ids)))
            if not cands:
                present |= piece
                continue
            if p.outline_target == "nearest":
                mc = orig[piece].mean(axis=0)
                li = min(cands, key=lambda c: float(np.sum((pal[color_ids[c]] - mc) ** 2)))
            else:
                li = cands[-1]
            m = cv2.dilate(piece.astype(np.uint8) * 255, k_sil)
            out[li] = m if li not in out else (out[li] | m)
            added += 1
    if stages:
        stroke = np.zeros(comp.shape, bool)
        for m in out.values():
            stroke |= m > 0
        return dict(st, stroke=stroke, present=present)
    return out, added


def _lines_into_layer(st: dict, sil, k: np.ndarray, li: int, painted: np.ndarray, stages: bool,
                      k_sil: np.ndarray | None = None):
    """Lines (and silhouette) cut into the colour layer at position `li`: only where the lightest
    layer under the stroke is painted before it (elsewhere the line is already cut or would be
    covered by a later layer). Same return values as `lines_into_layers`."""
    line = st["kept"] | sil if sil is not None else st["kept"].copy()
    keep = np.zeros_like(line)
    if li >= 0 and line.any():
        under = cv2.erode(painted.astype(np.int16), k)           # lightest layer under the stroke
        keep = line & (under < li)
        # pieces left by the cut shorter than the stroke width: dropped (nothing would show)
        n, comp, stats, _ = cv2.connectedComponentsWithStats(keep.astype(np.uint8), connectivity=8)
        size = np.maximum(stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT])
        ok = size >= max(2, k.shape[0])
        ok[0] = False
        keep = ok[comp]
    present = line & ~keep
    stroke = cv2.dilate(keep.astype(np.uint8) * 255, k)
    if sil is not None and k_sil is not None and k_sil.shape != k.shape:   # silhouette: its own width
        stroke = cv2.dilate(np.where(keep & ~sil, 255, 0).astype(np.uint8), k) \
            | cv2.dilate(np.where(keep & sil, 255, 0).astype(np.uint8), k_sil)
    if stages:
        return dict(st, stroke=stroke > 0, present=present)
    if not keep.any():
        return {}, 0
    return {li: stroke}, cv2.connectedComponents(keep.astype(np.uint8), connectivity=8)[0] - 1


def outline_keep_share(sensitivity: float) -> float:
    """Share of the candidate edge points kept by the outline sensitivity (0..100), progressive
    scale: 1 % at 0, 95 % at 100, each step multiplies it by ~1.05."""
    return 0.01 * 95.0 ** (float(np.clip(sensitivity, 0, 100)) / 100.0)


def hyst_seed_share(method: str, keep_share: float) -> float:
    """Share of the candidate points the selection starts from, before the hysteresis extends them
    along their lines. DoG: the share of the sensitivity (its scale was calibrated with the
    hysteresis). Canny / ridges: fewer (`HYST_SEED_DIV`), otherwise a sensitivity would keep far
    more line than when they were selected point by point."""
    d0 = HYST_SEED_DIV.get(method)
    if d0 is None:
        return keep_share
    # the divisor fades out towards the top of the scale, where nearly every point is kept anyway
    return keep_share / (1.0 + (d0 - 1.0) * (1.0 - keep_share) ** 3)


GAP_JITTER_MAX = 80.0                   # % (beyond, two interruptions may fall on the same place)


def _gap_spacings(n: int, jitter: float, seed: int) -> np.ndarray:
    """`n` relative distances between interruptions (1 = the distance set), each varying at random
    by up to `jitter` % either way. Seeded: the same settings give the same stencils."""
    if jitter <= 0 or n <= 0:
        return np.ones(max(n, 0))
    u = np.random.default_rng(seed).uniform(-1.0, 1.0, n)
    return 1.0 + (jitter / 100.0) * u


def gap_lines(line: np.ndarray, ppm: float, every_mm: float, gap_mm: float,
              width_mm: float, jitter: float = 0.0) -> tuple[np.ndarray, np.ndarray | None]:
    """Interrupts any line drawing (one-pixel lines, drawn `width_mm` wide) about every `every_mm`
    (distance between interruptions, measured in a straight line: close to the length along smooth
    lines; varying at random by up to `jitter` % so that the line does not look dotted) over
    `gap_mm`, so that closed loops do not cut islands out. Returns (line, zones of the
    interruptions)."""
    every = every_mm * ppm
    if every <= 0 or not line.any():
        return line, None
    r = max(1, int(round((gap_mm + width_mm) * ppm / 2)))
    ys, xs = np.nonzero(line)
    # greedy sampling, bucketed on a grid of cells of the largest distance: a point closer to a
    # centre already taken than that centre's own distance is skipped
    spac = every * _gap_spacings(len(ys), jitter, 1)    # distance of the centre each point may become
    cell = every * (1.0 + max(jitter, 0.0) / 100.0)
    taken: dict[tuple, list] = {}
    centres = []
    for i, (y, x) in enumerate(zip(ys.tolist(), xs.tolist())):
        cy, cx = int(y // cell), int(x // cell)
        near = False
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                for (yy, xx, e2) in taken.get((cy + dy, cx + dx), ()):
                    if (yy - y) ** 2 + (xx - x) ** 2 < e2:
                        near = True
                        break
                if near:
                    break
            if near:
                break
        if not near:
            taken.setdefault((cy, cx), []).append((y, x, float(spac[i]) ** 2))
            centres.append((x, y))
    zones = np.zeros(line.shape, np.uint8)
    for x, y in centres:
        cv2.circle(zones, (x, y), r, 1, -1)
    zones = zones > 0
    return line & ~zones, zones


def _absorb_small_zones(lab: np.ndarray, min_px: float) -> np.ndarray:
    """Label map where the zones smaller than `min_px` pixels take the tone of the zones around them
    (each pixel the one of the nearest pixel kept: a small zone between two may be shared)."""
    small = np.zeros(lab.shape, bool)
    for t in np.unique(lab).tolist():
        check_cancel()
        _, cc, st, _ = cv2.connectedComponentsWithStats((lab == t).astype(np.uint8), connectivity=4)
        tiny = st[:, cv2.CC_STAT_AREA] < min_px
        tiny[0] = False                 # component 0: the other tones
        if tiny.any():
            small |= tiny[cc]
    if not small.any() or small.all():
        return lab
    _, near = cv2.distanceTransformWithLabels(small.astype(np.uint8), cv2.DIST_L2, 5,
                                             labelType=cv2.DIST_LABEL_PIXEL)
    ys, xs = np.nonzero(~small)
    lut = np.zeros(int(near.max()) + 1, lab.dtype)
    lut[near[ys, xs]] = lab[ys, xs]
    out = lab.copy()
    out[small] = lut[near[small]]
    return out


def _smooth_zones(lab: np.ndarray, sigma: float) -> np.ndarray:
    """Label map with smoother limits: each pixel takes the tone most present around it (Gaussian
    weights of standard deviation `sigma` pixels)."""
    best = np.full(lab.shape, -1.0, np.float32)
    out = lab.copy()
    for t in np.unique(lab).tolist():
        check_cancel()
        s = cv2.GaussianBlur((lab == t).astype(np.float32), (0, 0), sigma)
        upd = s > best
        out[upd] = t
        best[upd] = s[upd]
    return out


def _nearest_line(line: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """For every pixel: distance (px) to the nearest pixel of `line` and the position of that pixel
    in (ys, xs) = np.nonzero(line). Returns (distance, position, ys, xs)."""
    dist, near = cv2.distanceTransformWithLabels((~line).astype(np.uint8), cv2.DIST_L2, 5,
                                                 labelType=cv2.DIST_LABEL_PIXEL)
    ys, xs = np.nonzero(line)
    pos = np.zeros(int(near.max()) + 1, np.int64)
    pos[near[ys, xs]] = np.arange(len(ys))
    return dist, pos[near], ys, xs


def zone_limits(prep: Prepared, p: Params, support) -> tuple:
    """Limits between the tone zones (colouring-book lines), with the colouring book's own settings:
    small zones absorbed, zones smoothed, limits between close tones left out, only inside the
    chosen mask, edge of the area coloured. Returns (one-pixel line, darker of the two tones each
    line pixel separates, contrast across it (ΔE), label map the limits were traced on)."""
    ppm = p.px_per_mm
    lab = prep.labels
    if p.line_zones_min_area_mm2 > 0:
        lab = _absorb_small_zones(lab, p.line_zones_min_area_mm2 * ppm * ppm)
    if p.line_zones_smooth_mm > 0:
        lab = _smooth_zones(lab, p.line_zones_smooth_mm * ppm)
    pal = _lab(prep.palette)
    Lt = pal[:, 0]
    dmat = np.linalg.norm(pal[:, None, :] - pal[None, :, :], axis=2).astype(np.float32)
    line = np.zeros(lab.shape, bool)
    darker = np.full(lab.shape, -1, np.int32)
    de = np.zeros(lab.shape, np.float32)
    for a, b, sl in ((lab[:, :-1], lab[:, 1:], (slice(None), slice(1, None))),
                     (lab[:-1], lab[1:], (slice(1, None), slice(None)))):
        diff = a != b
        dab = dmat[a, b]
        if p.line_zones_min_de > 0:     # close tones: no limit between them
            diff &= dab >= p.line_zones_min_de
        d = np.where(Lt[a] <= Lt[b], a, b)
        line[sl] |= diff
        darker[sl][diff] = d[diff]
        de[sl][diff] = dab[diff]
    area = None                         # removed background and chosen mask: only the limits inside
    for m in (prep.fg, getattr(prep, "zones_area", None)):
        if m is not None:
            area = m if area is None else (area & m)
    if area is not None:
        line &= cv2.erode(area.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    bd = max(2, int(round(0.5 * ppm)))  # the image's own edge is not a limit
    line[:bd, :] = False; line[-bd:, :] = False; line[:, :bd] = False; line[:, -bd:] = False
    if p.line_zones_frame:              # edge of the area coloured, its stroke inside it
        w = max(p.line_zones_width_mm, p.line_zones_width_max_mm, p.min_opening_mm * 1.05)
        r = max(bd, int(math.ceil(w * ppm / 2)) + 1)
        inside = np.ones(lab.shape, np.uint8) if area is None else area.astype(np.uint8)
        inside = cv2.distanceTransform(np.pad(inside, 1), cv2.DIST_L2, 5)[1:-1, 1:-1] > r
        edge = inside & ~(cv2.erode(inside.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)
        line |= edge
        darker[edge] = lab[edge]        # the tone of the zone it closes, its contrast with the support
        de[edge] = np.linalg.norm(pal - _lab(support)[0], axis=1).astype(np.float32)[lab[edge]]
    n, lab_c, stats, _ = cv2.connectedComponentsWithStats(line.astype(np.uint8), connectivity=8)
    length = np.maximum(stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT])
    keep = length >= max(2, int(p.line_zones_min_length_mm * ppm))
    keep[0] = False
    return keep[lab_c], darker, de, lab


def line_stencils(prep: Prepared, p: Params, entries: list, groups: dict, support: np.ndarray,
                  place, fx: dict | None = None, manual: "ManualBridges | None" = None) -> list:
    """Stencils of the "line" style: no flat colour, only lines, one stencil per line colour
    (sprayed from the lightest to the darkest). Sources: key details found against the bare
    support, silhouette of a mask, limits between the tone zones (colouring book).
    `fx`: effects per stencil number (see `effects.parse`), each applied to its stencil alone.
    `manual`: manual bridges, matched to the stencils by their colour."""
    H, W = prep.labels.shape
    ppm = p.px_per_mm
    flat = np.empty((H, W, 3), np.uint8)
    flat[:] = np.asarray(support, np.uint8)
    def gcol(g):                        # colour of an outline layer (no colour layers here:
        g = g if g in groups else next(iter(groups), None)   # darkest / nearest = the first one)
        pg = group_params(p, groups[g]) if g else p
        return g or "A", np.asarray(outline_color(prep, support, pg), np.uint8)

    def colour(h):                      # own colour, or automatic: the most contrast with the support
        return (hex_to_bgr(h).astype(np.uint8) if h.startswith("#")
                else np.asarray(outline_color(prep, support, replace(p, outline_auto_color=True)), np.uint8))

    parts = []                          # (one-pixel line, colour, source, line width mm)
    zones = np.zeros((H, W), bool)
    for pe, reg, dest, is_sil in entries:
        if is_sil:
            continue
        g, c = gcol(dest)
        st = detect_key_details(prep, flat, pe, reg, outline_is_dark(c, support), stages=True)
        w = max(pe.outline_width_mm, p.min_opening_mm * 1.05)
        ln, z = gap_lines(st["kept"], ppm, pe.outline_gap_every_mm, pe.outline_gap_mm, w,
                          pe.outline_gap_jitter)
        parts.append((ln, c, "details:" + g, w))
        if z is not None:
            zones |= z
    if p.outline_subject:
        ln, z = subject_silhouette(prep, p, None, gaps=True)
        if ln is not None:
            parts.append((ln, colour(p.line_sil_color), "silhouette", silhouette_width(p)))
            if z is not None:
                zones |= z
    if p.line_zones:                    # colouring book: its own settings only
        wz = max(p.line_zones_width_mm, p.min_opening_mm * 1.05)
        wz2 = max(p.line_zones_width_max_mm, wz) if p.line_zones_width_max_mm > 0 else wz
        ln, darker, de, zlab = zone_limits(prep, p, support)
        ln, z = gap_lines(ln, ppm, p.line_zones_gap_every_mm, p.line_zones_gap_mm, wz2,
                          p.line_zones_gap_jitter)
        if z is not None:
            zones |= z
        wmap = None                     # width (mm) of each line pixel, growing with the contrast
        if wz2 > wz and ln.any():
            lo, hi = float(de[ln].min()), float(de[ln].max())
            wmap = wz + (wz2 - wz) * np.clip((de - lo) / max(hi - lo, 1e-6), 0, 1)
        stroke = None                   # None: each line pixel drawn as a disk of the width, centred
        if wmap is not None or p.line_zones_pos != "center":
            def stroke(sub, wmap=wmap, wz=wz, darker=darker, zlab=zlab):
                """Stroke of the limits `sub`: their width at the nearest limit pixel, centred on the
                limit or all on the side of its darker / lighter zone."""
                dist, k, ys, xs = _nearest_line(sub)
                w = (wmap[ys, xs] if wmap is not None else np.full(len(ys), wz, np.float32)) * ppm
                wk = w[k]
                if p.line_zones_pos == "center":
                    band = dist <= wk / 2
                else:
                    # + 1 px: as wide as a centred stroke, which also holds the limit pixel
                    dark = zlab == darker[ys, xs][k]
                    band = (dist <= wk + 1) & (dark if p.line_zones_pos == "dark" else ~dark)
                return band.astype(np.uint8) * 255
        if p.line_zones_color == "tones":
            paints = parse_tone_colors(p.layer_colors)
            for t in np.unique(darker[ln]).tolist():
                if t >= 0:
                    c = paints.get(t, prep.palette[t].astype(np.uint8))
                    parts.append((ln & (darker == t), c, f"tone{t}", wz2, stroke))
        else:
            c = (hex_to_bgr(p.line_zones_color).astype(np.uint8) if p.line_zones_color.startswith("#")
                 else np.asarray(outline_color(prep, support, replace(p, outline_auto_color=True)), np.uint8))
            parts.append((ln, c, "zones", wz2, stroke))
    wmax = max([pt[3] for pt in parts] + [p.min_opening_mm * 1.05])
    # the interruptions of a line stay free of every other line
    free = ~(cv2.dilate(zones.astype(np.uint8), disk(wmax * ppm / 2)) > 0)
    by_colour: dict[str, list] = {}
    for ln, c, src, w, *make in parts:
        ln = ln & free
        if not ln.any():
            continue
        g = by_colour.setdefault(bgr_hex(c), [np.zeros((H, W), np.uint8), [], np.asarray(c, np.uint8)])
        if make and make[0] is not None:    # own stroke (colouring book: width, position)
            g[0] |= make[0](ln)
        else:                               # each line its own width
            g[0] |= cv2.dilate(ln.astype(np.uint8) * 255, disk(w * ppm / 2))
        if src not in g[1]:
            g[1].append(src)
    order = sorted(by_colour.values(), key=lambda g: -float(_lab(g[2].reshape(1, 3))[0, 0]))  # light first
    layers, done = [], []
    by_layer = manual.assign("line", [(i + 1, c) for i, (_t, _s, c) in enumerate(order)]) if manual else {}
    for i, (target, srcs, c) in enumerate(order):
        cleaned = clean_mask(target, p)
        lost = cv2.bitwise_xor(cleaned, target)
        if fx and fx.get(i + 1):
            cleaned = fxmod.apply_alone(cleaned, fx[i + 1], p, prep.resized, prep.off_px, done)
        done.append(cleaned)
        opening = place(cleaned)
        mine = by_layer.get(i + 1, [])
        s = opening.copy()
        m_drawn, forbid = manual.apply(opening, i + 1, mine) if manual else ([], None)
        cut = _cut_by(s, opening)
        merged, n_merged = merge_line_islands(opening, p, c, support)
        s = opening.copy()
        pre = s if manual and needs_pre(manual.segments(mine)) else None
        held = hold_material(opening, p, None, forbid, m_drawn)
        if manual:
            held, l_drawn = manual.late(opening, pre, i + 1, mine, held)
            m_drawn = m_drawn + l_drawn
        cut = _bridge_cut(cut | _cut_by(s, opening), opening, merged)
        bridges, n_isl, n_short, n_struct, n_weak = held
        layers.append(Layer(index=i + 1, color=c, target=target, opening=opening,
                            lost=lost, bridges=m_drawn + bridges,
                            paths=vectorize(opening, p), islands=n_isl, short_islands=n_short,
                            struct_bridges=n_struct, weak_spots=len(n_weak), weak_points=_weak_mm(n_weak, p),
                            merged=merged, n_merged=n_merged, kind="outline", sources=tuple(srcs),
                            manual_bridges=len(m_drawn), bridge_cut=cut))
    return layers


def silhouette_width(p: Params) -> float:
    """Width of the silhouette stroke: its own (0, settings of earlier versions: that of the key-detail
    lines); at least the minimum opening."""
    return max(p.outline_subject_width_mm or p.outline_width_mm, p.min_opening_mm * 1.05)


def subject_silhouette(prep: Prepared, p: Params, sim: np.ndarray | None = None,
                       gaps: bool = False):
    """One-pixel line along the outline of the selected subject (option `outline_subject`):
    smoothed, not drawn along the image edges, pieces shorter than the minimum line length removed.
    `outline_subject_pos`: the stroke is centred on the outline, or the line is moved by half the
    stroke width so that the stroke lies just outside or just inside it.

    `outline_subject_weak` + `sim` (simulated render of the colour layers): only the parts of the
    edge that the render shows poorly (colour difference across the edge below `outline_subject_de`).
    `outline_subject_gap_every_mm`: the line is interrupted at regular intervals along its length
    (varying at random by up to `outline_subject_gap_jitter` %), so that a closed silhouette does not cut the subject out as one big island (each interruption
    is a built-in bridge). With `gaps`, returns (line, zones of the interruptions) - the other
    lines must leave these zones empty too - otherwise the line alone."""
    none = (None, None) if gaps else None
    subj = prep.subject
    if subj is None and p.outline_subject_mask == "subject":
        subj = prep.fg                      # without the server: the cut-out subject
    if not p.outline_subject or subj is None or not subj.any() or subj.all():
        return none
    ppm = p.px_per_mm
    edge = cv2.GaussianBlur(subj.astype(np.float32), (0, 0), max(0.8, 0.4 * ppm)) > 0.5
    k3 = np.ones((3, 3), np.uint8)
    b = max(2, int(round(0.5 * ppm)))       # the image's own edge is not an outline

    def border(a: np.ndarray) -> np.ndarray:
        ln = a & ~cv2.erode(a.astype(np.uint8), k3).astype(bool)
        ln[:b, :] = False; ln[-b:, :] = False; ln[:, :b] = False; ln[:, -b:] = False
        return ln

    width = silhouette_width(p)
    half = disk(width * ppm / 2)            # the structuring element of the stroke
    shift = 0
    m = edge
    if p.outline_subject_pos != "center":
        shift = half.shape[0] // 2
        if p.outline_subject_pos == "outside":   # one pixel more: the stroke starts just past the edge
            m = cv2.dilate(cv2.dilate(edge.astype(np.uint8), half), k3) > 0
        else:
            m = cv2.erode(edge.astype(np.uint8), half) > 0
        if not m.any():                     # subject thinner than the stroke
            return none
    line = border(m)
    min_px = max(2, int(p.outline_subject_min_length_mm * ppm))
    if p.outline_subject_weak and sim is not None and line.any():
        # colour of the render on each side of the edge (normalised Gaussian means, ~1.5 mm)
        lab = cv2.cvtColor(np.asarray(sim, np.float32) / 255.0, cv2.COLOR_BGR2LAB)
        sig = max(1.0, 1.5 * ppm)
        sides = []
        for w in (edge.astype(np.float32), (~edge).astype(np.float32)):
            den = cv2.GaussianBlur(w, (0, 0), sig) + 1e-6
            sides.append(np.dstack([cv2.GaussianBlur(lab[..., c] * w, (0, 0), sig) / den for c in range(3)]))
        de = np.sqrt(((sides[0] - sides[1]) ** 2).sum(axis=2))
        weak = border(edge) & (de < p.outline_subject_de)   # measured on the outline itself
        # bridge short strong parts (a few mm) so the line is not chopped into dashes
        line &= cv2.dilate(weak.astype(np.uint8), disk(max(1.0, 1.5 * ppm) + shift)) > 0
    n, lab_c, stats, _ = cv2.connectedComponentsWithStats(line.astype(np.uint8), connectivity=8)
    length = np.maximum(stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT])
    keep = length >= min_px
    keep[0] = False
    line = keep[lab_c]
    if not line.any():
        return none
    zones = None
    every = p.outline_subject_gap_every_mm * ppm
    if every > 0:
        r = (p.outline_subject_gap_mm + width) * ppm / 2     # stroke half-width + half the gap
        centres = []
        contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        for ci, c in enumerate(contours):
            pts = c[:, 0, :]
            on = line[pts[:, 1], pts[:, 0]]
            if not on.any():
                continue
            closed = bool(on.all())
            # runs of contour points on the kept line (the contour is cyclic: start at a gap)
            if not closed:
                k0 = int(np.argmin(on))
                pts, on = np.roll(pts, -k0, axis=0), np.roll(on, -k0)
            idx = np.flatnonzero(on)
            runs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
            for ri, run in enumerate(runs):
                q = pts[run].astype(np.float64)
                steps = np.hypot(*np.diff(q, axis=0).T)
                if closed:
                    steps = np.append(steps, np.hypot(*(q[0] - q[-1])))
                s_ = np.concatenate([[0.0], np.cumsum(steps)])
                L = float(s_[-1])
                # k pieces of line between the interruptions, of random relative lengths w
                if closed:
                    k = max(1, int(round(L / every)))
                    w = _gap_spacings(k, p.outline_subject_gap_jitter, 1000 * ci + ri)
                    at = (np.concatenate([[0.0], np.cumsum(w)[:-1]]) + 0.5 * w[0]) * L / w.sum()
                else:
                    k = int(L // every) + 1
                    w = _gap_spacings(k, p.outline_subject_gap_jitter, 1000 * ci + ri)
                    at = np.cumsum(w)[:-1] * L / w.sum()
                for a in at:
                    j = min(int(np.searchsorted(s_, a)), len(q) - 1)
                    centres.append(q[j])
        if centres:
            zones = np.zeros(line.shape, np.uint8)
            for x, y in centres:
                cv2.circle(zones, (int(round(x)), int(round(y))), max(1, int(round(r))), 1, -1)
            zones = zones > 0
            line &= ~zones
    if gaps:
        return (line, zones) if line.any() else none
    return line if line.any() else None


def _zone_lines(entries: list) -> dict:
    """Line counts of the whole-line zones of the last search: {zone id: (kept, detected)}."""
    out = {}
    for pe, *_ in entries:
        c = getattr(pe, "_line_counts", None)
        if c is not None and getattr(pe, "_zone", None):
            out[pe._zone] = c
    return out


LINE_FLOOR = 0.05      # whole lines: weakest relative mark reached at sensitivity 100


def line_threshold(sens: float) -> float:
    """Whole lines: minimum mark of a line, as a share of the mark of the most marked lines of the
    zone (90th percentile of the line marks), for a detection sensitivity 0..100. Logarithmic
    (contrast is perceived as ratios): 100 % at 0, ~22 % at 50, 5 % at 100."""
    return LINE_FLOOR ** (float(np.clip(sens, 0, 100)) / 100.0)


def line_sensitivity(rel: np.ndarray) -> np.ndarray:
    """Inverse of line_threshold: sensitivity from which a line of relative mark `rel` is kept."""
    return np.clip(100.0 * np.log(np.maximum(rel, 1e-9)) / np.log(LINE_FLOOR), 0, 100)


def _line_marks(cand: np.ndarray, score: np.ndarray, ppm: float, min_len_mm: float):
    """Candidate points grouped into lines (small gaps rejoined): (labels, ids of the lines long
    enough, mark of each line = median importance of its points)."""
    joined = cv2.morphologyEx(cand.astype(np.uint8), cv2.MORPH_CLOSE, disk(max(1.0, 0.3 * ppm)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(joined, connectivity=8)
    if n <= 1:
        return lab, np.zeros(0, np.int64), np.zeros(max(n, 1))
    lab_c = lab[cand]
    cnt = np.bincount(lab_c, minlength=n)
    order = np.argsort(lab_c, kind="stable")
    labs, vals = lab_c[order], score[cand][order]
    starts = np.searchsorted(labs, np.arange(n))
    ends = np.searchsorted(labs, np.arange(n), side="right")
    med = np.zeros(n)
    for i in range(1, n):
        if ends[i] > starts[i]:
            med[i] = float(np.median(vals[starts[i]:ends[i]]))
    length = np.maximum(stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT])
    ok = (length >= max(2, int(min_len_mm * ppm))) & (cnt > 0)
    ok[0] = False
    return lab, np.flatnonzero(ok), med


def _whole_lines(cand: np.ndarray, score: np.ndarray, sens: float, ppm: float,
                 min_len_mm: float) -> tuple[np.ndarray, tuple]:
    """Selection by whole lines in the zone: each line whose mark reaches the detection threshold
    (`line_threshold(sens)` x the mark of the most marked lines of the zone) is kept, whole. Lines
    shorter than the minimum length are not candidates. Returns (points, (kept, detected, steps))."""
    lab, ids, med = _line_marks(cand, score, ppm, min_len_mm)
    if not len(ids):
        return np.zeros(cand.shape, bool), (0, 0, [])
    ref = float(np.percentile(med[ids], 90))
    keep = np.zeros(len(med), bool)
    keep[ids[med[ids] >= line_threshold(sens) * ref]] = True
    # sensitivity from which each line appears (marks on the slider)
    steps = sorted(round(float(x), 1) for x in line_sensitivity(med[ids] / max(ref, 1e-9)))
    return cand & keep[lab], (int(keep.sum()), len(ids), steps)


def _thin(m: np.ndarray) -> np.ndarray:
    """One-pixel centre lines of a mask (Zhang-Suen thinning)."""
    img = m.astype(np.uint8)
    while True:
        changed = False
        for step in (0, 1):
            P = np.pad(img, 1)
            p2, p3, p4, p5 = P[:-2, 1:-1], P[:-2, 2:], P[1:-1, 2:], P[2:, 2:]
            p6, p7, p8, p9 = P[2:, 1:-1], P[2:, :-2], P[1:-1, :-2], P[:-2, :-2]
            nb = [p2, p3, p4, p5, p6, p7, p8, p9, p2]
            B = sum(x.astype(np.int16) for x in nb[:8])
            A = sum(((nb[i] == 0) & (nb[i + 1] == 1)).astype(np.int16) for i in range(8))
            if step == 0:
                c1, c2 = p2 * p4 * p6, p4 * p6 * p8
            else:
                c1, c2 = p2 * p4 * p8, p2 * p6 * p8
            rm = (img == 1) & (B >= 2) & (B <= 6) & (A == 1) & (c1 == 0) & (c2 == 0)
            if rm.any():
                img[rm] = 0
                changed = True
        if not changed:
            return img > 0


def _close_lines(lines: np.ndarray, stroke: np.ndarray, p: Params, region: np.ndarray | None):
    """Lines of a zone closer to each other than the minimum material strip: the material limits
    would open the strip between them (a flat area). Strip by strip, the two pieces of line along
    it are replaced by one: their median line (`median`), or the piece farther from / closer to
    the centre of the zone (`outer` / `inner`); the other lines are left alone.
    Returns (lines, stroke)."""
    ppm = p.px_per_mm
    r_close = p.min_strip_mm * ppm / 2
    if r_close < 0.5 or not stroke.any():
        return lines, stroke
    closed = cv2.morphologyEx(stroke, cv2.MORPH_CLOSE, disk(r_close)) > 0
    fill = closed & (stroke == 0)
    if not fill.any():
        return lines, stroke
    w_px = max(p.outline_width_mm, p.min_opening_mm * 1.05) * ppm
    mode = p.outline_close_lines
    ref = region if region is not None and region.any() else lines
    cy, cx = (float(v.mean()) for v in np.nonzero(ref))       # centre of the zone
    out = lines.copy()
    H, W = lines.shape
    m = int(np.ceil(w_px + 4))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(fill.astype(np.uint8), connectivity=8)
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if a < 3:
            continue
        ya, yb, xa, xb = max(0, y - m), min(H, y + h + m), max(0, x - m), min(W, x + w + m)
        strip = (lab[ya:yb, xa:xb] == i).astype(np.uint8)
        along = cv2.dilate(strip, disk(w_px / 2 + 2)) > 0          # the strip and the two strokes beside it
        sides = lines[ya:yb, xa:xb] & along                        # the pieces of line along the strip
        if not sides.any():
            continue
        if mode == "median":
            band = (closed[ya:yb, xa:xb] & along) | (strip > 0)
            centre = _thin(band)
            out[ya:yb, xa:xb] = (out[ya:yb, xa:xb] & ~sides) | centre
            continue
        k, pl_ = cv2.connectedComponents(sides.astype(np.uint8), connectivity=8)
        if k <= 2:                       # a single piece (the two lines join here): left as is
            continue
        py, px = np.nonzero(sides)
        dist = np.hypot(px + xa - cx, py + ya - cy)
        lbl = pl_[py, px]
        mean = np.bincount(lbl, dist, minlength=k) / np.maximum(np.bincount(lbl, minlength=k), 1)
        cand = np.arange(1, k)
        keep = cand[np.argmax(mean[1:])] if mode == "outer" else cand[np.argmin(mean[1:])]
        drop = sides & (pl_ != keep)
        out[ya:yb, xa:xb] &= ~drop
    stroke = cv2.dilate(out.astype(np.uint8) * 255, disk(w_px / 2))
    return out, stroke


def detail_image(prep: Prepared, p: Params) -> np.ndarray:
    """Image the key-detail lines of a zone are looked for in: the preprocessed image, plus the
    zone's own local contrast (`outline_clahe`, same scale as `pre_clahe`), which the layers never see."""
    if p.outline_clahe <= 0:
        return prep.resized
    lab = cv2.cvtColor(prep.resized, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=1.0 + p.outline_clahe / 100.0 * 4.0, tileGridSize=(8, 8))
    lab[..., 0] = clahe.apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


_GRAD_CACHE: OrderedDict = OrderedDict()    # local contrast of the last images (see _image_grad)
_GRAD_LOCK = threading.Lock()


def _image_grad(img: np.ndarray, ppm: float, make) -> np.ndarray:
    """Local contrast of an image (`make()`: gradient magnitude of its L*a*b*), cached by content:
    the zones and the silhouette of one computation measure it on the same images (original,
    render of the layers), each zone once otherwise. Read-only (shared)."""
    h = blake2b(digest_size=16)
    h.update(np.ascontiguousarray(img).tobytes())
    key = (h.hexdigest(), img.shape, img.dtype.str, round(ppm, 3))
    with _GRAD_LOCK:
        if key in _GRAD_CACHE:
            _GRAD_CACHE.move_to_end(key)
            return _GRAD_CACHE[key]
    g = make()
    g.setflags(write=False)
    with _GRAD_LOCK:
        _GRAD_CACHE[key] = g
        while len(_GRAD_CACHE) > 4:
            _GRAD_CACHE.popitem(last=False)
    return g


def _lab01(img: np.ndarray) -> np.ndarray:
    """CIE L*a*b* (float) of a BGR 8-bit image."""
    return cv2.cvtColor(img.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)


def _lab_grad(labimg: np.ndarray, ppm: float) -> np.ndarray:
    """Gradient magnitude of a L*a*b* image (local contrast, the three channels together)."""
    sig = max(0.7, 0.12 * ppm)
    g2 = np.zeros(labimg.shape[:2], np.float32)
    for c in range(3):
        ch = cv2.GaussianBlur(labimg[..., c], (0, 0), sig)
        gx = cv2.Sobel(ch, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(ch, cv2.CV_32F, 0, 1, ksize=3)
        g2 += gx * gx + gy * gy
    return np.sqrt(g2)


# Candidate lines of the last zones (step 1 of detect_key_details, from the image alone): computed
# in advance while the colour layers are made (_prefetch_lines), and kept for the next computations
# (most changes leave the image and the zone's method as they are). Keyed by the identity of the
# prepared image (kept alive by the entry, so the id cannot be reused) and the settings involved.
_CAND_CACHE: OrderedDict = OrderedDict()     # key -> (prepared image, Future of (ref, strength, edges))
_CAND_LOCK = threading.Lock()
_CAND_MAX = 3


def _cand_key(prep: Prepared, p: Params, dark: bool, region: np.ndarray | None) -> tuple:
    method = p.outline_method
    rkey = None
    if method in ("xdog", "fdog") and region is not None:   # their noise floor is measured in the zone
        h = blake2b(digest_size=16)
        h.update(np.ascontiguousarray(region).tobytes())
        rkey = (h.hexdigest(), region.shape)
    return (id(prep.resized), round(p.px_per_mm, 3), float(p.outline_clahe), method,
            None if method == "teed" else p.outline_channel,
            bool(dark) if method in ("xdog", "fdog", "ridge") else None, rkey)


def _compute_candidates(prep: Prepared, p: Params, dark: bool, region: np.ndarray | None) -> tuple:
    """(image searched, line strength, candidate line points) of a zone, by its method (see
    detect_key_details); read-only arrays."""
    ppm = p.px_per_mm
    method = p.outline_method
    ref = detail_image(prep, p)
    lab_ref = None if method == "teed" else _lab01(ref)
    check_cancel()
    if method == "teed":
        # learned edges (TEED): one centre line per edge (crest of the network's smoothed
        # logits; a skeleton of its lines gave the same lines 5x slower); the strength is the
        # network's confidence, so the sensitivity keeps the surest lines first. The channel
        # setting is not used (the network looks at the colour image).
        lg = teed.logits(ref)
        strength = np.maximum(cv2.GaussianBlur(lg, (0, 0), 1.0), 0) * TEED_SCALE
        edges = _crest(strength, TEED_EDGE * TEED_SCALE)
    elif method == "canny":
        # Canny on the luminance AND on the colour channels (a, b), to also catch details as
        # bright as their surroundings; "colour" only: on a and b (a shadow mostly changes L*).
        lab8 = cv2.cvtColor(cv2.bilateralFilter(ref, 7, 40, 5), cv2.COLOR_BGR2LAB)
        edges = np.zeros(ref.shape[:2], bool)
        chans = (1, 2) if p.outline_channel == "colour" else (0, 1, 2)
        for c, (lo, hi) in enumerate(((12, 32), (8, 20), (8, 20))):
            if c in chans:
                edges |= cv2.Canny(lab8[..., c], lo, hi, L2gradient=True) > 0
        if p.outline_channel == "colour":
            g_ab = lab_ref.copy(); g_ab[..., 0] = 0
            strength = _lab_grad(g_ab, ppm)
        else:
            strength = _image_grad(ref, ppm, lambda: _lab_grad(lab_ref, ppm))
    else:
        # chroma (C*): a black / grey line is a low-chroma line on coloured areas, whatever the
        # lighting, while the edge of a shadow over a colour is weak in it
        chroma = np.hypot(lab_ref[..., 1], lab_ref[..., 2]).astype(np.float32)
        chans = {"colour": [chroma], "both": [lab_ref[..., 0], chroma]}.get(p.outline_channel, [lab_ref[..., 0]])
        if method == "fdog":
            strength = _fdog_strength(chans, ppm, dark)
        else:
            strength = _line_response(chans[0], ppm, method, dark)
            for ch in chans[1:]:
                check_cancel()
                strength = np.maximum(strength, _line_response(ch, ppm, method, dark))
        floor = _LINE_FLOOR[method]
        if method in ("xdog", "fdog"):   # faint lines of a low-contrast image stay candidates
            floor *= _contrast_factor(lab_ref[..., 0], region)
        edges = _crest(strength, floor)
    edges.setflags(write=False)          # shared by the zones and computations (copied to change)
    return ref, strength, edges


def _line_candidates(prep: Prepared, p: Params, dark: bool, region: np.ndarray | None) -> tuple:
    """Cached _compute_candidates: a computation already running (e.g. started in advance) is
    waited for instead of being run twice."""
    key = _cand_key(prep, p, dark, region)
    for _ in range(3):
        with _CAND_LOCK:
            ent = _CAND_CACHE.get(key)
            if ent is not None and ent[0] is prep.resized:
                _CAND_CACHE.move_to_end(key)
                fut, mine = ent[1], False
            else:
                fut, mine = Future(), True
                _CAND_CACHE[key] = (prep.resized, fut)
                while len(_CAND_CACHE) > _CAND_MAX:
                    _CAND_CACHE.popitem(last=False)
        if mine:
            try:
                fut.set_result(_compute_candidates(prep, p, dark, region))
            except BaseException as e:
                fut.set_exception(e)
                with _CAND_LOCK:
                    if _CAND_CACHE.get(key, (None, None))[1] is fut:
                        del _CAND_CACHE[key]
                raise
            return fut.result()
        try:
            return fut.result()
        except Cancelled:                # another computation, cancelled while making it: again
            continue
    return _compute_candidates(prep, p, dark, region)


def _prefetch_lines(prep: Prepared, p: Params, entries: list, groups: dict, support: np.ndarray) -> None:
    """Starts the candidate lines of the key-detail zones and the local contrast of the original
    in the background as soon as the computation starts (they depend on the image only): they are
    found while the colour layers are cleaned and cut, and the detection then finds them in the
    caches (_line_candidates, _image_grad).
    Not cancelled with the computation (the next one usually needs the same); errors are left to
    the detection itself."""
    if PARALLEL == "none" or not entries:
        return
    # local contrast of the original photo, which every detection compares the render with
    orig = prep.raw if prep.raw is not None else prep.resized
    ppm = p.px_per_mm
    rec = timings.current()             # computation timed: the prefetch reports to it too

    def contrast():
        try:
            with timings.use(rec):
                _image_grad(orig, ppm, lambda: _lab_grad(_lab01(orig), ppm))
        except Exception:
            pass
    _thread_pool().submit(contrast)
    for pe, reg, dest, _is_sil in entries:
        if not pe.outline:
            continue
        # polarity of the lines looked for, as the detection will ask for it (see build)
        if p.stencil_style == "line":
            g = dest if dest in groups else next(iter(groups), None)
            pg = group_params(p, groups[g]) if g else p
        elif into_layers(dest):
            pg = None                   # lines cut into the colour layers: dark lines
        elif dest in groups:
            pg = group_params(p, groups[dest])
        else:
            continue
        dark = True if pg is None else outline_is_dark(outline_color(prep, support, pg), support)

        def run(pe=pe, reg=reg, dark=dark):
            try:
                with timings.use(rec):
                    _line_candidates(prep, pe, dark, reg)
            except Exception:
                pass
        _thread_pool().submit(run)


def detect_key_details(prep: Prepared, sim: np.ndarray, p: Params,
                       region: np.ndarray | None = None, dark: bool = True,
                       stages: bool = False):
    """Key details of the image that the flat render (simplification + airbrush blur) makes disappear.

    1. Candidate lines of the original, by the chosen method (`p.outline_method`):
       canny = sharp edges (thresholds on L*, a*, b*); xdog = drawn line (difference of Gaussians);
       fdog = drawn line averaged along the flow of the contours; ridge = centre line of thin
       strokes (Hessian); teed = learned edges (TEED network). `dark`: polarity of the lines
       for xdog / fdog / ridge.
    2. Only those where the local contrast of the simulated render is clearly lower than the
       original's are kept (lost or faded detail).
    3. Lines that are too short are removed, then thickened to the line width.
    Returns a mask (image area) of the outline stencil openings; with `stages`, a dict of the
    intermediate masks (for the detection preview): detected, selected, kept, short, stroke.
    """
    ppm = p.px_per_mm
    method = p.outline_method
    check_cancel()
    # 1. Candidate lines (none when only the silhouette is drawn): from the image alone, cached
    #    (and usually computed in advance, see _prefetch_lines)
    if p.outline:
        ref, strength, edges = _line_candidates(prep, p, dark, region)
        edges = edges.copy()
    else:
        ref = detail_image(prep, p)
        strength = _image_grad(ref, ppm, lambda: _lab_grad(_lab01(ref), ppm))
        edges = np.zeros(ref.shape[:2], bool)
    check_cancel()
    g_sim = _image_grad(sim, ppm, lambda: _lab_grad(_lab01(sim), ppm))
    check_cancel()
    # the paints reproduce the original photo: the loss of contrast is measured against it
    # (lines are still detected on the preprocessed image, sharper and more contrasted)
    orig = prep.raw if prep.raw is not None else prep.resized
    g_orig = _image_grad(orig, ppm, lambda: _lab_grad(_lab01(orig), ppm))
    check_cancel()
    b = max(2, int(round(0.5 * ppm)))   # no line on the edge of the image itself
    edges[:b, :] = False; edges[-b:, :] = False; edges[:, :b] = False; edges[:, -b:] = False
    if region is not None:               # mask painted by the user: search only inside
        edges &= region.astype(bool)

    def result(sel, kept, short, stroke):
        if not stages:
            return stroke
        return dict(detected=edges, selected=sel, kept=kept, short=short, stroke=stroke > 0, sil=sil)

    # 2. Loss of sharpness: share of the original's local contrast missing from the simulated render
    loss = np.clip((g_orig - g_sim) / (g_orig + 1e-3), 0, 1)
    loss = cv2.dilate(loss, disk(max(1.0, 0.4 * ppm)))      # edge position tolerance

    # 3. Point-by-point selection: importance = line strength x lost share.
    #    The sensitivity sets the share of edge points kept (from most to least important), on a
    #    progressive scale (1 % at 0, ~10 % at 50, 95 % at 100: fine control of the few-lines end).
    #    Minimum lost share: 0.5 up to 15 % kept, then down to 0.2 at 95 % (as before version 1.2).
    keep_share = outline_keep_share(p.outline_sensitivity)
    loss_min = 0.5 - 0.3 * float(np.clip((keep_share - 0.15) / 0.8, 0, 1))
    cand = edges & (loss >= loss_min)
    # effective width: never less than the material's minimum opening (otherwise not cuttable)
    width = max(p.outline_width_mm, p.min_opening_mm * 1.05)
    check_cancel()
    sil, gaps = subject_silhouette(prep, p, sim, gaps=True)   # selected subject's outline, always kept
    check_cancel()
    if not cand.any():
        z = np.zeros(ref.shape[:2], bool)
        if sil is None:
            return result(z, z, z, np.zeros(ref.shape[:2], np.uint8))
        return result(z, sil, z, cv2.dilate(sil.astype(np.uint8) * 255, disk(silhouette_width(p) * ppm / 2)))
    score = strength * loss
    thr = float(np.percentile(score[cand], 100 * (1 - hyst_seed_share(method, keep_share))))
    sel = cand & (score >= thr)
    # Hysteresis along the lines: a selected point extends along its candidate line (up to
    # HYST_MM; DoG / FDoG / TEED: the zone setting outline_follow_mm) where the line stays nearly as
    # important. The importance of a line fluctuates
    # along it (the colour layers partly keep the edge it runs along): without this it is
    # selected in dashes that the minimum length removes. Canny and ridges start from fewer
    # points (`hyst_seed_share`), which sets how much line a sensitivity keeps.
    weak = edges & (loss >= HYST_LOSS * loss_min) & (score >= HYST_SCORE * thr)
    # (0 / 1 masks with OpenCV: the selection only grows, so the same count = nothing new)
    k3 = np.ones((3, 3), np.uint8)
    s8, w8 = sel.view(np.uint8), np.ascontiguousarray(weak).view(np.uint8)
    n_sel = cv2.countNonZero(s8)
    for _ in range(int(round((p.outline_follow_mm if method in FOLLOW_METHODS else HYST_MM) * ppm))):
        grown = cv2.bitwise_or(cv2.bitwise_and(cv2.dilate(s8, k3), w8), s8)
        n = cv2.countNonZero(grown)
        if n == n_sel:
            break
        s8, n_sel = grown, n
    sel = s8.view(bool)
    if p.outline_whole_lines:
        # whole lines: their own detection sensitivity, a minimum mark measured against the most
        # marked lines of the zone
        sel, counts = _whole_lines(cand, score, p.outline_line_sens, ppm, p.outline_min_length_mm)
        object.__setattr__(p, "_line_counts", counts)
    # rejoin small gaps along a same line, then remove pieces that are too short
    e8 = cv2.morphologyEx(sel.astype(np.uint8) * 255, cv2.MORPH_CLOSE, disk(max(1.0, 0.3 * ppm)))
    n, lab_e, stats, _ = cv2.connectedComponentsWithStats(e8, connectivity=8)
    min_px = max(2, int(p.outline_min_length_mm * ppm))
    # approximate length of a line = its largest dimension
    length = np.maximum(stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT])
    keep = length >= min_px
    keep[0] = False
    short = (e8 > 0) & ~keep[lab_e]
    e8 = keep[lab_e].astype(np.uint8) * 255
    if gaps is not None:                # interruptions of the silhouette: no other line there either
        e8[cv2.dilate(gaps.astype(np.uint8), disk(width * ppm / 2)) > 0] = 0
    if sil is not None:
        e8[sil] = 255
    stroke = cv2.dilate(e8, disk(width * ppm / 2))
    if sil is not None and silhouette_width(p) != width:     # the silhouette has its own width
        stroke = cv2.dilate(np.where((e8 > 0) & ~sil, 255, 0).astype(np.uint8), disk(width * ppm / 2)) \
            | cv2.dilate(sil.astype(np.uint8) * 255, disk(silhouette_width(p) * ppm / 2))
    if p.outline_close_lines:
        lines_1px = e8 > 0
        for _ in range(3):              # three lines close together need more than one pass
            before = int(lines_1px.sum())
            lines_1px, stroke = _close_lines(lines_1px, stroke, p, region)
            if int(lines_1px.sum()) == before:
                break
        e8 = lines_1px.astype(np.uint8) * 255
    return result(sel, e8 > 0, short, stroke)


def outline_detection(res: Result, p: Params, region: np.ndarray | None = None) -> dict:
    """Detection stages of the outline stencil for the preview, from a computed result and the
    current outline settings of `p` (colour layers and simulated render are the result's)."""
    p = replace(res.params, **{f.name: getattr(p, f.name) for f in fields(Params)
                               if f.name.startswith("outline")})
    prep = res.prepared
    if prep.fg is not None:
        region = prep.fg if region is None else (region.astype(bool) & prep.fg)
    sim = res.detect_sim_pre if (p.outline_ref == "layers" and res.detect_sim_pre is not None) else res.detect_sim
    if sim is None:      # outline off in the last computation: render of its colour layers
        sim = image_sim(replace(res, layers=[L for L in res.layers if L.kind not in ("outline", "primer")]))
    if p.outline_ref == "image":       # the image alone: compared with the bare support
        sim = np.empty(sim.shape, np.uint8)
        sim[:] = np.asarray(res.support_color, np.uint8)
    if p.outline_target == "stencil" or p.stencil_style == "line":
        if p.stencil_style == "line":   # lines searched against the bare support
            sim = np.empty(sim.shape, np.uint8)
            sim[:] = np.asarray(res.support_color, np.uint8)
        dark = outline_is_dark(outline_color(prep, res.support_color, p), res.support_color)
        return dict(detect_key_details(prep, sim, p, region, dark, stages=True), present=None)
    colour = [L for L in res.layers if L.kind not in ("outline", "primer")]
    color_ids = [i for i in range(len(prep.palette)) if i != res.support_label][:len(colour)]
    painted = res.detect_painted
    if painted is None:  # lines not merged in the last computation: its colour layers as cut
        H, W = prep.labels.shape
        X0, Y0 = res.img_off_px
        painted = np.full((H, W), -1, np.int32)
        for li, L in enumerate(colour):
            painted[L.opening[Y0:Y0 + H, X0:X0 + W] > 0] = li
    return lines_into_layers(prep, sim, p, region, color_ids, painted, stages=True)

# --------------------------------------------------------------------------
# Registration marks (exact geometry in mm, sheet coordinates)
# --------------------------------------------------------------------------

CORNER_NAMES = ("haut-gauche", "haut-droite", "bas-gauche", "bas-droite")


def _reg_layout(res: Result) -> tuple[str, list[tuple], float, list[str]]:
    """Effective shape, mark centres (sheet coordinates, mm) and width of the free band.

    Returns (shape, [(name, cx, cy, sx, sy)], band, warnings); (sx, sy) = inward direction.
    """
    p = res.params
    a = p.reg_size_mm
    t = max(1.0, min(a / 5, 2.5))
    W, H = res.sheet_mm
    warns = []
    shape = p.reg_shape
    if p.sheet_mode == "around":
        m = p.margin_mm
        ox, oy = res.sup_off_mm
        sw, sh = res.support_mm
        x0, y0, x1, y1 = ox, oy, ox + sw, oy + sh
        corners = [(x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)]
        if shape == "equerre" and p.reg_corners and m < t + 1:
            warns.append(f"Marge de la feuille trop petite pour des équerres (il faut au moins {t + 1:.1f} mm) : "
                         "repères en croix à la place.")
            shape = "croix"
        pts = []
        for name, (cx, cy, sx, sy) in zip(CORNER_NAMES, corners[:p.reg_corners]):
            if shape == "equerre":
                pts.append((name, cx, cy, sx, sy))
            elif m >= a + 2:
                pts.append((name, cx - sx * m / 2, cy - sy * m / 2, sx, sy))          # in the margin
            else:
                pts.append((name, cx + sx * (a / 2 + 1), cy + sy * (a / 2 + 1), sx, sy))  # in the support corner
                if p.inner_margin_mm < a + 2 and not warns:
                    warns.append("Les repères sont placés dans les angles du support et risquent de "
                                 "chevaucher l'image : augmentez la marge de la feuille ou la marge intérieure.")
        return shape, pts, m, warns
    # Sheet laid INSIDE the support: no support corner to aim at -> crosses (or holes) in the border,
    # transferred in pencil onto the support using the dimensions.
    if shape == "equerre":
        shape = "croix"
    bx = (res.img_off_px[0]) / p.px_per_mm                       # actual border on the left…
    by = (res.img_off_px[1]) / p.px_per_mm
    bx2 = W - bx - res.image_mm[0]
    by2 = H - by - res.image_mm[1]
    band = max(0.0, min(bx, by, bx2, by2))
    corners = [(0, 0, 1, 1), (W, 0, -1, 1), (0, H, 1, -1), (W, H, -1, -1)]
    off = band / 2 if band >= a + 2 else a / 2 + 1
    if band < a + 2:
        warns.append(f"Bordure de la feuille trop étroite pour les repères ({band:.0f} mm, il en faut "
                     f"{a + 2:.0f}) : ils risquent de chevaucher le motif. Augmentez la bordure.")
    pts = [(name, cx + sx * off, cy + sy * off, sx, sy)
           for name, (cx, cy, sx, sy) in zip(CORNER_NAMES, corners[:p.reg_corners])]
    return shape, pts, band, warns


def registration_shapes(res: Result, layer_index: int | None = None) -> tuple[list[dict], list[str]]:
    """Shapes to cut for registration.

    - equerre (L-bracket): L-shaped slot outside each support corner. The stencil is
      in place when the support corner exactly meets the inner corner of the L.
    - croix (cross): cross window -> the cross is transferred in pencil onto the support (at the
      given dimensions when the sheet is laid inside the support), then each layer is aligned on it.
    - rond (round): hole for registration pegs (diameter = size).
    3 corners: the bottom-right corner is omitted, which acts as a key.
    """
    p = res.params
    a = p.reg_size_mm
    t = max(1.0, min(a / 5, 2.5))
    shape, pts, band, warns = _reg_layout(res)
    shapes = []
    for _name, px, py, sx, sy in pts:
        if shape == "equerre":
            rel = [(-t, -t), (a, -t), (a, 0), (0, 0), (0, a), (-t, a)]
            shapes.append({"type": "poly", "pts": [(px + sx * u, py + sy * v) for u, v in rel]})
        elif shape == "rond":
            shapes.append({"type": "circle", "cx": px, "cy": py, "r": a / 2})
        else:
            half, w = a / 2, t / 2
            rel = [(-w, -half), (w, -half), (w, -w), (half, -w), (half, w), (w, w),
                   (w, half), (-w, half), (-w, w), (-half, w), (-half, -w), (-w, -w)]
            shapes.append({"type": "poly", "pts": [(px + u, py + v) for u, v in rel]})
    if p.layer_notches and layer_index and band >= 5:
        s = min(2.0, band / 6)
        gap = s * 0.8
        total = layer_index * s + (layer_index - 1) * gap
        xs = res.sheet_mm[0] / 2 - total / 2
        cy = band / 2
        for k in range(layer_index):
            x = xs + k * (s + gap)
            shapes.append({"type": "poly", "pts": [(x, cy - s / 2), (x + s, cy - s / 2),
                                                   (x + s, cy + s / 2), (x, cy + s / 2)]})
    return shapes, warns


def reg_shapes_json(shapes: list[dict]) -> list[dict]:
    """Registration shapes (sheet coordinates, mm) rounded for the browser."""
    out = []
    for s in shapes:
        if s["type"] == "poly":
            out.append({"type": "poly", "pts": [[round(x, 2), round(y, 2)] for x, y in s["pts"]]})
        else:
            out.append({"type": "circle", "cx": round(s["cx"], 2), "cy": round(s["cy"], 2), "r": round(s["r"], 2)})
    return out


def positioning(res: Result) -> dict:
    """Ruler dimensions (sheet laid inside the support), origin = support's top-left corner (mm).

    A cross is pencilled on the support at each dimension, then each layer is laid by
    aligning its cross windows on these marks.
    """
    p = res.params
    sw, sh = res.support_mm
    sx0, sy0, sx1, sy1 = res.sheet_rect_mm
    shape, pts, _band, _w = _reg_layout(res)
    marks = []
    for name, px, py, _sx, _sy in pts:
        x, y = sx0 + px, sy0 + py
        marks.append({"coin": name, "x": round(x, 1), "y": round(y, 1),
                      "depuis_droite": round(sw - x, 1), "depuis_bas": round(sh - y, 1),
                      "sur_support": bool(0 <= x <= sw and 0 <= y <= sh)})
    return {
        "mode": p.sheet_mode,
        "repere": "coin haut-gauche du support",
        "forme": shape,
        "feuille": {"gauche": round(sx0, 1), "haut": round(sy0, 1),
                    "droite": round(sw - sx1, 1), "bas": round(sh - sy1, 1),
                    "largeur": round(sx1 - sx0, 1), "hauteur": round(sy1 - sy0, 1)},
        "reperes": marks,
    }


# --------------------------------------------------------------------------
# 6. Previews
# --------------------------------------------------------------------------

# Render of the colour layers on the sheet (float canvas) kept from the detection of the outline
# stencils, which are painted last: the render of the whole result (ΔE) only adds them to it.
# One entry, taken (removed) by the render that uses it; it holds the openings it was made from,
# so their ids cannot be reused while it lives.
_SIM_PREFIX: list = []
_SIM_LOCK = threading.Lock()


def simulate_sheet(res: Result, keep: bool = False) -> np.ndarray:
    """Airbrush render simulation on the sheet (the only place paint goes through).
    `keep`: the float render is kept for a later render of the same layers plus others painted
    after them (see _SIM_PREFIX)."""
    p = res.params
    first = res.layers[0].opening if res.layers else None
    shape = first.shape if first is not None else (int(round(res.sheet_mm[1] * p.px_per_mm)),
                                                   int(round(res.sheet_mm[0] * p.px_per_mm)))
    sig = max(0.3, p.spray_mm * p.px_per_mm)
    setting = (shape, sig, tuple(np.asarray(res.support_color, np.float64).tolist()))
    layers = list(res.layers)
    start = 0
    canvas = None
    with _SIM_LOCK:
        if _SIM_PREFIX:
            pre_set, pre_layers, pre_canvas = _SIM_PREFIX[0]
            n = len(pre_layers)
            if (pre_set == setting and 0 < n <= len(layers)
                    and all(a[0] is L.opening and np.array_equal(a[1], L.color) for a, L in zip(pre_layers, layers))):
                _SIM_PREFIX.clear()
                canvas, start = pre_canvas, n
    if canvas is None:
        canvas = np.empty(shape + (3,), np.float32)
        canvas[:] = res.support_color
    # Only around the openings of each layer: the blur is zero farther than its kernel radius
    # (OpenCV: ksize = round(8 sigma + 1) | 1 for float images), and with a margin at least that
    # wide the blur inside the window is the same as on the whole sheet. Same arithmetic as
    # canvas * (1 - a) + colour * a, in place.
    r = (int(round(sig * 8 + 1)) | 1) // 2 + 2
    H, W = shape
    for L in layers[start:]:
        check_cancel()
        x, y, w, h = cv2.boundingRect(L.opening)
        if w == 0 or h == 0:
            continue
        x0, y0, x1, y1 = max(0, x - r), max(0, y - r), min(W, x + w + r), min(H, y + h + r)
        mask = cv2.GaussianBlur(L.opening[y0:y1, x0:x1].astype(np.float32) / 255.0, (0, 0), sig)
        check_cancel()
        # canvas * (1 - a) + colour * a with OpenCV on 3-channel copies of a: the same float32
        # operations as numpy, without its slow broadcast of a over the 3 channels
        a = 0.97 * mask
        sub = canvas[y0:y1, x0:x1]
        cv2.multiply(sub, cv2.merge([1 - a] * 3), dst=sub)
        col = tuple(float(c) for c in np.asarray(L.color, np.float32)) + (0.0,)
        cv2.add(sub, cv2.multiply(cv2.merge([a] * 3), col), dst=sub)
    out = np.clip(canvas, 0, 255).astype(np.uint8)
    if keep and layers:
        with _SIM_LOCK:
            _SIM_PREFIX[:] = [(setting, [(L.opening, np.array(L.color)) for L in layers], canvas)]
    return out


def image_sim(res: Result, sheet_sim: np.ndarray | None = None, keep: bool = False) -> np.ndarray:
    """Simulated render of the image area only (`keep`: see simulate_sheet)."""
    sim = simulate_sheet(res, keep) if sheet_sim is None else sheet_sim
    H, W = res.prepared.labels.shape
    X0, Y0 = res.img_off_px
    return sim[Y0:Y0 + H, X0:X0 + W]


def support_overlap(res: Result) -> tuple[tuple, tuple]:
    """Common area of sheet and support: (x0, y0, x1, y1) in the support and offset to the sheet."""
    SW, SH = res.support_px
    ox, oy = res.sup_off_px
    if res.layers:
        Hs, Ws = res.layers[0].opening.shape
    else:
        Ws, Hs = int(round(res.sheet_mm[0] * res.params.px_per_mm)), int(round(res.sheet_mm[1] * res.params.px_per_mm))
    x0, y0 = max(0, -ox), max(0, -oy)
    x1, y1 = min(SW, Ws - ox), min(SH, Hs - oy)
    return (x0, y0, max(x0, x1), max(y0, y1)), (ox, oy)


def sheet_to_support(res: Result, arr: np.ndarray, fill=0) -> np.ndarray:
    """Copies a "sheet" array to the support size (the rest is filled with `fill`)."""
    SW, SH = res.support_px
    out = np.empty((SH, SW) + arr.shape[2:], arr.dtype)
    out[:] = fill
    (x0, y0, x1, y1), (ox, oy) = support_overlap(res)
    if x1 > x0 and y1 > y0:
        out[y0:y1, x0:x1] = arr[y0 + oy:y1 + oy, x0 + ox:x1 + ox]
    return out


def simulate(res: Result) -> np.ndarray:
    """Airbrush render simulation over the whole support."""
    return sheet_to_support(res, simulate_sheet(res), res.support_color.astype(np.uint8))


def image_crop(res: Result, sim: np.ndarray) -> np.ndarray:
    """Image area in a support-sized render."""
    H, W = res.prepared.labels.shape
    ox, oy = res.prepared.off_px
    return sim[oy:oy + H, ox:ox + W]


def quantized_view(res: Result) -> np.ndarray:
    pal = res.prepared.palette.copy()
    if res.support_label is not None:
        pal[res.support_label] = res.support_color
    return pal[res.prepared.labels]


def layer_view(res: Result, L: Layer, sheet_bgr=(200, 211, 220), bridges: bool = True,
               lost: bool = True, marks: bool = True) -> np.ndarray:
    """Stencil view: material, openings, bridges (red), lost details (orange), registration.

    `bridges` / `lost` = False: view without these highlights (see `layer_overlays` to get them separately).
    `marks` = False: without the registration marks (see `reg_shapes_json` to draw them separately).
    """
    p = res.params
    ppm = p.px_per_mm
    SW, SH = res.support_px
    ox, oy = res.sup_off_px
    op = L.opening > 0
    img = np.empty(op.shape + (3,), np.uint8)
    img[:] = sheet_bgr
    # support outline (visual guide, not cut)
    cv2.rectangle(img, (ox, oy), (ox + SW - 1, oy + SH - 1), (150, 150, 150), 1)
    light = (L.color.astype(np.float32) * 0.55 + 255 * 0.45).astype(np.uint8)
    img[op] = light
    edge = cv2.morphologyEx(L.opening, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
    img[edge] = (40, 40, 40)
    if lost:
        sel = _lost_mask(res, L, edge)
        img[sel] = (img[sel] * (1 - LOST_ALPHA) + np.array(LOST_BGR) * LOST_ALPHA).astype(np.uint8)
    if bridges:
        _draw_bridges(img, res, L, BRIDGE_BGR)
    shapes, _ = registration_shapes(res, L.index) if marks else ([], [])
    for s in shapes:
        if s["type"] == "poly":
            pts = (np.array(s["pts"]) * ppm).round().astype(np.int32)
            cv2.fillPoly(img, [pts], (255, 255, 255))
            cv2.polylines(img, [pts], True, (40, 40, 40), 1)
        else:
            c = (int(round(s["cx"] * ppm)), int(round(s["cy"] * ppm)))
            r = int(round(s["r"] * ppm))
            cv2.circle(img, c, r, (255, 255, 255), -1)
            cv2.circle(img, c, r, (40, 40, 40), 1)
    if p.sheet_outline:
        cv2.rectangle(img, (0, 0), (img.shape[1] - 1, img.shape[0] - 1), (40, 40, 40), 1)
    return img


BRIDGE_BGR = (40, 40, 230)
LOST_BGR = (0, 140, 255)
LOST_ALPHA = 0.6
MERGE_BGR = (200, 60, 150)      # violet: zones merged instead of a bridge
MERGE_ALPHA = 0.75
FLAT_BGR = (166, 184, 20)       # teal: painted flat zones of the layer
FLAT_ALPHA = 0.35


def _flat_mask(res: Result, L: Layer) -> np.ndarray | None:
    """Painted flat zones that apply to a colour layer, sheet size: the "auto" zones where they
    ended in this layer's tone, and the zones forced into this layer. None when there is none."""
    prep = res.prepared
    if L.kind == "outline" or L.tone is None:
        return None
    m = None
    if prep.flat_zones is not None:
        m = prep.flat_zones & (prep.labels == L.tone)
    forced = (prep.flat_forced_zones or {}).get(L.tone)
    if forced is not None:
        m = forced if m is None else (m | forced)
    if m is None or not m.any():
        return None
    H, W = m.shape
    X0, Y0 = res.img_off_px
    out = np.zeros(L.opening.shape, bool)
    Hs, Ws = out.shape
    x0, y0, x1, y1 = max(0, X0), max(0, Y0), min(Ws, X0 + W), min(Hs, Y0 + H)
    if x1 > x0 and y1 > y0:
        out[y0:y1, x0:x1] = m[y0 - Y0:y1 - Y0, x0 - X0:x1 - X0]
    return out if out.any() else None


def _lost_mask(res: Result, L: Layer, edge: np.ndarray | None = None) -> np.ndarray:
    H, W = res.prepared.labels.shape
    X0, Y0 = res.img_off_px
    m = np.zeros(L.opening.shape, bool)
    m[Y0:Y0 + H, X0:X0 + W] = L.lost > 0
    if edge is None:
        edge = cv2.morphologyEx(L.opening, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
    return m & ~edge


def _draw_bridges(img: np.ndarray, res: Result, L: Layer, color) -> None:
    bw = max(1, int(round(res.params.bridge_width_mm * res.params.px_per_mm)))
    for (x1, y1, x2, y2) in L.bridges:
        cv2.line(img, (x1, y1), (x2, y2), color, thickness=bw)


def opening_without_bridges(res: Result, L: Layer) -> np.ndarray:
    """Openings of a layer with its bridges painted over, as after retouching them: the opening
    pixels the bridges cut (Layer.bridge_cut) are put back. Results saved before it existed: the
    bridge strokes limited to a closing of the opening at the bridge width (this missed the base
    of the bridges, where a disk of the closing fits across the bridge and the material it joins)."""
    if not L.bridges:
        return L.opening
    cut = getattr(L, "bridge_cut", None)
    if cut is not None:
        return np.where(cut, np.uint8(255), L.opening)
    a = np.zeros(L.opening.shape, np.uint8)
    _draw_bridges(a, res, L, 255)
    bw = max(1, int(round(res.params.bridge_width_mm * res.params.px_per_mm)))
    closed = cv2.morphologyEx(L.opening, cv2.MORPH_CLOSE, disk(bw / 2 + 1))
    return np.maximum(L.opening, cv2.bitwise_and(a, closed))


def layer_overlays(res: Result, L: Layer) -> tuple:
    """Transparent overlays (BGRA, sheet size) of the bridges, removed details, merged zones and
    painted flat zones (None when the layer has none), to stack on
    `layer_view(..., bridges=False, lost=False)` so they can be shown or hidden."""
    # colour everywhere, only transparency varies: sharp edges once the image is downscaled
    shape = L.opening.shape
    br = np.empty(shape + (4,), np.uint8)
    br[..., :3] = BRIDGE_BGR
    a = np.zeros(shape, np.uint8)
    _draw_bridges(a, res, L, 255)
    br[..., 3] = a
    lo = np.empty(shape + (4,), np.uint8)
    lo[..., :3] = LOST_BGR
    lo[..., 3] = np.where(_lost_mask(res, L), int(round(255 * LOST_ALPHA)), 0).astype(np.uint8)
    mg = np.empty(shape + (4,), np.uint8)
    mg[..., :3] = MERGE_BGR
    mg[..., 3] = 0 if L.merged is None else np.where(L.merged, int(round(255 * MERGE_ALPHA)), 0).astype(np.uint8)
    fz = _flat_mask(res, L)
    fl = None
    if fz is not None:
        # light fill and a solid outline: the zone stays readable over the layer
        u8 = fz.astype(np.uint8)
        edge = cv2.morphologyEx(u8, cv2.MORPH_GRADIENT, disk(max(1.0, res.params.px_per_mm * 0.25))) > 0
        fl = np.empty(shape + (4,), np.uint8)
        fl[..., :3] = FLAT_BGR
        fl[..., 3] = np.where(edge, 255, np.where(fz, int(round(255 * FLAT_ALPHA)), 0)).astype(np.uint8)
    return br, lo, mg, fl


def delta_e(res: Result, sheet_sim: np.ndarray | None = None) -> float:
    """Mean difference (Lab ΔE) between the original image and the simulated render, over the image area."""
    a = cv2.cvtColor(original(res.prepared).astype(np.float32) / 255, cv2.COLOR_BGR2LAB)
    b = cv2.cvtColor(image_sim(res, sheet_sim).astype(np.float32) / 255, cv2.COLOR_BGR2LAB)
    if res.prepared.gray:
        return float(np.abs(a[..., 0] - b[..., 0]).mean())
    return float(np.sqrt(((a - b) ** 2).sum(axis=2)).mean())


def layer_stats(L: Layer) -> dict:
    length = 0.0
    nodes = 0
    for pts in L.paths:
        d = np.diff(np.vstack([pts, pts[:1]]), axis=0)
        length += float(np.hypot(d[:, 0], d[:, 1]).sum())
        nodes += len(pts)
    return {
        "paths": len(L.paths),
        "nodes": nodes,
        "bridges": len(L.bridges),
        "merged": L.n_merged,
        "islands": L.islands,
        "kind": L.kind,
        "short_islands": L.short_islands,
        "struct_bridges": L.struct_bridges,
        "manual_bridges": getattr(L, "manual_bridges", 0),
        "weak_spots": L.weak_spots,
        "cut_length_m": round(length / 1000, 2),
        "lost_pct": round(100 * float((L.lost > 0).mean()), 1),
    }
