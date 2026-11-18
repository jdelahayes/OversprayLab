# OversprayLab – notes for Claude

See README.md for what the app does and how it is organised. This file only lists the rules that
are not obvious from the code.

## Run

- Server: `.venv/Scripts/python app.py` → http://127.0.0.1:5000 (or `lancer.bat`).
- Restart the server after any Python change (`app.py`, `overspraylab/*.py`); `static/` is served live.
- `OVERSPRAYLAB_PORT=5001` runs a second instance; tests go there, 5000 is the author's own instance.
- Layer computation is parallel; `OVERSPRAYLAB_PARALLEL=thread|process|none` (`none` helps debugging).

## Translations (main pitfall)

- **UI source strings are French.** `static/index.html` (HTML text, attributes, JS strings, TIPS)
  is written in French and translated at runtime by `static/i18n.js` (exact-match `EN` dictionary
  plus regex patterns, applied by a MutationObserver). Default language is English.
- Every new or changed UI text, tooltip or server message shown in the UI **must get an entry in
  `i18n.js`**, otherwise English users see French. Elements marked `data-noi18n` are skipped.
- `I18N.start()` must stay after the code that builds the DOM (setupColumns, applyTips,
  setupAccordion) in `init`.
- Exports work the other way round: `overspraylab/svg_export.py` strings are English, wrapped in
  `T()`, with French in `_FR`; the language comes from `?lang=` (contextvar `_LANG`).
- Keys of `infos.json` (`…_infos.json` in earlier exports) are a file format (project re-opening): never translate or rename them.
- `tools/extract_strings.py` lists French strings to check coverage.

## Silhouette Studio compatibility (Inkscape export)

Hard-won constraints, verified by the author in Silhouette Studio – do not "clean up":
- embedded images are JPEG at full resolution, with **no `clip-path`**;
- hidden layers carry both `style="display:none"` and the `display="none"` attribute;
- the occasional black/partial render image when zooming in Silhouette is a Silhouette bug;
- Silhouette skips part of the shapes of a layer made of many separate polygons: `silhouette.svg`
  keeps **one single compound `<path>` per layer** (openings + sheet outline + marks). Generating
  `.studio3` files directly was investigated and is not needed.

## Object selection (MobileSAM)

- `overspraylab/semantic.py`: optional `onnxruntime`; models downloaded on first use into `models/`
  (git-ignored), pinned by size + SHA-256. Never commit the `.onnx` files.
- Embedding cached per session (`sess["sem"]`); a subject mask containing an object selection is
  "exact" (`subject_exact`, project key `masque_sujet_exact`): used as is, no GrabCut.
- Exception: `overspraylab/edge_model/teed_biped7.onnx` (TEED, 250 kB, MIT, licence file next to it)
  is committed on purpose (author's choice: no official ONNX exists to download). Regenerate it
  with `tools/export_teed_onnx.py` (needs PyTorch in a separate environment, never in `.venv`).

## Key-detail selection

- Every method selects with a hysteresis along the lines. Canny and ridges start from fewer points
  (`hyst_seed_share`, `HYST_SEED_DIV`): the divisors set how much line a sensitivity keeps (the
  author chose about 2 to 3 times more than the former point-by-point selection at sensitivity 50);
  re-measure before changing either.
- DoG: the strength is averaged along the contours (`_along_contours`), which needs a longer
  hysteresis reach: the zone setting `outline_follow_mm` (zone key `follow_mm`, default
  `HYST_MM_DOG` = 12, also for zones saved before it existed). With the 4 mm of the other methods,
  long clean lines (logos) come out with wide gaps at medium sensitivity. Tried and rejected for the DoG: several scales and
  hue channels (double lines on sharp contours), denoising (no visible gain).
- FDoG (`fdog`, `_fdog_strength`): DoG averaged along curved streamlines of the structure-tensor
  flow (2 mm, 2 px steps); needs only 4 mm of following. Slow (~3 s on a 3 Mpx zone), so its
  strength map is cached (LRU of 4, keyed by a hash of the channels). An iterative ETF flow gave
  the same lines, rounded T-junctions and was 3x slower: not used.
- TEED (`teed`, `overspraylab/teed.py`): strength = network logits (not the sigmoid, which
  saturates at 1 on every line and leaves the sensitivity without effect); lines = crest of the
  smoothed logits (a skeleton gave the same lines 5x slower). Uses `outline_follow_mm`
  (advised 8 mm); the zone's channel setting is ignored (hidden in the UI).

## Projects folder and snapshots

- `overspraylab/projects.py`: one folder per project in `projets/` (git-ignored; `OVERSPRAYLAB_PROJECTS`
  moves it – tests must point it to a temporary folder, never to the author's `projets/`).
  `original.<ext>` = source image as loaded, never reduced (`sess["orig_raw"]`); each `snapshots/v<n>/`
  is written once (temp folder renamed) and never changed.
- `session.pkl.gz` = pickle of the session entries in `SESSION_DUMP` (app.py), restored as is so the
  next computation still reuses the geometry (identity of the masks in `sess["geom"]`). Bump
  `projects.SESSION_FORMAT` when those objects change incompatibly: older dumps are then ignored and
  the snapshot is recomputed. Unpickling an imported zip is not safe: accepted for now (author's
  choice), to be dealt with later.
- The client compares `snapKey()` with the key saved at the last snapshot / opening to know whether
  the current version is saved; display-only options (`VIEW_ONLY`) are not changes.

## Testing in the browser

- Browser tests run in **Chrome** (Claude in Chrome extension), on the port 5001 instance; do not use
  Edge, headless or not. If the extension is not connected, say so instead of switching browser.
- Never alter the author's saved state: set `NO_REMEMBER = true` in the page before changing
  settings (otherwise localStorage `overspraylab.*` is overwritten; keys of the former name
  `stencilit.*` are migrated at the top of `i18n.js`), and back up / restore
  `presets_utilisateur.json` if a test saves presets or palettes.
- Use `samples/test_visage.png` (or `test_synthetique.png`). `samples/2400x0.jpg` is a personal
  photo (git-ignored): it may be used for local tests and calibration, but **must never be
  committed nor appear in screenshots, the README or any published output**.

## Conventions

- Code, comments, README and commit messages in English.
- Geometry is in mm; rasterisation at `px_per_mm`.
- Commit, tag or push only when asked. The GitHub repo (jdelahayes/OversprayLab) is public.
