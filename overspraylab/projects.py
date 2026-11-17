"""Project folders and their snapshots, on disk.

    <projects>/<project folder>/
        projet.json              project: id, name, original image
        original.<ext>           source image as loaded, never reduced, stored once
        snapshots/v<n>/          one self-contained snapshot (written once, never changed)
            infos.json           settings (same format as the export)
            *.png                painted masks (size of the working image of the snapshot)
            reduite.<ext>        working image, only when the snapshot works on a reduced image
            snapshot.json        label, date, measurements
            thumb.jpg            simulated render, small
            session.pkl.gz       dump of the in-memory session (computed result): optional cache

The JSON keys are a file format (French, like infos.json): never translate or rename them.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import pickle
import re
import shutil
import uuid
from datetime import datetime
from pathlib import Path

ROOT = Path(os.environ.get("OVERSPRAYLAB_PROJECTS") or Path(__file__).resolve().parent.parent / "projets")
PROJECT_FILE = "projet.json"
SNAP_FILE = "snapshot.json"
SESSION_FILE = "session.pkl.gz"
SESSION_FORMAT = 1                  # format of the session dump: older or newer dumps are ignored
SNAP_ID = re.compile(r"^v(\d{1,5})$")
DIR_NAME = re.compile(r"^[^<>:\"/\\|?*\x00-\x1f]{1,80}$")


class ProjectError(Exception):
    pass


def _read_json(p: Path) -> dict:
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def _write_json(p: Path, data: dict) -> None:
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(p)


def project_dir(name: str) -> Path:
    """Folder of an existing project (name of the folder, as listed)."""
    if not (isinstance(name, str) and DIR_NAME.match(name)) or name in (".", "..") or name.strip(" .") != name:
        raise ProjectError("Projet inconnu")
    d = ROOT / name
    if not (d / PROJECT_FILE).is_file():
        raise ProjectError("Projet inconnu")
    return d


def snap_dir(d: Path, sid: str) -> Path:
    if not (isinstance(sid, str) and SNAP_ID.match(sid)):
        raise ProjectError("Snapshot inconnue")
    s = d / "snapshots" / sid
    if not (s / "infos.json").is_file():
        raise ProjectError("Snapshot inconnue")
    return s


def snap_ids(d: Path) -> list[str]:
    """Snapshots of a project, in creation order."""
    base = d / "snapshots"
    ids = [p.name for p in base.iterdir() if p.is_dir() and SNAP_ID.match(p.name)] if base.is_dir() else []
    return sorted(ids, key=lambda s: int(s[1:]))


def next_id(d: Path | None) -> str:
    """Number of the next snapshot: never one already used, even by a deleted snapshot."""
    if not d:
        return "v0"
    ids = snap_ids(d)
    last = max([int(i[1:]) for i in ids] + [int(project_info(d).get("dernier_numero", -1))])
    return f"v{last + 1}"


def delete_snapshot(d: Path, sid: str) -> None:
    s = snap_dir(d, sid)
    _note_number(d, sid)
    shutil.rmtree(s)


def set_label(d: Path, sid: str, label: str) -> None:
    s = snap_dir(d, sid)
    meta = _read_json(s / SNAP_FILE)
    meta["libelle"] = label
    _write_json(s / SNAP_FILE, meta)


def _note_number(d: Path, sid: str) -> None:
    """Highest snapshot number used, kept in projet.json (deleted numbers are not given again)."""
    info = project_info(d)
    if int(sid[1:]) > int(info.get("dernier_numero", -1)):
        info["dernier_numero"] = int(sid[1:])
        _write_json(d / PROJECT_FILE, info)


def project_info(d: Path) -> dict:
    return _read_json(d / PROJECT_FILE)


def snapshots(d: Path) -> list[dict]:
    out = []
    for sid in snap_ids(d):
        meta = _read_json(d / "snapshots" / sid / SNAP_FILE)
        out.append({"id": sid, "uid": meta.get("uid"), "label": str(meta.get("libelle") or ""), "date": meta.get("date"),
                    "delta_e": meta.get("delta_e"), "layers": meta.get("pochoirs"),
                    "reduced": bool(meta.get("image_reduite")),
                    "thumb": (d / "snapshots" / sid / "thumb.jpg").is_file()})
    return out


def touch(d: Path) -> None:
    """Notes that the project was just used (opened, or a snapshot taken): recent projects."""
    info = project_info(d)
    info["ouvert_le"] = now()
    _write_json(d / PROJECT_FILE, info)


def list_projects(limit: int | None = None) -> list[dict]:
    """Projects of the folder, most recently used first (opened or saved); `limit`: only the first ones."""
    out = []
    if not ROOT.is_dir():
        return out
    for d in ROOT.iterdir():
        if not (d.is_dir() and (d / PROJECT_FILE).is_file()):
            continue
        info, snaps = project_info(d), snapshots(d)
        last = snaps[-1] if snaps else None
        out.append({"dir": d.name, "name": str(info.get("nom") or d.name), "count": len(snaps),
                    "last": last["id"] if last else None,
                    "date": (last or {}).get("date") or info.get("date"),
                    "used": info.get("ouvert_le") or (last or {}).get("date") or info.get("date"),
                    "thumb": bool(last and last["thumb"])})
    out.sort(key=lambda p: p["used"] or "", reverse=True)
    return out[:limit] if limit else out


def _free_dir(name: str) -> Path:
    base = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', " ", name).strip().strip(".") or "projet"
    base = re.sub(r"\s+", " ", base)[:70]
    d, n = ROOT / base, 2
    while d.exists():
        d, n = ROOT / f"{base}_{n}", n + 1
    return d


def create_project(name: str, original: bytes, ext: str, image_name: str | None, pid: str | None = None) -> Path:
    """New project folder (named after the project, suffix _2, _3… if taken) with its original image."""
    ROOT.mkdir(parents=True, exist_ok=True)
    d = _free_dir(name)
    d.mkdir(parents=True)
    img = f"original{ext}"
    (d / img).write_bytes(original)
    _write_json(d / PROJECT_FILE, {"format": 1, "id": pid or uuid.uuid4().hex, "nom": name, "image": img,
                                   "nom_image": image_name, "sha256": hashlib.sha256(original).hexdigest(),
                                   "date": now()})
    return d


def rename_project(d: Path, name: str) -> None:
    """Project renamed in the interface: only projet.json changes (the folder keeps its name)."""
    info = project_info(d)
    if info.get("nom") != name:
        info["nom"] = name
        _write_json(d / PROJECT_FILE, info)


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def write_snapshot(d: Path, files: dict[str, bytes], meta: dict, thumb: bytes | None,
                   session: dict | None) -> str:
    """Writes a new snapshot (next number) and returns its id. Written into a temporary folder
    renamed at the end: a snapshot is complete or absent."""
    base = d / "snapshots"
    base.mkdir(exist_ok=True)
    tmp = base / f".tmp_{uuid.uuid4().hex[:8]}"
    tmp.mkdir()
    try:
        for f, data in files.items():
            (tmp / f).write_bytes(data)
        if thumb:
            (tmp / "thumb.jpg").write_bytes(thumb)
        if session is not None:
            try:
                blob = pickle.dumps({"format": SESSION_FORMAT, "session": session}, protocol=pickle.HIGHEST_PROTOCOL)
                (tmp / SESSION_FILE).write_bytes(gzip.compress(blob, 6))
            except (pickle.PicklingError, TypeError, AttributeError):
                pass                     # no cache: the snapshot is recomputed when opened
        while True:
            sid = next_id(d)
            _write_json(tmp / SNAP_FILE, {**meta, "id": sid, "date": meta.get("date") or now(),
                                          "uid": meta.get("uid") or uuid.uuid4().hex})
            try:
                tmp.rename(base / sid)
                _note_number(d, sid)
                touch(d)
                return sid
            except FileExistsError:      # taken meanwhile: next number
                continue
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


def read_session(s: Path) -> dict | None:
    """Session dump of a snapshot, or None (absent, unreadable or of another format)."""
    p = s / SESSION_FILE
    if not p.is_file():
        return None
    try:
        data = pickle.loads(gzip.decompress(p.read_bytes()))
    except Exception:                  # noqa: BLE001 - any failure: recomputed instead
        return None
    if not isinstance(data, dict) or data.get("format") != SESSION_FORMAT or not isinstance(data.get("session"), dict):
        return None
    return data["session"]


def folder_files(d: Path, only: str | None = None) -> dict[str, bytes]:
    """Every file of the project folder (relative paths with /), for the export zip; `only`: just
    this snapshot (with the project files)."""
    out = {}
    for p in sorted(d.rglob("*")):
        rel = p.relative_to(d).as_posix()
        if not p.is_file() or rel.split("/")[-1].startswith(".tmp") or "/.tmp_" in "/" + rel:
            continue
        if only and rel.startswith("snapshots/") and not rel.startswith(f"snapshots/{only}/"):
            continue
        out[rel] = p.read_bytes()
    return out


def import_folder(files: dict[str, bytes]) -> tuple[Path, dict[str, str]]:
    """Project folder found in an export zip, put into the projects folder. If a project with the
    same id is already there, the snapshots it lacks are added to it (under the next numbers; an
    existing snapshot is never overwritten), otherwise a new folder is created.
    Returns the folder and {snapshot id in the zip: id in the folder}."""
    info = json.loads(files[PROJECT_FILE].decode("utf-8"))
    img = info.get("image")
    if not (isinstance(img, str) and img in files and "/" not in img):
        raise ProjectError("Dossier de projet incomplet : image d'origine absente")
    pid = str(info.get("id") or "")
    target = None
    if pid and ROOT.is_dir():
        for d in ROOT.iterdir():
            if d.is_dir() and (d / PROJECT_FILE).is_file() and project_info(d).get("id") == pid:
                target = d
                break
    if target is None:
        target = create_project(str(info.get("nom") or "projet"), files[img], Path(img).suffix,
                                info.get("nom_image"), pid or None)
    # a snapshot is known by its uid (snapshots of earlier versions: by date and label)
    def key_of(meta):
        return meta.get("uid") or (meta.get("date"), str(meta.get("libelle") or meta.get("label") or ""))
    have = {key_of(m): m["id"] for m in snapshots(target)}
    groups: dict[str, dict[str, bytes]] = {}
    for rel, data in files.items():
        parts = rel.split("/")
        if len(parts) == 3 and parts[0] == "snapshots" and SNAP_ID.match(parts[1]) and DIR_NAME.match(parts[2]) \
                and parts[2] not in (".", ".."):
            groups.setdefault(parts[1], {})[parts[2]] = data
    mapping = {}
    for sid in sorted(groups, key=lambda s: int(s[1:])):
        g = groups[sid]
        if "infos.json" not in g:
            continue
        try:
            meta = json.loads(g.pop(SNAP_FILE, b"{}").decode("utf-8"))
        except ValueError:
            meta = {}
        meta = meta if isinstance(meta, dict) else {}
        key = key_of(meta)
        if key in have:                  # already there
            mapping[sid] = have[key]
            continue
        thumb = g.pop("thumb.jpg", None)
        session = g.pop(SESSION_FILE, None)
        new = write_snapshot(target, g, meta, thumb, None)
        if session:
            (target / "snapshots" / new / SESSION_FILE).write_bytes(session)
        mapping[sid] = new
        have[key] = new
    return target, mapping               # mapping: snapshot ids of the zip -> ids in the folder
