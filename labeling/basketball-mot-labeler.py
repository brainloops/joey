#!/usr/bin/env python3
"""
BasketballMOT DearPyGui labeler (MVP).

Primary workflow:
- Label one person per pass (active track).
- Assigned detections are hidden from candidate pools by default.
- Write MOT GT rows directly to gt/gt.txt in-place under BasketballMOT sequences.
"""

from __future__ import annotations

import argparse
import configparser
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
sv = None
tk = None

try:
    import supervision as sv  # type: ignore
except Exception:
    sv = None

try:
    import trackers as tk  # type: ignore
except Exception:
    tk = None

try:
    import dearpygui.dearpygui as dpg
except Exception as exc:  # pragma: no cover - runtime dependency guard
    raise RuntimeError(
        "DearPyGui is required. Install it before running the labeler: pip install dearpygui"
    ) from exc


@dataclass
class Detection:
    frame: int
    det_idx: int
    x: float
    y: float
    w: float
    h: float
    score: float
    class_id: int


@dataclass
class Assignment:
    track_id: int
    frame: int
    x: float
    y: float
    w: float
    h: float
    det_key: Optional[Tuple[int, int]]


@dataclass
class SequenceInfo:
    name: str
    root: Path
    img_dir: Path
    det_file: Path
    gt_file: Path
    legacy_state_file: Path
    absent_file: Path
    session_file: Path
    seq_len: int
    width: int
    height: int
    im_ext: str


def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def _bbox_iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _bbox_intersects(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> bool:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    return (ix2 - ix1) > 0.0 and (iy2 - iy1) > 0.0


def _get_screen_size() -> Tuple[int, int]:
    """Best-effort monitor size query with sane fallback."""
    try:
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        width = int(root.winfo_screenwidth())
        height = int(root.winfo_screenheight())
        root.destroy()
        if width > 0 and height > 0:
            return width, height
    except Exception:
        pass
    return 1920, 1080


def _atomic_write_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    if path.exists():
        bak_path = path.with_name(path.name + ".bak")
        shutil.copy2(path, bak_path)
    tmp_path.write_text(payload, encoding="utf-8")
    tmp_path.replace(path)


def _atomic_write_json(path: Path, obj: Any) -> None:
    _atomic_write_text(path, json.dumps(obj, indent=2))


def discover_sequences(dataset_root: Path, split: str) -> List[SequenceInfo]:
    split_root = dataset_root / split
    if not split_root.is_dir():
        raise FileNotFoundError(f"Split folder not found: {split_root}")

    seqs: List[SequenceInfo] = []
    for seq_root in sorted(p for p in split_root.iterdir() if p.is_dir()):
        img_dir = seq_root / "img1"
        det_file = seq_root / "det" / "det.txt"
        gt_file = seq_root / "gt" / "gt.txt"
        legacy_state_file = seq_root / "gt" / "labeler_state.json"
        absent_file = seq_root / "gt" / "absent_frames.json"
        session_file = seq_root / "gt" / "session_state.json"
        seqinfo_file = seq_root / "seqinfo.ini"
        if not (img_dir.is_dir() and det_file.is_file() and seqinfo_file.is_file()):
            continue
        cfg = configparser.ConfigParser()
        cfg.read(seqinfo_file)
        seq_cfg = cfg["Sequence"]
        seqs.append(
            SequenceInfo(
                name=seq_root.name,
                root=seq_root,
                img_dir=img_dir,
                det_file=det_file,
                gt_file=gt_file,
                legacy_state_file=legacy_state_file,
                absent_file=absent_file,
                session_file=session_file,
                seq_len=int(seq_cfg.get("seqLength", "0")),
                width=int(seq_cfg.get("imWidth", "0")),
                height=int(seq_cfg.get("imHeight", "0")),
                im_ext=seq_cfg.get("imExt", ".jpg"),
            )
        )
    if not seqs:
        raise RuntimeError(f"No valid sequences found under: {split_root}")
    return seqs


def parse_det_file(det_file: Path, min_score: float = 0.0) -> Dict[int, List[Detection]]:
    by_frame: Dict[int, List[Detection]] = {}
    det_counts: Dict[int, int] = {}
    for raw in det_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        fields = line.split(",")
        if len(fields) < 8:
            continue
        frame = int(float(fields[0]))
        x = float(fields[2])
        y = float(fields[3])
        w = float(fields[4])
        h = float(fields[5])
        score = float(fields[6])
        class_id = int(float(fields[7]))
        if class_id != 1:
            continue
        if score < min_score:
            continue
        det_idx = det_counts.get(frame, 0)
        det_counts[frame] = det_idx + 1
        det = Detection(frame=frame, det_idx=det_idx, x=x, y=y, w=w, h=h, score=score, class_id=class_id)
        by_frame.setdefault(frame, []).append(det)
    return by_frame


class BasketballMOTLabelerApp:
    def __init__(self, dataset_root: Path, split: str, min_score: float) -> None:
        self.track_id_start = 1
        self.dataset_root = dataset_root
        self.split = split
        self.split_root = dataset_root / split
        self.min_score = min_score
        self.sequences = discover_sequences(dataset_root=dataset_root, split=split)
        self.sequence_by_name = {s.name: s for s in self.sequences}
        self.app_state_file = self.split_root / "labeler_app_state.json"
        self.current_seq = self._initial_sequence_from_app_state()
        self.current_frame = 1
        self.active_track_id: Optional[int] = None
        self.next_track_id = 1

        self.detections_by_frame: Dict[int, List[Detection]] = {}
        self.assignments_by_track: Dict[int, Dict[int, Assignment]] = {}
        self.absent_frames_by_track: Dict[int, set[int]] = {}
        self.assigned_det_keys: set[Tuple[int, int]] = set()
        self.last_assigned_box_by_track: Dict[int, Tuple[float, float, float, float]] = {}
        self.undo_stack: List[Tuple[str, Any]] = []
        self.focus_mode = False
        self.skip_mode = False
        self.active_box_img: Optional[Tuple[float, float, float, float]] = None
        self.cycle_last_point_img: Optional[Tuple[float, float]] = None
        self.cycle_last_signature: Optional[Tuple[Tuple[int, int], ...]] = None
        self.cycle_step = 0
        self.playback_track_id: Optional[int] = None
        self.playback_active = False
        self.playback_fps = 30.0
        self.playback_last_ts = 0.0
        self.space_toggle_latch = False

        self.texture_tag = "bmot_texture"
        self.image_tag = "bmot_image"
        self.canvas_tag = "bmot_canvas"
        self.status_tag = "bmot_status"
        self.track_tag = "bmot_active_track"
        self.frame_slider_tag = "bmot_frame_slider"
        self.frame_input_tag = "bmot_frame_input"
        self.unreviewed_only_tag = "bmot_unreviewed_only"
        self.playback_loop_tag = "bmot_playback_loop"
        self.playback_button_tag = "bmot_playback_button"
        self.spotlight_tag = "bmot_spotlight"
        self.tracking_enabled_tag = "bmot_tracking_enabled"
        self.seq_combo_tag = "bmot_seq_combo"
        self.seq_meta_tag = "bmot_seq_meta"
        self.track_table_tag = "bmot_track_table"
        self.persist_tag = "bmot_persist_text"

        self.texture_w = 1
        self.texture_h = 1
        self.texture_alloc_w = 1
        self.texture_alloc_h = 1
        self.display_w = 1280
        self.display_h = 720
        self.scale_x = 1.0
        self.scale_y = 1.0
        self.ui_ready = False
        self.window_tag = "bmot_main_window"
        self.viewport_w = 1680
        self.viewport_h = 1020
        self.window_w = 1660
        self.window_h = 980
        self.window_padding_w = 40
        self.window_controls_h = 220
        self.side_panel_w = 540
        self.main_gap_w = 14
        self.left_pane_w = 1100
        self.frame_status_drawlist_tag = "bmot_frame_status_drawlist"
        self.tracker_name = "ByteTrack"
        self.tracker_available, self.tracker_unavailable_reason = self._check_tracker_runtime()
        self.last_tracker_seed_frame: Optional[int] = None
        self.last_tracker_seed_track_id: Optional[int] = None
        self.initial_full_pass_done_tracks: set[int] = set()
        self.tracking_enabled = True
        self.autotrack_active = False
        self.autotrack_tracker = None
        self.autotrack_tracker_id: Optional[int] = None

        self._load_sequence(self.current_seq.name)

    def _initial_sequence_from_app_state(self) -> SequenceInfo:
        default_seq = self.sequences[0]
        if not self.app_state_file.exists():
            return default_seq
        try:
            payload = json.loads(self.app_state_file.read_text(encoding="utf-8"))
        except Exception:
            return default_seq
        seq_name = payload.get("last_sequence")
        if not isinstance(seq_name, str) or seq_name not in self.sequence_by_name:
            return default_seq
        return self.sequence_by_name[seq_name]

    def _save_app_state(self) -> None:
        payload = {
            "version": 1,
            "split": self.split,
            "last_sequence": self.current_seq.name,
        }
        _atomic_write_json(self.app_state_file, payload)

    @staticmethod
    def _rgb_to_texture_data(rgb: np.ndarray) -> np.ndarray:
        """Convert uint8 RGB image to DearPyGui RGBA float texture payload."""
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("Expected HxWx3 RGB image.")
        h, w, _ = rgb.shape
        alpha = np.full((h, w, 1), 255, dtype=np.uint8)
        rgba = np.concatenate([rgb, alpha], axis=2)
        return (rgba.astype(np.float32) / 255.0).flatten()

    def _check_tracker_runtime(self) -> Tuple[bool, str]:
        if sv is None or tk is None:
            return (
                False,
                "Install tracker deps in this env: pip install trackers supervision",
            )
        return True, ""

    def _create_bytetrack_tracker(self):
        if tk is None:
            raise RuntimeError("Missing 'trackers' package.")
        for attr in ("ByteTrackTracker", "BYTESORTTracker", "BYTETrackTracker"):
            if hasattr(tk, attr):
                return getattr(tk, attr)()
        raise RuntimeError("No ByteTrack class found in installed trackers package.")

    @staticmethod
    def _to_tracker_detections(rows: List[Detection]):
        if sv is None:
            raise RuntimeError("Missing 'supervision' package.")
        if not rows:
            return sv.Detections.empty()
        xyxy = np.asarray([[r.x, r.y, r.x + r.w, r.y + r.h] for r in rows], dtype=np.float32)
        confidence = np.asarray([r.score for r in rows], dtype=np.float32)
        class_id = np.asarray([r.class_id for r in rows], dtype=np.int32)
        return sv.Detections(xyxy=xyxy, confidence=confidence, class_id=class_id)

    @staticmethod
    def _update_tracker(tracker, detections):
        if hasattr(tracker, "update_with_detections"):
            return tracker.update_with_detections(detections)
        if hasattr(tracker, "update"):
            return tracker.update(detections)
        raise RuntimeError("Tracker has no compatible update method.")

    @staticmethod
    def _extract_tracker_rows(tracked) -> List[Tuple[int, float, float, float, float, float]]:
        if tracked is None:
            return []
        tracker_ids = getattr(tracked, "tracker_id", None)
        if tracker_ids is None and hasattr(tracked, "data"):
            data = tracked.data if tracked.data is not None else {}
            tracker_ids = data.get("tracker_id")
        if tracker_ids is None:
            return []
        ids = np.asarray(tracker_ids)
        xyxy = getattr(tracked, "xyxy", None)
        if xyxy is None:
            return []
        conf = getattr(tracked, "confidence", None)
        rows: List[Tuple[int, float, float, float, float, float]] = []
        for i, raw_id in enumerate(ids):
            if raw_id is None:
                continue
            try:
                tid = int(raw_id)
            except Exception:
                continue
            if tid < 0 or i >= len(xyxy):
                continue
            b = xyxy[i]
            x = float(b[0])
            y = float(b[1])
            w = float(b[2] - b[0])
            h = float(b[3] - b[1])
            score = 1.0
            if conf is not None and len(conf) > i:
                try:
                    score = float(conf[i])
                except Exception:
                    score = 1.0
            rows.append((tid, x, y, w, h, score))
        return rows

    def _run_bytetrack_frames(
        self,
        start_frame: int,
    ) -> Dict[int, List[Tuple[int, float, float, float, float, float]]]:
        tracker = self._create_bytetrack_tracker()
        by_frame: Dict[int, List[Tuple[int, float, float, float, float, float]]] = {}
        for frame_idx in range(start_frame, self.current_seq.seq_len + 1):
            dets = self._to_tracker_detections(self.detections_by_frame.get(frame_idx, []))
            tracked = self._update_tracker(tracker, dets)
            by_frame[frame_idx] = self._extract_tracker_rows(tracked)
        return by_frame

    def _update_tracker_controls(self) -> None:
        if not self.ui_ready:
            return
        if dpg.does_item_exist(self.tracking_enabled_tag):
            dpg.configure_item(self.tracking_enabled_tag, enabled=self.tracker_available)
            if not self.tracker_available:
                self.tracking_enabled = False
                dpg.set_value(self.tracking_enabled_tag, False)

    def _log_status(self, msg: str) -> None:
        if self.ui_ready and dpg.does_item_exist(self.status_tag):
            dpg.set_value(self.status_tag, msg)

    def _set_active_track(self, track_id: Optional[int]) -> None:
        self.active_track_id = track_id
        if self.ui_ready and dpg.does_item_exist(self.track_tag):
            dpg.set_value(self.track_tag, f"Active track: {track_id if track_id is not None else 'None'}")
        self._update_playback_controls()
        self._update_tracker_controls()
        self._update_persist_text()

    def _load_sequence(self, seq_name: str) -> None:
        self._stop_playback()
        self.current_seq = self.sequence_by_name[seq_name]
        self._save_app_state()
        if self.ui_ready and dpg.does_item_exist(self.seq_meta_tag):
            dpg.set_value(self.seq_meta_tag, f"Sequence: {self.current_seq.name}")
        self.current_frame = 1
        self.assignments_by_track = {}
        self.absent_frames_by_track = {}
        self.assigned_det_keys = set()
        self.last_assigned_box_by_track = {}
        self.undo_stack = []
        self.focus_mode = False
        self.skip_mode = False
        self.initial_full_pass_done_tracks = set()
        self.autotrack_active = False
        self.autotrack_tracker = None
        self.autotrack_tracker_id = None
        self.cycle_last_point_img = None
        self.cycle_last_signature = None
        self.cycle_step = 0
        self._set_active_track(None)
        self.next_track_id = self.track_id_start
        self.detections_by_frame = parse_det_file(self.current_seq.det_file, min_score=self.min_score)
        self._load_existing_labels()
        if self.ui_ready:
            self._sync_frame_controls()
            self._prepare_texture_for_sequence()
            self._render_frame()
            self._refresh_track_table()
        self._log_status(
            f"Loaded {seq_name}: frames={self.current_seq.seq_len}, "
            f"det_rows={sum(len(v) for v in self.detections_by_frame.values())}"
        )
        self._update_persist_text()

    def _load_existing_labels(self) -> None:
        self._load_from_gt_file()
        loaded_absent = self._load_absent_file()
        loaded_session = self._load_session_file()
        # Legacy migration path: previous monolithic state file.
        migrated_from_legacy = False
        if not loaded_absent and not loaded_session and self.current_seq.legacy_state_file.exists():
            migrated_from_legacy = self._load_from_legacy_state_file()
        self._apply_resume_fallback()
        if migrated_from_legacy:
            # Write split files immediately once migrated from legacy state.
            self._persist_progress(silent=True)

    def _known_track_ids(self) -> set[int]:
        return set(self.assignments_by_track.keys()) | set(self.absent_frames_by_track.keys())

    def _apply_resume_fallback(self) -> None:
        if self.active_track_id is not None:
            return
        if self.assignments_by_track:
            latest_tid = max(
                self.assignments_by_track.keys(),
                key=lambda tid: max(self.assignments_by_track[tid].keys()),
            )
            self.active_track_id = latest_tid
            self.focus_mode = False
            self.skip_mode = False
            return
        if self.absent_frames_by_track:
            latest_tid = max(
                self.absent_frames_by_track.keys(),
                key=lambda tid: max(self.absent_frames_by_track[tid]) if self.absent_frames_by_track[tid] else -1,
            )
            self.active_track_id = latest_tid
            self.focus_mode = False
            self.skip_mode = True
            return
        self.focus_mode = False
        self.skip_mode = False

    def _load_absent_file(self) -> bool:
        if not self.current_seq.absent_file.exists():
            return False
        try:
            payload = json.loads(self.current_seq.absent_file.read_text(encoding="utf-8"))
        except Exception:
            return False
        absent_payload = payload.get("absent_frames", payload if isinstance(payload, dict) else {})
        if not isinstance(absent_payload, dict):
            return False
        loaded_any = False
        for k, frames in absent_payload.items():
            try:
                tid = int(k)
            except Exception:
                continue
            if tid < self.track_id_start or not isinstance(frames, list):
                continue
            norm: set[int] = set()
            for f in frames:
                try:
                    fi = int(f)
                except Exception:
                    continue
                if 1 <= fi <= self.current_seq.seq_len:
                    norm.add(fi)
            if norm:
                self.absent_frames_by_track[tid] = norm
                self.next_track_id = max(self.next_track_id, tid + 1)
                loaded_any = True
        return loaded_any

    def _load_session_file(self) -> bool:
        if not self.current_seq.session_file.exists():
            return False
        try:
            payload = json.loads(self.current_seq.session_file.read_text(encoding="utf-8"))
        except Exception:
            return False
        self.next_track_id = max(
            self.next_track_id,
            int(payload.get("next_track_id", self.track_id_start)),
        )
        saved_frame = int(payload.get("current_frame", 1))
        self.current_frame = _clamp(saved_frame, 1, max(1, self.current_seq.seq_len))
        saved_active_track = payload.get("active_track_id")
        known_ids = self._known_track_ids()
        if isinstance(saved_active_track, int) and saved_active_track >= self.track_id_start:
            if saved_active_track in known_ids or saved_active_track < self.next_track_id:
                self.active_track_id = saved_active_track
        self.focus_mode = bool(payload.get("focus_mode", False))
        self.skip_mode = bool(payload.get("skip_mode", False))
        return True

    def _load_from_legacy_state_file(self) -> bool:
        try:
            payload = json.loads(self.current_seq.legacy_state_file.read_text(encoding="utf-8"))
        except Exception:
            return False
        # Restore positives from legacy only if gt is empty/missing.
        if not self.assignments_by_track:
            rows = payload.get("assignments", [])
            if isinstance(rows, list):
                for item in rows:
                    if not isinstance(item, dict):
                        continue
                    frame = int(item.get("frame", 0))
                    track_id = int(item.get("track_id", 0))
                    x = float(item.get("x", 0.0))
                    y = float(item.get("y", 0.0))
                    w = float(item.get("w", 0.0))
                    h = float(item.get("h", 0.0))
                    if frame < 1 or track_id < self.track_id_start or w <= 0 or h <= 0:
                        continue
                    det_key_raw = item.get("det_key")
                    det_key: Optional[Tuple[int, int]] = None
                    if isinstance(det_key_raw, list) and len(det_key_raw) == 2:
                        det_key = (int(det_key_raw[0]), int(det_key_raw[1]))
                    asn = Assignment(track_id=track_id, frame=frame, x=x, y=y, w=w, h=h, det_key=det_key)
                    self.assignments_by_track.setdefault(track_id, {})[frame] = asn
                    if det_key is not None:
                        self.assigned_det_keys.add(det_key)
                    self.last_assigned_box_by_track[track_id] = (x, y, w, h)
                    self.next_track_id = max(self.next_track_id, track_id + 1)
        # Load legacy absent/session fields.
        absent_payload = payload.get("absent_frames", {})
        if isinstance(absent_payload, dict):
            for k, frames in absent_payload.items():
                try:
                    tid = int(k)
                except Exception:
                    continue
                if tid < self.track_id_start or not isinstance(frames, list):
                    continue
                norm: set[int] = set()
                for f in frames:
                    try:
                        fi = int(f)
                    except Exception:
                        continue
                    if 1 <= fi <= self.current_seq.seq_len:
                        norm.add(fi)
                if norm:
                    self.absent_frames_by_track[tid] = norm
                    self.next_track_id = max(self.next_track_id, tid + 1)
        self.next_track_id = max(self.next_track_id, int(payload.get("next_track_id", self.track_id_start)))
        self.current_frame = _clamp(int(payload.get("current_frame", 1)), 1, max(1, self.current_seq.seq_len))
        saved_active_track = payload.get("active_track_id")
        known_ids = self._known_track_ids()
        if isinstance(saved_active_track, int) and saved_active_track >= self.track_id_start:
            if saved_active_track in known_ids or saved_active_track < self.next_track_id:
                self.active_track_id = saved_active_track
        self.focus_mode = bool(payload.get("focus_mode", False))
        self.skip_mode = bool(payload.get("skip_mode", False))
        return True

    def _load_from_gt_file(self) -> None:
        if not self.current_seq.gt_file.exists():
            return
        txt = self.current_seq.gt_file.read_text(encoding="utf-8").strip()
        if not txt:
            return
        for raw in txt.splitlines():
            fields = raw.split(",")
            if len(fields) < 8:
                continue
            frame = int(float(fields[0]))
            track_id = int(float(fields[1]))
            x = float(fields[2])
            y = float(fields[3])
            w = float(fields[4])
            h = float(fields[5])
            class_id = int(float(fields[7]))
            if class_id != 1:
                continue
            det_key = self._match_detection_key(frame=frame, box=(x, y, w, h))
            asn = Assignment(track_id=track_id, frame=frame, x=x, y=y, w=w, h=h, det_key=det_key)
            self.assignments_by_track.setdefault(track_id, {})[frame] = asn
            if det_key is not None:
                self.assigned_det_keys.add(det_key)
            self.last_assigned_box_by_track[track_id] = (x, y, w, h)
            self.next_track_id = max(self.next_track_id, track_id + 1)

    def _match_detection_key(self, frame: int, box: Tuple[float, float, float, float]) -> Optional[Tuple[int, int]]:
        best_key: Optional[Tuple[int, int]] = None
        best_iou = 0.0
        for det in self.detections_by_frame.get(frame, []):
            key = (frame, det.det_idx)
            if key in self.assigned_det_keys:
                continue
            iou = _bbox_iou(box, (det.x, det.y, det.w, det.h))
            if iou > best_iou:
                best_iou = iou
                best_key = key
        return best_key if best_iou >= 0.5 else None

    def _clear_active_track_from_frame(self, start_frame: int) -> None:
        if self.active_track_id is None:
            return
        tid = self.active_track_id
        rows = self.assignments_by_track.get(tid, {})
        for frame in [f for f in rows.keys() if f >= start_frame]:
            asn = rows.pop(frame, None)
            if asn is not None and asn.det_key is not None and asn.det_key in self.assigned_det_keys:
                self.assigned_det_keys.remove(asn.det_key)
        absent = self.absent_frames_by_track.get(tid, set())
        self.absent_frames_by_track[tid] = {f for f in absent if f < start_frame}

    def _set_assignment_for_active_track(
        self,
        frame: int,
        box: Tuple[float, float, float, float],
    ) -> None:
        if self.active_track_id is None:
            return
        tid = self.active_track_id
        x, y, w, h = box
        if w <= 0 or h <= 0:
            return
        rows = self.assignments_by_track.setdefault(tid, {})
        prev = rows.get(frame)
        if prev is not None and prev.det_key is not None and prev.det_key in self.assigned_det_keys:
            self.assigned_det_keys.remove(prev.det_key)
        det_key = self._match_detection_key(frame=frame, box=(x, y, w, h))
        asn = Assignment(track_id=tid, frame=frame, x=x, y=y, w=w, h=h, det_key=det_key)
        rows[frame] = asn
        if det_key is not None:
            self.assigned_det_keys.add(det_key)
        self.last_assigned_box_by_track[tid] = (x, y, w, h)
        self.absent_frames_by_track.setdefault(tid, set()).discard(frame)

    def _start_tracker_session_from_seed(self, start_frame: int, clear_future: bool = False) -> bool:
        if not self.tracker_available:
            self._log_status(self.tracker_unavailable_reason)
            return False
        if self.active_track_id is None:
            self._log_status("Choose or create an active track first.")
            return False
        seed_asn = self._active_assignment(start_frame)
        if seed_asn is None:
            self._log_status("Select/correct a box on this frame first, then track.")
            return False
        try:
            tracker = self._create_bytetrack_tracker()
        except Exception as exc:
            self._log_status(f"{self.tracker_name} failed: {exc}")
            return False

        seed_box = (seed_asn.x, seed_asn.y, seed_asn.w, seed_asn.h)
        # Treat selected frame as first frame of a fresh tracker segment.
        if clear_future:
            self._clear_active_track_from_frame(start_frame=start_frame)
        self._set_assignment_for_active_track(frame=start_frame, box=seed_box)

        # Prime ByteTrack on full-frame detections, then map the clicked seed box
        # to the closest returned tracker row. This is more stable than seeding
        # with only one synthetic detection.
        seed_input_rows = self.detections_by_frame.get(start_frame, [])
        if not seed_input_rows:
            # Fallback: if detector has no rows on this frame, seed with selected box.
            seed_input_rows = [
                Detection(
                    frame=start_frame,
                    det_idx=0,
                    x=seed_box[0],
                    y=seed_box[1],
                    w=seed_box[2],
                    h=seed_box[3],
                    score=1.0,
                    class_id=1,
                )
            ]
        try:
            tracked_seed = self._update_tracker(tracker, self._to_tracker_detections(seed_input_rows))
            seed_rows = self._extract_tracker_rows(tracked_seed)
        except Exception as exc:
            self._log_status(f"{self.tracker_name} failed on seed frame: {exc}")
            return False
        if not seed_rows:
            self._log_status("Tracker did not return a valid ID for the selected seed box.")
            return False
        # Choose the returned track row that best matches clicked seed box.
        best_seed = max(seed_rows, key=lambda r: _bbox_iou(seed_box, (r[1], r[2], r[3], r[4])))
        best_iou = _bbox_iou(seed_box, (best_seed[1], best_seed[2], best_seed[3], best_seed[4]))
        if best_iou <= 0.0 and self.detections_by_frame.get(start_frame):
            self._log_status("Tracker could not align selected box to a seed ID on this frame.")
            return False
        seed_tid = int(best_seed[0])
        if seed_tid < 0:
            self._log_status("Tracker seed ID was invalid (<0).")
            return False

        self.last_tracker_seed_frame = start_frame
        self.last_tracker_seed_track_id = seed_tid
        self.autotrack_tracker = tracker
        self.autotrack_tracker_id = seed_tid
        self.autotrack_active = True
        return True

    def _autotrack_assign_current_frame(self) -> bool:
        if not self.autotrack_active or self.autotrack_tracker is None:
            return False
        if self.active_track_id is None:
            return False
        frame_idx = self.current_frame
        frame_rows = self.detections_by_frame.get(frame_idx, [])
        try:
            tracked = self._update_tracker(self.autotrack_tracker, self._to_tracker_detections(frame_rows))
            out_rows = self._extract_tracker_rows(tracked)
        except Exception as exc:
            self._log_status(f"{self.tracker_name} failed at frame {frame_idx}: {exc}")
            self.autotrack_active = False
            self.autotrack_tracker = None
            self.autotrack_tracker_id = None
            return False
        match = next((r for r in out_rows if int(r[0]) == int(self.autotrack_tracker_id)), None)
        if match is None:
            self.absent_frames_by_track.setdefault(self.active_track_id, set()).add(frame_idx)
            return True
        _rid, x, y, w, h, _score = match
        self._set_assignment_for_active_track(frame=frame_idx, box=(x, y, w, h))
        return True

    def _autotrack_step_forward_one_frame(self) -> bool:
        if self.current_frame >= self.current_seq.seq_len:
            return False
        self.current_frame += 1
        if self.autotrack_active:
            self._autotrack_assign_current_frame()
        self._sync_frame_controls()
        self._render_frame()
        self._persist_progress(silent=True)
        return True

    def _run_tracking_from_seed(self, start_frame: int) -> None:
        ok = self._start_tracker_session_from_seed(start_frame=start_frame, clear_future=False)
        if not ok:
            return
        self.focus_mode = True
        self.skip_mode = False
        self._persist_progress(silent=True)
        self._log_status(
            f"{self.tracker_name} tracking initialized at frame {start_frame} for track {self.active_track_id} "
            f"(tracker id={self.autotrack_tracker_id})."
        )
        if self.active_track_id is not None:
            self._start_track_playback(self.active_track_id, start_frame=start_frame)

    def _run_initial_full_pass(self) -> None:
        # Alias retained for existing callbacks/hotkeys.
        self._run_tracking_from_seed(start_frame=self.current_frame)

    def _sequence_frame_path(self, frame: int) -> Path:
        ext = self.current_seq.im_ext or ".jpg"
        return self.current_seq.img_dir / f"{frame:06d}{ext}"

    def _prepare_texture_for_sequence(self) -> None:
        if not self.ui_ready:
            return
        frame_path = self._sequence_frame_path(1)
        image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if image is None:
            self.texture_w = max(1, self.current_seq.width)
            self.texture_h = max(1, self.current_seq.height)
            rgb = np.zeros((self.texture_h, self.texture_w, 3), dtype=np.uint8)
        else:
            self.texture_h, self.texture_w = image.shape[:2]
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        max_canvas_w = max(480, self.left_pane_w - self.window_padding_w)
        max_canvas_h = max(320, self.window_h - self.window_controls_h)
        scale = min(max_canvas_w / max(1, self.texture_w), max_canvas_h / max(1, self.texture_h))
        self.display_w = max(1, int(self.texture_w * scale))
        self.display_h = max(1, int(self.texture_h * scale))
        self.scale_x = self.display_w / max(1, self.texture_w)
        self.scale_y = self.display_h / max(1, self.texture_h)
        initial = self._rgb_to_texture_data(rgb)
        # Reallocate only when dimensions actually change. Recreating texture
        # objects repeatedly during sequence switches can destabilize DearPyGui.
        needs_realloc = (
            (not dpg.does_item_exist(self.texture_tag))
            or self.texture_alloc_w != self.texture_w
            or self.texture_alloc_h != self.texture_h
        )
        if needs_realloc:
            if dpg.does_item_exist(self.texture_tag):
                dpg.delete_item(self.texture_tag)
            elif dpg.does_alias_exist(self.texture_tag):
                # Defensive cleanup for stale alias-only state.
                dpg.remove_alias(self.texture_tag)
            dpg.add_dynamic_texture(
                width=self.texture_w,
                height=self.texture_h,
                default_value=initial,
                tag=self.texture_tag,
                parent="bmot_texture_registry",
            )
            self.texture_alloc_w = self.texture_w
            self.texture_alloc_h = self.texture_h
        else:
            dpg.set_value(self.texture_tag, initial)
        dpg.configure_item(self.canvas_tag, width=self.display_w, height=self.display_h)

    def _sync_frame_controls(self) -> None:
        if not self.ui_ready:
            return
        unreviewed_only = dpg.does_item_exist(self.unreviewed_only_tag) and bool(dpg.get_value(self.unreviewed_only_tag))
        frames = self._unreviewed_frames_for_active_track() if unreviewed_only else []
        if frames:
            dpg.configure_item(self.frame_slider_tag, min_value=1, max_value=len(frames))
            if self.current_frame in frames:
                slider_value = frames.index(self.current_frame) + 1
            else:
                # Keep slider in a reasonable nearest position when current frame is reviewed.
                nearest_idx = min(range(len(frames)), key=lambda i: abs(frames[i] - self.current_frame))
                slider_value = nearest_idx + 1
            dpg.set_value(self.frame_slider_tag, slider_value)
            dpg.set_value(self.frame_input_tag, slider_value)
        else:
            dpg.configure_item(self.frame_slider_tag, min_value=1, max_value=max(1, self.current_seq.seq_len))
            dpg.set_value(self.frame_slider_tag, self.current_frame)
            dpg.set_value(self.frame_input_tag, self.current_frame)

    def _is_frame_reviewed_for_track(self, track_id: int, frame: int) -> bool:
        if frame in self.assignments_by_track.get(track_id, {}):
            return True
        if frame in self.absent_frames_by_track.get(track_id, set()):
            return True
        return False

    def _unreviewed_frames_for_active_track(self) -> List[int]:
        if self.active_track_id is None:
            return []
        tid = self.active_track_id
        return [f for f in range(1, self.current_seq.seq_len + 1) if not self._is_frame_reviewed_for_track(tid, f)]

    def _frame_assignments(self, frame: int) -> Dict[int, Assignment]:
        out: Dict[int, Assignment] = {}
        for tid, rows in self.assignments_by_track.items():
            if frame in rows:
                out[tid] = rows[frame]
        return out

    def _active_assignment(self, frame: int) -> Optional[Assignment]:
        if self.active_track_id is None:
            return None
        return self.assignments_by_track.get(self.active_track_id, {}).get(frame)

    def _active_det_key_for_frame(self, frame: int) -> Optional[Tuple[int, int]]:
        active_asn = self._active_assignment(frame)
        if active_asn is not None and active_asn.det_key is not None:
            return active_asn.det_key
        if self.active_track_id is None or self.skip_mode:
            return None
        if frame in self.absent_frames_by_track.get(self.active_track_id, set()):
            return None
        if self.active_track_id not in self.last_assigned_box_by_track:
            return None
        suggested = self._best_iou_detection(frame=frame, ref_box=self.last_assigned_box_by_track[self.active_track_id])
        if suggested is None:
            return None
        return (suggested.frame, suggested.det_idx)

    def _predicted_detection_for_frame(self, frame: int) -> Optional[Detection]:
        if self.active_track_id is None:
            return None
        if self.skip_mode:
            return None
        if frame in self.absent_frames_by_track.get(self.active_track_id, set()):
            return None
        if self._active_assignment(frame) is not None:
            return None
        ref_box = self.last_assigned_box_by_track.get(self.active_track_id)
        if ref_box is None:
            return None
        return self._best_iou_detection(frame=frame, ref_box=ref_box)

    def _candidate_detections(self, frame: int) -> List[Detection]:
        active_asn = self._active_assignment(frame)
        active_key = active_asn.det_key if active_asn is not None else None
        dets: List[Detection] = []
        for det in self.detections_by_frame.get(frame, []):
            key = (frame, det.det_idx)
            if key in self.assigned_det_keys and key != active_key:
                continue
            dets.append(det)
        return dets

    def _best_iou_detection(self, frame: int, ref_box: Tuple[float, float, float, float]) -> Optional[Detection]:
        best_det: Optional[Detection] = None
        best_iou = -1.0
        for det in self._candidate_detections(frame):
            iou = _bbox_iou(ref_box, (det.x, det.y, det.w, det.h))
            if iou > best_iou:
                best_iou = iou
                best_det = det
        return best_det

    def _hit_detections_at_point(self, img_x: float, img_y: float) -> List[Detection]:
        # Reverse draw order so first hit is the visually top-most box.
        ordered = list(reversed(self._candidate_detections(self.current_frame)))
        hits: List[Detection] = []
        for det in ordered:
            if det.x <= img_x <= det.x + det.w and det.y <= img_y <= det.y + det.h:
                hits.append(det)
        return hits

    def _reset_click_cycle(self) -> None:
        self.cycle_last_point_img = None
        self.cycle_last_signature = None
        self.cycle_step = 0

    def _is_same_click_region(
        self, img_x: float, img_y: float, signature: Tuple[Tuple[int, int], ...], tol_px: float = 3.0
    ) -> bool:
        if self.cycle_last_signature is None or self.cycle_last_point_img is None:
            return False
        if self.cycle_last_signature != signature:
            return False
        last_x, last_y = self.cycle_last_point_img
        return abs(last_x - img_x) <= tol_px and abs(last_y - img_y) <= tol_px

    def _render_frame(self) -> None:
        if not self.ui_ready:
            return
        frame_path = self._sequence_frame_path(self.current_frame)
        image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if image is None:
            image = np.zeros((self.texture_h, self.texture_w, 3), dtype=np.uint8)
        else:
            h, w = image.shape[:2]
            if h != self.texture_h or w != self.texture_w:
                # Rare mismatch fallback: resize to texture dimensions for texture stability.
                image = cv2.resize(image, (self.texture_w, self.texture_h), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        texture = self._rgb_to_texture_data(rgb)
        dpg.set_value(self.texture_tag, texture)

        dpg.delete_item(self.canvas_tag, children_only=True)
        dpg.draw_image(self.texture_tag, pmin=(0, 0), pmax=(self.display_w, self.display_h), parent=self.canvas_tag)

        base_thickness = 4
        active_thickness = 6
        active_asn = self._active_assignment(self.current_frame)
        predicted_det = self._predicted_detection_for_frame(self.current_frame)
        predicted_overlap_colors = [
            (255, 205, 135, 255),  # light orange
            (215, 170, 255, 255),  # light purple
            (155, 220, 255, 255),  # light blue
            (255, 170, 220, 255),  # light pink
        ]
        active_key = (
            active_asn.det_key
            if (active_asn is not None and active_asn.det_key is not None)
            else ((predicted_det.frame, predicted_det.det_idx) if predicted_det is not None else None)
        )
        active_is_locked = active_asn is not None and active_asn.det_key is not None
        self.active_box_img = None
        active_rect_px: Optional[Tuple[float, float, float, float]] = None
        overlap_color_by_key: Dict[Tuple[int, int], Tuple[int, int, int, int]] = {}
        if predicted_det is not None:
            pred_box = (predicted_det.x, predicted_det.y, predicted_det.w, predicted_det.h)
            overlaps: List[Tuple[int, int]] = []
            for det in self._candidate_detections(self.current_frame):
                key = (det.frame, det.det_idx)
                if key == active_key:
                    continue
                if _bbox_intersects(pred_box, (det.x, det.y, det.w, det.h)):
                    overlaps.append(key)
            overlaps.sort()
            for i, key in enumerate(overlaps):
                overlap_color_by_key[key] = predicted_overlap_colors[i % len(predicted_overlap_colors)]

        spotlight_on = self.ui_ready and dpg.does_item_exist(self.spotlight_tag) and bool(dpg.get_value(self.spotlight_tag))
        if spotlight_on and active_asn is not None:
            sx1 = active_asn.x * self.scale_x
            sy1 = active_asn.y * self.scale_y
            sx2 = (active_asn.x + active_asn.w) * self.scale_x
            sy2 = (active_asn.y + active_asn.h) * self.scale_y
            sx1 = max(0.0, min(float(self.display_w), sx1))
            sy1 = max(0.0, min(float(self.display_h), sy1))
            sx2 = max(0.0, min(float(self.display_w), sx2))
            sy2 = max(0.0, min(float(self.display_h), sy2))
            if sx2 > sx1 and sy2 > sy1:
                active_rect_px = (sx1, sy1, sx2, sy2)
                mask_fill = (90, 90, 90, 120)
                # Dim everything outside active labeled box.
                dpg.draw_rectangle((0, 0), (self.display_w, sy1), color=(0, 0, 0, 0), fill=mask_fill, parent=self.canvas_tag)
                dpg.draw_rectangle((0, sy2), (self.display_w, self.display_h), color=(0, 0, 0, 0), fill=mask_fill, parent=self.canvas_tag)
                dpg.draw_rectangle((0, sy1), (sx1, sy2), color=(0, 0, 0, 0), fill=mask_fill, parent=self.canvas_tag)
                dpg.draw_rectangle((sx2, sy1), (self.display_w, sy2), color=(0, 0, 0, 0), fill=mask_fill, parent=self.canvas_tag)

        candidates = self._candidate_detections(self.current_frame)
        for det in candidates:
            key = (det.frame, det.det_idx)
            if self.playback_active and (self.active_track_id is None or key != active_key):
                continue
            x1 = det.x * self.scale_x
            y1 = det.y * self.scale_y
            x2 = (det.x + det.w) * self.scale_x
            y2 = (det.y + det.h) * self.scale_y
            is_active = self.active_track_id is not None and key == active_key
            if is_active:
                # Dark green = locked/confirmed, light green = predicted.
                color = (0, 150, 0, 255) if active_is_locked else (130, 255, 130, 255)
            elif key in overlap_color_by_key:
                color = overlap_color_by_key[key]
            else:
                color = (210, 210, 210, 230)
            thickness = active_thickness if is_active else base_thickness
            dpg.draw_rectangle((x1, y1), (x2, y2), color=color, thickness=thickness, parent=self.canvas_tag)
            if is_active:
                self.active_box_img = (det.x, det.y, det.w, det.h)
            if not is_active:
                dpg.draw_text((x1, y1), f"{det.score:.2f}", color=(230, 230, 230, 255), size=12, parent=self.canvas_tag)

        if active_asn is not None and self.active_track_id is not None:
            label_x = active_asn.x * self.scale_x + 3
            label_y = max(0, active_asn.y * self.scale_y - 18)
            dpg.draw_text((label_x, label_y), f"T{self.active_track_id}", color=(0, 150, 0, 255), size=16, parent=self.canvas_tag)
        elif predicted_det is not None and self.active_track_id is not None:
            label_x = predicted_det.x * self.scale_x + 3
            label_y = max(0, predicted_det.y * self.scale_y - 18)
            dpg.draw_text(
                (label_x, label_y),
                f"T{self.active_track_id} (pred)",
                color=(130, 255, 130, 255),
                size=16,
                parent=self.canvas_tag,
            )

        assigned_count = len(self.assigned_det_keys)
        all_count = sum(len(v) for v in self.detections_by_frame.values())
        active_boxes = 0
        active_reviewed = 0
        if self.active_track_id is not None:
            active_boxes = len(self.assignments_by_track.get(self.active_track_id, {}))
            active_absent = len(self.absent_frames_by_track.get(self.active_track_id, set()))
            active_reviewed = active_boxes + active_absent
        self._log_status(
            f"{self.current_seq.name} | frame {self.current_frame}/{self.current_seq.seq_len} | "
            f"active={self.active_track_id if self.active_track_id is not None else '-'} | "
            f"track_boxes={active_boxes}/{self.current_seq.seq_len} reviewed={active_reviewed}/{self.current_seq.seq_len} | "
            f"global_assigned={assigned_count}/{all_count}"
        )
        self._refresh_track_table()
        self._render_frame_status_badge()
        self._update_tracker_controls()

    def _step_frame(self, delta: int) -> None:
        unreviewed_only = self.ui_ready and dpg.does_item_exist(self.unreviewed_only_tag) and bool(
            dpg.get_value(self.unreviewed_only_tag)
        )
        if unreviewed_only:
            frames = self._unreviewed_frames_for_active_track()
            if frames:
                if delta >= 0:
                    nxt = next((f for f in frames if f > self.current_frame), None)
                    self.current_frame = nxt if nxt is not None else frames[-1]
                else:
                    prev = next((f for f in reversed(frames) if f < self.current_frame), None)
                    self.current_frame = prev if prev is not None else frames[0]
            else:
                self.current_frame = _clamp(self.current_frame + delta, 1, max(1, self.current_seq.seq_len))
        else:
            self.current_frame = _clamp(self.current_frame + delta, 1, max(1, self.current_seq.seq_len))
        self._sync_frame_controls()
        self._render_frame()
        self._persist_session_state()

    def _set_frame(self, frame: int) -> None:
        self.current_frame = _clamp(frame, 1, max(1, self.current_seq.seq_len))
        self._sync_frame_controls()
        self._render_frame()
        self._persist_session_state()

    def _first_available_track_id(self) -> int:
        used = set(self.assignments_by_track.keys()) | set(self.absent_frames_by_track.keys())
        if self.active_track_id is not None:
            used.add(self.active_track_id)
        tid = self.track_id_start
        while tid in used:
            tid += 1
        return tid

    def _next_monotonic_track_id(self) -> int:
        known = self._known_track_ids()
        if self.active_track_id is not None:
            known.add(self.active_track_id)
        min_from_known = (max(known) + 1) if known else self.track_id_start
        return max(self.track_id_start, int(self.next_track_id), min_from_known)

    def _current_track_id_for_assignment(self) -> int:
        if self.active_track_id is None:
            tid = self._next_monotonic_track_id()
            self.next_track_id = max(self.next_track_id, tid + 1)
            self._set_active_track(tid)
            self._persist_session_state()
            return tid
        return self.active_track_id

    def _assign_detection(self, det: Detection) -> None:
        key = (det.frame, det.det_idx)
        if key in self.assigned_det_keys and key != (self._active_assignment(det.frame).det_key if self._active_assignment(det.frame) else None):
            return
        tid = self._current_track_id_for_assignment()
        if tid in self.absent_frames_by_track and det.frame in self.absent_frames_by_track[tid]:
            self.absent_frames_by_track[tid].discard(det.frame)
        prev = self.assignments_by_track.get(tid, {}).get(det.frame)
        if prev is not None and prev.det_key is not None and prev.det_key in self.assigned_det_keys:
            self.assigned_det_keys.remove(prev.det_key)
        asn = Assignment(track_id=tid, frame=det.frame, x=det.x, y=det.y, w=det.w, h=det.h, det_key=key)
        self.assignments_by_track.setdefault(tid, {})[det.frame] = asn
        self.assigned_det_keys.add(key)
        self.last_assigned_box_by_track[tid] = (det.x, det.y, det.w, det.h)
        self.undo_stack.append(("assign", (tid, det.frame, prev, asn)))
        self.focus_mode = True
        self.skip_mode = False
        self._render_frame()
        self._persist_progress(silent=True)

    def _deselect_current_active(self) -> None:
        if self.active_track_id is None:
            return
        asn = self._active_assignment(self.current_frame)
        if asn is not None:
            rows = self.assignments_by_track.get(self.active_track_id, {})
            if self.current_frame in rows:
                del rows[self.current_frame]
            if asn.det_key is not None and asn.det_key in self.assigned_det_keys:
                self.assigned_det_keys.remove(asn.det_key)
        self.focus_mode = False
        self.skip_mode = True
        self._reset_click_cycle()
        self._render_frame()
        self._persist_progress(silent=True)

    def _delete_active_frame_assignment(self) -> None:
        if self.active_track_id is None:
            self._log_status("No active track selected.")
            return
        self.absent_frames_by_track.setdefault(self.active_track_id, set()).add(self.current_frame)
        rows = self.assignments_by_track.get(self.active_track_id, {})
        asn = rows.get(self.current_frame)
        removed = False
        if asn is not None:
            del rows[self.current_frame]
            if asn.det_key is not None and asn.det_key in self.assigned_det_keys:
                self.assigned_det_keys.remove(asn.det_key)
            removed = True
        self.focus_mode = False
        self.skip_mode = True
        self._reset_click_cycle()
        self._render_frame()
        self._persist_progress(silent=True)
        if removed:
            self._log_status(f"Removed assignment for track {self.active_track_id} on frame {self.current_frame}.")
        else:
            self._log_status(
                f"No explicit assignment on frame {self.current_frame}; switched to skip mode for track {self.active_track_id}."
            )

    def _mark_absent_current_frame(self) -> None:
        if self.active_track_id is None:
            return
        # Do not overwrite explicit positive assignment with absent label.
        if self._active_assignment(self.current_frame) is not None:
            return
        self.absent_frames_by_track.setdefault(self.active_track_id, set()).add(self.current_frame)

    def _confirm_current_frame(self, emit_missing_msg: bool = True) -> bool:
        """Lock in current frame selection for active track. Returns True if saved."""
        if self.active_track_id is None:
            self._log_status("No active track selected.")
            return False
        if self._active_assignment(self.current_frame) is not None:
            # Already confirmed on this frame.
            self.absent_frames_by_track.setdefault(self.active_track_id, set()).discard(self.current_frame)
            self.focus_mode = True
            self.skip_mode = False
            self._render_frame()
            self._persist_session_state()
            return True
        predicted = self._predicted_detection_for_frame(self.current_frame)
        if predicted is None:
            if emit_missing_msg:
                self._log_status(f"No predicted box to save on frame {self.current_frame}.")
            return False
        self._assign_detection(predicted)
        return True

    def _save_or_mark_absent_current_frame(self) -> None:
        if self.active_track_id is None:
            self._log_status("No active track selected.")
            return
        saved = self._confirm_current_frame(emit_missing_msg=False)
        if saved:
            return
        self._mark_absent_current_frame()
        self.focus_mode = False
        self.skip_mode = True
        self._render_frame()
        self._persist_progress(silent=True)
        self._log_status(f"Marked frame {self.current_frame} absent for track {self.active_track_id}.")

    def _pick_detection_at_mouse(self, mouse_x: float, mouse_y: float) -> Optional[Detection]:
        dx = mouse_x / self.scale_x
        dy = mouse_y / self.scale_y
        for det in self._candidate_detections(self.current_frame):
            if det.x <= dx <= det.x + det.w and det.y <= dy <= det.y + det.h:
                if (det.frame, det.det_idx) not in self.assigned_det_keys:
                    return det
        return None

    def _undo(self) -> None:
        if not self.undo_stack:
            self._log_status("Undo stack is empty.")
            return
        kind, payload = self.undo_stack.pop()
        if kind == "assign":
            tid, frame, prev, asn = payload
            rows = self.assignments_by_track.setdefault(tid, {})
            if frame in rows:
                del rows[frame]
            if prev is not None:
                rows[frame] = prev
            if asn.det_key is not None and asn.det_key in self.assigned_det_keys:
                self.assigned_det_keys.remove(asn.det_key)
            if prev is not None and prev.det_key is not None:
                self.assigned_det_keys.add(prev.det_key)
            self.current_frame = frame
            self._sync_frame_controls()
            self._render_frame()
            self._persist_progress(silent=True)

    def _save_gt(self, silent: bool = False) -> None:
        rows: List[Assignment] = []
        for tid, by_frame in self.assignments_by_track.items():
            for frame, asn in by_frame.items():
                if frame < 1 or frame > self.current_seq.seq_len:
                    continue
                if asn.w <= 0 or asn.h <= 0:
                    continue
                rows.append(asn)
        rows.sort(key=lambda r: (r.frame, r.track_id))
        out_lines = [
            f"{r.frame},{r.track_id},{r.x:.3f},{r.y:.3f},{r.w:.3f},{r.h:.3f},1,1,1"
            for r in rows
        ]
        payload = "\n".join(out_lines) + ("\n" if out_lines else "")
        _atomic_write_text(self.current_seq.gt_file, payload)
        if not silent:
            self._log_status(f"Saved {len(rows)} GT rows: {self.current_seq.gt_file}")

    def _save_absent_frames(self, silent: bool = False) -> None:
        payload = {
            "version": 1,
            "track_id_start": self.track_id_start,
            "absent_frames": {str(tid): sorted(list(frames)) for tid, frames in self.absent_frames_by_track.items()},
        }
        _atomic_write_json(self.current_seq.absent_file, payload)
        if not silent:
            self._log_status(f"Saved absent-frame state: {self.current_seq.absent_file}")

    def _save_session_state(self, silent: bool = False) -> None:
        payload = {
            "version": 1,
            "track_id_start": self.track_id_start,
            "next_track_id": self.next_track_id,
            "current_frame": self.current_frame,
            "active_track_id": self.active_track_id,
            "focus_mode": self.focus_mode,
            "skip_mode": self.skip_mode,
        }
        _atomic_write_json(self.current_seq.session_file, payload)
        if not silent:
            self._log_status(f"Saved session state: {self.current_seq.session_file}")

    def _persist_progress(self, silent: bool = True) -> None:
        self._save_gt(silent=silent)
        self._save_absent_frames(silent=silent)
        self._save_session_state(silent=silent)
        self._update_persist_text()

    def _persist_session_state(self) -> None:
        self._save_session_state(silent=True)
        self._update_persist_text()

    def _refresh_track_table(self) -> None:
        if not self.ui_ready or not dpg.does_item_exist(self.track_table_tag):
            return
        dpg.delete_item(self.track_table_tag, children_only=True, slot=1)
        track_ids = sorted(set(self.assignments_by_track.keys()) | set(self.absent_frames_by_track.keys()))
        for tid in track_ids:
            count = len(self.assignments_by_track.get(tid, {}))
            absent = len(self.absent_frames_by_track.get(tid, set()))
            reviewed = count + absent
            pct = (100.0 * reviewed / max(1, self.current_seq.seq_len))
            active = "*" if self.active_track_id == tid else ""
            with dpg.table_row(parent=self.track_table_tag):
                if self.active_track_id == tid:
                    dpg.add_text(str(tid), color=(40, 220, 60, 255))
                else:
                    dpg.add_text(str(tid))
                dpg.add_text(str(count))
                dpg.add_text(f"{reviewed}/{self.current_seq.seq_len}")
                dpg.add_text(f"{pct:.1f}%")
                dpg.add_text(active)
                with dpg.group(horizontal=True):
                    dpg.add_button(label="Use", callback=self.cb_use_track_row, user_data=tid, width=54)
                    dpg.add_button(label="Del", callback=self.cb_delete_track_row, user_data=tid, width=54)

    def _start_track_playback(self, track_id: int, start_frame: Optional[int] = None) -> None:
        if track_id not in self.assignments_by_track and track_id not in self.absent_frames_by_track:
            return
        self.playback_track_id = track_id
        self.playback_active = True
        self.playback_last_ts = time.perf_counter()
        self._set_active_track(track_id)
        self.focus_mode = False
        self.skip_mode = False
        if start_frame is not None:
            self.current_frame = _clamp(start_frame, 1, max(1, self.current_seq.seq_len))
        self._sync_frame_controls()
        self._render_frame()
        self._update_playback_controls()
        self._refresh_track_table()

    def _stop_playback(self) -> None:
        self.playback_active = False
        self.playback_track_id = None
        self._update_playback_controls()

    def _tick_playback(self) -> None:
        if not self.playback_active:
            return
        if self.playback_track_id is None:
            self._stop_playback()
            return
        interval = 1.0 / max(1e-6, self.playback_fps)
        now = time.perf_counter()
        elapsed = now - self.playback_last_ts
        if elapsed < interval:
            return
        steps = int(elapsed / interval)
        self.playback_last_ts += steps * interval
        advanced = False
        for _ in range(steps):
            if self.current_frame >= self.current_seq.seq_len:
                loop_enabled = self.ui_ready and dpg.does_item_exist(self.playback_loop_tag) and bool(
                    dpg.get_value(self.playback_loop_tag)
                )
                if loop_enabled:
                    self.current_frame = 1
                    advanced = True
                    continue
                self._stop_playback()
                break
            if self.tracking_enabled and self.active_track_id is not None and not self.autotrack_active:
                seeded = self._start_tracker_session_from_seed(start_frame=self.current_frame, clear_future=False)
                if not seeded:
                    # Keep forward-only autoplay alive even when seeding fails.
                    # This matches repeated Right-arrow semantics.
                    self._save_or_mark_absent_current_frame()
                    self.current_frame += 1
                    advanced = True
                    continue
            self.current_frame += 1
            if self.autotrack_active:
                self._autotrack_assign_current_frame()
            advanced = True
        if advanced:
            self._sync_frame_controls()
            self._render_frame()
            self._refresh_track_table()
            self._persist_progress(silent=True)

    def _toggle_playback_active_track(self) -> None:
        if self.playback_active:
            self._stop_playback()
            return
        if self.active_track_id is None:
            self._log_status("Select/Use a track first, then press Play track.")
            return
        if self.tracking_enabled:
            if not self._start_tracker_session_from_seed(start_frame=self.current_frame, clear_future=False):
                # Do not block autoplay start on seed failure; tick loop will keep
                # advancing and retry seeding frame-by-frame.
                self.autotrack_active = False
                self.autotrack_tracker = None
                self.autotrack_tracker_id = None
            self.focus_mode = True
            self.skip_mode = False
            self._persist_progress(silent=True)
        else:
            self.autotrack_active = False
            self.autotrack_tracker = None
            self.autotrack_tracker_id = None
        self._start_track_playback(self.active_track_id, start_frame=self.current_frame)

    def _update_playback_controls(self) -> None:
        if not self.ui_ready or not dpg.does_item_exist(self.playback_button_tag):
            return
        label = "Stop track" if self.playback_active else "Play track"
        dpg.configure_item(self.playback_button_tag, label=label)

    def _use_track(self, track_id: int) -> None:
        if track_id not in self.assignments_by_track and track_id not in self.absent_frames_by_track:
            return
        self._stop_playback()
        self._set_active_track(track_id)
        self.focus_mode = False
        self.skip_mode = False
        self._render_frame()
        self._persist_session_state()

    def _delete_track(self, track_id: int) -> None:
        if track_id not in self.assignments_by_track and track_id not in self.absent_frames_by_track:
            return
        if self.playback_track_id == track_id:
            self._stop_playback()
        rows = self.assignments_by_track.pop(track_id, {})
        self.absent_frames_by_track.pop(track_id, None)
        for asn in rows.values():
            if asn.det_key is not None and asn.det_key in self.assigned_det_keys:
                self.assigned_det_keys.remove(asn.det_key)
        if track_id in self.last_assigned_box_by_track:
            del self.last_assigned_box_by_track[track_id]
        if track_id in self.initial_full_pass_done_tracks:
            self.initial_full_pass_done_tracks.remove(track_id)
        if self.active_track_id == track_id:
            self._set_active_track(None)
            self.focus_mode = False
            self.skip_mode = False
        self.next_track_id = max(self.next_track_id, self._next_monotonic_track_id())
        self._reset_click_cycle()
        self._render_frame()
        self._persist_progress(silent=True)
        self._log_status(f"Deleted track {track_id}.")

    def cb_use_track_row(self, _sender: Any, _app_data: Any, user_data: Any) -> None:
        try:
            track_id = int(user_data)
        except Exception:
            return
        self._use_track(track_id)

    def cb_delete_track_row(self, _sender: Any, _app_data: Any, user_data: Any) -> None:
        try:
            track_id = int(user_data)
        except Exception:
            return
        self._delete_track(track_id)

    def cb_toggle_playback_button(self, _sender: Any, _app_data: Any) -> None:
        self._toggle_playback_active_track()

    def cb_toggle_tracking_enabled(self, _sender: Any, app_data: Any) -> None:
        requested = bool(app_data)
        if requested and not self.tracker_available:
            self.tracking_enabled = False
            if self.ui_ready and dpg.does_item_exist(self.tracking_enabled_tag):
                dpg.set_value(self.tracking_enabled_tag, False)
            self._log_status(f"Tracking unavailable: {self.tracker_unavailable_reason}")
            return
        self.tracking_enabled = requested
        if not self.tracking_enabled:
            self.autotrack_active = False
            self.autotrack_tracker = None
            self.autotrack_tracker_id = None
            self._log_status("Tracking OFF: preview/review mode.")
        else:
            self._log_status("Tracking ON: autoplay and right-arrow accumulate labels.")

    def _update_persist_text(self) -> None:
        if not self.ui_ready or not dpg.does_item_exist(self.persist_tag):
            return
        active = self.active_track_id if self.active_track_id is not None else "-"
        dpg.set_value(
            self.persist_tag,
            (
                f"Autosave: {self.current_seq.session_file.name}, "
                f"{self.current_seq.absent_file.name}, gt.txt | frame={self.current_frame} active={active}"
            ),
        )

    def _render_frame_status_badge(self) -> None:
        if not self.ui_ready or not dpg.does_item_exist(self.frame_status_drawlist_tag):
            return
        dpg.delete_item(self.frame_status_drawlist_tag, children_only=True)
        if self.active_track_id is None:
            return
        has_pos = self._active_assignment(self.current_frame) is not None
        has_neg = self.current_frame in self.absent_frames_by_track.get(self.active_track_id, set())
        if has_pos:
            dpg.draw_text((8, 8), "LABELED +", color=(40, 220, 60, 255), size=44, parent=self.frame_status_drawlist_tag)
        elif has_neg:
            dpg.draw_text((8, 8), "LABELED -", color=(220, 50, 50, 255), size=44, parent=self.frame_status_drawlist_tag)

    def _end_track(self) -> None:
        self._set_active_track(None)
        self.focus_mode = False
        self.skip_mode = False
        self._render_frame()
        self._persist_session_state()

    def _start_new_track(self) -> None:
        self._stop_playback()
        next_monotonic_id = self._next_monotonic_track_id()
        self._set_active_track(next_monotonic_id)
        self.next_track_id = next_monotonic_id + 1
        self.current_frame = 1
        self.focus_mode = False
        self.skip_mode = False
        self._reset_click_cycle()
        self._sync_frame_controls()
        self._render_frame()
        self._persist_session_state()
        self._log_status(f"Started new track {next_monotonic_id} at frame 1.")

    def _advance_with_iou(self) -> None:
        if self.active_track_id is None:
            if self.current_frame >= self.current_seq.seq_len:
                return
            self.current_frame += 1
            self._sync_frame_controls()
            self._render_frame()
            self._persist_session_state()
            return
        if self.skip_mode:
            self._mark_absent_current_frame()
            if self.current_frame >= self.current_seq.seq_len:
                self._render_frame()
                self._persist_session_state()
                return
            self.current_frame += 1
            self._sync_frame_controls()
            self.focus_mode = False
            self._render_frame()
            self._persist_session_state()
            return
        saved = self._confirm_current_frame(emit_missing_msg=False)
        if not saved:
            # Spacebar linear pass: no confirmed box on this frame => explicit absent.
            self._mark_absent_current_frame()
            if self.current_frame >= self.current_seq.seq_len:
                self._render_frame()
                self._persist_session_state()
                return
            self.current_frame += 1
            self._sync_frame_controls()
            self.focus_mode = False
            self.skip_mode = True
            self._render_frame()
            self._persist_session_state()
            return
        if self.current_frame >= self.current_seq.seq_len:
            return
        self.current_frame += 1
        self._sync_frame_controls()
        self.focus_mode = True
        self.skip_mode = False
        self._render_frame()
        self._persist_session_state()

    # DearPyGui callbacks
    def cb_sequence_changed(self, _sender: Any, app_data: Any) -> None:
        self._load_sequence(str(app_data))

    def cb_frame_slider(self, _sender: Any, app_data: Any) -> None:
        self._stop_playback()
        unreviewed_only = self.ui_ready and dpg.does_item_exist(self.unreviewed_only_tag) and bool(
            dpg.get_value(self.unreviewed_only_tag)
        )
        if unreviewed_only:
            frames = self._unreviewed_frames_for_active_track()
            if frames:
                idx = _clamp(int(app_data), 1, len(frames))
                self._set_frame(frames[idx - 1])
                return
        self._set_frame(int(app_data))

    def cb_frame_input(self, _sender: Any, app_data: Any) -> None:
        self._stop_playback()
        unreviewed_only = self.ui_ready and dpg.does_item_exist(self.unreviewed_only_tag) and bool(
            dpg.get_value(self.unreviewed_only_tag)
        )
        if unreviewed_only:
            frames = self._unreviewed_frames_for_active_track()
            if frames:
                idx = _clamp(int(app_data), 1, len(frames))
                self._set_frame(frames[idx - 1])
                return
        self._set_frame(int(app_data))

    def cb_toggle_unreviewed_only(self, _sender: Any, app_data: Any) -> None:
        if bool(app_data):
            frames = self._unreviewed_frames_for_active_track()
            if frames and self.current_frame not in frames:
                next_frame = next((f for f in frames if f >= self.current_frame), None)
                self.current_frame = next_frame if next_frame is not None else frames[-1]
        self._sync_frame_controls()
        self._render_frame()
        self._persist_session_state()

    def cb_canvas_click(self, _sender: Any, _app_data: Any) -> None:
        self._stop_playback()
        if not dpg.is_item_hovered(self.canvas_tag):
            return
        mx, my = dpg.get_mouse_pos(local=False)
        x0, y0 = dpg.get_item_rect_min(self.canvas_tag)
        local_x = mx - x0
        local_y = my - y0
        if local_x < 0 or local_y < 0 or local_x > self.display_w or local_y > self.display_h:
            return
        img_x = local_x / self.scale_x
        img_y = local_y / self.scale_y
        hits = self._hit_detections_at_point(img_x, img_y)
        if hits:
            if len(hits) == 1:
                self._assign_detection(hits[0])
                self._reset_click_cycle()
                return
            signature: Tuple[Tuple[int, int], ...] = tuple((det.frame, det.det_idx) for det in hits)
            if self._is_same_click_region(img_x, img_y, signature):
                self.cycle_step += 1
            else:
                self.cycle_step = 0
                self.cycle_last_signature = signature
                self.cycle_last_point_img = (img_x, img_y)

            pick_idx = self.cycle_step % (len(hits) + 1)
            if pick_idx == len(hits):
                if self.active_track_id is not None:
                    self._deselect_current_active()
                else:
                    self.focus_mode = False
                    self.skip_mode = True
                    self._render_frame()
                    self._persist_session_state()
                return

            self._assign_detection(hits[pick_idx])
            return
        # Clicking empty space reveals all available detections on this frame.
        self._reset_click_cycle()
        self.focus_mode = False
        self._render_frame()
        self._persist_session_state()

    def cb_hotkey(self, _sender: Any, app_data: Any, _user_data: Any) -> None:
        key = int(app_data)
        if key == dpg.mvKey_Spacebar:
            if self.space_toggle_latch:
                return
            self.space_toggle_latch = True
            self._toggle_playback_active_track()
            return

        # For non-playback hotkeys, freeze playback first to avoid edits while running.
        self._stop_playback()
        if key == dpg.mvKey_Q:
            self._end_track()
        elif key == dpg.mvKey_Left or key == dpg.mvKey_A:
            self._step_frame(-1)
        elif key == dpg.mvKey_Right:
            if self.active_track_id is None:
                self._step_frame(1)
            else:
                if not self.tracking_enabled:
                    self._step_frame(1)
                else:
                    if not self._start_tracker_session_from_seed(start_frame=self.current_frame, clear_future=False):
                        self._save_or_mark_absent_current_frame()
                        self._step_frame(1)
                        return
                    self._autotrack_step_forward_one_frame()
        elif key == dpg.mvKey_D or key == dpg.mvKey_Delete:
            self._delete_active_frame_assignment()
        elif key == dpg.mvKey_E:
            self._start_new_track()
        elif key == dpg.mvKey_S:
            self._save_or_mark_absent_current_frame()
        elif key == dpg.mvKey_R:
            self._undo()
        elif key == dpg.mvKey_T:
            if self.tracking_enabled:
                self._run_tracking_from_seed(start_frame=self.current_frame)
            else:
                self._log_status("Tracking is OFF. Enable tracking to run from here.")
        elif key == dpg.mvKey_Y:
            if self.tracking_enabled:
                self._run_tracking_from_seed(start_frame=self.current_frame)
            else:
                self._log_status("Tracking is OFF. Enable tracking to run from here.")

    def cb_hotkey_release(self, _sender: Any, app_data: Any, _user_data: Any) -> None:
        key = int(app_data)
        if key == dpg.mvKey_Spacebar:
            self.space_toggle_latch = False

    def build_ui(self) -> None:
        screen_w, screen_h = _get_screen_size()
        target_w = int(screen_w * 0.70)
        target_h = int(screen_h * 0.85)
        self.viewport_w = _clamp(target_w, 1200, max(1200, screen_w - 40))
        self.viewport_h = _clamp(target_h, 820, max(820, screen_h - 40))
        self.window_w = max(800, self.viewport_w - 20)
        self.window_h = max(700, self.viewport_h - 20)
        self.left_pane_w = max(520, self.window_w - self.side_panel_w - self.main_gap_w - 20)
        pos_x = max(0, int((screen_w - self.viewport_w) / 2))
        pos_y = max(0, int((screen_h - self.viewport_h) / 2))

        dpg.create_context()
        dpg.create_viewport(title="BasketballMOT Labeler", width=self.viewport_w, height=self.viewport_h)
        with dpg.texture_registry(tag="bmot_texture_registry"):
            dpg.add_dynamic_texture(
                width=1,
                height=1,
                default_value=[0.0, 0.0, 0.0, 1.0],
                tag=self.texture_tag,
            )

        with dpg.window(label="BasketballMOT Labeler", width=self.window_w, height=self.window_h, tag=self.window_tag):
            dpg.add_text(f"Sequence: {self.current_seq.name}", tag=self.seq_meta_tag)
            dpg.add_text("", tag=self.track_tag)
            dpg.add_text("", tag=self.status_tag)
            dpg.add_text("", tag=self.persist_tag)
            if self.tracker_available:
                dpg.add_text("Tracker: ByteTrack ready")
            else:
                dpg.add_text(f"Tracker unavailable: {self.tracker_unavailable_reason}")
            dpg.add_spacer(height=6)
            with dpg.group(horizontal=True):
                dpg.add_drawlist(width=self.left_pane_w, height=self.display_h, tag=self.canvas_tag)
                dpg.add_spacer(width=self.main_gap_w)
                with dpg.child_window(width=self.side_panel_w, height=self.display_h, border=True):
                    dpg.add_text("Tracks")
                    dpg.add_drawlist(
                        width=max(120, self.side_panel_w - 24),
                        height=70,
                        tag=self.frame_status_drawlist_tag,
                    )
                    dpg.add_spacer(height=6)
                    with dpg.table(
                        header_row=True,
                        borders_innerH=True,
                        borders_outerH=True,
                        borders_innerV=True,
                        borders_outerV=True,
                        row_background=True,
                        resizable=False,
                        policy=dpg.mvTable_SizingStretchProp,
                        tag=self.track_table_tag,
                    ):
                        dpg.add_table_column(label="Track ID")
                        dpg.add_table_column(label="Boxes")
                        dpg.add_table_column(label="Reviewed")
                        dpg.add_table_column(label="%")
                        dpg.add_table_column(label="Active")
                        dpg.add_table_column(label="Actions")

            dpg.add_spacer(height=8)
            with dpg.group(horizontal=True):
                dpg.add_slider_int(
                    tag=self.frame_slider_tag,
                    default_value=1,
                    min_value=1,
                    max_value=max(1, self.current_seq.seq_len),
                    callback=self.cb_frame_slider,
                    width=max(360, self.left_pane_w - 160),
                )
                dpg.add_input_int(
                    tag=self.frame_input_tag,
                    default_value=1,
                    callback=self.cb_frame_input,
                    width=100,
                )

            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="Play/Pause track",
                    tag=self.playback_button_tag,
                    callback=self.cb_toggle_playback_button,
                    width=120,
                )
                dpg.add_checkbox(
                    label="Unreviewed only",
                    tag=self.unreviewed_only_tag,
                    default_value=False,
                    callback=self.cb_toggle_unreviewed_only,
                )
                dpg.add_checkbox(
                    label="Loop playback",
                    tag=self.playback_loop_tag,
                    default_value=False,
                )
                dpg.add_checkbox(
                    label="Spotlight",
                    tag=self.spotlight_tag,
                    default_value=False,
                    callback=lambda _s, _a: self._render_frame(),
                )
                dpg.add_checkbox(
                    label="Tracking enabled",
                    tag=self.tracking_enabled_tag,
                    default_value=True,
                    callback=self.cb_toggle_tracking_enabled,
                )

            dpg.add_spacer(height=6)
            with dpg.group(horizontal=True):
                dpg.add_text("Sequence")
                dpg.add_combo(
                    [s.name for s in self.sequences],
                    default_value=self.current_seq.name,
                    tag=self.seq_combo_tag,
                    callback=self.cb_sequence_changed,
                    width=340,
                )
                dpg.add_button(label="New track (E)", callback=lambda: self._start_new_track())
                dpg.add_button(label="End track (Q)", callback=lambda: self._end_track())
                dpg.add_button(label="Undo (R)", callback=lambda: self._undo())
                dpg.add_button(label="Save (S)", callback=lambda: self._save_gt())
                dpg.add_button(label="Save All", callback=lambda: self._persist_progress(silent=False))

        with dpg.item_handler_registry(tag="bmot_canvas_handlers"):
            dpg.add_item_clicked_handler(callback=self.cb_canvas_click)
        dpg.bind_item_handler_registry(self.canvas_tag, "bmot_canvas_handlers")

        with dpg.handler_registry():
            dpg.add_key_press_handler(key=dpg.mvKey_D, callback=self.cb_hotkey)
            dpg.add_key_press_handler(key=dpg.mvKey_Delete, callback=self.cb_hotkey)
            dpg.add_key_press_handler(key=dpg.mvKey_A, callback=self.cb_hotkey)
            dpg.add_key_press_handler(key=dpg.mvKey_Left, callback=self.cb_hotkey)
            dpg.add_key_press_handler(key=dpg.mvKey_Right, callback=self.cb_hotkey)
            dpg.add_key_press_handler(key=dpg.mvKey_W, callback=self.cb_hotkey)
            dpg.add_key_press_handler(key=dpg.mvKey_Q, callback=self.cb_hotkey)
            dpg.add_key_press_handler(key=dpg.mvKey_E, callback=self.cb_hotkey)
            dpg.add_key_press_handler(key=dpg.mvKey_S, callback=self.cb_hotkey)
            dpg.add_key_press_handler(key=dpg.mvKey_R, callback=self.cb_hotkey)
            dpg.add_key_press_handler(key=dpg.mvKey_T, callback=self.cb_hotkey)
            dpg.add_key_press_handler(key=dpg.mvKey_Y, callback=self.cb_hotkey)
            dpg.add_key_press_handler(key=dpg.mvKey_Spacebar, callback=self.cb_hotkey)
            dpg.add_key_release_handler(key=dpg.mvKey_Spacebar, callback=self.cb_hotkey_release)

        self.ui_ready = True
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_viewport_pos([pos_x, pos_y])
        self._set_active_track(self.active_track_id)
        self._sync_frame_controls()
        self._prepare_texture_for_sequence()
        self._render_frame()
        self._refresh_track_table()
        self._update_persist_text()
        while dpg.is_dearpygui_running():
            self._tick_playback()
            dpg.render_dearpygui_frame()
        dpg.destroy_context()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BasketballMOT DearPyGui labeler (MVP).")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("benchmarks/datasets/BasketballMOT"),
        help="BasketballMOT dataset root",
    )
    parser.add_argument(
        "--split",
        default="train",
        choices=["train", "val", "test"],
        help="Dataset split",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.25,
        help="Minimum detector confidence to show candidate detections",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    app = BasketballMOTLabelerApp(
        dataset_root=args.dataset_root.expanduser().resolve(),
        split=args.split,
        min_score=max(0.0, float(args.min_score)),
    )
    app.build_ui()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
