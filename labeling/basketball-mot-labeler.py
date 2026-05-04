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
        self.kept_track_info: Dict[int, Dict[str, Any]] = {}
        self.track_lineage_info: Dict[int, Dict[str, Any]] = {}
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
        self.keep_modal_track_id: Optional[int] = None
        self.merge_modal_source_track_id: Optional[int] = None
        self.merge_target_label_to_track_id: Dict[str, int] = {}

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
        self.seq_combo_tag = "bmot_seq_combo"
        self.tracker_combo_tag = "bmot_tracker_combo"
        self.seq_meta_tag = "bmot_seq_meta"
        self.track_table_tag = "bmot_track_table"
        self.kept_track_table_tag = "bmot_kept_track_table"
        self.persist_tag = "bmot_persist_text"
        self.track_all_button_tag = "bmot_track_all_button"
        self.clear_all_button_tag = "bmot_clear_all_button"
        self.split_button_tag = "bmot_split_button"
        self.keep_modal_tag = "bmot_keep_modal"
        self.keep_team_tag = "bmot_keep_team"
        self.keep_jersey_tag = "bmot_keep_jersey"
        self.keep_name_tag = "bmot_keep_name"
        self.keep_title_tag = "bmot_keep_title"
        self.merge_modal_tag = "bmot_merge_modal"
        self.merge_title_tag = "bmot_merge_title"
        self.merge_target_combo_tag = "bmot_merge_target_combo"

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
        self.tracker_name_to_internal = {
            "ByteTrack": "bytetrack",
            "OCSort": "ocsort",
            "McByte": "mcbyte",
        }
        self.available_tracker_names = list(self.tracker_name_to_internal.keys())
        self.tracker_available, self.tracker_unavailable_reason = self._check_tracker_runtime()
        self.initial_full_pass_done_tracks: set[int] = set()

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
        tracker_key = self.tracker_name_to_internal.get(self.tracker_name, "bytetrack")
        if tracker_key in ("bytetrack", "ocsort"):
            if sv is None or tk is None:
                return (
                    False,
                    "Install tracker deps in this env: pip install trackers supervision",
                )
            if tracker_key == "bytetrack":
                for attr in ("ByteTrackTracker", "BYTESORTTracker", "BYTETrackTracker"):
                    if hasattr(tk, attr):
                        return True, ""
                return False, "No ByteTrack class found in installed trackers package."
            for attr in ("OCSORTTracker", "OCSortTracker", "OcSortTracker"):
                if hasattr(tk, attr):
                    return True, ""
            return False, "No OCSort class found in installed trackers package."
        if tracker_key == "mcbyte":
            if sv is None:
                return False, "Install supervision: pip install supervision"
            try:
                from yolox.tracker import rfdetr_adapter as _mcbyte_adapter  # type: ignore

                if not hasattr(_mcbyte_adapter, "McByteRfDetrAdapter"):
                    return False, "McByte adapter class missing in yolox.tracker.rfdetr_adapter."
            except Exception:
                return (
                    False,
                    "Install McByte runtime (yolox.tracker.rfdetr_adapter) in this env.",
                )
            return True, ""
        return False, f"Unsupported tracker: {self.tracker_name}"

    def _create_tracker(self):
        tracker_key = self.tracker_name_to_internal.get(self.tracker_name, "bytetrack")
        if tracker_key == "bytetrack":
            if tk is None:
                raise RuntimeError("Missing 'trackers' package.")
            for attr in ("ByteTrackTracker", "BYTESORTTracker", "BYTETrackTracker"):
                if hasattr(tk, attr):
                    return getattr(tk, attr)()
            raise RuntimeError("No ByteTrack class found in installed trackers package.")
        if tracker_key == "ocsort":
            if tk is None:
                raise RuntimeError("Missing 'trackers' package.")
            for attr in ("OCSORTTracker", "OCSortTracker", "OcSortTracker"):
                if hasattr(tk, attr):
                    return getattr(tk, attr)()
            raise RuntimeError("No OCSort class found in installed trackers package.")
        if tracker_key == "mcbyte":
            try:
                from yolox.tracker import rfdetr_adapter as mcbyte_adapter  # type: ignore
            except Exception as exc:
                raise RuntimeError(
                    "Missing McByte runtime (yolox.tracker.rfdetr_adapter)."
                ) from exc
            cfg = mcbyte_adapter.McByteRfDetrConfig()
            return mcbyte_adapter.McByteRfDetrAdapter(config=cfg, save_folder=".")
        raise RuntimeError(f"Unsupported tracker: {self.tracker_name}")

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

    @staticmethod
    def _rows_to_xyxy_conf(rows: List[Detection]) -> np.ndarray:
        if not rows:
            return np.zeros((0, 5), dtype=np.float32)
        out = np.zeros((len(rows), 5), dtype=np.float32)
        for i, r in enumerate(rows):
            out[i, 0] = float(r.x)
            out[i, 1] = float(r.y)
            out[i, 2] = float(r.x + r.w)
            out[i, 3] = float(r.y + r.h)
            out[i, 4] = float(r.score)
        return out

    @staticmethod
    def _extract_mcbyte_rows(tracked) -> List[Tuple[int, float, float, float, float, float]]:
        if tracked is None:
            return []
        rows: List[Tuple[int, float, float, float, float, float]] = []
        for tr in tracked:
            try:
                tid = int(tr.get("track_id", -1))
                if tid < 0:
                    continue
                tlwh = tr.get("tlwh")
                if tlwh is None or len(tlwh) < 4:
                    continue
                x, y, w, h = float(tlwh[0]), float(tlwh[1]), float(tlwh[2]), float(tlwh[3])
                score = float(tr.get("score", 1.0))
                rows.append((tid, x, y, w, h, score))
            except Exception:
                continue
        return rows

    def _read_frame_or_blank(self, frame_idx: int) -> np.ndarray:
        frame_path = self._sequence_frame_path(frame_idx)
        frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if frame is not None:
            return frame
        return np.zeros((max(1, self.current_seq.height), max(1, self.current_seq.width), 3), dtype=np.uint8)

    @staticmethod
    def _dedupe_rows_by_track_id(
        rows: List[Tuple[int, float, float, float, float, float]],
    ) -> List[Tuple[int, float, float, float, float, float]]:
        best: Dict[int, Tuple[int, float, float, float, float, float]] = {}
        for row in rows:
            tid, *_rest, score = row
            prev = best.get(tid)
            if prev is None or score > prev[5]:
                best[tid] = row
        return [best[tid] for tid in sorted(best.keys())]

    def _run_tracker_frames(
        self,
        start_frame: int,
    ) -> Dict[int, List[Tuple[int, float, float, float, float, float]]]:
        tracker = self._create_tracker()
        tracker_key = self.tracker_name_to_internal.get(self.tracker_name, "bytetrack")
        by_frame: Dict[int, List[Tuple[int, float, float, float, float, float]]] = {}
        for frame_idx in range(start_frame, self.current_seq.seq_len + 1):
            frame_rows = self.detections_by_frame.get(frame_idx, [])
            if tracker_key == "mcbyte":
                frame_img = self._read_frame_or_blank(frame_idx)
                dets_xyxy_conf = self._rows_to_xyxy_conf(frame_rows)
                tracked = tracker.step(frame_img, dets_xyxy_conf)
                rows = self._extract_mcbyte_rows(tracked)
            else:
                dets = self._to_tracker_detections(frame_rows)
                tracked = self._update_tracker(tracker, dets)
                rows = self._extract_tracker_rows(tracked)
            by_frame[frame_idx] = self._dedupe_rows_by_track_id(rows)
        return by_frame

    def _update_tracker_controls(self) -> None:
        if not self.ui_ready:
            return
        if dpg.does_item_exist(self.track_all_button_tag):
            dpg.configure_item(self.track_all_button_tag, enabled=self.tracker_available)

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
        self.kept_track_info = {}
        self.track_lineage_info = {}
        self.assigned_det_keys = set()
        self.last_assigned_box_by_track = {}
        self.undo_stack = []
        self.focus_mode = False
        self.skip_mode = False
        self.initial_full_pass_done_tracks = set()
        self.cycle_last_point_img = None
        self.cycle_last_signature = None
        self.cycle_step = 0
        self.keep_modal_track_id = None
        self.merge_modal_source_track_id = None
        self.merge_target_label_to_track_id = {}
        self._set_active_track(None)
        self.next_track_id = self.track_id_start
        self.detections_by_frame = parse_det_file(self.current_seq.det_file, min_score=self.min_score)
        self._load_existing_labels()
        self._sync_lineage_with_tracks()
        self._prune_kept_tracks()
        if self.ui_ready:
            if dpg.does_item_exist(self.keep_modal_tag):
                dpg.configure_item(self.keep_modal_tag, show=False)
            if dpg.does_item_exist(self.merge_modal_tag):
                dpg.configure_item(self.merge_modal_tag, show=False)
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

    def _track_start_frame(self, track_id: int) -> int:
        rows = self.assignments_by_track.get(track_id, {})
        return min(rows.keys()) if rows else 10**9

    def _track_lineage_defaults(self, track_id: int) -> Dict[str, Any]:
        return {
            "group_id": int(track_id),
            "segment_index": 1,
            "parent_track_id": None,
            "split_from_frame": None,
        }

    def _lineage_for_track(self, track_id: int) -> Dict[str, Any]:
        if track_id not in self.track_lineage_info:
            self.track_lineage_info[track_id] = self._track_lineage_defaults(track_id)
        info = self.track_lineage_info[track_id]
        group_id = int(info.get("group_id", track_id))
        segment_index = int(info.get("segment_index", 1))
        parent_track_id = info.get("parent_track_id")
        split_from_frame = info.get("split_from_frame")
        self.track_lineage_info[track_id] = {
            "group_id": group_id,
            "segment_index": max(1, segment_index),
            "parent_track_id": int(parent_track_id) if isinstance(parent_track_id, int) else None,
            "split_from_frame": int(split_from_frame) if isinstance(split_from_frame, int) else None,
        }
        return self.track_lineage_info[track_id]

    def _sync_lineage_with_tracks(self) -> None:
        valid_ids = set(self.assignments_by_track.keys())
        self.track_lineage_info = {
            tid: info for tid, info in self.track_lineage_info.items() if tid in valid_ids
        }
        for tid in sorted(valid_ids):
            self._lineage_for_track(tid)

    def _prune_kept_tracks(self) -> None:
        valid_ids = set(self.assignments_by_track.keys())
        self.kept_track_info = {tid: info for tid, info in self.kept_track_info.items() if tid in valid_ids}

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
        kept_payload = payload.get("kept_tracks", {})
        if isinstance(kept_payload, dict):
            for k, raw in kept_payload.items():
                try:
                    tid = int(k)
                except Exception:
                    continue
                if tid < self.track_id_start:
                    continue
                if not isinstance(raw, dict):
                    continue
                team = str(raw.get("team", "home")).strip().lower()
                if team not in {"home", "away"}:
                    team = "home"
                jersey_raw = raw.get("jersey_number", 0)
                try:
                    jersey_number = int(jersey_raw)
                except Exception:
                    jersey_number = 0
                name = str(raw.get("name", ""))
                self.kept_track_info[tid] = {
                    "team": team,
                    "jersey_number": jersey_number,
                    "name": name,
                }
        lineage_payload = payload.get("track_lineage", {})
        if isinstance(lineage_payload, dict):
            for k, raw in lineage_payload.items():
                try:
                    tid = int(k)
                except Exception:
                    continue
                if tid < self.track_id_start or not isinstance(raw, dict):
                    continue
                try:
                    group_id = int(raw.get("group_id", tid))
                except Exception:
                    group_id = tid
                try:
                    segment_index = int(raw.get("segment_index", 1))
                except Exception:
                    segment_index = 1
                parent_track_id = raw.get("parent_track_id")
                split_from_frame = raw.get("split_from_frame")
                self.track_lineage_info[tid] = {
                    "group_id": group_id,
                    "segment_index": max(1, segment_index),
                    "parent_track_id": int(parent_track_id) if isinstance(parent_track_id, int) else None,
                    "split_from_frame": int(split_from_frame) if isinstance(split_from_frame, int) else None,
                }
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

    def _match_detection_key_from_pool(
        self,
        frame: int,
        box: Tuple[float, float, float, float],
        used_det_keys: set[Tuple[int, int]],
    ) -> Optional[Tuple[int, int]]:
        best_key: Optional[Tuple[int, int]] = None
        best_iou = 0.0
        for det in self.detections_by_frame.get(frame, []):
            key = (frame, det.det_idx)
            if key in used_det_keys:
                continue
            iou = _bbox_iou(box, (det.x, det.y, det.w, det.h))
            if iou > best_iou:
                best_iou = iou
                best_key = key
        return best_key if best_iou >= 0.5 else None

    def _replace_all_tracks_from_tracker_output(
        self,
        by_frame: Dict[int, List[Tuple[int, float, float, float, float, float]]],
    ) -> int:
        self.assignments_by_track = {}
        self.absent_frames_by_track = {}
        self.kept_track_info = {}
        self.track_lineage_info = {}
        self.assigned_det_keys = set()
        self.last_assigned_box_by_track = {}
        self.undo_stack = []
        self.focus_mode = False
        self.skip_mode = False

        used_det_keys: set[Tuple[int, int]] = set()
        tracker_id_map: Dict[int, int] = {}
        next_tid = self.track_id_start
        generated_rows = 0
        for frame_idx in range(1, self.current_seq.seq_len + 1):
            frame_rows = by_frame.get(frame_idx, [])
            for raw_tid, x, y, w, h, _score in frame_rows:
                if w <= 0.0 or h <= 0.0:
                    continue
                if raw_tid not in tracker_id_map:
                    tracker_id_map[raw_tid] = next_tid
                    next_tid += 1
                track_id = tracker_id_map[raw_tid]
                det_key = self._match_detection_key_from_pool(frame_idx, (x, y, w, h), used_det_keys=used_det_keys)
                asn = Assignment(
                    track_id=track_id,
                    frame=frame_idx,
                    x=x,
                    y=y,
                    w=w,
                    h=h,
                    det_key=det_key,
                )
                self.assignments_by_track.setdefault(track_id, {})[frame_idx] = asn
                if det_key is not None:
                    used_det_keys.add(det_key)
                    self.assigned_det_keys.add(det_key)
                self.last_assigned_box_by_track[track_id] = (x, y, w, h)
                generated_rows += 1

        self.next_track_id = max(self.track_id_start, next_tid)
        self._sync_lineage_with_tracks()
        self._prune_kept_tracks()
        self._set_active_track(min(self.assignments_by_track.keys()) if self.assignments_by_track else None)
        self._reset_click_cycle()
        return generated_rows

    def _run_track_all(self) -> None:
        if not self.tracker_available:
            self._log_status(f"{self.tracker_name} unavailable: {self.tracker_unavailable_reason}")
            return
        self._stop_playback()
        self._log_status(f"Running {self.tracker_name} over all frames...")
        try:
            tracked_by_frame = self._run_tracker_frames(start_frame=1)
        except Exception as exc:
            self._log_status(f"{self.tracker_name} failed during Track all: {exc}")
            return
        generated_rows = self._replace_all_tracks_from_tracker_output(tracked_by_frame)
        self.current_frame = 1
        self._sync_frame_controls()
        self._render_frame()
        self._persist_progress(silent=True)
        self._log_status(
            f"Track all complete with {self.tracker_name}: "
            f"{len(self.assignments_by_track)} tracks, {generated_rows} boxes."
        )

    def _clear_all_tracks(self) -> None:
        self._stop_playback()
        self.assignments_by_track = {}
        self.absent_frames_by_track = {}
        self.kept_track_info = {}
        self.track_lineage_info = {}
        self.assigned_det_keys = set()
        self.last_assigned_box_by_track = {}
        self.undo_stack = []
        self.focus_mode = False
        self.skip_mode = False
        self.current_frame = 1
        self.next_track_id = self.track_id_start
        self.keep_modal_track_id = None
        self.merge_modal_source_track_id = None
        self.merge_target_label_to_track_id = {}
        self._set_active_track(None)
        self._reset_click_cycle()
        self._sync_frame_controls()
        self._render_frame()
        self._persist_progress(silent=True)
        self._log_status(f"Cleared all tracks for sequence {self.current_seq.name}.")

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

    def _hit_track_assignments_at_point(self, img_x: float, img_y: float) -> List[int]:
        # Reverse track-id order for stable "top-most" style picking.
        hits: List[int] = []
        frame_rows = self._frame_assignments(self.current_frame)
        for track_id, asn in sorted(frame_rows.items(), key=lambda x: x[0], reverse=True):
            if asn.x <= img_x <= asn.x + asn.w and asn.y <= img_y <= asn.y + asn.h:
                hits.append(track_id)
        return hits

    def _reset_click_cycle(self) -> None:
        self.cycle_last_point_img = None
        self.cycle_last_signature = None
        self.cycle_step = 0

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

        base_thickness = 3
        active_thickness = 5
        frame_asns = self._frame_assignments(self.current_frame)
        active_asn = frame_asns.get(self.active_track_id) if self.active_track_id is not None else None

        spotlight_on = self.ui_ready and dpg.does_item_exist(self.spotlight_tag) and bool(dpg.get_value(self.spotlight_tag))
        if spotlight_on and active_asn is not None:
            sx1 = max(0.0, min(float(self.display_w), active_asn.x * self.scale_x))
            sy1 = max(0.0, min(float(self.display_h), active_asn.y * self.scale_y))
            sx2 = max(0.0, min(float(self.display_w), (active_asn.x + active_asn.w) * self.scale_x))
            sy2 = max(0.0, min(float(self.display_h), (active_asn.y + active_asn.h) * self.scale_y))
            if sx2 > sx1 and sy2 > sy1:
                mask_fill = (90, 90, 90, 120)
                dpg.draw_rectangle((0, 0), (self.display_w, sy1), color=(0, 0, 0, 0), fill=mask_fill, parent=self.canvas_tag)
                dpg.draw_rectangle((0, sy2), (self.display_w, self.display_h), color=(0, 0, 0, 0), fill=mask_fill, parent=self.canvas_tag)
                dpg.draw_rectangle((0, sy1), (sx1, sy2), color=(0, 0, 0, 0), fill=mask_fill, parent=self.canvas_tag)
                dpg.draw_rectangle((sx2, sy1), (self.display_w, sy2), color=(0, 0, 0, 0), fill=mask_fill, parent=self.canvas_tag)

        for tid, asn in sorted(frame_asns.items(), key=lambda x: x[0]):
            x1 = asn.x * self.scale_x
            y1 = asn.y * self.scale_y
            x2 = (asn.x + asn.w) * self.scale_x
            y2 = (asn.y + asn.h) * self.scale_y
            is_active = self.active_track_id == tid
            color = (0, 150, 0, 255) if is_active else (210, 210, 210, 235)
            thickness = active_thickness if is_active else base_thickness
            dpg.draw_rectangle((x1, y1), (x2, y2), color=color, thickness=thickness, parent=self.canvas_tag)
            label_y = max(0, y1 - 18)
            dpg.draw_text((x1 + 3, label_y), f"T{tid}", color=color, size=16, parent=self.canvas_tag)

        total_boxes = sum(len(v) for v in self.assignments_by_track.values())
        active_boxes = 0
        if self.active_track_id is not None:
            active_boxes = len(self.assignments_by_track.get(self.active_track_id, {}))
        self._log_status(
            f"{self.current_seq.name} | frame {self.current_frame}/{self.current_seq.seq_len} | "
            f"active={self.active_track_id if self.active_track_id is not None else '-'} | "
            f"track_boxes={active_boxes}/{self.current_seq.seq_len} | "
            f"total_boxes={total_boxes}"
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

    def _next_monotonic_track_id(self) -> int:
        known = self._known_track_ids()
        if self.active_track_id is not None:
            known.add(self.active_track_id)
        min_from_known = (max(known) + 1) if known else self.track_id_start
        return max(self.track_id_start, int(self.next_track_id), min_from_known)

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
            "version": 2,
            "track_id_start": self.track_id_start,
            "next_track_id": self.next_track_id,
            "current_frame": self.current_frame,
            "active_track_id": self.active_track_id,
            "focus_mode": self.focus_mode,
            "skip_mode": self.skip_mode,
            "kept_tracks": {
                str(tid): {
                    "team": str(info.get("team", "home")),
                    "jersey_number": int(info.get("jersey_number", 0)),
                    "name": str(info.get("name", "")),
                }
                for tid, info in sorted(self.kept_track_info.items())
                if isinstance(tid, int)
            },
            "track_lineage": {
                str(tid): {
                    "group_id": int(info.get("group_id", tid)),
                    "segment_index": int(info.get("segment_index", 1)),
                    "parent_track_id": info.get("parent_track_id"),
                    "split_from_frame": info.get("split_from_frame"),
                }
                for tid, info in sorted(self.track_lineage_info.items())
                if isinstance(tid, int)
            },
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
        if not self.ui_ready:
            return
        if not dpg.does_item_exist(self.track_table_tag) or not dpg.does_item_exist(self.kept_track_table_tag):
            return
        self._sync_lineage_with_tracks()
        dpg.delete_item(self.track_table_tag, children_only=True, slot=1)
        dpg.delete_item(self.kept_track_table_tag, children_only=True, slot=1)
        track_ids = sorted(
            self.assignments_by_track.keys(),
            key=lambda tid: (
                int(self._lineage_for_track(tid).get("group_id", tid)),
                int(self._lineage_for_track(tid).get("segment_index", 1)),
                self._track_start_frame(tid),
                tid,
            ),
        )
        proposed_ids = [tid for tid in track_ids if tid not in self.kept_track_info]
        kept_ids = [tid for tid in track_ids if tid in self.kept_track_info]

        for tid in proposed_ids:
            count = len(self.assignments_by_track.get(tid, {}))
            pct = (100.0 * count / max(1, self.current_seq.seq_len))
            lineage = self._lineage_for_track(tid)
            group_id = int(lineage.get("group_id", tid))
            segment_index = int(lineage.get("segment_index", 1))
            with dpg.table_row(parent=self.track_table_tag):
                dpg.add_selectable(
                    label=str(tid),
                    default_value=self.active_track_id == tid,
                    callback=self.cb_select_track_row,
                    user_data=tid,
                    span_columns=False,
                )
                dpg.add_text(str(group_id))
                dpg.add_text(str(segment_index))
                dpg.add_text(str(count))
                dpg.add_text(f"{count}/{self.current_seq.seq_len}")
                dpg.add_text(f"{pct:.1f}%")
                with dpg.group(horizontal=True):
                    dpg.add_button(label="Play", callback=self.cb_play_track_row, user_data=tid, width=46)
                    dpg.add_button(label="Keep", callback=self.cb_keep_track_row, user_data=tid, width=46)
                    dpg.add_button(label="Merge", callback=self.cb_merge_track_row, user_data=tid, width=50)
                    dpg.add_button(label="Del", callback=self.cb_delete_track_row, user_data=tid, width=42)

        for tid in kept_ids:
            count = len(self.assignments_by_track.get(tid, {}))
            pct = (100.0 * count / max(1, self.current_seq.seq_len))
            info = self.kept_track_info.get(tid, {})
            name = str(info.get("name", ""))
            jersey = str(info.get("jersey_number", ""))
            team = str(info.get("team", "home"))
            with dpg.table_row(parent=self.kept_track_table_tag):
                dpg.add_selectable(
                    label=str(tid),
                    default_value=self.active_track_id == tid,
                    callback=self.cb_select_track_row,
                    user_data=tid,
                    span_columns=False,
                )
                dpg.add_text(name)
                dpg.add_text(jersey)
                dpg.add_text(team)
                dpg.add_text(f"{count}/{self.current_seq.seq_len}")
                dpg.add_text(f"{pct:.1f}%")
                with dpg.group(horizontal=True):
                    dpg.add_button(label="Play", callback=self.cb_play_track_row, user_data=tid, width=44)
                    dpg.add_button(label="Edit", callback=self.cb_keep_track_row, user_data=tid, width=44)
                    dpg.add_button(label="Del", callback=self.cb_delete_track_row, user_data=tid, width=40)

    def _stop_playback(self) -> None:
        self.playback_active = False
        self.playback_track_id = None
        self._update_playback_controls()

    def _tick_playback(self) -> None:
        if not self.playback_active:
            return
        interval = 1.0 / max(1e-6, self.playback_fps)
        now = time.perf_counter()
        elapsed = now - self.playback_last_ts
        if elapsed < interval:
            return
        steps = int(elapsed / interval)
        self.playback_last_ts += steps * interval
        advanced = False
        track_end_frame: Optional[int] = None
        if self.playback_track_id is not None:
            track_rows = self.assignments_by_track.get(self.playback_track_id, {})
            if track_rows:
                track_end_frame = max(track_rows.keys())
                if self.current_frame >= track_end_frame:
                    self._stop_playback()
                    self._log_status(
                        f"Stopped at end of track {self.playback_track_id} (frame {track_end_frame})."
                    )
                    return
        for _ in range(steps):
            if track_end_frame is not None and self.current_frame >= track_end_frame:
                self._stop_playback()
                break
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
            self.current_frame += 1
            if track_end_frame is not None and self.current_frame >= track_end_frame:
                self._stop_playback()
                advanced = True
                break
            advanced = True
        if advanced:
            self._sync_frame_controls()
            self._render_frame()
            self._refresh_track_table()
            self._persist_session_state()

    def _toggle_playback_active_track(self) -> None:
        if self.playback_active:
            self._stop_playback()
            return
        if self.active_track_id is None:
            self._log_status("Select a track first; playback now runs track-to-end.")
            return
        if self.active_track_id not in self.assignments_by_track:
            self._log_status(f"Track {self.active_track_id} has no boxes to play.")
            return
        self.playback_track_id = self.active_track_id
        self.playback_active = True
        self.playback_last_ts = time.perf_counter()
        self._update_playback_controls()

    def _update_playback_controls(self) -> None:
        if not self.ui_ready or not dpg.does_item_exist(self.playback_button_tag):
            return
        label = "Stop" if self.playback_active else "Play"
        dpg.configure_item(self.playback_button_tag, label=label)

    def _use_track(self, track_id: int) -> None:
        if track_id not in self.assignments_by_track:
            return
        self._stop_playback()
        self._set_active_track(track_id)
        self.focus_mode = False
        self.skip_mode = False
        self._render_frame()
        self._persist_session_state()

    def _play_track_once_from_start(self, track_id: int) -> None:
        if track_id not in self.assignments_by_track:
            self._log_status(f"Track {track_id} not found.")
            return
        rows = self.assignments_by_track.get(track_id, {})
        if not rows:
            self._log_status(f"Track {track_id} has no boxes to play.")
            return
        self._stop_playback()
        self._set_active_track(track_id)
        self.current_frame = min(rows.keys())
        if self.ui_ready and dpg.does_item_exist(self.playback_loop_tag):
            dpg.set_value(self.playback_loop_tag, False)
        self.playback_track_id = track_id
        self.playback_active = True
        self.playback_last_ts = time.perf_counter()
        self._sync_frame_controls()
        self._render_frame()
        self._update_playback_controls()
        self._persist_session_state()

    def _open_keep_modal(self, track_id: int) -> None:
        if track_id not in self.assignments_by_track:
            self._log_status(f"Track {track_id} not found.")
            return
        self.keep_modal_track_id = int(track_id)
        info = self.kept_track_info.get(track_id, {})
        team = str(info.get("team", "home")).strip().lower()
        if team not in {"home", "away"}:
            team = "home"
        jersey_raw = info.get("jersey_number", 0)
        try:
            jersey_number = int(jersey_raw)
        except Exception:
            jersey_number = 0
        name = str(info.get("name", ""))
        if self.ui_ready:
            if dpg.does_item_exist(self.keep_title_tag):
                dpg.set_value(self.keep_title_tag, f"Keep Track {track_id}")
            if dpg.does_item_exist(self.keep_team_tag):
                dpg.set_value(self.keep_team_tag, team)
            if dpg.does_item_exist(self.keep_jersey_tag):
                dpg.set_value(self.keep_jersey_tag, jersey_number)
            if dpg.does_item_exist(self.keep_name_tag):
                dpg.set_value(self.keep_name_tag, name)
            if dpg.does_item_exist(self.keep_modal_tag):
                modal_w = 360
                modal_h = 220
                center_x = max(0, int((self.viewport_w - modal_w) / 2))
                center_y = max(0, int((self.viewport_h - modal_h) / 2))
                dpg.configure_item(self.keep_modal_tag, pos=(center_x, center_y))
                dpg.configure_item(self.keep_modal_tag, show=True)

    def _save_keep_modal(self) -> None:
        if self.keep_modal_track_id is None:
            return
        track_id = int(self.keep_modal_track_id)
        if track_id not in self.assignments_by_track:
            if dpg.does_item_exist(self.keep_modal_tag):
                dpg.configure_item(self.keep_modal_tag, show=False)
            self.keep_modal_track_id = None
            return
        team = "home"
        if dpg.does_item_exist(self.keep_team_tag):
            team = str(dpg.get_value(self.keep_team_tag)).strip().lower()
        if team not in {"home", "away"}:
            team = "home"
        try:
            jersey_number = int(dpg.get_value(self.keep_jersey_tag)) if dpg.does_item_exist(self.keep_jersey_tag) else 0
        except Exception:
            self._log_status("Jersey number must be an integer.")
            return
        name = str(dpg.get_value(self.keep_name_tag)) if dpg.does_item_exist(self.keep_name_tag) else ""
        self.kept_track_info[track_id] = {
            "team": team,
            "jersey_number": jersey_number,
            "name": name,
        }
        self._refresh_track_table()
        self._persist_session_state()
        if dpg.does_item_exist(self.keep_modal_tag):
            dpg.configure_item(self.keep_modal_tag, show=False)
        self.keep_modal_track_id = None
        self._log_status(
            f"Kept track {track_id}: team={team}, jersey={jersey_number}, name='{name}'."
        )

    def _open_merge_modal(self, source_track_id: int) -> None:
        if source_track_id not in self.assignments_by_track:
            self._log_status(f"Track {source_track_id} not found.")
            return
        if source_track_id in self.kept_track_info:
            self._log_status("Merge source must be a proposed track, not a kept track.")
            return
        candidates = sorted(
            tid
            for tid in self.kept_track_info.keys()
            if tid in self.assignments_by_track and tid != source_track_id
        )
        if not candidates:
            self._log_status("No kept tracks available as merge targets.")
            return
        self.merge_modal_source_track_id = int(source_track_id)
        items: List[str] = []
        self.merge_target_label_to_track_id = {}
        for tid in candidates:
            info = self.kept_track_info.get(tid, {})
            name = str(info.get("name", "")).strip()
            jersey = str(info.get("jersey_number", "")).strip()
            team = str(info.get("team", "")).strip()
            label = f"T{tid} | #{jersey or '?'} | {name or 'Unknown'} | {team or '-'}"
            self.merge_target_label_to_track_id[label] = tid
            items.append(label)
        if dpg.does_item_exist(self.merge_title_tag):
            dpg.set_value(self.merge_title_tag, f"Merge Proposed Track {source_track_id} Into Kept Track...")
        if dpg.does_item_exist(self.merge_target_combo_tag):
            dpg.configure_item(self.merge_target_combo_tag, items=items)
            dpg.set_value(self.merge_target_combo_tag, items[0])
        if dpg.does_item_exist(self.merge_modal_tag):
            modal_w = 360
            modal_h = 170
            center_x = max(0, int((self.viewport_w - modal_w) / 2))
            center_y = max(0, int((self.viewport_h - modal_h) / 2))
            dpg.configure_item(self.merge_modal_tag, pos=(center_x, center_y))
            dpg.configure_item(self.merge_modal_tag, show=True)

    def _merge_tracks(self, source_track_id: int, target_track_id: int) -> None:
        if source_track_id == target_track_id:
            self._log_status("Choose two different tracks to merge.")
            return
        if source_track_id not in self.assignments_by_track or target_track_id not in self.assignments_by_track:
            self._log_status("Source or target track not found.")
            return
        if source_track_id in self.kept_track_info:
            self._log_status("Source must be a proposed track.")
            return
        if target_track_id not in self.kept_track_info:
            self._log_status("Target must be a kept track.")
            return

        self._stop_playback()
        source_rows = self.assignments_by_track.get(source_track_id, {})
        target_rows = self.assignments_by_track.setdefault(target_track_id, {})
        moved_count = 0
        overlap_skipped = 0
        for frame in sorted(source_rows.keys()):
            asn = source_rows[frame]
            if frame in target_rows:
                overlap_skipped += 1
                continue
            target_rows[frame] = Assignment(
                track_id=target_track_id,
                frame=asn.frame,
                x=asn.x,
                y=asn.y,
                w=asn.w,
                h=asn.h,
                det_key=asn.det_key,
            )
            if asn.det_key is not None:
                self.assigned_det_keys.add(asn.det_key)
            moved_count += 1

        target_absent = set(self.absent_frames_by_track.get(target_track_id, set()))
        target_absent |= set(self.absent_frames_by_track.get(source_track_id, set()))
        for frame in target_rows.keys():
            target_absent.discard(frame)
        if target_absent:
            self.absent_frames_by_track[target_track_id] = target_absent
        else:
            self.absent_frames_by_track.pop(target_track_id, None)
        self.absent_frames_by_track.pop(source_track_id, None)

        self.assignments_by_track.pop(source_track_id, None)
        self.kept_track_info.pop(source_track_id, None)
        self.track_lineage_info.pop(source_track_id, None)

        if target_rows:
            last = target_rows[max(target_rows.keys())]
            self.last_assigned_box_by_track[target_track_id] = (last.x, last.y, last.w, last.h)
        self.last_assigned_box_by_track.pop(source_track_id, None)

        self._set_active_track(target_track_id)
        self.next_track_id = max(self.next_track_id, self._next_monotonic_track_id())
        self._sync_lineage_with_tracks()
        self._prune_kept_tracks()
        self._sync_frame_controls()
        self._render_frame()
        self._persist_progress(silent=True)
        self._log_status(
            f"Merged track {source_track_id} into {target_track_id}: moved={moved_count}, "
            f"overlap_skipped={overlap_skipped}."
        )

    def _save_merge_modal(self) -> None:
        if self.merge_modal_source_track_id is None:
            return
        source_track_id = int(self.merge_modal_source_track_id)
        if not dpg.does_item_exist(self.merge_target_combo_tag):
            return
        target_raw = str(dpg.get_value(self.merge_target_combo_tag)).strip()
        target_track_id = self.merge_target_label_to_track_id.get(target_raw)
        if target_track_id is None:
            self._log_status("Select a valid target track.")
            return
        if dpg.does_item_exist(self.merge_modal_tag):
            dpg.configure_item(self.merge_modal_tag, show=False)
        self.merge_modal_source_track_id = None
        self.merge_target_label_to_track_id = {}
        self._merge_tracks(source_track_id=source_track_id, target_track_id=target_track_id)

    def _split_active_track_at_current_frame(self) -> None:
        if self.active_track_id is None:
            self._log_status("Select a track before splitting.")
            return
        source_track_id = int(self.active_track_id)
        rows = self.assignments_by_track.get(source_track_id, {})
        if not rows:
            self._log_status(f"Track {source_track_id} has no boxes to split.")
            return

        split_frame = int(self.current_frame)
        move_frames = sorted(f for f in rows.keys() if f >= split_frame)
        keep_frames = sorted(f for f in rows.keys() if f < split_frame)
        if not move_frames:
            self._log_status(
                f"Track {source_track_id} has no boxes at or after frame {split_frame}."
            )
            return
        if not keep_frames:
            self._log_status(
                f"Split frame {split_frame} is at/before the first box of track {source_track_id}; choose a later frame."
            )
            return

        new_track_id = self._next_monotonic_track_id()
        self.next_track_id = max(self.next_track_id, new_track_id + 1)
        self._stop_playback()

        new_rows: Dict[int, Assignment] = {}
        for frame in move_frames:
            asn = rows.pop(frame)
            new_rows[frame] = Assignment(
                track_id=new_track_id,
                frame=asn.frame,
                x=asn.x,
                y=asn.y,
                w=asn.w,
                h=asn.h,
                det_key=asn.det_key,
            )
        self.assignments_by_track[new_track_id] = new_rows

        src_lineage = self._lineage_for_track(source_track_id)
        group_id = int(src_lineage.get("group_id", source_track_id))
        max_seg = 1
        for tid in self.assignments_by_track.keys():
            info = self._lineage_for_track(tid)
            if int(info.get("group_id", tid)) == group_id:
                max_seg = max(max_seg, int(info.get("segment_index", 1)))
        self.track_lineage_info[new_track_id] = {
            "group_id": group_id,
            "segment_index": max_seg + 1,
            "parent_track_id": source_track_id,
            "split_from_frame": split_frame,
        }

        source_absent = set(self.absent_frames_by_track.get(source_track_id, set()))
        if source_absent:
            move_absent = {f for f in source_absent if f >= split_frame}
            keep_absent = {f for f in source_absent if f < split_frame}
            if keep_absent:
                self.absent_frames_by_track[source_track_id] = keep_absent
            else:
                self.absent_frames_by_track.pop(source_track_id, None)
            if move_absent:
                self.absent_frames_by_track[new_track_id] = move_absent

        source_last = rows[max(rows.keys())]
        self.last_assigned_box_by_track[source_track_id] = (
            source_last.x,
            source_last.y,
            source_last.w,
            source_last.h,
        )
        new_last = new_rows[max(new_rows.keys())]
        self.last_assigned_box_by_track[new_track_id] = (
            new_last.x,
            new_last.y,
            new_last.w,
            new_last.h,
        )

        self._set_active_track(new_track_id)
        self._sync_lineage_with_tracks()
        self._sync_frame_controls()
        self._render_frame()
        self._persist_progress(silent=True)
        self._log_status(
            f"Split track {source_track_id} at frame {split_frame} -> new track {new_track_id} "
            f"({len(new_rows)} boxes moved)."
        )

    def _delete_track(self, track_id: int) -> None:
        if track_id not in self.assignments_by_track:
            return
        if self.playback_track_id == track_id:
            self._stop_playback()
        rows = self.assignments_by_track.pop(track_id, {})
        self.kept_track_info.pop(track_id, None)
        self.track_lineage_info.pop(track_id, None)
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

    def cb_select_track_row(self, _sender: Any, app_data: Any, user_data: Any) -> None:
        if not bool(app_data):
            return
        try:
            track_id = int(user_data)
        except Exception:
            return
        self._use_track(track_id)

    def cb_play_track_row(self, _sender: Any, _app_data: Any, user_data: Any) -> None:
        try:
            track_id = int(user_data)
        except Exception:
            return
        self._play_track_once_from_start(track_id)

    def cb_keep_track_row(self, _sender: Any, _app_data: Any, user_data: Any) -> None:
        try:
            track_id = int(user_data)
        except Exception:
            return
        self._use_track(track_id)
        self._open_keep_modal(track_id)

    def cb_keep_modal_save(self, _sender: Any, _app_data: Any) -> None:
        self._save_keep_modal()

    def cb_keep_modal_cancel(self, _sender: Any, _app_data: Any) -> None:
        if dpg.does_item_exist(self.keep_modal_tag):
            dpg.configure_item(self.keep_modal_tag, show=False)
        self.keep_modal_track_id = None

    def cb_merge_track_row(self, _sender: Any, _app_data: Any, user_data: Any) -> None:
        try:
            track_id = int(user_data)
        except Exception:
            return
        self._use_track(track_id)
        self._open_merge_modal(track_id)

    def cb_merge_modal_save(self, _sender: Any, _app_data: Any) -> None:
        self._save_merge_modal()

    def cb_merge_modal_cancel(self, _sender: Any, _app_data: Any) -> None:
        if dpg.does_item_exist(self.merge_modal_tag):
            dpg.configure_item(self.merge_modal_tag, show=False)
        self.merge_modal_source_track_id = None
        self.merge_target_label_to_track_id = {}

    def cb_delete_track_row(self, _sender: Any, _app_data: Any, user_data: Any) -> None:
        try:
            track_id = int(user_data)
        except Exception:
            return
        self._delete_track(track_id)

    def cb_toggle_playback_button(self, _sender: Any, _app_data: Any) -> None:
        self._toggle_playback_active_track()

    def cb_track_all_button(self, _sender: Any, _app_data: Any) -> None:
        self._run_track_all()

    def cb_clear_all_tracks_button(self, _sender: Any, _app_data: Any) -> None:
        self._clear_all_tracks()

    def cb_split_button(self, _sender: Any, _app_data: Any) -> None:
        self._split_active_track_at_current_frame()

    def cb_tracker_changed(self, _sender: Any, app_data: Any) -> None:
        next_name = str(app_data)
        if next_name not in self.tracker_name_to_internal:
            return
        self.tracker_name = next_name
        self.tracker_available, self.tracker_unavailable_reason = self._check_tracker_runtime()
        if not self.tracker_available:
            self._log_status(f"{self.tracker_name} unavailable: {self.tracker_unavailable_reason}")
        else:
            self._log_status(f"Selected tracker: {self.tracker_name}")
        self._update_tracker_controls()

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
        has_box = self._active_assignment(self.current_frame) is not None
        if has_box:
            dpg.draw_text(
                (8, 8),
                f"ACTIVE T{self.active_track_id}",
                color=(40, 220, 60, 255),
                size=18,
                parent=self.frame_status_drawlist_tag,
            )

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
        hits = self._hit_track_assignments_at_point(img_x, img_y)
        if hits:
            self._set_active_track(hits[0])
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
        if key == dpg.mvKey_Left or key == dpg.mvKey_A:
            self._step_frame(-1)
        elif key == dpg.mvKey_Right:
            self._step_frame(1)
        elif key == dpg.mvKey_S:
            self._split_active_track_at_current_frame()
        elif key == dpg.mvKey_M:
            if self.active_track_id is None:
                self._log_status("Select a track first, then press M to merge.")
            else:
                self._open_merge_modal(int(self.active_track_id))
        elif key == dpg.mvKey_T:
            self._run_track_all()

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
                dpg.add_text(f"Tracker: {self.tracker_name} ready")
            else:
                dpg.add_text(f"Tracker unavailable: {self.tracker_unavailable_reason}")
            dpg.add_spacer(height=6)
            with dpg.group(horizontal=True):
                dpg.add_drawlist(width=self.left_pane_w, height=self.display_h, tag=self.canvas_tag)
                dpg.add_spacer(width=self.main_gap_w)
                with dpg.child_window(
                    width=self.side_panel_w,
                    height=self.display_h,
                    border=True,
                    no_scrollbar=True,
                ):
                    dpg.add_text("Track Editor")
                    dpg.add_drawlist(
                        width=max(120, self.side_panel_w - 24),
                        height=70,
                        tag=self.frame_status_drawlist_tag,
                    )
                    dpg.add_spacer(height=6)
                    dpg.add_text("Kept tracks")
                    with dpg.table(
                        header_row=True,
                        borders_innerH=True,
                        borders_outerH=True,
                        borders_innerV=True,
                        borders_outerV=True,
                        row_background=True,
                        resizable=False,
                        policy=dpg.mvTable_SizingStretchProp,
                        tag=self.kept_track_table_tag,
                    ):
                        dpg.add_table_column(label="Track ID")
                        dpg.add_table_column(label="Name")
                        dpg.add_table_column(label="Jersey")
                        dpg.add_table_column(label="Team")
                        dpg.add_table_column(label="Frames")
                        dpg.add_table_column(label="Coverage")
                        dpg.add_table_column(label="Actions")
                    dpg.add_spacer(height=8)
                    dpg.add_text("Proposed tracks")
                    with dpg.child_window(
                        width=max(120, self.side_panel_w - 24),
                        height=-1,
                        border=True,
                    ):
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
                            dpg.add_table_column(label="Group")
                            dpg.add_table_column(label="Seg")
                            dpg.add_table_column(label="Boxes")
                            dpg.add_table_column(label="Frames")
                            dpg.add_table_column(label="Coverage")
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
                    label="Play",
                    tag=self.playback_button_tag,
                    callback=self.cb_toggle_playback_button,
                    width=90,
                )
                dpg.add_button(
                    label="Track all",
                    tag=self.track_all_button_tag,
                    callback=self.cb_track_all_button,
                    width=90,
                )
                dpg.add_button(
                    label="Clear all tracks",
                    tag=self.clear_all_button_tag,
                    callback=self.cb_clear_all_tracks_button,
                    width=120,
                )
                dpg.add_button(
                    label="Split (S)",
                    tag=self.split_button_tag,
                    callback=self.cb_split_button,
                    width=90,
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
                dpg.add_text("Tracker")
                dpg.add_combo(
                    self.available_tracker_names,
                    default_value=self.tracker_name,
                    tag=self.tracker_combo_tag,
                    callback=self.cb_tracker_changed,
                    width=140,
                )
                dpg.add_button(label="Export GT", callback=lambda: self._save_gt())
                dpg.add_button(label="Save All", callback=lambda: self._persist_progress(silent=False))

        with dpg.window(
            label="Keep Track",
            tag=self.keep_modal_tag,
            modal=True,
            show=False,
            no_resize=True,
            no_collapse=True,
            no_move=True,
            width=360,
            height=220,
        ):
            dpg.add_text("Keep Track", tag=self.keep_title_tag)
            dpg.add_spacer(height=6)
            with dpg.group(horizontal=True):
                dpg.add_text("Team")
                dpg.add_combo(["home", "away"], default_value="home", tag=self.keep_team_tag, width=220)
            with dpg.group(horizontal=True):
                dpg.add_text("Jersey")
                dpg.add_input_int(default_value=0, tag=self.keep_jersey_tag, width=220)
            with dpg.group(horizontal=True):
                dpg.add_text("Name")
                dpg.add_input_text(default_value="", tag=self.keep_name_tag, width=220)
            dpg.add_spacer(height=10)
            with dpg.group(horizontal=True):
                dpg.add_button(label="Save", callback=self.cb_keep_modal_save, width=100)
                dpg.add_button(label="Cancel", callback=self.cb_keep_modal_cancel, width=100)

        with dpg.window(
            label="Merge Track",
            tag=self.merge_modal_tag,
            modal=True,
            show=False,
            no_resize=True,
            no_collapse=True,
            no_move=True,
            width=360,
            height=170,
        ):
            dpg.add_text("Merge Track", tag=self.merge_title_tag)
            dpg.add_spacer(height=8)
            with dpg.group(horizontal=True):
                dpg.add_text("Target")
                dpg.add_combo([], default_value="", tag=self.merge_target_combo_tag, width=240)
            dpg.add_spacer(height=12)
            with dpg.group(horizontal=True):
                dpg.add_button(label="Merge", callback=self.cb_merge_modal_save, width=100)
                dpg.add_button(label="Cancel", callback=self.cb_merge_modal_cancel, width=100)

        with dpg.item_handler_registry(tag="bmot_canvas_handlers"):
            dpg.add_item_clicked_handler(callback=self.cb_canvas_click)
        dpg.bind_item_handler_registry(self.canvas_tag, "bmot_canvas_handlers")

        with dpg.handler_registry():
            dpg.add_key_press_handler(key=dpg.mvKey_A, callback=self.cb_hotkey)
            dpg.add_key_press_handler(key=dpg.mvKey_Left, callback=self.cb_hotkey)
            dpg.add_key_press_handler(key=dpg.mvKey_Right, callback=self.cb_hotkey)
            dpg.add_key_press_handler(key=dpg.mvKey_S, callback=self.cb_hotkey)
            dpg.add_key_press_handler(key=dpg.mvKey_M, callback=self.cb_hotkey)
            dpg.add_key_press_handler(key=dpg.mvKey_T, callback=self.cb_hotkey)
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
