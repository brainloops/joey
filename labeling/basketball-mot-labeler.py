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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

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
    state_file: Path
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


def discover_sequences(dataset_root: Path, split: str) -> List[SequenceInfo]:
    split_root = dataset_root / split
    if not split_root.is_dir():
        raise FileNotFoundError(f"Split folder not found: {split_root}")

    seqs: List[SequenceInfo] = []
    for seq_root in sorted(p for p in split_root.iterdir() if p.is_dir()):
        img_dir = seq_root / "img1"
        det_file = seq_root / "det" / "det.txt"
        gt_file = seq_root / "gt" / "gt.txt"
        state_file = seq_root / "gt" / "labeler_state.json"
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
                state_file=state_file,
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
        self.min_score = min_score
        self.sequences = discover_sequences(dataset_root=dataset_root, split=split)
        self.sequence_by_name = {s.name: s for s in self.sequences}
        self.current_seq = self.sequences[0]
        self.current_frame = 1
        self.active_track_id: Optional[int] = None
        self.next_track_id = 1

        self.detections_by_frame: Dict[int, List[Detection]] = {}
        self.assignments_by_track: Dict[int, Dict[int, Assignment]] = {}
        self.assigned_det_keys: set[Tuple[int, int]] = set()
        self.last_assigned_box_by_track: Dict[int, Tuple[float, float, float, float]] = {}
        self.undo_stack: List[Tuple[str, Any]] = []
        self.focus_mode = False
        self.skip_mode = False
        self.active_box_img: Optional[Tuple[float, float, float, float]] = None
        self.cycle_last_point_img: Optional[Tuple[float, float]] = None
        self.cycle_last_signature: Optional[Tuple[Tuple[int, int], ...]] = None
        self.cycle_step = 0

        self.texture_tag = "bmot_texture"
        self.image_tag = "bmot_image"
        self.canvas_tag = "bmot_canvas"
        self.status_tag = "bmot_status"
        self.track_tag = "bmot_active_track"
        self.frame_slider_tag = "bmot_frame_slider"
        self.frame_input_tag = "bmot_frame_input"
        self.seq_combo_tag = "bmot_seq_combo"
        self.track_table_tag = "bmot_track_table"
        self.persist_tag = "bmot_persist_text"

        self.texture_w = 1
        self.texture_h = 1
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
        self.side_panel_w = 360
        self.main_gap_w = 14
        self.left_pane_w = 1100

        self._load_sequence(self.current_seq.name)

    @staticmethod
    def _rgb_to_texture_data(rgb: np.ndarray) -> np.ndarray:
        """Convert uint8 RGB image to DearPyGui RGBA float texture payload."""
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("Expected HxWx3 RGB image.")
        h, w, _ = rgb.shape
        alpha = np.full((h, w, 1), 255, dtype=np.uint8)
        rgba = np.concatenate([rgb, alpha], axis=2)
        return (rgba.astype(np.float32) / 255.0).flatten()

    def _log_status(self, msg: str) -> None:
        if self.ui_ready and dpg.does_item_exist(self.status_tag):
            dpg.set_value(self.status_tag, msg)

    def _set_active_track(self, track_id: Optional[int]) -> None:
        self.active_track_id = track_id
        if self.ui_ready and dpg.does_item_exist(self.track_tag):
            dpg.set_value(self.track_tag, f"Active track: {track_id if track_id is not None else 'None'}")
        self._update_persist_text()

    def _load_sequence(self, seq_name: str) -> None:
        self.current_seq = self.sequence_by_name[seq_name]
        self.current_frame = 1
        self.assignments_by_track = {}
        self.assigned_det_keys = set()
        self.last_assigned_box_by_track = {}
        self.undo_stack = []
        self.focus_mode = False
        self.skip_mode = False
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
        if self.current_seq.state_file.exists():
            self._load_from_state_file()
        else:
            self._load_from_gt_file()

    def _load_from_state_file(self) -> None:
        try:
            payload = json.loads(self.current_seq.state_file.read_text(encoding="utf-8"))
        except Exception:
            self._load_from_gt_file()
            return
        rows = payload.get("assignments", [])
        if not isinstance(rows, list):
            rows = []
        self.next_track_id = int(payload.get("next_track_id", self.track_id_start))
        if self.next_track_id < self.track_id_start:
            self.next_track_id = self.track_id_start
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
        saved_frame = int(payload.get("current_frame", 1))
        self.current_frame = _clamp(saved_frame, 1, max(1, self.current_seq.seq_len))
        saved_active_track = payload.get("active_track_id")
        if isinstance(saved_active_track, int) and saved_active_track >= self.track_id_start:
            if saved_active_track in self.assignments_by_track or saved_active_track < self.next_track_id:
                self.active_track_id = saved_active_track
        self.focus_mode = bool(payload.get("focus_mode", False))
        self.skip_mode = bool(payload.get("skip_mode", False))
        if self.active_track_id is None:
            if self.assignments_by_track:
                # Sensible resume fallback: continue the track with latest labeled frame.
                latest_tid = max(
                    self.assignments_by_track.keys(),
                    key=lambda tid: max(self.assignments_by_track[tid].keys()),
                )
                self.active_track_id = latest_tid
                self.focus_mode = False
                self.skip_mode = False
            else:
                self.focus_mode = False
                self.skip_mode = False

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
        if dpg.does_item_exist(self.texture_tag):
            dpg.delete_item(self.texture_tag)
        dpg.add_dynamic_texture(
            width=self.texture_w,
            height=self.texture_h,
            default_value=initial,
            tag=self.texture_tag,
            parent="bmot_texture_registry",
        )
        dpg.configure_item(self.canvas_tag, width=self.display_w, height=self.display_h)

    def _sync_frame_controls(self) -> None:
        if not self.ui_ready:
            return
        dpg.configure_item(self.frame_slider_tag, min_value=1, max_value=max(1, self.current_seq.seq_len))
        dpg.set_value(self.frame_slider_tag, self.current_frame)
        dpg.set_value(self.frame_input_tag, self.current_frame)

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
        if self.active_track_id not in self.last_assigned_box_by_track:
            return None
        suggested = self._best_iou_detection(frame=frame, ref_box=self.last_assigned_box_by_track[self.active_track_id])
        if suggested is None:
            return None
        return (suggested.frame, suggested.det_idx)

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
        active_key = self._active_det_key_for_frame(self.current_frame)
        self.active_box_img = None

        candidates = self._candidate_detections(self.current_frame)
        for det in candidates:
            key = (det.frame, det.det_idx)
            if self.focus_mode and self.active_track_id is not None and key != active_key:
                continue
            x1 = det.x * self.scale_x
            y1 = det.y * self.scale_y
            x2 = (det.x + det.w) * self.scale_x
            y2 = (det.y + det.h) * self.scale_y
            is_active = self.active_track_id is not None and key == active_key
            color = (40, 220, 60, 255) if is_active else (210, 210, 210, 230)
            thickness = active_thickness if is_active else base_thickness
            dpg.draw_rectangle((x1, y1), (x2, y2), color=color, thickness=thickness, parent=self.canvas_tag)
            if is_active:
                self.active_box_img = (det.x, det.y, det.w, det.h)
            if not is_active:
                dpg.draw_text((x1, y1), f"{det.score:.2f}", color=(230, 230, 230, 255), size=12, parent=self.canvas_tag)

        if active_asn is not None and self.active_track_id is not None:
            label_x = active_asn.x * self.scale_x + 3
            label_y = max(0, active_asn.y * self.scale_y - 18)
            dpg.draw_text((label_x, label_y), f"T{self.active_track_id}", color=(40, 220, 60, 255), size=16, parent=self.canvas_tag)

        assigned_count = len(self.assigned_det_keys)
        all_count = sum(len(v) for v in self.detections_by_frame.values())
        self._log_status(
            f"{self.current_seq.name} | frame {self.current_frame}/{self.current_seq.seq_len} | "
            f"active={self.active_track_id if self.active_track_id is not None else '-'} | "
            f"assigned={assigned_count}/{all_count}"
        )
        self._refresh_track_table()

    def _step_frame(self, delta: int) -> None:
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
        used = set(self.assignments_by_track.keys())
        tid = self.track_id_start
        while tid in used:
            tid += 1
        return tid

    def _current_track_id_for_assignment(self) -> int:
        if self.active_track_id is None:
            tid = self._first_available_track_id()
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
        self.current_seq.gt_file.parent.mkdir(parents=True, exist_ok=True)
        self.current_seq.gt_file.write_text(payload, encoding="utf-8")
        if not silent:
            self._log_status(f"Saved {len(rows)} GT rows: {self.current_seq.gt_file}")

    def _save_state(self, silent: bool = False) -> None:
        rows: List[dict[str, Any]] = []
        for tid in sorted(self.assignments_by_track):
            for frame in sorted(self.assignments_by_track[tid]):
                asn = self.assignments_by_track[tid][frame]
                rows.append(
                    {
                        "track_id": tid,
                        "frame": asn.frame,
                        "x": asn.x,
                        "y": asn.y,
                        "w": asn.w,
                        "h": asn.h,
                        "det_key": [asn.det_key[0], asn.det_key[1]] if asn.det_key is not None else None,
                    }
                )
        payload = {
            "version": 1,
            "track_id_start": self.track_id_start,
            "next_track_id": self.next_track_id,
            "current_frame": self.current_frame,
            "active_track_id": self.active_track_id,
            "focus_mode": self.focus_mode,
            "skip_mode": self.skip_mode,
            "assignments": rows,
        }
        self.current_seq.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.current_seq.state_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        if not silent:
            self._log_status(f"Saved labeler state: {self.current_seq.state_file}")

    def _persist_progress(self, silent: bool = True) -> None:
        self._save_gt(silent=silent)
        self._save_state(silent=silent)
        self._update_persist_text()

    def _persist_session_state(self) -> None:
        self._save_state(silent=True)
        self._update_persist_text()

    def _refresh_track_table(self) -> None:
        if not self.ui_ready or not dpg.does_item_exist(self.track_table_tag):
            return
        dpg.delete_item(self.track_table_tag, children_only=True, slot=1)
        track_ids = sorted(self.assignments_by_track.keys())
        for tid in track_ids:
            count = len(self.assignments_by_track.get(tid, {}))
            active = "*" if self.active_track_id == tid else ""
            with dpg.table_row(parent=self.track_table_tag):
                dpg.add_text(str(tid))
                dpg.add_text(str(count))
                dpg.add_text(active)
                with dpg.group(horizontal=True):
                    dpg.add_button(label="Use", callback=self.cb_use_track_row, user_data=tid, width=54)
                    dpg.add_button(label="Del", callback=self.cb_delete_track_row, user_data=tid, width=54)

    def _use_track(self, track_id: int) -> None:
        if track_id not in self.assignments_by_track:
            return
        self._set_active_track(track_id)
        self.focus_mode = False
        self.skip_mode = False
        self._render_frame()
        self._persist_session_state()

    def _delete_track(self, track_id: int) -> None:
        if track_id not in self.assignments_by_track:
            return
        rows = self.assignments_by_track.pop(track_id, {})
        for asn in rows.values():
            if asn.det_key is not None and asn.det_key in self.assigned_det_keys:
                self.assigned_det_keys.remove(asn.det_key)
        if track_id in self.last_assigned_box_by_track:
            del self.last_assigned_box_by_track[track_id]
        if self.active_track_id == track_id:
            self._set_active_track(None)
            self.focus_mode = False
            self.skip_mode = False
        self.next_track_id = self._first_available_track_id()
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

    def _update_persist_text(self) -> None:
        if not self.ui_ready or not dpg.does_item_exist(self.persist_tag):
            return
        active = self.active_track_id if self.active_track_id is not None else "-"
        dpg.set_value(
            self.persist_tag,
            f"Autosave: {self.current_seq.state_file.name} + gt.txt | frame={self.current_frame} active={active}",
        )

    def _end_track(self) -> None:
        self._set_active_track(None)
        self.focus_mode = False
        self.skip_mode = False
        self._render_frame()
        self._persist_session_state()

    def _advance_with_iou(self) -> None:
        if self.current_frame >= self.current_seq.seq_len:
            return
        next_frame = self.current_frame + 1
        self.current_frame = next_frame
        self._sync_frame_controls()
        if self.active_track_id is None:
            self._render_frame()
            self._persist_session_state()
            return
        if self.skip_mode:
            self.focus_mode = False
            self._render_frame()
            self._persist_session_state()
            return
        ref_box = self.last_assigned_box_by_track.get(self.active_track_id)
        if ref_box is None:
            self._render_frame()
            self._persist_session_state()
            return
        best = self._best_iou_detection(frame=next_frame, ref_box=ref_box)
        if best is not None:
            self._assign_detection(best)
        else:
            self._render_frame()
            self._persist_session_state()

    # DearPyGui callbacks
    def cb_sequence_changed(self, _sender: Any, app_data: Any) -> None:
        self._load_sequence(str(app_data))

    def cb_frame_slider(self, _sender: Any, app_data: Any) -> None:
        self._set_frame(int(app_data))

    def cb_frame_input(self, _sender: Any, app_data: Any) -> None:
        self._set_frame(int(app_data))

    def cb_canvas_click(self, _sender: Any, _app_data: Any) -> None:
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
            signature: Tuple[Tuple[int, int], ...] = tuple((det.frame, det.det_idx) for det in hits)
            if self._is_same_click_region(img_x, img_y, signature):
                self.cycle_step += 1
            else:
                self.cycle_step = 0
                self.cycle_last_signature = signature
                self.cycle_last_point_img = (img_x, img_y)

            active_key = self._active_det_key_for_frame(self.current_frame)
            if len(hits) == 1 and active_key == (hits[0].frame, hits[0].det_idx):
                self._deselect_current_active()
                return

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
        if key == dpg.mvKey_Q:
            self._end_track()
        elif key == dpg.mvKey_E:
            tid = self._first_available_track_id()
            self._set_active_track(tid)
            self.next_track_id = max(self.next_track_id, tid + 1)
            self.focus_mode = False
            self._render_frame()
            self._persist_session_state()
        elif key == dpg.mvKey_S:
            self._persist_progress(silent=False)
        elif key == dpg.mvKey_R:
            self._undo()
        elif key == dpg.mvKey_Spacebar:
            self._advance_with_iou()

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
            with dpg.group(horizontal=True):
                dpg.add_text("Sequence")
                dpg.add_combo(
                    [s.name for s in self.sequences],
                    default_value=self.current_seq.name,
                    tag=self.seq_combo_tag,
                    callback=self.cb_sequence_changed,
                    width=340,
                )
                dpg.add_text("", tag=self.track_tag)
                dpg.add_button(label="Save (S)", callback=lambda: self._save_gt())
                dpg.add_button(label="Undo (R)", callback=lambda: self._undo())
                dpg.add_button(label="End track (Q)", callback=lambda: self._end_track())
                dpg.add_button(label="Save All", callback=lambda: self._persist_progress(silent=False))

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

            dpg.add_text("", tag=self.status_tag)
            dpg.add_text("", tag=self.persist_tag)
            dpg.add_spacer(height=6)
            with dpg.group(horizontal=True):
                dpg.add_drawlist(width=self.left_pane_w, height=self.display_h, tag=self.canvas_tag)
                dpg.add_spacer(width=self.main_gap_w)
                with dpg.child_window(width=self.side_panel_w, height=self.display_h, border=True):
                    dpg.add_text("Tracks")
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
                        dpg.add_table_column(label="Frames")
                        dpg.add_table_column(label="Active")
                        dpg.add_table_column(label="Actions")

        with dpg.item_handler_registry(tag="bmot_canvas_handlers"):
            dpg.add_item_clicked_handler(callback=self.cb_canvas_click)
        dpg.bind_item_handler_registry(self.canvas_tag, "bmot_canvas_handlers")

        with dpg.handler_registry():
            dpg.add_key_press_handler(key=dpg.mvKey_D, callback=self.cb_hotkey)
            dpg.add_key_press_handler(key=dpg.mvKey_A, callback=self.cb_hotkey)
            dpg.add_key_press_handler(key=dpg.mvKey_W, callback=self.cb_hotkey)
            dpg.add_key_press_handler(key=dpg.mvKey_Q, callback=self.cb_hotkey)
            dpg.add_key_press_handler(key=dpg.mvKey_E, callback=self.cb_hotkey)
            dpg.add_key_press_handler(key=dpg.mvKey_S, callback=self.cb_hotkey)
            dpg.add_key_press_handler(key=dpg.mvKey_R, callback=self.cb_hotkey)
            dpg.add_key_press_handler(key=dpg.mvKey_Spacebar, callback=self.cb_hotkey)

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
        dpg.start_dearpygui()
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
