from __future__ import annotations # to enable postponed evaluation of type annotations
from dataclasses import dataclass, field # generate classes
from typing import Dict, Optional, List, Tuple # import typing helpers

import numpy as np
import cv2 # color conversion (BGR -> HSV)

# a helper function that takes a BGR image crop and returns another crop (HSV)
def _torso_crop(bgr) -> Optional[np.ndarray]:
    if bgr is None or bgr.size == 0:
        return None
    h, w = bgr.shape[:2]
    if h < 10 or w < 10:
        return None
    
    y1 = int(h * 0.2)
    y2 = int(h * 0.6)
    x1 = int(w * 0.2)
    x2 = int(w * 0.8)
    torso = bgr[y1:y2, x1:x2]
    if torso.size == 0:
        return None
    return torso

# a helper that returns a single numeric feature (hue) from a BGR crop
def _sv_feature(bgr: np.ndarray) -> Optional[Tuple[float, float]]:
    torso = _torso_crop(bgr)
    if torso is None:
        return None

    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1].astype(np.float32)
    v = hsv[:, :, 2].astype(np.float32)

    # Keep pixels that are not too dark. (White jerseys have low S, so don't filter by high S.)
    mask = (v >= 40)

    s_vals = s[mask]
    v_vals = v[mask]

    if s_vals.size < 50:
        return None

    return (float(np.median(s_vals)), float(np.median(v_vals)))

@dataclass
class TeamClassifier:
    """
    Collect huye smaples per track_id, then cluster 2 teams using 1D k-means (implemented manually)
    Returns stable team_id per track_id once enough evidence is collected.
    """
    
    min_samples_per_track: int = 10 # a track must have at least 10 hue samples before it's used for clustering
    recompute_every: int = 30 # re-run clustering every 30 updates (roughly every 30 frames processed)
    _track_feats: Dict[int, List[Tuple[float, float]]] = field(default_factory=dict)
    _track_team: Dict[int, int] = field(default_factory=dict) # mapping track_id -> team_id (0 or 1)
    _num_updates: int = 0 # counter of how many times update()
    _centers: Optional[Tuple[float, float]] = None # store 2 cluster centers (team hue centers) if clustering has been computed; otherwise None

    def update(self, track_id: int, bgr_bbox_crop: np.ndarray) -> Optional[int]:
        feat = _sv_feature(bgr_bbox_crop)  # (s_med, v_med)
        self._num_updates += 1

        if feat is not None:
            self._track_feats.setdefault(track_id, []).append(feat)

        # periodically re-cluster once enough tracks have enough samples
        if self._num_updates % self.recompute_every == 0:
            self._recluster()
        return self._track_team.get(track_id)

    def _recluster(self) -> None:
        reps = []      # list of [s_rep, v_rep]
        rep_ids = []   # list of track_ids

        # Build 2D representative per track: (median S, median V)
        for tid, feats in self._track_feats.items():
            if len(feats) >= self.min_samples_per_track:
                s_list = [sv[0] for sv in feats]
                v_list = [sv[1] for sv in feats]
                reps.append([float(np.median(s_list)), float(np.median(v_list))])
                rep_ids.append(tid)

        if len(reps) < 2:
            return

        X = np.array(reps, dtype=np.float32)  # shape (N,2)

        # Initialize centers using saturation extremes (white vs dark usually separates on S)
        i0 = int(np.argmin(X[:, 0]))  # smallest S
        i1 = int(np.argmax(X[:, 0]))  # largest S
        c0 = X[i0].copy()
        c1 = X[i1].copy()

        # If centers are identical, no clustering possible
        if np.max(np.abs(c1 - c0)) < 1e-3:
            return

        # 2D k-means
        for _ in range(20):
            d0 = np.sum((X - c0) ** 2, axis=1)  # squared distance
            d1 = np.sum((X - c1) ** 2, axis=1)
            labels = (d1 < d0).astype(np.int32)

            if np.any(labels == 0):
                new_c0 = X[labels == 0].mean(axis=0)
            else:
                new_c0 = c0

            if np.any(labels == 1):
                new_c1 = X[labels == 1].mean(axis=0)
            else:
                new_c1 = c1

            if np.max(np.abs(new_c0 - c0)) < 1e-3 and np.max(np.abs(new_c1 - c1)) < 1e-3:
                break

            c0, c1 = new_c0, new_c1

        # Stable labeling: team 0 = lower saturation center
        if c0[0] <= c1[0]:
            self._centers = ((float(c0[0]), float(c0[1])), (float(c1[0]), float(c1[1])))
        else:
            self._centers = ((float(c1[0]), float(c1[1])), (float(c0[0]), float(c0[1])))
            labels = 1 - labels

        for tid, lab in zip(rep_ids, labels.tolist()):
            self._track_team[tid] = int(lab)
        
    def centers(self):
        return self._centers

