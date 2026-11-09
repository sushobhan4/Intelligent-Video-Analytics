"""
================================================================================
Neural CCTV Analyser - Semantic Activity & Natural Language Event Engine
================================================================================
Synthesizes high-level, human-like activity summaries from raw spatial coordinates,
tracking histories, facial identities, zone containment, and package detection.
Examples:
  - "Father is home. He arrived at your doorstep."
  - "An unknown person was at your door carrying a package."
  - "A vehicle paused at your doorstep."
"""

import math
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import cv2
import numpy as np


class HouseholdZone:
    """Represents a named polygonal zone (e.g. doorstep, driveway, walkway)."""
    def __init__(self, zone_id: str, name: str, polygon_normalized: List[List[float]], alert_level: str = "INFO"):
        self.zone_id = zone_id
        self.name = name
        self.polygon_norm = polygon_normalized
        self.alert_level = alert_level

    def contains_point(self, pt_norm: Tuple[float, float]) -> bool:
        """Returns True if point (x_norm, y_norm) lies inside the polygon."""
        poly = np.array(self.polygon_norm, dtype=np.float32)
        res = cv2.pointPolygonTest(poly, pt_norm, False)
        return res >= 0


class EntityTrackState:
    """State tracking for a single moving entity (person or vehicle)."""
    def __init__(self, track_id: int, entity_type: str, initial_bbox: List[int], timestamp: float):
        self.track_id = track_id
        self.entity_type = entity_type  # 'person', 'car', 'truck', etc.
        self.identity_name = "Unknown Person" if entity_type == "person" else entity_type.capitalize()
        self.is_identified = False
        self.confidence = 0.0

        self.first_seen = timestamp
        self.last_seen = timestamp
        self.positions: List[Tuple[float, float, float]] = []  # (x, y, timestamp)
        self.bboxes: List[List[int]] = []

        self.current_zone: Optional[HouseholdZone] = None
        self.zone_entry_time: float = timestamp
        self.is_stationary = False
        self.carrying_package = False

        # Alert deduplication records
        self.logged_actions: set = set()
        self.last_log_time: float = 0.0

        self.update(initial_bbox, timestamp)

    @property
    def dwell_time_seconds(self) -> float:
        return self.last_seen - self.zone_entry_time

    def update(self, bbox: List[int], timestamp: float):
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = y2  # Ground contact point
        self.positions.append((cx, cy, timestamp))
        self.bboxes.append(bbox)
        self.last_seen = timestamp

        # Keep history compact (last 60 entries)
        if len(self.positions) > 60:
            self.positions = self.positions[-60:]
            self.bboxes = self.bboxes[-60:]

        # Calculate recent velocity
        if len(self.positions) >= 5:
            p_old = self.positions[-5]
            p_new = self.positions[-1]
            dt = max(p_new[2] - p_old[2], 0.001)
            dist = math.hypot(p_new[0] - p_old[0], p_new[1] - p_old[1])
            speed = dist / dt
            self.is_stationary = speed < 18.0  # Pixels per second
        else:
            self.is_stationary = False


class SemanticActivityEngine:
    """
    Evaluates tracking history and produces natural language household events.
    """

    def __init__(self, zones_config: List[Dict[str, Any]], log_cooldown_seconds: float = 4.0):
        self.zones: List[HouseholdZone] = []
        for z in zones_config:
            self.zones.append(HouseholdZone(
                zone_id=z["zone_id"],
                name=z["name"],
                polygon_normalized=z["polygon"],
                alert_level=z.get("alert_level", "INFO")
            ))

        self.tracks: Dict[int, EntityTrackState] = {}
        self.log_cooldown = log_cooldown_seconds

    def check_zone(self, cx_norm: float, cy_norm: float) -> Optional[HouseholdZone]:
        """Checks which zone a normalized coordinate falls inside."""
        for zone in self.zones:
            if zone.contains_point((cx_norm, cy_norm)):
                return zone
        return None

    def update_track(
        self,
        track_id: int,
        entity_type: str,
        bbox: List[int],
        frame_w: int,
        frame_h: int,
        timestamp: float,
        detected_packages: List[List[int]] = None
    ) -> EntityTrackState:
        """Updates entity track and checks package overlap."""
        if track_id not in self.tracks:
            self.tracks[track_id] = EntityTrackState(track_id, entity_type, bbox, timestamp)

        track = self.tracks[track_id]
        track.update(bbox, timestamp)

        # Check if coordinates fall into a zone
        x1, y1, x2, y2 = bbox
        cx_norm = ((x1 + x2) / 2.0) / frame_w
        cy_norm = float(y2) / frame_h

        new_zone = self.check_zone(cx_norm, cy_norm)
        if new_zone != track.current_zone:
            track.current_zone = new_zone
            track.zone_entry_time = timestamp

        # Check if person is carrying a package (overlap with backpack, parcel, bag)
        if entity_type == "person" and detected_packages:
            for pbox in detected_packages:
                px1, py1, px2, py2 = pbox
                # Check for significant intersection with person bounding box
                ix1 = max(x1, px1)
                iy1 = max(y1, py1)
                ix2 = min(x2, px2)
                iy2 = min(y2, py2)
                if ix2 > ix1 and iy2 > iy1:
                    track.carrying_package = True
                    break

        return track

    def set_identity(self, track_id: int, name: str, is_known: bool, confidence: float):
        """Binds a facial recognition result to a track."""
        if track_id in self.tracks:
            track = self.tracks[track_id]
            track.identity_name = name
            track.is_identified = is_known
            track.confidence = confidence

    def generate_events(self, timestamp: float) -> List[Dict[str, Any]]:
        """
        Analyzes track states and yields semantic event notifications.
        """
        events = []
        now = timestamp

        for track_id, track in list(self.tracks.items()):
            # Prune lost tracks inactive for > 15 seconds
            if now - track.last_seen > 15.0:
                continue

            zone_name = track.current_zone.name if track.current_zone else "perimeter area"
            zone_id = track.current_zone.zone_id if track.current_zone else "perimeter"

            # -------------------------------------------------------------
            # Case 1: Known Family / Household Member Arrived
            # -------------------------------------------------------------
            if track.entity_type == "person" and track.is_identified:
                action_key = f"{track.identity_name}_arrived"
                if action_key not in track.logged_actions and (now - track.last_log_time >= self.log_cooldown):
                    if track.carrying_package:
                        sentence = f"{track.identity_name} is home. Carrying a package at your {zone_name}."
                    else:
                        sentence = f"{track.identity_name} is home. Arrived at your {zone_name}."

                    events.append({
                        "severity": "INFO",
                        "category": "KNOWN_PERSON_ARRIVED",
                        "track_id": track_id,
                        "identity": track.identity_name,
                        "zone": zone_name,
                        "description": sentence,
                        "timestamp": now
                    })
                    track.logged_actions.add(action_key)
                    track.last_log_time = now

            # -------------------------------------------------------------
            # Case 2: Person at Doorstep Carrying a Package (Delivery / Courier)
            # -------------------------------------------------------------
            elif track.entity_type == "person" and track.carrying_package:
                action_key = f"delivery_package_{zone_id}"
                if action_key not in track.logged_actions and (now - track.last_log_time >= self.log_cooldown):
                    if not track.is_identified:
                        sentence = f"A person was on your {zone_name} carrying a package."
                        severity = "NOTICE"
                        category = "PACKAGE_DELIVERY"
                    else:
                        sentence = f"{track.identity_name} was on your {zone_name} carrying a package."
                        severity = "INFO"
                        category = "PACKAGE_HANDLING"

                    events.append({
                        "severity": severity,
                        "category": category,
                        "track_id": track_id,
                        "identity": track.identity_name,
                        "zone": zone_name,
                        "description": sentence,
                        "timestamp": now
                    })
                    track.logged_actions.add(action_key)
                    track.last_log_time = now

            # -------------------------------------------------------------
            # Case 3: Vehicle Paused at Doorstep / Driveway
            # -------------------------------------------------------------
            elif track.entity_type in ["car", "truck", "motorcycle", "vehicle"]:
                if track.is_stationary and track.dwell_time_seconds >= 2.5:
                    action_key = f"vehicle_paused_{zone_id}"
                    if action_key not in track.logged_actions and (now - track.last_log_time >= self.log_cooldown):
                        sentence = f"A vehicle paused at your {zone_name}."
                        events.append({
                            "severity": "NOTICE",
                            "category": "VEHICLE_PAUSED",
                            "track_id": track_id,
                            "identity": "Vehicle",
                            "zone": zone_name,
                            "description": sentence,
                            "timestamp": now
                        })
                        track.logged_actions.add(action_key)
                        track.last_log_time = now

            # -------------------------------------------------------------
            # Case 4: Unknown Person Paused / Lingering at Doorstep or Walkway
            # -------------------------------------------------------------
            elif track.entity_type == "person" and not track.is_identified:
                if track.is_stationary and track.dwell_time_seconds >= 2.0:
                    action_key = f"unknown_paused_{zone_id}"
                    if action_key not in track.logged_actions and (now - track.last_log_time >= self.log_cooldown):
                        sentence = f"An unknown person paused at your {zone_name}."
                        events.append({
                            "severity": "ALERT",
                            "category": "UNKNOWN_PERSON_PAUSED",
                            "track_id": track_id,
                            "identity": "Unknown Person",
                            "zone": zone_name,
                            "description": sentence,
                            "timestamp": now
                        })
                        track.logged_actions.add(action_key)
                        track.last_log_time = now
                elif "initial_entry" not in track.logged_actions and (now - track.last_log_time >= self.log_cooldown):
                    sentence = f"An unknown person entered your {zone_name}."
                    events.append({
                        "severity": "NOTICE",
                        "category": "UNKNOWN_PERSON_ENTERED",
                        "track_id": track_id,
                        "identity": "Unknown Person",
                        "zone": zone_name,
                        "description": sentence,
                        "timestamp": now
                    })
                    track.logged_actions.add("initial_entry")
                    track.last_log_time = now

        return events
