"""
Voice Scene Assistant - FastAPI Backend
Wraps the existing voice_assistant_enhanced.py logic as REST endpoints.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import base64
import json
import time
import os
import io
import tempfile
from datetime import datetime
from collections import defaultdict, deque
from typing import Optional
import asyncio
from contextlib import asynccontextmanager

# ── Optional heavy deps ────────────────────────────────────────────────────────
try:
    import pytesseract
    TESSERACT_OK = True
except ImportError:
    TESSERACT_OK = False

try:
    from gtts import gTTS
    GTTS_OK = True
except ImportError:
    GTTS_OK = False

# ── App state (loaded once at startup) ────────────────────────────────────────
app_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load heavy models once at startup."""
    print("Loading YOLO model...")
    weights = os.environ.get("YOLO_WEIGHTS", "yolov3.weights")
    cfg     = os.environ.get("YOLO_CFG",     "yolov3.cfg")
    names   = os.environ.get("COCO_NAMES",   "coco.names")

    if not os.path.exists(weights):
        print(f"WARNING: {weights} not found – detection will be skipped.")
        app_state["net"] = None
        app_state["classes"] = []
        app_state["output_layers"] = []
    else:
        net = cv2.dnn.readNet(weights, cfg)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        layer_names   = net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
        with open(names) as f:
            classes = [l.strip() for l in f.readlines()]
        app_state["net"]           = net
        app_state["classes"]       = classes
        app_state["output_layers"] = output_layers
        print(f"YOLO loaded — {len(classes)} classes.")

    # Face cascade
    app_state["face_cascade"] = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    # Scene memory (in-memory for this session)
    app_state["scene_memory"]      = defaultdict(list)  # label → [timestamps]
    app_state["object_alerts"]     = {}                  # label → "appear"|"disappear"
    app_state["alert_cooldown"]    = {}
    app_state["currently_visible"] = set()

    print("Backend ready!")
    yield
    print("Shutting down.")

app = FastAPI(title="Voice Scene Assistant API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Helpers ────────────────────────────────────────────────────────────────────

KNOWN_WIDTHS = {
    "person": 0.5, "car": 1.8, "bicycle": 0.6, "chair": 0.5,
    "laptop": 0.35, "cell phone": 0.07, "book": 0.15,
    "bottle": 0.08, "cup": 0.08, "tv": 1.0,
    "keyboard": 0.4, "mouse": 0.1,
}
FOCAL_LENGTH = 800
CONF_THRESH  = 0.5
NMS_THRESH   = 0.4


def decode_image(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def estimate_distance(label: str, pixel_width: int) -> Optional[float]:
    kw = KNOWN_WIDTHS.get(label.lower())
    if kw and pixel_width > 0:
        return (kw * FOCAL_LENGTH) / pixel_width
    return None


def distance_desc(d: Optional[float]) -> str:
    if d is None:          return "unknown distance"
    if d < 0.5:            return f"very close, about {int(d*100)} cm away"
    if d < 1.5:            return f"close, about {d:.1f} m away"
    if d < 3.0:            return f"medium distance, about {d:.1f} m away"
    if d < 6.0:            return f"far, about {int(d)} m away"
    return f"very far, about {int(d)} m away"


def position_label(cx: float, width: int) -> str:
    if cx < width / 3:      return "left"
    if cx > 2 * width / 3:  return "right"
    return "center"


def run_yolo(frame: np.ndarray):
    net           = app_state["net"]
    classes       = app_state["classes"]
    output_layers = app_state["output_layers"]

    if net is None:
        return []

    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (320, 320), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids, confidences, boxes = [], [], []
    for out in outs:
        for det in out:
            scores     = det[5:]
            class_id   = int(np.argmax(scores))
            confidence = float(scores[class_id])
            if confidence > CONF_THRESH:
                cx = int(det[0] * w); cy = int(det[1] * h)
                bw = int(det[2] * w); bh = int(det[3] * h)
                x  = cx - bw // 2;   y  = cy - bh // 2
                boxes.append([x, y, bw, bh])
                confidences.append(confidence)
                class_ids.append(class_id)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESH, NMS_THRESH)
    results = []
    if len(idxs) > 0:
        for i in idxs.flatten():
            x, y, bw, bh = boxes[i]
            label  = classes[class_ids[i]]
            dist_m = estimate_distance(label, bw)
            results.append({
                "label":      label,
                "confidence": round(confidences[i], 2),
                "position":   position_label(x + bw / 2, w),
                "distance":   distance_desc(dist_m),
                "distance_m": round(dist_m, 2) if dist_m else None,
                "box":        [x, y, bw, bh],
            })
    return results


def detect_faces(frame: np.ndarray):
    """Detect faces with strict parameters to avoid false positives."""
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = app_state["face_cascade"].detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=8,    # higher = stricter, fewer false positives
        minSize=(60, 60)   # ignore anything smaller than 60x60 pixels
    )
    if len(faces) == 0:
        return []
    return [{"label": "face", "box": [int(x), int(y), int(w), int(h)]}
            for (x, y, w, h) in faces]


def update_memory(objects):
    now = datetime.now().isoformat()
    mem = app_state["scene_memory"]
    for obj in objects:
        mem[obj["label"]].append(now)
        if len(mem[obj["label"]]) > 200:
            mem[obj["label"]] = mem[obj["label"]][-200:]


def check_alerts(objects) -> list[str]:
    alerts         = app_state["object_alerts"]
    cooldown       = app_state["alert_cooldown"]
    visible        = app_state["currently_visible"]
    now            = time.time()
    triggered      = []
    current_labels = {o["label"].lower() for o in objects}

    for name, atype in alerts.items():
        if now - cooldown.get(name, 0) < 5:
            continue
        matched = any(name in lbl for lbl in current_labels)
        if atype == "appear" and matched and name not in visible:
            triggered.append(f"Alert! {name.title()} detected!")
            cooldown[name] = now
            visible.add(name)
        elif atype == "disappear" and not matched and name in visible:
            triggered.append(f"Alert! {name.title()} has disappeared!")
            cooldown[name] = now
            visible.discard(name)
        if not matched and name in visible and atype == "appear":
            visible.discard(name)

    return triggered


def build_scene_description(objects, faces) -> str:
    if not objects and not faces:
        return "I don't see anything recognizable right now."
    parts = []
    for obj in objects:
        parts.append(f"a {obj['label']} {obj['distance']} on the {obj['position']}")
    if faces:
        count = len(faces)
        parts.append(f"{'a human face' if count == 1 else f'{count} human faces'}")
    return "I can see: " + ", ".join(parts) + "."


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status":      "ok",
        "yolo_loaded": app_state.get("net") is not None,
        "classes":     len(app_state.get("classes", [])),
    }


@app.post("/analyze")
async def analyze_frame(
    image: UploadFile = File(...),
    mode:  str        = Form("describe"),
):
    """
    Accepts a JPEG/PNG frame, runs YOLO + face detection.
    Returns detected objects + scene description + any triggered alerts.
    """
    raw   = await image.read()
    frame = decode_image(raw)
    if frame is None:
        raise HTTPException(400, "Could not decode image")

    objects     = run_yolo(frame)
    faces       = detect_faces(frame)
    update_memory(objects + [{"label": "face"} for _ in faces])
    alert_msgs  = check_alerts(objects)
    description = build_scene_description(objects, faces)

    return {
        "description": description,
        "objects":     objects,
        "faces":       len(faces),
        "alerts":      alert_msgs,
        "timestamp":   datetime.now().isoformat(),
    }


@app.post("/command")
async def process_command(body: dict):
    """
    Process a voice command string from the app.
    Returns a text response the app should speak aloud.
    """
    command = body.get("command", "").lower().strip()
    mem     = app_state["scene_memory"]
    alerts  = app_state["object_alerts"]

    # ── Memory queries ─────────────────────────────────────────────────────────
    if "when did you see" in command or "when did i last see" in command:
        target = command.replace("when did you see", "").replace("when did i last see", "").strip()
        for label, timestamps in mem.items():
            if target in label.lower():
                if timestamps:
                    last = datetime.fromisoformat(timestamps[-1])
                    secs = (datetime.now() - last).total_seconds()
                    if secs < 60:
                        t = f"{int(secs)} seconds ago"
                    elif secs < 3600:
                        t = f"{int(secs/60)} minutes ago"
                    else:
                        t = f"{int(secs/3600)} hours ago"
                    return {"response": f"I last saw {label} {t}."}
        return {"response": f"I haven't seen {target} in this session."}

    if "history" in command:
        if not mem:
            return {"response": "No objects recorded yet."}
        recent = list(mem.keys())[-10:]
        return {"response": "Recently seen: " + ", ".join(recent) + "."}

    if "statistics" in command or "stats" in command:
        if not mem:
            return {"response": "No data yet."}
        sorted_objs = sorted(mem.items(), key=lambda x: len(x[1]), reverse=True)[:5]
        parts = [f"{k} ({len(v)} times)" for k, v in sorted_objs]
        return {"response": "Most seen objects: " + ", ".join(parts) + "."}

    # ── Alert management ────────────────────────────────────────────────────────
    if "alert me when" in command and "appears" in command:
        obj = command.replace("alert me when", "").replace("appears", "").strip()
        alerts[obj] = "appear"
        return {"response": f"Alert set. I'll notify you when {obj} appears."}

    if "alert me if" in command and "disappears" in command:
        obj = command.replace("alert me if", "").replace("disappears", "").strip()
        alerts[obj] = "disappear"
        return {"response": f"Alert set. I'll notify you if {obj} disappears."}

    if "list alerts" in command:
        if not alerts:
            return {"response": "No active alerts."}
        parts = [f"{k} ({v})" for k, v in alerts.items()]
        return {"response": "Active alerts: " + ", ".join(parts) + "."}

    if "clear alerts" in command:
        alerts.clear()
        return {"response": "All alerts cleared."}

    if "remove alert" in command:
        obj = command.replace("remove alert for", "").replace("remove alert", "").strip()
        if obj in alerts:
            del alerts[obj]
            return {"response": f"Alert for {obj} removed."}
        return {"response": f"No alert found for {obj}."}

    # ── Settings ────────────────────────────────────────────────────────────────
    if "status" in command:
        return {"response": f"Backend running. YOLO loaded: {app_state['net'] is not None}. Alerts: {len(alerts)}."}

    if "help" in command:
        return {"response": (
            "You can say: describe, read text, when did you see [object], "
            "history, statistics, alert me when [object] appears, "
            "alert me if [object] disappears, list alerts, clear alerts, status."
        )}

    return {"response": "I didn't understand that command. Say 'help' for a list of commands."}


@app.post("/ocr")
async def ocr_frame(image: UploadFile = File(...)):
    """Extract text from a frame using Tesseract."""
    if not TESSERACT_OK:
        return {"text": None, "error": "Tesseract not installed on server."}
    raw   = await image.read()
    frame = decode_image(raw)
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    try:
        text = pytesseract.image_to_string(gray, config="--oem 3 --psm 6").strip()
        return {"text": text or None}
    except Exception as e:
        return {"text": None, "error": str(e)}


@app.get("/memory")
async def get_memory():
    """Return the full scene memory for this session."""
    mem = app_state["scene_memory"]
    summary = {
        label: {
            "count":      len(timestamps),
            "last_seen":  timestamps[-1] if timestamps else None,
            "first_seen": timestamps[0]  if timestamps else None,
        }
        for label, timestamps in mem.items()
    }
    return {"memory": summary}


@app.delete("/memory")
async def clear_memory():
    app_state["scene_memory"].clear()
    return {"status": "memory cleared"}