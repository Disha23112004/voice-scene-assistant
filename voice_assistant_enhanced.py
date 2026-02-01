"""
Enhanced Voice-Controlled Scene Assistant - ADVANCED VERSION
Features: Object Tracking, Depth Estimation, OCR, Scene Memory, Object Alerts, Custom Training
"""

import cv2
import numpy as np
import requests
import base64
from gtts import gTTS
import threading
import queue
import speech_recognition as sr
import tempfile
import time
import os
import json
import pickle
from collections import defaultdict, deque
from datetime import datetime, timedelta
import pytesseract

# Configure Tesseract OCR path for Windows
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\pdish\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

# Try to import pygame for audio
try:
    import pygame
    USE_PYGAME = True
except ImportError:
    USE_PYGAME = False

print(f"Audio method: {'pygame' if USE_PYGAME else 'system default'}")


class SceneMemory:
    """Remember and query past scenes"""
    
    def __init__(self, max_history=1000):
        self.history = deque(maxlen=max_history)
        self.object_log = defaultdict(list)  # object_name: [timestamps]
        self.object_first_seen = {}
        self.object_last_seen = {}
        
        print(" Scene Memory initialized")
    
    def add_scene(self, objects, timestamp=None):
        """Record a scene"""
        if timestamp is None:
            timestamp = datetime.now()
        
        # Add to history
        scene = {
            'timestamp': timestamp,
            'objects': objects.copy()
        }
        self.history.append(scene)
        
        # Update object logs
        for obj in objects:
            label = obj['label']
            self.object_log[label].append(timestamp)
            
            # Track first and last seen
            if label not in self.object_first_seen:
                self.object_first_seen[label] = timestamp
            self.object_last_seen[label] = timestamp
    
    def query_object(self, object_name):
        """Query when an object was last seen - handles custom objects"""
        object_name = object_name.lower()
        
        # Remove common words
        object_name = object_name.replace('my ', '').replace('the ', '').strip()
        
        for label, timestamps in self.object_log.items():
            label_lower = label.lower()
            
            # Remove " (custom)" suffix for comparison
            label_clean = label_lower.replace(' (custom)', '')
            
            # Check if it's a match (either exact or partial)
            if object_name in label_clean or label_clean in object_name:
                if timestamps:
                    last = timestamps[-1]
                    first = timestamps[0]
                    count = len(timestamps)
                    
                    time_ago = (datetime.now() - last).total_seconds()
                    
                    if time_ago < 60:
                        time_str = f"{int(time_ago)} seconds ago"
                    elif time_ago < 3600:
                        time_str = f"{int(time_ago / 60)} minutes ago"
                    else:
                        time_str = f"{int(time_ago / 3600)} hours ago"
                    
                    return {
                        'found': True,
                        'label': label,
                        'last_seen': last,
                        'time_ago': time_str,
                        'count': count,
                        'first_seen': first
                    }
        
        return {'found': False}
    
    def get_recent_objects(self, minutes=5):
        """Get objects seen in the last N minutes"""
        cutoff = datetime.now() - timedelta(minutes=minutes)
        recent = set()
        
        for scene in reversed(self.history):
            if scene['timestamp'] < cutoff:
                break
            for obj in scene['objects']:
                recent.add(obj['label'])
        
        return list(recent)
    
    def save_to_file(self, filename='scene_memory.pkl'):
        """Save memory to disk"""
        data = {
            'history': list(self.history),
            'object_log': dict(self.object_log),
            'object_first_seen': self.object_first_seen,
            'object_last_seen': self.object_last_seen
        }
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        print(f" Memory saved to {filename}")
    
    def load_from_file(self, filename='scene_memory.pkl'):
        """Load memory from disk"""
        try:
            with open(filename, 'rb') as f:
                data = pickle.load(f)
            self.history = deque(data['history'], maxlen=1000)
            self.object_log = defaultdict(list, data['object_log'])
            self.object_first_seen = data['object_first_seen']
            self.object_last_seen = data['object_last_seen']
            print(f" Memory loaded from {filename}")
        except FileNotFoundError:
            print(" No saved memory found")


class ObjectAlerts:
    """Alert system for specific objects"""
    
    def __init__(self, callback):
        self.alerts = {}  # object_name: {'type': 'appear/disappear', 'active': True}
        self.callback = callback  # Function to call when alert triggers
        self.currently_visible = set()
        self.alert_cooldown = {}  # Prevent spam
        self.cooldown_duration = 5  # seconds
        
        print(" Object Alerts initialized")
    
    def add_alert(self, object_name, alert_type='appear'):
        """Add alert for an object"""
        object_name = object_name.lower()
        self.alerts[object_name] = {
            'type': alert_type,
            'active': True,
            'triggered_count': 0
        }
        print(f" Alert set: {object_name} ({alert_type})")
    
    def remove_alert(self, object_name):
        """Remove alert"""
        object_name = object_name.lower()
        if object_name in self.alerts:
            del self.alerts[object_name]
            print(f" Alert removed: {object_name}")
    
    def check_alerts(self, detected_objects):
        """Check if any alerts should trigger"""
        current_time = time.time()
        current_labels = set(obj['label'].lower() for obj in detected_objects)
        
        for alert_name, alert_info in self.alerts.items():
            if not alert_info['active']:
                continue
            
            # Check cooldown
            if alert_name in self.alert_cooldown:
                if current_time - self.alert_cooldown[alert_name] < self.cooldown_duration:
                    continue
            
            # Check for matches
            for label in current_labels:
                if alert_name in label:
                    # APPEAR alert
                    if alert_info['type'] == 'appear' and alert_name not in self.currently_visible:
                        message = f"Alert! {label.title()} detected!"
                        self.callback(message)
                        self.alert_cooldown[alert_name] = current_time
                        alert_info['triggered_count'] += 1
                        print(f" {message}")
                    
                    self.currently_visible.add(alert_name)
                    break
            else:
                # DISAPPEAR alert
                if alert_info['type'] == 'disappear' and alert_name in self.currently_visible:
                    message = f"Alert! {alert_name.title()} disappeared!"
                    self.callback(message)
                    self.alert_cooldown[alert_name] = current_time
                    alert_info['triggered_count'] += 1
                    print(f" {message}")
                
                if alert_name in self.currently_visible:
                    self.currently_visible.discard(alert_name)
    
    def list_alerts(self):
        """List all active alerts"""
        return [(name, info['type'], info['triggered_count']) 
                for name, info in self.alerts.items() if info['active']]
    
    def toggle_alert(self, object_name, active):
        """Enable or disable an alert"""
        object_name = object_name.lower()
        if object_name in self.alerts:
            self.alerts[object_name]['active'] = active


class CustomObjectTrainer:
    """Train custom objects for recognition - IMPROVED VERSION"""
    
    def __init__(self):
        self.custom_objects = {}  # name: {'features': [], 'histograms': [], 'sizes': []}
        self.feature_extractor = cv2.ORB_create(nfeatures=1000)  # More features
        self.similarity_threshold = 0.35  # Lower threshold for better matching
        
        print(" Custom Object Trainer initialized")
    
    def train_object(self, name, image, bbox=None):
        """Train on a new custom object"""
        name = name.lower()
        
        # Extract region if bbox provided
        if bbox:
            x, y, w, h = bbox
            # Add padding to capture more context
            padding = 10
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(image.shape[1] - x, w + 2*padding)
            h = min(image.shape[0] - y, h + 2*padding)
            roi = image[y:y+h, x:x+w]
        else:
            roi = image
        
        if roi.size == 0:
            print(f" Invalid ROI for {name}")
            return False
        
        # Store original size for comparison
        original_size = (roi.shape[1], roi.shape[0])
        
        # Resize for consistency
        roi_resized = cv2.resize(roi, (256, 256))  # Larger size for better features
        
        # Extract ORB features
        gray = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE for better feature detection
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        
        keypoints, descriptors = self.feature_extractor.detectAndCompute(gray, None)
        
        if descriptors is None or len(descriptors) < 20:
            print(f" Could not extract enough features from {name}")
            return False
        
        # Calculate color histogram (BGR and HSV)
        hist_bgr = []
        for i in range(3):
            hist = cv2.calcHist([roi_resized], [i], None, [32], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            hist_bgr.extend(hist)
        
        # HSV histogram for color
        hsv = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2HSV)
        hist_hsv = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
        hist_hsv = cv2.normalize(hist_hsv, hist_hsv).flatten()
        
        combined_hist = np.concatenate([hist_bgr, hist_hsv])
        
        # Store
        if name not in self.custom_objects:
            self.custom_objects[name] = {
                'features': [],
                'histograms': [],
                'sizes': [],
                'images': [],
                'count': 0
            }
        
        self.custom_objects[name]['features'].append(descriptors)
        self.custom_objects[name]['histograms'].append(combined_hist)
        self.custom_objects[name]['sizes'].append(original_size)
        self.custom_objects[name]['images'].append(roi_resized)
        self.custom_objects[name]['count'] += 1
        
        print(f" Trained on {name} (sample #{self.custom_objects[name]['count']}, {len(descriptors)} features)")
        return True
    
    def recognize_custom_objects(self, frame, detected_objects):
        """Try to recognize custom objects in the frame - IMPROVED"""
        if not self.custom_objects:
            return []
        
        custom_detections = []
        
        # Also scan the whole frame for custom objects (not just detected objects)
        all_regions_to_check = []
        
        # Check detected objects
        for obj in detected_objects:
            all_regions_to_check.append(('detected', obj))
        
        # Sliding window for undetected custom objects (optional - can be slow)
        # Skip this for now to maintain performance
        
        for region_type, obj in all_regions_to_check:
            x, y, w, h = obj['box']
            
            # Add padding
            padding = 10
            x_pad = max(0, x - padding)
            y_pad = max(0, y - padding)
            w_pad = min(frame.shape[1] - x_pad, w + 2*padding)
            h_pad = min(frame.shape[0] - y_pad, h + 2*padding)
            
            roi = frame[y_pad:y_pad+h_pad, x_pad:x_pad+w_pad]
            
            if roi.size == 0:
                continue
            
            roi_resized = cv2.resize(roi, (256, 256))
            gray = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2GRAY)
            
            # Apply CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            gray = clahe.apply(gray)
            
            keypoints, descriptors = self.feature_extractor.detectAndCompute(gray, None)
            
            if descriptors is None or len(descriptors) < 20:
                continue
            
            # Calculate histograms
            hist_bgr = []
            for i in range(3):
                hist = cv2.calcHist([roi_resized], [i], None, [32], [0, 256])
                hist = cv2.normalize(hist, hist).flatten()
                hist_bgr.extend(hist)
            
            hsv = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2HSV)
            hist_hsv = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
            hist_hsv = cv2.normalize(hist_hsv, hist_hsv).flatten()
            
            combined_hist = np.concatenate([hist_bgr, hist_hsv])
            
            # Match against all custom objects
            best_match = None
            best_score = 0
            
            for custom_name, custom_data in self.custom_objects.items():
                total_feature_score = 0
                total_hist_score = 0
                num_samples = len(custom_data['features'])
                
                # Compare with all training samples
                for i in range(num_samples):
                    stored_descriptors = custom_data['features'][i]
                    stored_histogram = custom_data['histograms'][i]
                    
                    # Feature matching with FLANN for speed
                    try:
                        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
                        matches = bf.knnMatch(descriptors, stored_descriptors, k=2)
                        
                        # Apply ratio test (Lowe's ratio test)
                        good_matches = []
                        for match_pair in matches:
                            if len(match_pair) == 2:
                                m, n = match_pair
                                if m.distance < 0.75 * n.distance:
                                    good_matches.append(m)
                        
                        if len(good_matches) > 10:  # Need minimum matches
                            # Score based on good matches
                            feature_score = len(good_matches) / min(len(descriptors), len(stored_descriptors))
                            total_feature_score += feature_score
                        
                    except Exception as e:
                        continue
                    
                    # Histogram comparison
                    hist_score = cv2.compareHist(combined_hist, stored_histogram, cv2.HISTCMP_CORREL)
                    if hist_score > 0:
                        total_hist_score += hist_score
                
                # Average scores across all samples
                if num_samples > 0:
                    avg_feature_score = total_feature_score / num_samples
                    avg_hist_score = total_hist_score / num_samples
                    
                    # Combined score (70% features, 30% histogram)
                    combined_score = (avg_feature_score * 0.7) + (avg_hist_score * 0.3)
                    
                    if combined_score > best_score and combined_score > self.similarity_threshold:
                        best_score = combined_score
                        best_match = custom_name
            
            if best_match:
                # Check if this detection overlaps with existing custom detection
                is_duplicate = False
                for existing in custom_detections:
                    ex, ey, ew, eh = existing['box']
                    # Calculate IoU (Intersection over Union)
                    x1 = max(x, ex)
                    y1 = max(y, ey)
                    x2 = min(x + w, ex + ew)
                    y2 = min(y + h, ey + eh)
                    
                    if x1 < x2 and y1 < y2:
                        intersection = (x2 - x1) * (y2 - y1)
                        union = (w * h) + (ew * eh) - intersection
                        iou = intersection / union if union > 0 else 0
                        
                        if iou > 0.5:  # 50% overlap
                            is_duplicate = True
                            break
                
                if not is_duplicate:
                    custom_detections.append({
                        'label': f"{best_match} (custom)",
                        'confidence': best_score,
                        'box': (x, y, w, h),
                        'is_custom': True
                    })
                    print(f" Recognized custom object: {best_match} (confidence: {best_score:.2f})")
        
        return custom_detections
    
    def list_custom_objects(self):
        """List all trained custom objects"""
        return [(name, data['count']) for name, data in self.custom_objects.items()]
    
    def save_training(self, filename='custom_objects.pkl'):
        """Save trained objects"""
        try:
            with open(filename, 'wb') as f:
                pickle.dump(self.custom_objects, f)
            print(f" Custom training saved to {filename}")
        except Exception as e:
            print(f" Could not save training: {e}")
    
    def load_training(self, filename='custom_objects.pkl'):
        """Load trained objects with backward compatibility"""
        try:
            if os.path.exists(filename):
                with open(filename, 'rb') as f:
                    loaded_data = pickle.load(f)
                
                # Check if old format (missing histograms/sizes)
                needs_migration = False
                for name, data in loaded_data.items():
                    if 'histograms' not in data or 'sizes' not in data:
                        needs_migration = True
                        break
                
                if needs_migration:
                    print(f" Old training format detected. Please retrain your objects for best results.")
                    print(f"  Run: delete custom_objects.pkl and train again")
                    # Skip loading old incompatible data
                    self.custom_objects = {}
                else:
                    self.custom_objects = loaded_data
                    print(f" Custom training loaded from {filename} ({len(self.custom_objects)} objects)")
        except Exception as e:
            print(f"Could not load training: {e}")
            print(f"  Delete custom_objects.pkl and train again")


class ObjectTracker:
    """Track objects across frames using centroid tracking"""
    
    def __init__(self, max_disappeared=30):
        self.next_object_id = 0
        self.objects = {}
        self.disappeared = {}
        self.object_history = defaultdict(lambda: deque(maxlen=30))
        self.max_disappeared = max_disappeared
    
    def register(self, centroid, label, distance):
        """Register a new object"""
        self.objects[self.next_object_id] = {
            'centroid': centroid,
            'label': label,
            'distance': distance,
            'first_seen': time.time()
        }
        self.disappeared[self.next_object_id] = 0
        self.object_history[self.next_object_id].append(centroid)
        self.next_object_id += 1
    
    def deregister(self, object_id):
        """Remove an object"""
        del self.objects[object_id]
        del self.disappeared[object_id]
        if object_id in self.object_history:
            del self.object_history[object_id]
    
    def update(self, detections):
        """Update tracked objects with new detections"""
        if len(detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects
        
        if len(self.objects) == 0:
            for detection in detections:
                self.register(detection['centroid'], detection['label'], detection['distance'])
        else:
            object_ids = list(self.objects.keys())
            object_centroids = [self.objects[oid]['centroid'] for oid in object_ids]
            detection_centroids = [d['centroid'] for d in detections]
            
            D = np.zeros((len(object_centroids), len(detection_centroids)))
            for i, obj_centroid in enumerate(object_centroids):
                for j, det_centroid in enumerate(detection_centroids):
                    D[i, j] = np.linalg.norm(np.array(obj_centroid) - np.array(det_centroid))
            
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            
            used_rows = set()
            used_cols = set()
            
            for row, col in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                
                if D[row, col] > 100:
                    continue
                
                object_id = object_ids[row]
                self.objects[object_id]['centroid'] = detections[col]['centroid']
                self.objects[object_id]['label'] = detections[col]['label']
                self.objects[object_id]['distance'] = detections[col]['distance']
                self.disappeared[object_id] = 0
                self.object_history[object_id].append(detections[col]['centroid'])
                
                used_rows.add(row)
                used_cols.add(col)
            
            unused_rows = set(range(D.shape[0])) - used_rows
            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            
            unused_cols = set(range(D.shape[1])) - used_cols
            for col in unused_cols:
                self.register(detections[col]['centroid'], detections[col]['label'], detections[col]['distance'])
        
        return self.objects
    
    def get_movement(self, object_id):
        """Get movement direction and speed for an object"""
        if object_id not in self.object_history:
            return None
        
        history = list(self.object_history[object_id])
        if len(history) < 5:
            return "stationary"
        
        start = np.array(history[0])
        end = np.array(history[-1])
        movement = end - start
        distance = np.linalg.norm(movement)
        
        if distance < 20:
            return "stationary"
        
        angle = np.arctan2(movement[1], movement[0]) * 180 / np.pi
        
        if -45 <= angle < 45:
            direction = "moving right"
        elif 45 <= angle < 135:
            direction = "moving down"
        elif -135 <= angle < -45:
            direction = "moving up"
        else:
            direction = "moving left"
        
        speed = "slowly" if distance < 50 else "quickly"
        
        return f"{direction} {speed}"


class DepthEstimator:
    """Estimate depth using monocular cues"""
    
    def __init__(self):
        self.focal_length = 800
        self.known_widths = {
            'person': 0.5,
            'car': 1.8,
            'bicycle': 0.6,
            'chair': 0.5,
            'laptop': 0.35,
            'cell phone': 0.07,
            'book': 0.15,
            'bottle': 0.08,
            'cup': 0.08,
            'tv': 1.0,
            'keyboard': 0.4,
            'mouse': 0.1
        }
    
    def estimate_distance(self, label, pixel_width, frame_width):
        """Estimate distance in meters using object width"""
        if label.lower() not in self.known_widths:
            return None
        
        known_width = self.known_widths[label.lower()]
        
        if pixel_width > 0:
            distance = (known_width * self.focal_length) / pixel_width
            return distance
        
        return None
    
    def get_distance_description(self, distance_meters):
        """Convert distance to human-readable description"""
        if distance_meters is None:
            return "unknown distance"
        
        if distance_meters < 0.5:
            return f"very close, about {int(distance_meters * 100)} centimeters away"
        elif distance_meters < 1.5:
            return f"close, about {distance_meters:.1f} meters away"
        elif distance_meters < 3.0:
            return f"medium distance, about {distance_meters:.1f} meters away"
        elif distance_meters < 6.0:
            return f"far, about {int(distance_meters)} meters away"
        else:
            return f"very far, about {int(distance_meters)} meters away"


class HybridOCR:
    """Hybrid OCR using both Tesseract (offline) and OCR.space (online)"""
    
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.api_url = "https://api.ocr.space/parse/image"
        
        try:
            pytesseract.get_tesseract_version()
            self.tesseract_available = True
            print(" Tesseract OCR (offline) available")
        except:
            self.tesseract_available = False
            print(" Tesseract not found")
        
        if api_key:
            print(" OCR.space API (online) configured")
    
    def read_text_tesseract(self, frame):
        """Read text using Tesseract (offline)"""
        if not self.tesseract_available:
            return None, 0
        
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            gray = cv2.medianBlur(gray, 3)
            
            configs = [
                '--oem 3 --psm 6',
                '--oem 3 --psm 11',
                '--oem 3 --psm 3',
            ]
            
            best_text = ""
            best_conf = 0
            
            for config in configs:
                data = pytesseract.image_to_data(gray, config=config, output_type=pytesseract.Output.DICT)
                
                text_parts = []
                confidences = []
                
                for i, conf in enumerate(data['conf']):
                    if conf > 30 and data['text'][i].strip():
                        text_parts.append(data['text'][i])
                        confidences.append(conf)
                
                if confidences:
                    avg_conf = sum(confidences) / len(confidences)
                    text = ' '.join(text_parts)
                    
                    if avg_conf > best_conf and len(text) > len(best_text):
                        best_text = text
                        best_conf = avg_conf
            
            if best_text:
                return best_text, best_conf
            
            return None, 0
            
        except Exception as e:
            print(f"Tesseract error: {e}")
            return None, 0
    
    def read_text_ocrspace(self, frame):
        """Read text using OCR.space API (online)"""
        if not self.api_key:
            return None, 0
        
        try:
            success, encoded = cv2.imencode('.jpg', frame)
            if not success:
                return None, 0
            
            image_base64 = base64.b64encode(encoded).decode('utf-8')
            
            payload = {
                'apikey': self.api_key,
                'base64Image': f'data:image/jpeg;base64,{image_base64}',
                'language': 'eng',
                'isOverlayRequired': False,
                'detectOrientation': True,
                'scale': True,
                'OCREngine': 2,
            }
            
            response = requests.post(self.api_url, data=payload, timeout=10)
            result = response.json()
            
            if result.get('IsErroredOnProcessing'):
                return None, 0
            
            if result.get('ParsedResults'):
                text = result['ParsedResults'][0].get('ParsedText', '').strip()
                if text:
                    confidence = 85 if len(text) > 3 else 70
                    return text, confidence
            
            return None, 0
            
        except Exception as e:
            print(f"OCR.space error: {e}")
            return None, 0
    
    def read_text(self, frame, prefer_online=True):
        """Read text using best available method"""
        if prefer_online and self.api_key:
            text, conf = self.read_text_ocrspace(frame)
            if text and conf > 50:
                print(f"   OCR.space (online): {conf:.1f}% confidence")
                return text, conf
        
        if self.tesseract_available:
            text, conf = self.read_text_tesseract(frame)
            if text and conf > 30:
                print(f"   Tesseract (offline): {conf:.1f}% confidence")
                return text, conf
        
        return None, 0


class EnhancedVoiceSceneAssistant:
    def __init__(self, ocr_api_key=None):
        self.running = True
        self.last_frame = None
        self.last_result = None
        
        # Time-based throttling for memory updates
        self.last_memory_update = time.time()
        self.memory_update_interval = 2.0  # Save to memory every 2 seconds
        
        # Initialize new features
        print("Initializing Scene Memory...")
        self.scene_memory = SceneMemory()
        
        print("Initializing Object Alerts...")
        self.object_alerts = ObjectAlerts(callback=self.speak)
        
        print("Initializing Custom Object Trainer...")
        self.custom_trainer = CustomObjectTrainer()
        
        # Training mode
        self.training_mode = False
        self.training_object_name = None
        self.training_samples_collected = 0
        self.training_samples_needed = 5
        
        # Initialize trackers
        print("Initializing object tracker...")
        self.tracker = ObjectTracker(max_disappeared=20)
        
        print("Initializing depth estimator...")
        self.depth_estimator = DepthEstimator()
        
        print("Initializing hybrid OCR...")
        self.ocr = HybridOCR(ocr_api_key)
        
        # Initialize pygame
        if USE_PYGAME:
            pygame.mixer.init()
            print(" Using pygame for audio")
        
        print("Loading YOLO model...")
        self.net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
        
        with open("coco.names", "r") as f:
            self.classes = [line.strip() for line in f.readlines()]
        
        layer_names = self.net.getLayerNames()
        self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        
        # Face detection
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Audio queue
        self.audio_queue = queue.Queue()
        self.audio_thread = threading.Thread(target=self._audio_worker, daemon=True)
        self.audio_thread.start()
        
        # Speech recognition
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
        print("Calibrating microphone...")
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=2)
        print(" Microphone ready!")
        
        # Voice command queue
        self.command_queue = queue.Queue()
        self.voice_thread = threading.Thread(target=self._voice_worker, daemon=True)
        self.voice_thread.start()
        
        self.confidence_threshold = 0.5
        self.nms_threshold = 0.4
        
        # Settings
        self.tracking_enabled = True
        self.depth_enabled = True
        self.memory_enabled = True
        self.alerts_enabled = True
        
        # Load saved data
        self.scene_memory.load_from_file()
        self.custom_trainer.load_training()
        
        print(" Enhanced Voice Assistant with Advanced Features ready!")
        print("    Object tracking: ON")
        print("    Depth estimation: ON")
        print("    Scene memory: ON")
        print("    Object alerts: ON")
        print("    Custom learning: Available")
    
    def _audio_worker(self):
        """Background audio playback worker"""
        temp_dir = tempfile.gettempdir()
        
        while True:
            text = self.audio_queue.get()
            if text is None:
                break
            
            try:
                tts = gTTS(text=text, lang='en', slow=False)
                temp_file = os.path.join(temp_dir, f"scene_audio_{os.getpid()}.mp3")
                tts.save(temp_file)
                
                if USE_PYGAME:
                    pygame.mixer.music.load(temp_file)
                    pygame.mixer.music.play()
                    while pygame.mixer.music.get_busy():
                        time.sleep(0.1)
                    pygame.mixer.music.unload()
                else:
                    if os.name == 'nt':
                        import subprocess
                        cmd = f'powershell -c "(New-Object Media.SoundPlayer \'{temp_file}\').PlaySync()"'
                        subprocess.run(cmd, shell=True, capture_output=True)
                
                time.sleep(0.5)
                for _ in range(5):
                    try:
                        if os.path.exists(temp_file):
                            os.remove(temp_file)
                            break
                    except:
                        time.sleep(0.2)
                        
            except Exception as e:
                print(f" Audio error: {e}")
            
            self.audio_queue.task_done()
    
    def _voice_worker(self):
        """Background voice recognition worker"""
        while self.running:
            try:
                with self.microphone as source:
                    audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=3)
                
                try:
                    command = self.recognizer.recognize_google(audio).lower()
                    print(f" Heard: '{command}'")
                    self.command_queue.put(command)
                except sr.UnknownValueError:
                    pass
                except sr.RequestError as e:
                    print(f" Recognition error: {e}")
                    
            except sr.WaitTimeoutError:
                pass
            except Exception as e:
                if self.running:
                    print(f" Voice error: {e}")
    
    def speak(self, text):
        """Queue text for speech"""
        print(f" Speaking: {text}")
        self.audio_queue.put(text)
    
    def detect_objects(self, frame):
        """Detect objects using YOLO"""
        height, width, _ = frame.shape
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (320, 320), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)
        
        class_ids, confidences, boxes = [], [], []
        
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > self.confidence_threshold:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, self.nms_threshold)
        
        detected_objects = []
        detections_for_tracking = []
        
        if len(indexes) > 0:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                label = str(self.classes[class_ids[i]])
                
                center_x = x + w / 2
                center_y = y + h / 2
                position = "left" if center_x < width / 3 else "right" if center_x > 2 * width / 3 else "center"
                
                if self.depth_enabled:
                    distance_meters = self.depth_estimator.estimate_distance(label, w, width)
                    distance_desc = self.depth_estimator.get_distance_description(distance_meters)
                else:
                    size_ratio = (w * h) / (width * height)
                    distance_desc = "very close" if size_ratio > 0.3 else "close" if size_ratio > 0.15 else "medium distance" if size_ratio > 0.05 else "far away"
                    distance_meters = None
                
                detected_objects.append({
                    'label': label,
                    'position': position,
                    'distance': distance_desc,
                    'distance_meters': distance_meters,
                    'box': (x, y, w, h),
                    'centroid': (center_x, center_y)
                })
                
                detections_for_tracking.append({
                    'centroid': (center_x, center_y),
                    'label': label,
                    'distance': distance_desc
                })
        
        # Add custom object detections with all required fields
        if self.custom_trainer.custom_objects:
            custom_detections = self.custom_trainer.recognize_custom_objects(frame, detected_objects)
            
            # Add missing fields to custom detections
            for custom_obj in custom_detections:
                if 'distance' not in custom_obj:
                    custom_obj['distance'] = 'unknown distance'
                if 'distance_meters' not in custom_obj:
                    custom_obj['distance_meters'] = None
                if 'centroid' not in custom_obj:
                    x, y, w, h = custom_obj['box']
                    custom_obj['centroid'] = (x + w/2, y + h/2)
                if 'position' not in custom_obj:
                    x, y, w, h = custom_obj['box']
                    center_x = x + w / 2
                    custom_obj['position'] = "left" if center_x < width / 3 else "right" if center_x > 2 * width / 3 else "center"
            
            detected_objects.extend(custom_detections)
        
        if self.tracking_enabled:
            tracked = self.tracker.update(detections_for_tracking)
        else:
            tracked = {}
        
        return detected_objects, tracked
    
    def detect_faces(self, frame):
        """Detect faces"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        face_info = []
        width = frame.shape[1]
        
        for (x, y, w, h) in faces:
            center_x = x + w / 2
            position = "left" if center_x < width / 3 else "right" if center_x > 2 * width / 3 else "center"
            face_info.append({'position': position, 'box': (x, y, w, h)})
        
        return face_info
    
    def generate_description(self, objects, faces, tracked_objects):
        """Generate enhanced scene description with tracking"""
        parts = []
        
        if objects:
            sorted_objs = sorted(objects, key=lambda x: x.get('distance_meters', 999) if x.get('distance_meters') else 999)
            
            for obj in sorted_objs[:3]:
                desc = f"A {obj['label']} {obj['distance']} on the {obj['position']}"
                
                if self.tracking_enabled and tracked_objects:
                    for track_id, track_data in tracked_objects.items():
                        if track_data['label'] == obj['label']:
                            movement = self.tracker.get_movement(track_id)
                            if movement and movement != "stationary":
                                desc += f", {movement}"
                            break
                
                parts.append(desc)
        
        if faces:
            parts.append(f"{len(faces)} person{'s' if len(faces) > 1 else ''} detected")
        
        return ". ".join(parts) + "." if parts else "No objects detected"
    
    def process_frame(self, frame):
        """Process a frame"""
        objects, tracked = self.detect_objects(frame)
        faces = self.detect_faces(frame)
        description = self.generate_description(objects, faces, tracked)
        
        # Update scene memory (time-based throttling)
        if self.memory_enabled and objects:
            current_time = time.time()
            if current_time - self.last_memory_update >= self.memory_update_interval:
                self.scene_memory.add_scene(objects)
                self.last_memory_update = current_time
        
        # Check alerts
        if self.alerts_enabled:
            self.object_alerts.check_alerts(objects)
        
        # Training mode - wait for capture command (don't auto-train)
        # Training happens only when user says "capture"
        
        # Draw detections
        annotated = frame.copy()
        
        # Draw tracked objects with IDs
        if self.tracking_enabled:
            for track_id, track_data in tracked.items():
                cx, cy = track_data['centroid']
                cv2.circle(annotated, (int(cx), int(cy)), 4, (0, 255, 255), -1)
                cv2.putText(annotated, f"ID:{track_id}", (int(cx) - 20, int(cy) - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        # Draw object boxes
        for obj in objects:
            x, y, w, h = obj['box']
            
            # Different color for custom objects
            color = (255, 0, 255) if obj.get('is_custom') else (0, 255, 0)
            
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
            
            label_text = obj['label']
            if obj.get('distance_meters'):
                label_text += f" ({obj['distance_meters']:.1f}m)"
            
            cv2.putText(annotated, label_text, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw faces
        for face in faces:
            x, y, w, h = face['box']
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        # Training mode indicator
        if self.training_mode:
            cv2.putText(annotated, f"LEARNING: {self.training_object_name} ({self.training_samples_collected}/{self.training_samples_needed})",
                       (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return {
            'description': description,
            'annotated_frame': annotated,
            'objects': objects,
            'faces': faces,
            'tracked': tracked
        }
    
    def process_command(self, command, frame):
        """Process voice commands"""
        print(f"\n{'='*60}")
        print(f" COMMAND: '{command}'")
        print(f"{'='*60}")
        
        # Exit commands
        if any(w in command for w in ['stop', 'exit', 'quit']):
            print(" EXIT")
            self.scene_memory.save_to_file()
            self.custom_trainer.save_training()
            self.speak("Saving data and exiting. Goodbye")
            self.running = False
            return
        
        # Scene description
        if any(w in command for w in ['describe', 'what', 'see', 'look']):
            print(" DESCRIBE")
            result = self.process_frame(frame)
            print(f" {result['description']}")
            self.speak(result['description'])
            return
        
        # OCR
        if 'read' in command or 'text' in command:
            print(" READ TEXT")
            self.speak("Reading text, please hold steady")
            text, conf = self.ocr.read_text(frame, prefer_online=True)
            if text:
                print(f"Text found: {text}")
                self.speak(f"The text says: {text}")
            else:
                print(" No text detected")
                self.speak("No text found. Try moving closer or improving lighting")
            return
        
        # Memory queries
        if 'when' in command and 'see' in command:
            # "When did you see my phone?"
            words = command.split()
            for i, word in enumerate(words):
                if word in ['see', 'saw']:
                    if i + 1 < len(words):
                        object_name = ' '.join(words[i+1:])
                        print(f"   DEBUG: Looking for object: '{object_name}'")
                        
                        result = self.scene_memory.query_object(object_name)
                        
                        print(f"   DEBUG: Found: {result['found']}")
                        if result['found']:
                            print(f"   DEBUG: Label: '{result['label']}'")
                            print(f"   DEBUG: Time ago: {result['time_ago']}")
                            # Remove (custom) from label when speaking
                            clean_label = result['label'].replace(' (custom)', '')
                            self.speak(f"I last saw {clean_label} {result['time_ago']}")
                        else:
                            print(f"   DEBUG: Available objects: {list(self.scene_memory.object_log.keys())[:5]}")
                            self.speak(f"I haven't seen {object_name} recently")
            return
        
        if 'history' in command or 'recent' in command:
            recent = self.scene_memory.get_recent_objects(5)
            num_scenes = len(self.scene_memory.history)
            print(f"   DEBUG: Total scenes in memory: {num_scenes}")
            print(f"   DEBUG: Recent objects found: {len(recent)}")
            
            if recent:
                # Clean up labels for speaking
                cleaned = []
                for obj in recent[:10]:
                    # Remove (custom) suffix for cleaner speech
                    clean_name = obj.replace(' (custom)', '')
                    cleaned.append(clean_name)
                
                objects_str = ', '.join(cleaned)
                self.speak(f"In the last 5 minutes I've seen: {objects_str}")
            else:
                self.speak("No objects recorded in the last 5 minutes")
            return
        
        # Alert commands - FIXED AND IMPROVED
        if 'alert' in command or 'notify' in command:
            if 'list' in command or 'show' in command:
                alerts = self.object_alerts.list_alerts()
                if alerts:
                    alert_str = ', '.join([f"{name} ({atype})" for name, atype, _ in alerts])
                    self.speak(f"Active alerts: {alert_str}")
                else:
                    self.speak("No active alerts")
                return
            
            elif 'clear' in command or 'remove all' in command:
                self.object_alerts.alerts.clear()
                self.speak("All alerts cleared")
                return
            
            elif 'remove' in command and 'for' in command:
                # "remove alert for person"
                words = command.split('for')
                if len(words) > 1:
                    obj_name = words[1].strip()
                    self.object_alerts.remove_alert(obj_name)
                    self.speak(f"Alert removed for {obj_name}")
                return
            
            else:
                # Set new alert - WORKS WITH ANY OBJECT!
                words = command.split()
                
                # Find "when", "whenever", or "if"
                keyword_idx = -1
                for i, word in enumerate(words):
                    if word in ['when', 'whenever', 'if']:
                        keyword_idx = i
                        break
                
                if keyword_idx >= 0 and keyword_idx + 1 < len(words):
                    # Get words after keyword
                    remaining = words[keyword_idx + 1:]
                    
                    # Remove filler words
                    stop_words = {'a', 'an', 'the', 'my', 'your', 'appears', 'appear', 
                                 'disappears', 'disappear', 'is', 'are', 'goes', 'gone', 
                                 'leaves', 'shows', 'up', 'down'}
                    
                    object_words = [w for w in remaining if w.lower() not in stop_words]
                    
                    if object_words:
                        object_name = ' '.join(object_words)
                        
                        # Determine type
                        disappear_keywords = ['disappear', 'disappears', 'gone', 'leaves', 
                                            'missing', 'lost', 'vanishes']
                        alert_type = 'disappear' if any(kw in command for kw in disappear_keywords) else 'appear'
                        
                        self.object_alerts.add_alert(object_name, alert_type)
                        self.speak(f"Alert set for {object_name} to {alert_type}")
                        return
                
                # Couldn't parse
                self.speak("Say: alert me when, followed by object name")
                return
        
        # Custom training
        if 'learn' in command:
            if 'this is' in command or 'this as' in command:
                # "Learn this as my mug"
                # Extract the object name after "this is" or "this as"
                if 'this is' in command:
                    object_name = command.split('this is')[-1].strip()
                elif 'this as' in command:
                    object_name = command.split('this as')[-1].strip()
                else:
                    object_name = 'object'
                
                # Remove "learn" from the beginning if it's still there
                object_name = object_name.replace('learn', '').strip()
                
                if not object_name:
                    object_name = 'object'
                
                self.training_mode = True
                self.training_object_name = object_name
                self.training_samples_collected = 0
                self.speak(f"Learning mode activated for {object_name}. Hold the object and say capture to train, or say done when finished")
                return
            
            elif 'stop' in command or 'done' in command:
                self.training_mode = False
                if self.training_samples_collected > 0:
                    self.speak(f"Learning complete for {self.training_object_name} with {self.training_samples_collected} samples")
                    self.custom_trainer.save_training()
                else:
                    self.speak("Learning cancelled")
                self.training_object_name = None
                return
            
            elif 'list' in command:
                custom_objects = self.custom_trainer.list_custom_objects()
                if custom_objects:
                    obj_str = ', '.join([f"{name} ({count} samples)" for name, count in custom_objects])
                    self.speak(f"Learned objects: {obj_str}")
                else:
                    self.speak("No custom objects learned yet")
                return
        
        # Capture command during training
        if self.training_mode and ('capture' in command or 'sample' in command):
            if not self.training_object_name:
                self.speak("No object name set. Say learn this as, followed by the object name first")
                return
                
            result = self.process_frame(frame)
            objects = result['objects']
            
            if objects:
                # Get the object closest to center (what user is pointing at)
                frame_center_x = frame.shape[1] / 2
                frame_center_y = frame.shape[0] / 2
                
                def distance_to_center(obj):
                    cx, cy = obj['centroid']
                    return ((cx - frame_center_x)**2 + (cy - frame_center_y)**2)**0.5
                
                closest_to_center = min(objects, key=distance_to_center)
                
                if self.custom_trainer.train_object(self.training_object_name, frame, closest_to_center['box']):
                    self.training_samples_collected += 1
                    remaining = self.training_samples_needed - self.training_samples_collected
                    
                    if remaining > 0:
                        self.speak(f"Sample {self.training_samples_collected} captured. {remaining} more needed")
                    else:
                        self.training_mode = False
                        self.speak(f"Learning complete for {self.training_object_name}")
                        self.custom_trainer.save_training()
            else:
                self.speak("No object detected. Try again")
            return
        
        # Toggle features
        if 'track' in command:
            if 'off' in command or 'disable' in command or 'stop' in command:
                if self.tracking_enabled:
                    self.tracking_enabled = False
                    self.speak("Object tracking disabled")
                else:
                    self.speak("Object tracking is already off")
            else:
                if not self.tracking_enabled:
                    self.tracking_enabled = True
                    self.speak("Object tracking enabled")
                else:
                    self.speak("Object tracking is already on")
            return
        
        if 'depth' in command:
            if 'off' in command or 'disable' in command:
                if self.depth_enabled:
                    self.depth_enabled = False
                    self.speak("Depth estimation disabled")
                else:
                    self.speak("Depth estimation is already off")
            else:
                if not self.depth_enabled:
                    self.depth_enabled = True
                    self.speak("Depth estimation enabled")
                else:
                    self.speak("Depth estimation is already on")
            return
        
        if 'memory' in command:
            if 'off' in command or 'disable' in command:
                self.memory_enabled = False
                self.speak("Scene memory disabled")
            elif 'save' in command:
                # Manual save
                self.scene_memory.save_to_file()
                self.speak("Memory saved to disk")
            elif 'clear' in command or 'reset' in command:
                # Clear memory
                self.scene_memory = SceneMemory()
                self.speak("Memory cleared")
            else:
                self.memory_enabled = True
                self.speak("Scene memory enabled")
            return
        
        # Help
        if 'help' in command or 'commands' in command:
            self.speak("Voice commands: describe, read text, track on or off, when did you see, history, alert me when, learn this as, list learned objects, status, help, stop")
            return
        
        # Status
        if 'status' in command or 'settings' in command:
            track_status = "on" if self.tracking_enabled else "off"
            depth_status = "on" if self.depth_enabled else "off"
            memory_status = "on" if self.memory_enabled else "off"
            self.speak(f"Tracking is {track_status}, depth is {depth_status}, memory is {memory_status}")
            return
    
    def run(self):
        """Run the assistant"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        self.speak("Enhanced assistant with advanced features started. Say help for commands.")
        
        frame_count = 0
        
        while self.running:
            ret, frame = cap.read()
            if not ret:
                break
            
            self.last_frame = frame
            
            if frame_count % 3 == 0:
                result = self.process_frame(frame)
                self.last_result = result
            
            if self.last_result:
                # Add status text with color coding
                status_text = []
                status_colors = []
                
                if self.tracking_enabled:
                    tracked_count = len(self.last_result.get('tracked', {}))
                    status_text.append(f"TRACK:ON({tracked_count})")
                    status_colors.append((0, 255, 0))
                else:
                    status_text.append("TRACK:OFF")
                    status_colors.append((0, 0, 255))
                
                if self.depth_enabled:
                    status_text.append("DEPTH:ON")
                    status_colors.append((0, 255, 0))
                else:
                    status_text.append("DEPTH:OFF")
                    status_colors.append((0, 0, 255))
                
                if self.memory_enabled:
                    status_text.append("MEMORY:ON")
                    status_colors.append((0, 255, 0))
                else:
                    status_text.append("MEMORY:OFF")
                    status_colors.append((0, 0, 255))
                
                # Show alert count
                alert_count = len([a for a in self.object_alerts.alerts.values() if a['active']])
                if alert_count > 0:
                    status_text.append(f"ALERTS:{alert_count}")
                    status_colors.append((0, 255, 255))
                
                # Show custom object count
                custom_count = len(self.custom_trainer.custom_objects)
                if custom_count > 0:
                    status_text.append(f"CUSTOM:{custom_count}")
                    status_colors.append((255, 0, 255))
                
                display_frame = self.last_result['annotated_frame'].copy()
                
                # Draw status
                x_offset = 10
                for i, text in enumerate(status_text):
                    cv2.putText(display_frame, text, (x_offset, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_colors[i], 2)
                    text_width = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0][0]
                    x_offset += text_width + 15
                
                cv2.imshow('Enhanced Voice Assistant - Advanced', display_frame)
            
            # Process voice commands
            try:
                command = self.command_queue.get_nowait()
                self.process_command(command, frame)
            except queue.Empty:
                pass
            
            if cv2.waitKey(1) & 0xFF == 27:
                break
            
            frame_count += 1
        
        # Save on exit
        self.scene_memory.save_to_file()
        self.custom_trainer.save_training()
        
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    print("ENHANCED VOICE ASSISTANT")
    print("\n ADVANCED FEATURES:")
    print("   Scene Memory - Remembers what it has seen")
    print("   Object Alerts - Notifications for specific objects")
    print("   Custom Learning - Teach it new objects")
    print("   Object tracking & detection")
    print("   Depth/distance estimation")
    print("   Hybrid OCR (offline + online)")
    print("   Movement detection")
    print("   Voice control")
    API_KEY = "K82391425288957"
    
    print("\nStarting...\n")
    assistant = EnhancedVoiceSceneAssistant(API_KEY)
    assistant.run()