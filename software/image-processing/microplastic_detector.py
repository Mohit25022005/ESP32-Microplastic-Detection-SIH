"""
ESP32-CAM Microplastic Detection - Image Processing Module
Smart India Hackathon 2025 -Tech Enchante Team

Advanced OpenCV-based microplastic detection system with:
- Real-time image processing from ESP32-CAM
- Multi-stage particle detection pipeline
- Size distribution analysis
- Confidence scoring and validation
- Automated annotation and reporting
- MQTT/HTTP integration for IoT connectivity

Author: SIH AquaGuard Team
Version: 2.0
Date: September 2025
"""

import cv2
import numpy as np
import json
import time
import threading
import queue
import requests
import paho.mqtt.client as mqtt
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass, asdict
import base64
import io
from PIL import Image
import sqlite3
import os
from pathlib import Path
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('microplastic_detector.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ParticleData:
    """Data structure for individual particle detection"""
    id: int
    center_x: float
    center_y: float
    area: float
    perimeter: float
    equivalent_diameter: float
    aspect_ratio: float
    circularity: float
    solidity: float
    brightness_mean: float
    brightness_std: float
    confidence: float
    particle_type: str = "unknown"
    
@dataclass
class DetectionResult:
    """Complete detection result structure"""
    timestamp: str
    device_id: str
    image_path: str
    total_particles: int
    particles: List[ParticleData]
    size_distribution: Dict[str, int]
    image_quality_score: float
    processing_time_ms: float
    photodiode_voltage: float
    confidence_score: float
    anomaly_detected: bool

class MicroplasticDetector:
    """
    Advanced microplastic detection system using OpenCV
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the detection system with configuration"""
        self.config = self._load_config(config_path)
        self.setup_directories()
        self.setup_database()
        self.setup_mqtt()
        
        # Detection parameters
        self.min_particle_area = self.config['detection']['min_particle_area']
        self.max_particle_area = self.config['detection']['max_particle_area']
        self.blur_kernel_size = self.config['detection']['blur_kernel_size']
        self.threshold_value = self.config['detection']['threshold_value']
        self.morphology_kernel_size = self.config['detection']['morphology_kernel_size']
        
        # Size classification ranges (micrometers)
        self.size_ranges = {
            'nano': (0.001, 1.0),
            'micro_small': (1.0, 100.0),
            'micro_large': (100.0, 1000.0),
            'meso': (1000.0, 5000.0),
            'macro': (5000.0, float('inf'))
        }
        
        # Calibration parameters (pixels to micrometers)
        self.pixel_to_micron_ratio = self.config['calibration']['pixel_to_micron_ratio']
        
        # Processing queue for real-time operation
        self.image_queue = queue.Queue(maxsize=10)
        self.result_queue = queue.Queue(maxsize=50)
        
        # Statistics tracking
        self.total_processed = 0
        self.total_particles_detected = 0
        self.processing_start_time = time.time()
        
        logger.info("MicroplasticDetector initialized successfully")
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file"""
        default_config = {
            'detection': {
                'min_particle_area': 5,
                'max_particle_area': 10000,
                'blur_kernel_size': 5,
                'threshold_value': 127,
                'morphology_kernel_size': 3
            },
            'calibration': {
                'pixel_to_micron_ratio': 0.5  # Default: 1 pixel = 0.5 micrometers
            },
            'mqtt': {
                'broker': 'mqtt.microplastic.cloud',
                'port': 1883,
                'username': 'processor',
                'password': 'processor_2025',
                'topics': {
                    'images': 'sensors/microplastic/images',
                    'results': 'sensors/microplastic/results',
                    'status': 'processor/microplastic/status'
                }
            },
            'storage': {
                'images_dir': 'processed_images',
                'results_dir': 'detection_results',
                'database': 'microplastic_data.db'
            }
        }
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            # Merge with defaults
            return {**default_config, **config}
        else:
            # Create default config file
            with open(config_path, 'w') as f:
                yaml.dump(default_config, f, default_flow_style=False)
            logger.info(f"Created default config file: {config_path}")
            return default_config
    
    def setup_directories(self):
        """Create necessary directories"""
        for dir_name in ['images_dir', 'results_dir']:
            dir_path = Path(self.config['storage'][dir_name])
            dir_path.mkdir(parents=True, exist_ok=True)
        
        logger.info("Storage directories created")
    
    def setup_database(self):
        """Initialize SQLite database for results storage"""
        db_path = self.config['storage']['database']
        self.db_connection = sqlite3.connect(db_path, check_same_thread=False)
        
        # Create tables
        cursor = self.db_connection.cursor()
        
        # Detection results table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS detection_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                device_id TEXT NOT NULL,
                image_path TEXT NOT NULL,
                total_particles INTEGER NOT NULL,
                confidence_score REAL NOT NULL,
                processing_time_ms REAL NOT NULL,
                photodiode_voltage REAL,
                anomaly_detected BOOLEAN,
                size_distribution TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Individual particles table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS particles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                detection_id INTEGER NOT NULL,
                particle_id INTEGER NOT NULL,
                center_x REAL NOT NULL,
                center_y REAL NOT NULL,
                area REAL NOT NULL,
                equivalent_diameter REAL NOT NULL,
                aspect_ratio REAL NOT NULL,
                circularity REAL NOT NULL,
                confidence REAL NOT NULL,
                particle_type TEXT,
                FOREIGN KEY (detection_id) REFERENCES detection_results (id)
            )
        ''')
        
        self.db_connection.commit()
        logger.info("Database initialized successfully")
    
    def setup_mqtt(self):
        """Initialize MQTT client for IoT communication"""
        self.mqtt_client = mqtt.Client()
        self.mqtt_client.username_pw_set(
            self.config['mqtt']['username'],
            self.config['mqtt']['password']
        )
        
        # Set up callbacks
        self.mqtt_client.on_connect = self._on_mqtt_connect
        self.mqtt_client.on_message = self._on_mqtt_message
        self.mqtt_client.on_disconnect = self._on_mqtt_disconnect
        
        try:
            self.mqtt_client.connect(
                self.config['mqtt']['broker'],
                self.config['mqtt']['port'],
                60
            )
            self.mqtt_client.loop_start()
            logger.info("MQTT client connected successfully")
        except Exception as e:
            logger.error(f"Failed to connect to MQTT broker: {e}")
    
    def _on_mqtt_connect(self, client, userdata, flags, rc):
        """MQTT connection callback"""
        if rc == 0:
            logger.info("Connected to MQTT broker")
            # Subscribe to image topic
            client.subscribe(self.config['mqtt']['topics']['images'])
            # Publish status
            self._publish_status("online")
        else:
            logger.error(f"Failed to connect to MQTT broker: {rc}")
    
    def _on_mqtt_message(self, client, userdata, msg):
        """Handle incoming MQTT messages"""
        try:
            if msg.topic == self.config['mqtt']['topics']['images']:
                # Decode image data
                message_data = json.loads(msg.payload.decode())
                image_data = base64.b64decode(message_data['image_data'])
                
                # Convert to OpenCV format
                image_array = np.frombuffer(image_data, np.uint8)
                image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                
                if image is not None:
                    # Add to processing queue
                    metadata = {
                        'device_id': message_data.get('device_id', 'unknown'),
                        'timestamp': message_data.get('timestamp', datetime.now().isoformat()),
                        'photodiode_voltage': message_data.get('photodiode_voltage', 0.0)
                    }
                    
                    if not self.image_queue.full():
                        self.image_queue.put((image, metadata))
                        logger.info(f"Image queued for processing from {metadata['device_id']}")
                    else:
                        logger.warning("Image queue full, dropping frame")
                
        except Exception as e:
            logger.error(f"Error processing MQTT message: {e}")
    
    def _on_mqtt_disconnect(self, client, userdata, rc):
        """MQTT disconnection callback"""
        logger.warning(f"Disconnected from MQTT broker: {rc}")
    
    def _publish_status(self, status: str):
        """Publish processor status to MQTT"""
        status_data = {
            'status': status,
            'timestamp': datetime.now().isoformat(),
            'total_processed': self.total_processed,
            'total_particles': self.total_particles_detected,
            'uptime_seconds': int(time.time() - self.processing_start_time)
        }
        
        self.mqtt_client.publish(
            self.config['mqtt']['topics']['status'],
            json.dumps(status_data),
            retain=True
        )
    
    def _publish_results(self, result: DetectionResult):
        """Publish detection results to MQTT"""
        result_data = asdict(result)
        
        self.mqtt_client.publish(
            self.config['mqtt']['topics']['results'],
            json.dumps(result_data, default=str),
            qos=1
        )
    
    def preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Preprocess image for optimal particle detection
        Returns: (processed_image, quality_score)
        """
        start_time = time.time()
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Calculate image quality metrics
        quality_score = self._calculate_image_quality(gray)
        
        # Noise reduction with bilateral filter
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Enhance contrast using CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        
        # Gaussian blur for smoothing
        blurred = cv2.GaussianBlur(enhanced, 
                                 (self.blur_kernel_size, self.blur_kernel_size), 0)
        
        processing_time = (time.time() - start_time) * 1000
        logger.debug(f"Image preprocessing completed in {processing_time:.2f}ms")
        
        return blurred, quality_score
    
    def _calculate_image_quality(self, image: np.ndarray) -> float:
        """Calculate image quality score based on various metrics"""
        # Sharpness (Laplacian variance)
        laplacian_var = cv2.Laplacian(image, cv2.CV_64F).var()
        
        # Contrast (standard deviation)
        contrast = np.std(image)
        
        # Brightness distribution
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        hist_norm = hist.flatten() / hist.sum()
        entropy = -np.sum(hist_norm * np.log2(hist_norm + 1e-10))
        
        # Normalize and combine metrics
        quality_score = (
            min(laplacian_var / 1000, 1.0) * 0.4 +  # Sharpness weight
            min(contrast / 128, 1.0) * 0.3 +        # Contrast weight
            min(entropy / 8, 1.0) * 0.3             # Entropy weight
        )
        
        return min(quality_score, 1.0)
    
    def detect_particles(self, image: np.ndarray) -> Tuple[List[np.ndarray], List[Dict]]:
        """
        Detect particles using advanced computer vision techniques
        Returns: (contours, particle_properties)
        """
        # Adaptive thresholding for varying illumination
        binary = cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Morphological operations to clean up the binary image
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, 
            (self.morphology_kernel_size, self.morphology_kernel_size)
        )
        
        # Remove noise
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Fill gaps
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by size and shape
        valid_contours = []
        particle_properties = []
        
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            
            # Size filtering
            if self.min_particle_area <= area <= self.max_particle_area:
                # Calculate particle properties
                properties = self._calculate_particle_properties(contour, image, i)
                
                # Shape filtering (remove obvious non-particles)
                if self._is_valid_particle(properties):
                    valid_contours.append(contour)
                    particle_properties.append(properties)
        
        return valid_contours, particle_properties
    
    def _calculate_particle_properties(self, contour: np.ndarray, 
                                     image: np.ndarray, particle_id: int) -> Dict:
        """Calculate comprehensive properties for a detected particle"""
        # Basic geometric properties
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        # Moments for centroid
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
        else:
            cx, cy = 0, 0
        
        # Equivalent diameter
        equivalent_diameter = np.sqrt(4 * area / np.pi) * self.pixel_to_micron_ratio
        
        # Bounding rectangle for aspect ratio
        rect = cv2.boundingRect(contour)
        aspect_ratio = float(rect[3]) / rect[2] if rect[2] != 0 else 0
        
        # Circularity (4π*Area/Perimeter²)
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
        
        # Solidity (Area/ConvexHull Area)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        
        # Intensity properties
        mask = np.zeros(image.shape, dtype=np.uint8)
        cv2.fillPoly(mask, [contour], 255)
        masked_pixels = image[mask > 0]
        
        brightness_mean = np.mean(masked_pixels)
        brightness_std = np.std(masked_pixels)
        
        # Confidence calculation based on multiple factors
        confidence = self._calculate_particle_confidence(
            area, circularity, solidity, brightness_std
        )
        
        return {
            'id': particle_id,
            'center_x': cx,
            'center_y': cy,
            'area': area,
            'perimeter': perimeter,
            'equivalent_diameter': equivalent_diameter,
            'aspect_ratio': aspect_ratio,
            'circularity': circularity,
            'solidity': solidity,
            'brightness_mean': brightness_mean,
            'brightness_std': brightness_std,
            'confidence': confidence
        }
    
    def _calculate_particle_confidence(self, area: float, circularity: float, 
                                     solidity: float, brightness_std: float) -> float:
        """Calculate confidence score for particle detection"""
        # Size confidence (optimal around 50-500 pixels)
        size_conf = 1.0 - abs(np.log10(area) - np.log10(200)) / 2
        size_conf = max(0, min(1, size_conf))
        
        # Shape confidence (prefer circular to slightly elongated)
        shape_conf = circularity * 0.7 + solidity * 0.3
        
        # Texture confidence (some variation expected)
        texture_conf = min(brightness_std / 50, 1.0)
        
        # Combined confidence
        confidence = (size_conf * 0.4 + shape_conf * 0.4 + texture_conf * 0.2)
        
        return min(max(confidence, 0.0), 1.0)
    
    def _is_valid_particle(self, properties: Dict) -> bool:
        """Validate if detected object is likely a microplastic particle"""
        # Minimum confidence threshold
        if properties['confidence'] < 0.3:
            return False
        
        # Shape constraints
        if properties['circularity'] < 0.1:  # Too irregular
            return False
        
        if properties['aspect_ratio'] > 5.0:  # Too elongated
            return False
        
        if properties['solidity'] < 0.5:  # Too concave
            return False
        
        return True
    
    def classify_particles(self, particles: List[Dict]) -> Dict[str, int]:
        """Classify particles by size into different categories"""
        size_distribution = {category: 0 for category in self.size_ranges.keys()}
        
        for particle in particles:
            diameter = particle['equivalent_diameter']
            
            for category, (min_size, max_size) in self.size_ranges.items():
                if min_size <= diameter < max_size:
                    size_distribution[category] += 1
                    particle['particle_type'] = category
                    break
        
        return size_distribution
    
    def create_annotated_image(self, original_image: np.ndarray, 
                             contours: List[np.ndarray], 
                             particles: List[Dict]) -> np.ndarray:
        """Create annotated image with detection results"""
        # Create a copy for annotation
        annotated = original_image.copy()
        
        # Color scheme for different particle sizes
        colors = {
            'nano': (255, 0, 255),      # Magenta
            'micro_small': (0, 255, 255), # Cyan
            'micro_large': (0, 255, 0),   # Green
            'meso': (255, 255, 0),        # Yellow
            'macro': (255, 0, 0)          # Red
        }
        
        for contour, particle in zip(contours, particles):
            particle_type = particle.get('particle_type', 'unknown')
            color = colors.get(particle_type, (128, 128, 128))
            confidence = particle['confidence']
            
            # Draw contour with thickness based on confidence
            thickness = max(1, int(confidence * 3))
            cv2.drawContours(annotated, [contour], -1, color, thickness)
            
            # Add particle ID and size
            center = (int(particle['center_x']), int(particle['center_y']))
            diameter_text = f"ID:{particle['id']} ({particle['equivalent_diameter']:.1f}μm)"
            
            # Add text with background for readability
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.4
            text_thickness = 1
            
            (text_w, text_h), _ = cv2.getTextSize(diameter_text, font, font_scale, text_thickness)
            cv2.rectangle(annotated, 
                         (center[0] - text_w//2, center[1] - text_h - 5),
                         (center[0] + text_w//2, center[1] + 5),
                         (0, 0, 0), -1)
            
            cv2.putText(annotated, diameter_text, 
                       (center[0] - text_w//2, center[1] - 2),
                       font, font_scale, color, text_thickness)
        
        # Add summary information
        summary_text = [
            f"Total Particles: {len(particles)}",
            f"Processing Time: {time.time():.0f}s",
            f"Image Quality: {self._calculate_image_quality(cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)):.2f}"
        ]
        
        y_offset = 25
        for text in summary_text:
            cv2.putText(annotated, text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 25
        
        return annotated
    
    def process_image(self, image: np.ndarray, metadata: Dict) -> DetectionResult:
        """
        Complete image processing pipeline
        """
        start_time = time.time()
        
        # Preprocess image
        processed_image, quality_score = self.preprocess_image(image)
        
        # Detect particles
        contours, particle_properties = self.detect_particles(processed_image)
        
        # Convert properties to ParticleData objects
        particles = []
        for props in particle_properties:
            particle = ParticleData(**props)
            particles.append(particle)
        
        # Classify particles by size
        size_distribution = self.classify_particles(particle_properties)
        
        # Create annotated image
        annotated_image = self.create_annotated_image(image, contours, particle_properties)
        
        # Save annotated image
        timestamp = metadata.get('timestamp', datetime.now().isoformat())
        device_id = metadata.get('device_id', 'unknown')
        
        # Generate filename
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{device_id}_{timestamp_str}_detected.jpg"
        image_path = os.path.join(self.config['storage']['images_dir'], filename)
        
        cv2.imwrite(image_path, annotated_image)
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000
        
        # Calculate overall confidence
        if particles:
            confidence_score = np.mean([p.confidence for p in particles])
        else:
            confidence_score = 0.0
        
        # Anomaly detection (simple threshold-based)
        total_particles = len(particles)
        anomaly_detected = total_particles > 100 or confidence_score < 0.5
        
        # Create result object
        result = DetectionResult(
            timestamp=timestamp,
            device_id=device_id,
            image_path=image_path,
            total_particles=total_particles,
            particles=particles,
            size_distribution=size_distribution,
            image_quality_score=quality_score,
            processing_time_ms=processing_time,
            photodiode_voltage=metadata.get('photodiode_voltage', 0.0),
            confidence_score=confidence_score,
            anomaly_detected=anomaly_detected
        )
        
        # Store in database
        self._store_result(result)
        
        # Update statistics
        self.total_processed += 1
        self.total_particles_detected += total_particles
        
        logger.info(f"Processed image from {device_id}: {total_particles} particles detected in {processing_time:.2f}ms")
        
        return result
    
    def _store_result(self, result: DetectionResult):
        """Store detection result in database"""
        cursor = self.db_connection.cursor()
        
        # Insert detection result
        cursor.execute('''
            INSERT INTO detection_results 
            (timestamp, device_id, image_path, total_particles, confidence_score, 
             processing_time_ms, photodiode_voltage, anomaly_detected, size_distribution)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            result.timestamp, result.device_id, result.image_path,
            result.total_particles, result.confidence_score, result.processing_time_ms,
            result.photodiode_voltage, result.anomaly_detected,
            json.dumps(result.size_distribution)
        ))
        
        detection_id = cursor.lastrowid
        
        # Insert individual particles
        for particle in result.particles:
            cursor.execute('''
                INSERT INTO particles 
                (detection_id, particle_id, center_x, center_y, area, 
                 equivalent_diameter, aspect_ratio, circularity, confidence, particle_type)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                detection_id, particle.id, particle.center_x, particle.center_y,
                particle.area, particle.equivalent_diameter, particle.aspect_ratio,
                particle.circularity, particle.confidence, particle.particle_type
            ))
        
        self.db_connection.commit()
    
    def process_from_url(self, image_url: str, device_id: str = "http_client") -> DetectionResult:
        """Process image from HTTP URL"""
        try:
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()
            
            # Convert to OpenCV format
            image_array = np.frombuffer(response.content, np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            
            if image is None:
                raise ValueError("Failed to decode image from URL")
            
            metadata = {
                'device_id': device_id,
                'timestamp': datetime.now().isoformat(),
                'photodiode_voltage': 0.0
            }
            
            return self.process_image(image, metadata)
            
        except Exception as e:
            logger.error(f"Error processing image from URL {image_url}: {e}")
            raise
    
    def start_processing_loop(self):
        """Start the main processing loop for real-time operation"""
        logger.info("Starting image processing loop...")
        
        def processing_worker():
            while True:
                try:
                    # Get image from queue (blocking)
                    image, metadata = self.image_queue.get(timeout=1.0)
                    
                    # Process image
                    result = self.process_image(image, metadata)
                    
                    # Publish results
                    self._publish_results(result)
                    
                    # Add to result queue for API access
                    if not self.result_queue.full():
                        self.result_queue.put(result)
                    
                    # Publish status update every 10 processed images
                    if self.total_processed % 10 == 0:
                        self._publish_status("processing")
                    
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"Error in processing worker: {e}")
        
        # Start processing thread
        processing_thread = threading.Thread(target=processing_worker, daemon=True)
        processing_thread.start()
        
        logger.info("Processing loop started successfully")
        return processing_thread
    
    def get_statistics(self) -> Dict:
        """Get processing statistics"""
        uptime = time.time() - self.processing_start_time
        
        return {
            'total_processed': self.total_processed,
            'total_particles_detected': self.total_particles_detected,
            'uptime_seconds': uptime,
            'average_particles_per_image': self.total_particles_detected / max(self.total_processed, 1),
            'processing_rate_per_minute': (self.total_processed / uptime) * 60 if uptime > 0 else 0,
            'queue_size': self.image_queue.qsize(),
            'result_queue_size': self.result_queue.qsize()
        }
    
    def get_recent_results(self, limit: int = 10) -> List[DetectionResult]:
        """Get recent detection results"""
        results = []
        temp_queue = queue.Queue()
        
        # Extract results from queue without losing them
        while not self.result_queue.empty() and len(results) < limit:
            try:
                result = self.result_queue.get_nowait()
                results.append(result)
                temp_queue.put(result)
            except queue.Empty:
                break
        
        # Put results back
        while not temp_queue.empty():
            self.result_queue.put(temp_queue.get())
        
        return results[::-1]  # Return most recent first
    
    def calibrate_pixel_size(self, calibration_image: np.ndarray, 
                           known_object_size_microns: float) -> float:
        """
        Calibrate pixel-to-micron ratio using a known size object
        Returns the new pixel_to_micron_ratio
        """
        logger.info("Starting pixel size calibration...")
        
        # Process calibration image
        processed_image, _ = self.preprocess_image(calibration_image)
        contours, properties = self.detect_particles(processed_image)
        
        if not properties:
            raise ValueError("No objects detected in calibration image")
        
        # Find the largest object (assuming it's the calibration standard)
        largest_particle = max(properties, key=lambda p: p['area'])
        largest_diameter_pixels = np.sqrt(4 * largest_particle['area'] / np.pi)
        
        # Calculate new ratio
        new_ratio = known_object_size_microns / largest_diameter_pixels
        
        # Update configuration
        self.pixel_to_micron_ratio = new_ratio
        
        logger.info(f"Calibration complete. New pixel-to-micron ratio: {new_ratio:.4f}")
        
        return new_ratio
    
    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, 'mqtt_client'):
            self._publish_status("offline")
            self.mqtt_client.loop_stop()
            self.mqtt_client.disconnect()
        
        if hasattr(self, 'db_connection'):
            self.db_connection.close()

def main():
    """Main function for standalone operation"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Microplastic Detection System')
    parser.add_argument('--config', default='config.yaml', help='Configuration file path')
    parser.add_argument('--image', help='Process single image file')
    parser.add_argument('--url', help='Process image from URL')
    parser.add_argument('--loop', action='store_true', help='Start processing loop')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = MicroplasticDetector(args.config)
    
    if args.image:
        # Process single image file
        image = cv2.imread(args.image)
        if image is None:
            print(f"Error: Could not load image {args.image}")
            return
        
        metadata = {
            'device_id': 'file_input',
            'timestamp': datetime.now().isoformat(),
            'photodiode_voltage': 0.0
        }
        
        result = detector.process_image(image, metadata)
        print(f"Detection complete: {result.total_particles} particles detected")
        print(f"Size distribution: {result.size_distribution}")
        print(f"Confidence score: {result.confidence_score:.3f}")
        print(f"Annotated image saved to: {result.image_path}")
        
    elif args.url:
        # Process image from URL
        try:
            result = detector.process_from_url(args.url)
            print(f"Detection complete: {result.total_particles} particles detected")
            print(f"Size distribution: {result.size_distribution}")
        except Exception as e:
            print(f"Error processing image from URL: {e}")
    
    elif args.loop:
        # Start processing loop
        detector.start_processing_loop()
        
        try:
            while True:
                time.sleep(1)
                stats = detector.get_statistics()
                if stats['total_processed'] % 10 == 0 and stats['total_processed'] > 0:
                    print(f"Processed: {stats['total_processed']} images, "
                          f"Particles: {stats['total_particles_detected']}")
        except KeyboardInterrupt:
            print("\nShutting down...")
    
    else:
        print("Please specify --image, --url, or --loop")

if __name__ == "__main__":
    main()