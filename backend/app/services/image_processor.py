"""
OpenCV Image Processing Service for Microplastic Detection
Handles image analysis, particle detection, and morphological analysis
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
import base64
import io
from PIL import Image
from dataclasses import dataclass, asdict
from datetime import datetime
import json

logger = logging.getLogger(__name__)

@dataclass
class ParticleData:
    """Data structure for individual particle measurements"""
    id: str
    center_x: float
    center_y: float
    area: float
    perimeter: float
    circularity: float
    equivalent_diameter: float
    aspect_ratio: float
    solidity: float
    particle_type: str = "unknown"
    confidence: float = 0.0
    size_category: str = "unknown"

@dataclass
class ImageAnalysisResult:
    """Complete image analysis result"""
    timestamp: str
    device_id: str
    image_id: str
    total_particles: int
    particles: List[ParticleData]
    size_distribution: Dict[str, int]
    contamination_level: str
    confidence_score: float
    processing_time: float
    image_metadata: Dict

class MicroplasticImageProcessor:
    """
    Advanced OpenCV-based microplastic detection and analysis
    Optimized for ESP32-CAM integration with real-time processing
    """
    
    def __init__(self):
        # Detection parameters optimized for microplastics
        self.min_particle_area = 15      # pixels²
        self.max_particle_area = 8000    # pixels²
        self.min_circularity = 0.2       # 0-1 scale
        self.gaussian_kernel_size = 5    # noise reduction
        self.canny_threshold1 = 50       # edge detection
        self.canny_threshold2 = 150      # edge detection
        
        # Size categories (micrometers) - calibrated for ESP32-CAM
        self.size_categories = {
            "nano": (0, 1),      # < 1 μm
            "micro": (1, 100),   # 1-100 μm  
            "small": (100, 500), # 100-500 μm
            "large": (500, 1000) # 500-1000 μm
        }
        
        # Pixel to micrometer conversion (needs calibration per device)
        self.pixel_to_micron = 2.5  # default calibration value
        
        logger.info("🔬 Microplastic Image Processor initialized")
    
    def decode_base64_image(self, base64_string: str) -> Optional[np.ndarray]:
        """
        Decode base64 image string to OpenCV format
        
        Args:
            base64_string: Base64 encoded image
            
        Returns:
            OpenCV image array or None if failed
        """
        try:
            # Remove data URL prefix if present
            if base64_string.startswith('data:image'):
                base64_string = base64_string.split(',')[1]
            
            # Decode base64 to bytes
            img_bytes = base64.b64decode(base64_string)
            
            # Convert to PIL Image
            pil_image = Image.open(io.BytesIO(img_bytes))
            
            # Convert to OpenCV format
            opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            
            logger.info(f"✅ Image decoded: {opencv_image.shape}")
            return opencv_image
            
        except Exception as e:
            logger.error(f"❌ Failed to decode base64 image: {e}")
            return None
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Advanced image preprocessing for optimal particle detection
        
        Args:
            image: Input image from ESP32-CAM
            
        Returns:
            Preprocessed binary image
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (self.gaussian_kernel_size, self.gaussian_kernel_size), 0)
        
        # Enhance contrast using CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(blurred)
        
        # Apply adaptive threshold for better results with varying lighting
        binary = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Morphological operations to clean up the image
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        
        # Remove noise with opening
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Fill holes with closing
        filled = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
        
        return filled
    
    def detect_particles(self, binary_image: np.ndarray, device_id: str = "unknown") -> List[ParticleData]:
        """
        Detect and analyze particles in binary image
        
        Args:
            binary_image: Preprocessed binary image
            device_id: ESP32 device identifier
            
        Returns:
            List of detected particles with measurements
        """
        particles = []
        
        # Find contours (particle boundaries)
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        logger.info(f"🔍 Found {len(contours)} potential particles for device {device_id}")
        
        for i, contour in enumerate(contours):
            # Calculate basic measurements
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            # Filter by area (remove noise and large debris)
            if area < self.min_particle_area or area > self.max_particle_area:
                continue
                
            # Calculate additional shape parameters
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
            else:
                circularity = 0
                
            # Filter by circularity (microplastics tend to be somewhat circular)
            if circularity < self.min_circularity:
                continue
            
            # Calculate centroid
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
            else:
                cx, cy = 0, 0
            
            # Calculate equivalent diameter in micrometers
            equivalent_diameter = np.sqrt(4 * area / np.pi) * self.pixel_to_micron
            
            # Calculate bounding rectangle for aspect ratio
            rect = cv2.minAreaRect(contour)
            width, height = rect[1]
            if height > 0 and width > 0:
                aspect_ratio = max(width, height) / min(width, height)
            else:
                aspect_ratio = 1.0
            
            # Calculate solidity (convexity measure)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            if hull_area > 0:
                solidity = area / hull_area
            else:
                solidity = 0
            
            # Classify particle type based on morphological features
            particle_type, confidence, size_category = self.classify_particle(
                circularity, aspect_ratio, solidity, equivalent_diameter
            )
            
            # Create particle data object
            particle = ParticleData(
                id=f"{device_id}_particle_{i}_{int(datetime.now().timestamp())}",
                center_x=cx,
                center_y=cy,
                area=area,
                perimeter=perimeter,
                circularity=circularity,
                equivalent_diameter=equivalent_diameter,
                aspect_ratio=aspect_ratio,
                solidity=solidity,
                particle_type=particle_type,
                confidence=confidence,
                size_category=size_category
            )
            
            particles.append(particle)
        
        logger.info(f"✅ Detected {len(particles)} valid particles for device {device_id}")
        return particles
    
    def classify_particle(self, circularity: float, aspect_ratio: float, 
                         solidity: float, size_microns: float) -> Tuple[str, float, str]:
        """
        Classify particle type based on morphological features
        
        Args:
            circularity: Shape circularity (0-1)
            aspect_ratio: Width/height ratio
            solidity: Convexity measure
            size_microns: Equivalent diameter in micrometers
            
        Returns:
            Tuple of (particle_type, confidence, size_category)
        """
        confidence = 0.0
        particle_type = "unknown"
        
        # Enhanced rule-based classification
        if circularity > 0.8 and solidity > 0.9 and aspect_ratio < 1.5:
            particle_type = "spherical_microplastic"
            confidence = 0.85 + (circularity - 0.8) * 0.75  # Higher confidence for more circular
        elif aspect_ratio > 3.0 and circularity < 0.4:
            particle_type = "fiber_microplastic"
            confidence = 0.75 + min((aspect_ratio - 3.0) * 0.05, 0.15)
        elif 0.4 <= circularity <= 0.8 and solidity > 0.6:
            particle_type = "fragment_microplastic"
            confidence = 0.65 + (solidity - 0.6) * 0.5
        elif solidity < 0.5:
            particle_type = "organic_debris"
            confidence = 0.50
        else:
            particle_type = "unknown_particle"
            confidence = 0.30
        
        # Adjust confidence based on size (microplastic typical range)
        if 1 <= size_microns <= 1000:
            confidence += 0.1
        elif size_microns < 0.1 or size_microns > 5000:
            confidence -= 0.2
        
        # Determine size category
        size_category = "unknown"
        for category, (min_size, max_size) in self.size_categories.items():
            if min_size <= size_microns < max_size:
                size_category = category
                break
        
        # Ensure confidence is within bounds
        confidence = max(0.0, min(1.0, confidence))
        
        return particle_type, confidence, size_category
    
    def calculate_size_distribution(self, particles: List[ParticleData]) -> Dict[str, int]:
        """
        Calculate size distribution of detected particles
        
        Args:
            particles: List of classified particles
            
        Returns:
            Dictionary with size category counts
        """
        distribution = {category: 0 for category in self.size_categories.keys()}
        distribution["unknown"] = 0
        
        for particle in particles:
            if particle.size_category in distribution:
                distribution[particle.size_category] += 1
            else:
                distribution["unknown"] += 1
        
        return distribution
    
    def assess_contamination_level(self, particles: List[ParticleData]) -> Tuple[str, float]:
        """
        Assess contamination level based on particle analysis
        
        Args:
            particles: List of detected particles
            
        Returns:
            Tuple of (contamination_level, confidence_score)
        """
        microplastic_count = sum(1 for p in particles 
                               if "microplastic" in p.particle_type)
        
        total_particles = len(particles)
        
        if total_particles == 0:
            return "clean", 0.95
        
        microplastic_ratio = microplastic_count / total_particles
        
        # Enhanced contamination assessment
        if microplastic_count == 0:
            level = "clean"
            confidence = 0.90
        elif microplastic_count <= 3:
            level = "low"
            confidence = 0.75
        elif microplastic_count <= 10:
            level = "moderate"
            confidence = 0.80
        elif microplastic_count <= 25:
            level = "high"
            confidence = 0.85
        else:
            level = "severe"
            confidence = 0.90
        
        # Adjust confidence based on particle classification quality
        if particles:
            avg_particle_confidence = np.mean([p.confidence for p in particles])
            final_confidence = (confidence + avg_particle_confidence) / 2
        else:
            final_confidence = confidence
        
        return level, final_confidence
    
    def create_annotated_image(self, original_image: np.ndarray, 
                             particles: List[ParticleData]) -> np.ndarray:
        """
        Create annotated image showing detected particles
        
        Args:
            original_image: Original input image
            particles: List of detected particles
            
        Returns:
            Annotated image with particle markers
        """
        annotated = original_image.copy()
        
        # Color mapping for particle types
        colors = {
            "spherical_microplastic": (0, 255, 0),    # Green
            "fiber_microplastic": (255, 0, 0),        # Blue  
            "fragment_microplastic": (0, 165, 255),   # Orange
            "organic_debris": (128, 128, 128),        # Gray
            "unknown_particle": (255, 255, 255)      # White
        }
        
        for particle in particles:
            center = (int(particle.center_x), int(particle.center_y))
            color = colors.get(particle.particle_type, (255, 255, 255))
            
            # Circle size based on particle area
            radius = max(3, int(np.sqrt(particle.area / np.pi)))
            cv2.circle(annotated, center, radius, color, 2)
            
            # Add confidence label for high-confidence detections
            if particle.confidence > 0.7:
                label = f"{particle.particle_type[:8]}"
                cv2.putText(annotated, label, 
                          (center[0] + radius + 2, center[1]), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        return annotated
    
    def process_image(self, image_data: str, device_id: str = "unknown", 
                     image_id: str = None) -> ImageAnalysisResult:
        """
        Complete image processing pipeline for ESP32-CAM integration
        
        Args:
            image_data: Base64 encoded image from ESP32
            device_id: ESP32 device identifier
            image_id: Unique image identifier
            
        Returns:
            Complete analysis results
        """
        start_time = datetime.now()
        
        if image_id is None:
            image_id = f"img_{device_id}_{int(start_time.timestamp())}"
        
        try:
            # Decode image
            image = self.decode_base64_image(image_data)
            if image is None:
                raise ValueError("Failed to decode image")
            
            # Store image metadata
            image_metadata = {
                "width": image.shape[1],
                "height": image.shape[0],
                "channels": image.shape[2] if len(image.shape) > 2 else 1,
                "size_bytes": len(image_data),
                "format": "JPEG"  # Assuming JPEG from ESP32
            }
            
            # Preprocess image
            binary_image = self.preprocess_image(image)
            
            # Detect particles
            particles = self.detect_particles(binary_image, device_id)
            
            # Calculate size distribution
            size_distribution = self.calculate_size_distribution(particles)
            
            # Assess contamination level
            contamination_level, confidence_score = self.assess_contamination_level(particles)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Create result object
            result = ImageAnalysisResult(
                timestamp=start_time.isoformat(),
                device_id=device_id,
                image_id=image_id,
                total_particles=len(particles),
                particles=particles,
                size_distribution=size_distribution,
                contamination_level=contamination_level,
                confidence_score=confidence_score,
                processing_time=processing_time,
                image_metadata=image_metadata
            )
            
            logger.info(f"🎯 Image analysis complete for {device_id}: "
                       f"{result.total_particles} particles, "
                       f"{result.contamination_level} contamination, "
                       f"{processing_time:.2f}s processing time")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Image processing failed for {device_id}: {e}")
            # Return error result
            return ImageAnalysisResult(
                timestamp=start_time.isoformat(),
                device_id=device_id,
                image_id=image_id or f"error_{int(start_time.timestamp())}",
                total_particles=0,
                particles=[],
                size_distribution={},
                contamination_level="error",
                confidence_score=0.0,
                processing_time=(datetime.now() - start_time).total_seconds(),
                image_metadata={"error": str(e)}
            )

# Global instance for use in FastAPI endpoints
image_processor = MicroplasticImageProcessor()