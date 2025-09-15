#!/usr/bin/env python3
"""
ESP32 Device Simulator for Microplastic Detection System
Simulates multiple ESP32-CAM devices sending data via MQTT
"""

import paho.mqtt.client as mqtt
import json
import time
import random
import os
from datetime import datetime
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ESP32Simulator:
    def __init__(self):
        self.device_id = os.getenv('DEVICE_ID', f'esp32_sim_{random.randint(1000, 9999)}')
        self.mqtt_broker = os.getenv('MQTT_BROKER_URL', 'localhost').replace('mqtt://', '')
        self.lat = float(os.getenv('LOCATION_LAT', '28.6139'))
        self.lon = float(os.getenv('LOCATION_LON', '77.2090'))
        
        # MQTT client setup
        self.mqtt_client = mqtt.Client()
        self.mqtt_client.on_connect = self.on_connect
        self.mqtt_client.on_publish = self.on_publish
        
        # Simulation parameters
        self.running = False
        self.measurement_interval = 30  # seconds
        
        logger.info(f"ESP32 Simulator initialized: {self.device_id}")
        logger.info(f"Location: {self.lat}, {self.lon}")
        logger.info(f"MQTT Broker: {self.mqtt_broker}")
    
    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            logger.info(f"Connected to MQTT broker: {self.mqtt_broker}")
            self.running = True
            
            # Subscribe to command topics
            command_topic = f"devices/{self.device_id}/commands/+"
            client.subscribe(command_topic)
            logger.info(f"Subscribed to commands: {command_topic}")
        else:
            logger.error(f"Failed to connect to MQTT broker: {rc}")
    
    def on_publish(self, client, userdata, mid):
        logger.debug(f"Message published: {mid}")
    
    def generate_sensor_data(self):
        """Generate realistic sensor data"""
        # Simulate particle detection with some randomness
        base_count = random.randint(5, 25)
        noise = random.gauss(0, 3)
        particle_count = max(0, int(base_count + noise))
        
        # Photodiode reading (inverse correlation with particles)
        photodiode_base = 2500
        photodiode_value = photodiode_base - (particle_count * 50) + random.uniform(-200, 200)
        photodiode_value = max(100, min(4000, photodiode_value))
        
        # Additional environmental data
        water_temp = 20 + random.uniform(-5, 15)  # 15-35°C
        ph_level = 7.0 + random.uniform(-1.5, 1.5)  # 5.5-8.5
        turbidity = random.uniform(1, 100)  # NTU
        
        return {
            'device_id': self.device_id,
            'timestamp': datetime.now().isoformat(),
            'particle_count': particle_count,
            'photodiode_value': round(photodiode_value, 2),
            'water_temperature': round(water_temp, 1),
            'ph_level': round(ph_level, 2),
            'turbidity': round(turbidity, 1),
            'location': {
                'lat': self.lat + random.uniform(-0.001, 0.001),
                'lon': self.lon + random.uniform(-0.001, 0.001)
            },
            'battery_level': random.randint(60, 100),
            'signal_strength': random.randint(-80, -40)
        }
    
    def generate_image_metadata(self):
        """Generate image capture metadata"""
        return {
            'device_id': self.device_id,
            'timestamp': datetime.now().isoformat(),
            'image_id': f"img_{int(time.time())}_{random.randint(1000, 9999)}",
            'image_size': random.randint(50000, 200000),  # bytes
            'image_format': 'JPEG',
            'resolution': '640x480',
            'compression_quality': random.randint(70, 95),
            'processing_time': round(random.uniform(0.5, 3.0), 2)  # seconds
        }
    
    def generate_classification_result(self):
        """Generate AI classification results"""
        particle_types = ['spherical_microplastic', 'fiber_microplastic', 'fragment_microplastic', 'organic_debris']
        size_categories = ['nano', 'micro', 'small', 'large']
        
        # Generate multiple particle classifications
        classifications = []
        num_particles = random.randint(0, 10)
        
        for _ in range(num_particles):
            classification = {
                'particle_type': random.choice(particle_types),
                'size_category': random.choice(size_categories),
                'confidence': round(random.uniform(0.6, 0.95), 3),
                'equivalent_diameter': round(random.uniform(1, 500), 1),  # micrometers
                'morphological_features': {
                    'circularity': round(random.uniform(0.3, 1.0), 3),
                    'aspect_ratio': round(random.uniform(1.0, 5.0), 2),
                    'solidity': round(random.uniform(0.5, 1.0), 3)
                }
            }
            classifications.append(classification)
        
        return {
            'device_id': self.device_id,
            'timestamp': datetime.now().isoformat(),
            'total_particles': num_particles,
            'classifications': classifications,
            'processing_time': round(random.uniform(0.1, 2.0), 2),
            'model_version': 'tflite_v1.0'
        }
    
    def publish_sensor_data(self):
        """Publish sensor data to MQTT"""
        data = self.generate_sensor_data()
        topic = f"devices/{self.device_id}/sensor_data"
        
        try:
            result = self.mqtt_client.publish(topic, json.dumps(data))
            if result.rc == 0:
                logger.info(f"Sensor data published: {data['particle_count']} particles detected")
            else:
                logger.error(f"Failed to publish sensor data: {result.rc}")
        except Exception as e:
            logger.error(f"Error publishing sensor data: {e}")
    
    def publish_image_metadata(self):
        """Publish image capture metadata"""
        data = self.generate_image_metadata()
        topic = f"devices/{self.device_id}/images"
        
        try:
            result = self.mqtt_client.publish(topic, json.dumps(data))
            if result.rc == 0:
                logger.info(f"Image metadata published: {data['image_id']}")
        except Exception as e:
            logger.error(f"Error publishing image metadata: {e}")
    
    def publish_classification_results(self):
        """Publish AI classification results"""
        data = self.generate_classification_result()
        topic = f"devices/{self.device_id}/analysis"
        
        try:
            result = self.mqtt_client.publish(topic, json.dumps(data))
            if result.rc == 0:
                logger.info(f"Classification results published: {data['total_particles']} particles analyzed")
        except Exception as e:
            logger.error(f"Error publishing classification: {e}")
    
    def publish_status(self):
        """Publish device status"""
        status_data = {
            'device_id': self.device_id,
            'timestamp': datetime.now().isoformat(),
            'status': 'online',
            'uptime': random.randint(1000, 100000),  # seconds
            'free_memory': random.randint(50000, 200000),  # bytes
            'cpu_usage': random.randint(20, 80),  # percent
            'network_quality': random.choice(['excellent', 'good', 'fair', 'poor']),
            'last_calibration': datetime.now().isoformat(),
            'measurements_today': random.randint(100, 1000)
        }
        
        topic = f"devices/{self.device_id}/status"
        
        try:
            result = self.mqtt_client.publish(topic, json.dumps(status_data))
            if result.rc == 0:
                logger.info(f"Status published: {status_data['status']}")
        except Exception as e:
            logger.error(f"Error publishing status: {e}")
    
    def start_simulation(self):
        """Start the simulation loop"""
        try:
            # Connect to MQTT broker
            logger.info(f"Connecting to MQTT broker at {self.mqtt_broker}...")
            self.mqtt_client.connect(self.mqtt_broker, 1883, 60)
            
            # Start MQTT loop in background
            self.mqtt_client.loop_start()
            
            # Wait for connection
            timeout = 10
            while not self.running and timeout > 0:
                time.sleep(1)
                timeout -= 1
            
            if not self.running:
                logger.error("Failed to connect to MQTT broker within timeout")
                return
            
            logger.info("Starting sensor simulation...")
            measurement_counter = 0
            
            while True:
                try:
                    # Publish sensor data every interval
                    self.publish_sensor_data()
                    
                    # Publish image metadata less frequently
                    if measurement_counter % 3 == 0:
                        self.publish_image_metadata()
                    
                    # Publish AI classification results
                    if measurement_counter % 2 == 0:
                        self.publish_classification_results()
                    
                    # Publish status every 10 measurements
                    if measurement_counter % 10 == 0:
                        self.publish_status()
                    
                    measurement_counter += 1
                    
                    # Wait for next measurement
                    time.sleep(self.measurement_interval)
                    
                except KeyboardInterrupt:
                    logger.info("Simulation stopped by user")
                    break
                except Exception as e:
                    logger.error(f"Error in simulation loop: {e}")
                    time.sleep(5)  # Wait before retrying
                    
        except Exception as e:
            logger.error(f"Failed to start simulation: {e}")
        finally:
            self.mqtt_client.loop_stop()
            self.mqtt_client.disconnect()
            logger.info("ESP32 simulator stopped")

def main():
    """Main function"""
    logger.info("Starting ESP32 Microplastic Detection Simulator")
    
    # Create and start simulator
    simulator = ESP32Simulator()
    simulator.start_simulation()

if __name__ == "__main__":
    main()