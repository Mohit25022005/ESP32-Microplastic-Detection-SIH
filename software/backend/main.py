"""
ESP32-CAM Microplastic Detection - Backend API Server
Smart India Hackathon 2025 - AquaGuard Team

FastAPI-based backend service with:
- REST API endpoints for device management
- Real-time MQTT communication with ESP32 devices  
- Database storage and analytics
- WebSocket support for real-time dashboards
- Authentication and authorization
- Data export and reporting
- Integration with ML forecasting models

Author: SIH AquaGuard Team
Version: 2.0
Date: September 2025
"""

from fastapi import FastAPI, HTTPException, Depends, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import uvicorn
import asyncio
import logging
import json
import time
import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field
import sqlite3
import aiosqlite
from pathlib import Path
import base64
import io
from PIL import Image
import paho.mqtt.client as mqtt
import threading
import queue
import yaml
import hashlib
import jwt
from passlib.context import CryptContext
import pandas as pd
import numpy as np
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import smtplib
import ssl

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('backend_server.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Security
security = HTTPBearer()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
SECRET_KEY = "your-secret-key-change-in-production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Pydantic Models
class DeviceRegistration(BaseModel):
    device_id: str = Field(..., description="Unique device identifier")
    device_name: str = Field(..., description="Human-readable device name")
    location: str = Field(..., description="Device installation location")
    coordinates: Optional[Dict[str, float]] = Field(None, description="GPS coordinates")
    owner_email: str = Field(..., description="Device owner email")

class SensorData(BaseModel):
    device_id: str
    timestamp: str
    photodiode_voltage: float
    particle_count: int
    confidence_score: float
    image_hash: Optional[str] = None
    water_temperature: Optional[float] = None
    flow_rate: Optional[float] = None
    anomaly_detected: bool = False

class DetectionResult(BaseModel):
    id: Optional[int] = None
    device_id: str
    timestamp: str
    total_particles: int
    size_distribution: Dict[str, int]
    confidence_score: float
    processing_time_ms: float
    image_path: Optional[str] = None
    anomaly_detected: bool = False

class AlertConfig(BaseModel):
    device_id: str
    max_particles: int = 100
    min_confidence: float = 0.5
    email_notifications: bool = True
    sms_notifications: bool = False
    notification_emails: List[str] = []

class User(BaseModel):
    username: str
    email: str
    full_name: Optional[str] = None
    disabled: Optional[bool] = False

class UserInDB(User):
    hashed_password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class ForecastRequest(BaseModel):
    device_id: str
    forecast_hours: int = Field(default=24, ge=1, le=168)  # 1 hour to 1 week
    include_weather: bool = Field(default=True)
    include_historical: bool = Field(default=True)

# Global variables
app_config = {}
db_pool = None
mqtt_client = None
websocket_connections = []
data_queue = queue.Queue(maxsize=1000)

# WebSocket Connection Manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket client connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info(f"WebSocket client disconnected. Total connections: {len(self.active_connections)}")

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.error(f"Error broadcasting to WebSocket: {e}")

manager = ConnectionManager()

# Application Lifecycle
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown events"""
    # Startup
    logger.info("Starting Microplastic Detection Backend Server...")
    
    # Load configuration
    global app_config
    app_config = load_config()
    
    # Initialize database
    await initialize_database()
    
    # Start MQTT client
    start_mqtt_client()
    
    # Start background tasks
    asyncio.create_task(process_data_queue())
    asyncio.create_task(cleanup_old_data())
    
    logger.info("Backend server started successfully!")
    
    yield
    
    # Shutdown
    logger.info("Shutting down backend server...")
    if mqtt_client:
        mqtt_client.loop_stop()
        mqtt_client.disconnect()
    logger.info("Backend server shutdown complete")

# Initialize FastAPI app
app = FastAPI(
    title="Microplastic Detection API",
    description="Backend API for ESP32-CAM Microplastic Detection System",
    version="2.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (images, etc.)
if not os.path.exists("static"):
    os.makedirs("static")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Configuration Management
def load_config():
    """Load application configuration"""
    config_path = "backend_config.yaml"
    
    default_config = {
        'database': {
            'path': 'microplastic_backend.db',
            'backup_interval': 3600
        },
        'mqtt': {
            'broker': 'localhost',
            'port': 1883,
            'username': 'backend',
            'password': 'backend_2025',
            'topics': {
                'sensor_data': 'sensors/microplastic/data',
                'device_status': 'sensors/microplastic/status',
                'commands': 'sensors/microplastic/commands',
                'alerts': 'alerts/microplastic'
            }
        },
        'api': {
            'host': '0.0.0.0',
            'port': 8000,
            'workers': 1
        },
        'alerts': {
            'smtp_server': 'smtp.gmail.com',
            'smtp_port': 587,
            'email_user': 'alerts@microplastic.com',
            'email_password': 'your_email_password'
        },
        'storage': {
            'images_dir': 'static/images',
            'max_file_size': 10485760,  # 10MB
            'allowed_extensions': ['.jpg', '.jpeg', '.png']
        }
    }
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return {**default_config, **config}
    else:
        with open(config_path, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False)
        return default_config

# Database Management
async def initialize_database():
    """Initialize SQLite database with required tables"""
    db_path = app_config['database']['path']
    
    async with aiosqlite.connect(db_path) as db:
        # Devices table
        await db.execute('''
            CREATE TABLE IF NOT EXISTS devices (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                device_id TEXT UNIQUE NOT NULL,
                device_name TEXT NOT NULL,
                location TEXT NOT NULL,
                coordinates_lat REAL,
                coordinates_lng REAL,
                owner_email TEXT NOT NULL,
                status TEXT DEFAULT 'active',
                last_seen TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Sensor data table
        await db.execute('''
            CREATE TABLE IF NOT EXISTS sensor_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                device_id TEXT NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                photodiode_voltage REAL NOT NULL,
                particle_count INTEGER NOT NULL,
                confidence_score REAL NOT NULL,
                image_hash TEXT,
                water_temperature REAL,
                flow_rate REAL,
                anomaly_detected BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (device_id) REFERENCES devices(device_id)
            )
        ''')
        
        # Detection results table
        await db.execute('''
            CREATE TABLE IF NOT EXISTS detection_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                device_id TEXT NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                total_particles INTEGER NOT NULL,
                size_distribution TEXT NOT NULL,
                confidence_score REAL NOT NULL,
                processing_time_ms REAL NOT NULL,
                image_path TEXT,
                anomaly_detected BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (device_id) REFERENCES devices(device_id)
            )
        ''')
        
        # Alerts table
        await db.execute('''
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                device_id TEXT NOT NULL,
                alert_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                message TEXT NOT NULL,
                resolved BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                resolved_at TIMESTAMP,
                FOREIGN KEY (device_id) REFERENCES devices(device_id)
            )
        ''')
        
        # Users table
        await db.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                full_name TEXT,
                hashed_password TEXT NOT NULL,
                disabled BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Alert configurations table
        await db.execute('''
            CREATE TABLE IF NOT EXISTS alert_configs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                device_id TEXT NOT NULL,
                max_particles INTEGER DEFAULT 100,
                min_confidence REAL DEFAULT 0.5,
                email_notifications BOOLEAN DEFAULT TRUE,
                sms_notifications BOOLEAN DEFAULT FALSE,
                notification_emails TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (device_id) REFERENCES devices(device_id)
            )
        ''')
        
        await db.commit()
        logger.info("Database initialized successfully")

# MQTT Client Setup
def start_mqtt_client():
    """Initialize and start MQTT client"""
    global mqtt_client
    
    mqtt_client = mqtt.Client()
    mqtt_client.username_pw_set(
        app_config['mqtt']['username'],
        app_config['mqtt']['password']
    )
    
    mqtt_client.on_connect = on_mqtt_connect
    mqtt_client.on_message = on_mqtt_message
    mqtt_client.on_disconnect = on_mqtt_disconnect
    
    try:
        mqtt_client.connect(
            app_config['mqtt']['broker'],
            app_config['mqtt']['port'],
            60
        )
        mqtt_client.loop_start()
        logger.info("MQTT client started successfully")
    except Exception as e:
        logger.error(f"Failed to start MQTT client: {e}")

def on_mqtt_connect(client, userdata, flags, rc):
    """MQTT connection callback"""
    if rc == 0:
        logger.info("Connected to MQTT broker")
        # Subscribe to topics
        topics = app_config['mqtt']['topics']
        for topic_name, topic in topics.items():
            client.subscribe(topic)
            logger.info(f"Subscribed to {topic}")
    else:
        logger.error(f"Failed to connect to MQTT broker: {rc}")

def on_mqtt_message(client, userdata, msg):
    """Handle incoming MQTT messages"""
    try:
        topic = msg.topic
        payload = json.loads(msg.payload.decode())
        
        # Add to processing queue
        if not data_queue.full():
            data_queue.put({
                'topic': topic,
                'payload': payload,
                'timestamp': datetime.now().isoformat()
            })
        else:
            logger.warning("Data queue full, dropping message")
            
    except Exception as e:
        logger.error(f"Error processing MQTT message: {e}")

def on_mqtt_disconnect(client, userdata, rc):
    """MQTT disconnection callback"""
    logger.warning(f"Disconnected from MQTT broker: {rc}")

# Background Tasks
async def process_data_queue():
    """Process incoming data from MQTT"""
    while True:
        try:
            if not data_queue.empty():
                data = data_queue.get_nowait()
                await handle_mqtt_data(data)
            else:
                await asyncio.sleep(0.1)  # Small delay to prevent busy waiting
        except Exception as e:
            logger.error(f"Error processing data queue: {e}")
            await asyncio.sleep(1)

async def handle_mqtt_data(data: Dict):
    """Handle different types of MQTT data"""
    topic = data['topic']
    payload = data['payload']
    
    if 'sensor_data' in topic or 'data' in topic:
        await store_sensor_data(payload)
        await check_alerts(payload)
        await broadcast_to_websockets(payload)
        
    elif 'status' in topic:
        await update_device_status(payload)
        
    elif 'results' in topic:
        await store_detection_result(payload)

async def store_sensor_data(data: Dict):
    """Store sensor data in database"""
    try:
        db_path = app_config['database']['path']
        async with aiosqlite.connect(db_path) as db:
            await db.execute('''
                INSERT INTO sensor_data 
                (device_id, timestamp, photodiode_voltage, particle_count, 
                 confidence_score, image_hash, water_temperature, flow_rate, anomaly_detected)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                data.get('device_id'),
                data.get('timestamp'),
                data.get('photodiode_voltage', 0.0),
                data.get('particle_count', 0),
                data.get('confidence_score', 0.0),
                data.get('image_hash'),
                data.get('water_temperature'),
                data.get('flow_rate'),
                data.get('anomaly_detected', False)
            ))
            await db.commit()
    except Exception as e:
        logger.error(f"Error storing sensor data: {e}")

async def store_detection_result(data: Dict):
    """Store detection result in database"""
    try:
        db_path = app_config['database']['path']
        async with aiosqlite.connect(db_path) as db:
            await db.execute('''
                INSERT INTO detection_results 
                (device_id, timestamp, total_particles, size_distribution,
                 confidence_score, processing_time_ms, image_path, anomaly_detected)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                data.get('device_id'),
                data.get('timestamp'),
                data.get('total_particles', 0),
                json.dumps(data.get('size_distribution', {})),
                data.get('confidence_score', 0.0),
                data.get('processing_time_ms', 0.0),
                data.get('image_path'),
                data.get('anomaly_detected', False)
            ))
            await db.commit()
    except Exception as e:
        logger.error(f"Error storing detection result: {e}")

async def check_alerts(data: Dict):
    """Check if data triggers any alerts"""
    device_id = data.get('device_id')
    particle_count = data.get('particle_count', 0)
    confidence_score = data.get('confidence_score', 1.0)
    anomaly_detected = data.get('anomaly_detected', False)
    
    try:
        db_path = app_config['database']['path']
        async with aiosqlite.connect(db_path) as db:
            # Get alert configuration for device
            async with db.execute('''
                SELECT max_particles, min_confidence, email_notifications, notification_emails
                FROM alert_configs WHERE device_id = ?
            ''', (device_id,)) as cursor:
                config = await cursor.fetchone()
                
                if config:
                    max_particles, min_confidence, email_notifications, notification_emails = config
                    
                    alert_triggered = False
                    alert_message = ""
                    severity = "low"
                    
                    if particle_count > max_particles:
                        alert_triggered = True
                        alert_message = f"High particle count detected: {particle_count} particles"
                        severity = "high"
                    
                    if confidence_score < min_confidence:
                        alert_triggered = True
                        alert_message += f" Low confidence score: {confidence_score:.2f}"
                        severity = "medium" if severity == "low" else severity
                    
                    if anomaly_detected:
                        alert_triggered = True
                        alert_message += " Anomaly detected in water sample"
                        severity = "high"
                    
                    if alert_triggered:
                        # Store alert
                        await db.execute('''
                            INSERT INTO alerts (device_id, alert_type, severity, message)
                            VALUES (?, ?, ?, ?)
                        ''', (device_id, "particle_detection", severity, alert_message))
                        
                        # Send notifications if enabled
                        if email_notifications and notification_emails:
                            emails = json.loads(notification_emails) if notification_emails else []
                            await send_email_alert(device_id, alert_message, emails)
                        
                        await db.commit()
                        logger.info(f"Alert triggered for device {device_id}: {alert_message}")
                        
    except Exception as e:
        logger.error(f"Error checking alerts: {e}")

async def send_email_alert(device_id: str, message: str, recipients: List[str]):
    """Send email alert to recipients"""
    try:
        smtp_config = app_config['alerts']
        
        msg = MIMEMultipart()
        msg['From'] = smtp_config['email_user']
        msg['Subject'] = f"Microplastic Detection Alert - Device {device_id}"
        
        body = f"""
        Alert from Microplastic Detection System
        
        Device: {device_id}
        Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        Alert: {message}
        
        Please check the dashboard for more details.
        
        Best regards,
        AquaGuard Monitoring System
        """
        
        msg.attach(MIMEText(body, 'plain'))
        
        context = ssl.create_default_context()
        with smtplib.SMTP(smtp_config['smtp_server'], smtp_config['smtp_port']) as server:
            server.starttls(context=context)
            server.login(smtp_config['email_user'], smtp_config['email_password'])
            
            for recipient in recipients:
                msg['To'] = recipient
                server.send_message(msg)
                
        logger.info(f"Email alert sent to {len(recipients)} recipients")
        
    except Exception as e:
        logger.error(f"Error sending email alert: {e}")

async def update_device_status(data: Dict):
    """Update device status and last seen timestamp"""
    try:
        device_id = data.get('device_id')
        status = data.get('status', 'online')
        
        db_path = app_config['database']['path']
        async with aiosqlite.connect(db_path) as db:
            await db.execute('''
                UPDATE devices 
                SET status = ?, last_seen = CURRENT_TIMESTAMP 
                WHERE device_id = ?
            ''', (status, device_id))
            await db.commit()
            
    except Exception as e:
        logger.error(f"Error updating device status: {e}")

async def broadcast_to_websockets(data: Dict):
    """Broadcast real-time data to WebSocket connections"""
    if manager.active_connections:
        message = json.dumps({
            'type': 'sensor_data',
            'data': data
        })
        await manager.broadcast(message)

async def cleanup_old_data():
    """Periodic cleanup of old data"""
    while True:
        try:
            # Run cleanup every 24 hours
            await asyncio.sleep(86400)
            
            # Delete data older than 90 days
            cutoff_date = datetime.now() - timedelta(days=90)
            
            db_path = app_config['database']['path']
            async with aiosqlite.connect(db_path) as db:
                await db.execute('''
                    DELETE FROM sensor_data 
                    WHERE created_at < ?
                ''', (cutoff_date,))
                
                await db.execute('''
                    DELETE FROM detection_results 
                    WHERE created_at < ?
                ''', (cutoff_date,))
                
                await db.commit()
                logger.info("Old data cleanup completed")
                
        except Exception as e:
            logger.error(f"Error during data cleanup: {e}")

# Authentication
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get current user from JWT token"""
    try:
        token = credentials.credentials
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid token")
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    # Get user from database
    db_path = app_config['database']['path']
    async with aiosqlite.connect(db_path) as db:
        async with db.execute('''
            SELECT username, email, full_name, disabled 
            FROM users WHERE username = ?
        ''', (username,)) as cursor:
            user_data = await cursor.fetchone()
            
            if user_data is None:
                raise HTTPException(status_code=401, detail="User not found")
            
            return User(
                username=user_data[0],
                email=user_data[1],
                full_name=user_data[2],
                disabled=bool(user_data[3])
            )

# API Endpoints

# Health Check
@app.get("/health", tags=["System"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0",
        "mqtt_connected": mqtt_client.is_connected() if mqtt_client else False,
        "active_websockets": len(manager.active_connections)
    }

# Device Management
@app.post("/api/devices/register", tags=["Devices"])
async def register_device(device: DeviceRegistration, current_user: User = Depends(get_current_user)):
    """Register a new ESP32-CAM device"""
    try:
        db_path = app_config['database']['path']
        async with aiosqlite.connect(db_path) as db:
            await db.execute('''
                INSERT INTO devices 
                (device_id, device_name, location, coordinates_lat, coordinates_lng, owner_email)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                device.device_id,
                device.device_name,
                device.location,
                device.coordinates.get('lat') if device.coordinates else None,
                device.coordinates.get('lng') if device.coordinates else None,
                device.owner_email
            ))
            await db.commit()
            
        logger.info(f"Device registered: {device.device_id}")
        return {"message": "Device registered successfully", "device_id": device.device_id}
        
    except Exception as e:
        logger.error(f"Error registering device: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/devices", tags=["Devices"])
async def get_devices(current_user: User = Depends(get_current_user)):
    """Get all registered devices"""
    try:
        db_path = app_config['database']['path']
        async with aiosqlite.connect(db_path) as db:
            async with db.execute('''
                SELECT device_id, device_name, location, coordinates_lat, 
                       coordinates_lng, owner_email, status, last_seen
                FROM devices ORDER BY created_at DESC
            ''') as cursor:
                devices = []
                async for row in cursor:
                    devices.append({
                        'device_id': row[0],
                        'device_name': row[1],
                        'location': row[2],
                        'coordinates': {
                            'lat': row[3],
                            'lng': row[4]
                        } if row[3] and row[4] else None,
                        'owner_email': row[5],
                        'status': row[6],
                        'last_seen': row[7]
                    })
                
        return {'devices': devices}
        
    except Exception as e:
        logger.error(f"Error getting devices: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/devices/{device_id}/status", tags=["Devices"])
async def get_device_status(device_id: str, current_user: User = Depends(get_current_user)):
    """Get device status and recent activity"""
    try:
        db_path = app_config['database']['path']
        async with aiosqlite.connect(db_path) as db:
            # Get device info
            async with db.execute('''
                SELECT device_name, location, status, last_seen
                FROM devices WHERE device_id = ?
            ''', (device_id,)) as cursor:
                device_info = await cursor.fetchone()
                
                if not device_info:
                    raise HTTPException(status_code=404, detail="Device not found")
            
            # Get recent data
            async with db.execute('''
                SELECT COUNT(*) as data_points,
                       AVG(particle_count) as avg_particles,
                       MAX(particle_count) as max_particles,
                       AVG(confidence_score) as avg_confidence
                FROM sensor_data 
                WHERE device_id = ? AND created_at > datetime('now', '-24 hours')
            ''', (device_id,)) as cursor:
                stats = await cursor.fetchone()
        
        return {
            'device_id': device_id,
            'device_name': device_info[0],
            'location': device_info[1],
            'status': device_info[2],
            'last_seen': device_info[3],
            'stats_24h': {
                'data_points': stats[0] or 0,
                'avg_particles': round(stats[1], 2) if stats[1] else 0,
                'max_particles': stats[2] or 0,
                'avg_confidence': round(stats[3], 3) if stats[3] else 0
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting device status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Data Endpoints
@app.get("/api/data/{device_id}", tags=["Data"])
async def get_sensor_data(
    device_id: str,
    hours: int = 24,
    current_user: User = Depends(get_current_user)
):
    """Get sensor data for a device"""
    try:
        start_time = datetime.now() - timedelta(hours=hours)
        
        db_path = app_config['database']['path']
        async with aiosqlite.connect(db_path) as db:
            async with db.execute('''
                SELECT timestamp, photodiode_voltage, particle_count, 
                       confidence_score, water_temperature, flow_rate, anomaly_detected
                FROM sensor_data 
                WHERE device_id = ? AND created_at > ?
                ORDER BY timestamp DESC
                LIMIT 1000
            ''', (device_id, start_time)) as cursor:
                data = []
                async for row in cursor:
                    data.append({
                        'timestamp': row[0],
                        'photodiode_voltage': row[1],
                        'particle_count': row[2],
                        'confidence_score': row[3],
                        'water_temperature': row[4],
                        'flow_rate': row[5],
                        'anomaly_detected': bool(row[6])
                    })
        
        return {'device_id': device_id, 'data': data}
        
    except Exception as e:
        logger.error(f"Error getting sensor data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/detection-results/{device_id}", tags=["Data"])
async def get_detection_results(
    device_id: str,
    limit: int = 100,
    current_user: User = Depends(get_current_user)
):
    """Get detection results for a device"""
    try:
        db_path = app_config['database']['path']
        async with aiosqlite.connect(db_path) as db:
            async with db.execute('''
                SELECT id, timestamp, total_particles, size_distribution,
                       confidence_score, processing_time_ms, image_path, anomaly_detected
                FROM detection_results 
                WHERE device_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (device_id, limit)) as cursor:
                results = []
                async for row in cursor:
                    results.append({
                        'id': row[0],
                        'timestamp': row[1],
                        'total_particles': row[2],
                        'size_distribution': json.loads(row[3]) if row[3] else {},
                        'confidence_score': row[4],
                        'processing_time_ms': row[5],
                        'image_path': row[6],
                        'anomaly_detected': bool(row[7])
                    })
        
        return {'device_id': device_id, 'results': results}
        
    except Exception as e:
        logger.error(f"Error getting detection results: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Alert Management
@app.post("/api/alerts/config", tags=["Alerts"])
async def configure_alerts(
    config: AlertConfig,
    current_user: User = Depends(get_current_user)
):
    """Configure alert settings for a device"""
    try:
        db_path = app_config['database']['path']
        async with aiosqlite.connect(db_path) as db:
            # Check if config exists
            async with db.execute('''
                SELECT id FROM alert_configs WHERE device_id = ?
            ''', (config.device_id,)) as cursor:
                existing = await cursor.fetchone()
            
            if existing:
                # Update existing config
                await db.execute('''
                    UPDATE alert_configs 
                    SET max_particles = ?, min_confidence = ?, 
                        email_notifications = ?, sms_notifications = ?,
                        notification_emails = ?
                    WHERE device_id = ?
                ''', (
                    config.max_particles,
                    config.min_confidence,
                    config.email_notifications,
                    config.sms_notifications,
                    json.dumps(config.notification_emails),
                    config.device_id
                ))
            else:
                # Create new config
                await db.execute('''
                    INSERT INTO alert_configs 
                    (device_id, max_particles, min_confidence, 
                     email_notifications, sms_notifications, notification_emails)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    config.device_id,
                    config.max_particles,
                    config.min_confidence,
                    config.email_notifications,
                    config.sms_notifications,
                    json.dumps(config.notification_emails)
                ))
            
            await db.commit()
        
        return {"message": "Alert configuration updated successfully"}
        
    except Exception as e:
        logger.error(f"Error configuring alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/alerts/{device_id}", tags=["Alerts"])
async def get_alerts(
    device_id: str,
    resolved: bool = None,
    limit: int = 50,
    current_user: User = Depends(get_current_user)
):
    """Get alerts for a device"""
    try:
        db_path = app_config['database']['path']
        
        query = '''
            SELECT id, alert_type, severity, message, resolved, created_at, resolved_at
            FROM alerts WHERE device_id = ?
        '''
        params = [device_id]
        
        if resolved is not None:
            query += ' AND resolved = ?'
            params.append(resolved)
        
        query += ' ORDER BY created_at DESC LIMIT ?'
        params.append(limit)
        
        async with aiosqlite.connect(db_path) as db:
            async with db.execute(query, params) as cursor:
                alerts = []
                async for row in cursor:
                    alerts.append({
                        'id': row[0],
                        'alert_type': row[1],
                        'severity': row[2],
                        'message': row[3],
                        'resolved': bool(row[4]),
                        'created_at': row[5],
                        'resolved_at': row[6]
                    })
        
        return {'device_id': device_id, 'alerts': alerts}
        
    except Exception as e:
        logger.error(f"Error getting alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# WebSocket for Real-time Updates
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time data updates"""
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# Data Export
@app.get("/api/export/{device_id}", tags=["Export"])
async def export_data(
    device_id: str,
    start_date: str,
    end_date: str,
    format: str = "csv",
    current_user: User = Depends(get_current_user)
):
    """Export sensor data in various formats"""
    try:
        start_dt = datetime.fromisoformat(start_date)
        end_dt = datetime.fromisoformat(end_date)
        
        db_path = app_config['database']['path']
        async with aiosqlite.connect(db_path) as db:
            async with db.execute('''
                SELECT timestamp, photodiode_voltage, particle_count,
                       confidence_score, water_temperature, flow_rate, anomaly_detected
                FROM sensor_data
                WHERE device_id = ? AND timestamp BETWEEN ? AND ?
                ORDER BY timestamp
            ''', (device_id, start_dt, end_dt)) as cursor:
                data = []
                async for row in cursor:
                    data.append({
                        'timestamp': row[0],
                        'photodiode_voltage': row[1],
                        'particle_count': row[2],
                        'confidence_score': row[3],
                        'water_temperature': row[4],
                        'flow_rate': row[5],
                        'anomaly_detected': row[6]
                    })
        
        if format.lower() == "csv":
            df = pd.DataFrame(data)
            filename = f"microplastic_data_{device_id}_{start_date}_{end_date}.csv"
            filepath = f"static/exports/{filename}"
            
            os.makedirs("static/exports", exist_ok=True)
            df.to_csv(filepath, index=False)
            
            return FileResponse(
                filepath,
                media_type="application/octet-stream",
                filename=filename
            )
        else:
            return {"data": data, "format": format}
            
    except Exception as e:
        logger.error(f"Error exporting data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Command Interface
@app.post("/api/devices/{device_id}/command", tags=["Control"])
async def send_device_command(
    device_id: str,
    command: str,
    parameters: Dict = {},
    current_user: User = Depends(get_current_user)
):
    """Send command to ESP32-CAM device"""
    try:
        if not mqtt_client or not mqtt_client.is_connected():
            raise HTTPException(status_code=503, detail="MQTT client not connected")
        
        command_data = {
            'device_id': device_id,
            'command': command,
            'parameters': parameters,
            'timestamp': datetime.now().isoformat()
        }
        
        topic = f"sensors/microplastic/commands/{device_id}"
        mqtt_client.publish(topic, json.dumps(command_data))
        
        logger.info(f"Command sent to device {device_id}: {command}")
        return {"message": f"Command '{command}' sent to device {device_id}"}
        
    except Exception as e:
        logger.error(f"Error sending command: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Create default admin user if not exists
    async def create_default_user():
        db_path = app_config.get('database', {}).get('path', 'microplastic_backend.db')
        async with aiosqlite.connect(db_path) as db:
            # Check if admin user exists
            async with db.execute('SELECT id FROM users WHERE username = ?', ('admin',)) as cursor:
                if not await cursor.fetchone():
                    hashed_password = get_password_hash('admin123')
                    await db.execute('''
                        INSERT INTO users (username, email, full_name, hashed_password)
                        VALUES (?, ?, ?, ?)
                    ''', ('admin', 'admin@microplastic.com', 'System Administrator', hashed_password))
                    await db.commit()
                    print("Default admin user created: admin/admin123")
    
    # Run the server
    config = load_config()
    uvicorn.run(
        "main:app",
        host=config['api']['host'],
        port=config['api']['port'],
        workers=config['api']['workers'],
        reload=True  # Set to False in production
    )