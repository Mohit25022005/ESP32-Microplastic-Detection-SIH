## Project Overview

This is the **Integrated Microplastic Detection and Forecasting System** - a comprehensive hardware-software platform for real-time in-situ detection, classification, and forecasting of microplastics in rivers. The system evolved from a cost-effective SIH hackathon solution (₹4,000) into a production-ready IoT platform that combines ESP32-CAM edge computing, cloud-based AI analytics, satellite data integration, and decision support systems for environmental monitoring at scale.

## Architecture

### High-Level System Design
The project follows a **3-tier IoT architecture** with edge computing, cloud backend, and web frontend:

1. **Hardware/Edge Layer**: ESP32-CAM with optical sensors, edge AI processing
2. **Backend Layer**: FastAPI cloud services, PostgreSQL database, MQTT broker, AI analytics
3. **Frontend Layer**: React dashboard with real-time monitoring, GIS mapping, and forecasting
4. **Integration Layer**: Satellite data fusion, external APIs, decision support system

### System Components

#### Hardware Layer (`hardware/`, `firmware/`)
- **ESP32-CAM Module**: Camera capture, laser control, photodiode sensing
- **Optical Detection**: 650nm laser diode + photodiode for scattering analysis
- **Transparent Flow Cell**: Continuous water sample processing
- **Edge AI**: OpenCV + TensorFlow Lite for real-time image analysis
- **IoT Communication**: MQTT/HTTP data transmission with OTA update support

#### Backend Layer (`backend/`)
- **FastAPI Server**: RESTful APIs with authentication and real-time data ingestion
- **MQTT Broker**: IoT device communication and streaming data pipeline
- **PostgreSQL Database**: Structured sensor data, metadata, and analytics
- **MinIO/S3 Storage**: Raw and processed image storage
- **AI Engine**: Advanced particle classification and pollution forecasting
- **External Integrations**: Satellite data APIs and environmental datasets

#### Frontend Layer (`frontend/`)
- **React Dashboard**: Responsive web application with Tailwind CSS
- **Real-time Visualization**: Live charts with Recharts for particle analysis
- **GIS Integration**: Leaflet/Mapbox for geographic monitoring sites
- **Alert System**: SMS/email notifications for pollution events
- **Forecasting UI**: AI-driven pollution pattern predictions

### Multi-Modal Detection Approach
The system integrates multiple detection and analysis methods:
1. **Edge Processing**: Real-time OpenCV analysis on ESP32-CAM
2. **Optical Scattering**: Laser + photodiode for particle counting and sizing
3. **Cloud AI**: Advanced TensorFlow classification and pattern recognition
4. **Satellite Fusion**: Integration with remote sensing data for context
5. **Forecasting Models**: AI-based prediction of pollution spread patterns

## Development Commands

### Environment Setup
```bash
# Install Python dependencies for edge processing
pip install -r requirements.txt

# Install backend dependencies
cd backend/
pip install -r requirements.txt

# Install frontend dependencies
cd frontend/
npm install

# Development with all features
pip install -e ".[ml,dev,viz,web,cloud]"
```

### Backend Development
```bash
# Start PostgreSQL and MinIO services
docker-compose up -d postgres minio mqtt-broker

# Run FastAPI development server
cd backend/
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Run database migrations
alembic upgrade head

# Seed database with test data
python scripts/seed_database.py

# Start MQTT broker and worker
celery -A app.tasks worker --loglevel=info
```

### Frontend Development
```bash
# Start React development server
cd frontend/
npm start

# Build for production
npm run build

# Run Storybook for component development
npm run storybook

# Run frontend tests
npm test
```

### Testing
```bash
# Run all backend tests
cd backend/
pytest tests/ --cov=app --cov-report=html

# Run frontend tests
cd frontend/
npm test -- --coverage

# Run integration tests (requires full stack)
pytest tests/integration/ -m integration

# Run hardware tests (requires ESP32 connected)
pytest tests/hardware/ -m hardware

# End-to-end testing
cd e2e/
npm run cypress:open
```

### Code Quality
```bash
# Backend code formatting
cd backend/
black app/ tests/
isort app/ tests/
flake8 app/ tests/
mypy app/

# Frontend code quality
cd frontend/
npm run lint
npm run format
npm run type-check
```

### Running the Complete System
```bash
# Start full development environment
docker-compose up -d

# Deploy to production
docker-compose -f docker-compose.prod.yml up -d

# Scale ESP32 device simulation
docker-compose up --scale esp32-simulator=5

# Monitor system health
docker-compose logs -f --tail=50
```

### Firmware Development
```bash
# Upload firmware to ESP32-CAM (PlatformIO recommended)
pio run --target upload --environment esp32cam

# OTA firmware update
curl -X POST http://device-ip/update -F "firmware=@firmware.bin"

# Monitor serial output
pio device monitor --port COM3 --baud 115200

# Flash multiple devices
pio run --target upload --upload-port COM3,COM4,COM5
```

### AI/ML Pipeline
```bash
# Train classification models
cd ml/
python train_classifier.py --dataset data/microplastics/ --epochs 100

# Convert to TensorFlow Lite for ESP32
python convert_to_tflite.py --model models/classifier.h5

# Run forecasting pipeline
python forecasting/train_forecaster.py --historical-data 30days

# Evaluate model performance
python evaluate_models.py --test-dataset data/test/
```

## ESP32 Edge Computing Workflow

### Hardware Setup Sequence
1. **Device Assembly**: Follow `hardware/assembly_guide.md` for flow cell integration
2. **Firmware Deployment**: PlatformIO-based build and OTA update system
3. **MQTT Configuration**: Connect to cloud MQTT broker for data streaming
4. **TensorFlow Lite Setup**: Deploy edge AI models for real-time classification
5. **System Calibration**: Remote calibration via cloud dashboard
6. **Multi-device Registration**: Register multiple river monitoring sites

### Key ESP32 Integration Points
- **Pin Configuration**: GPIO12 (laser), GPIO2 (photodiode), GPIO4 (status LED)
- **Camera Settings**: 2MP capture optimized for particle detection with auto-exposure
- **Edge AI Processing**: TensorFlow Lite inference directly on ESP32-CAM
- **Flow Cell Interface**: Continuous water sampling with transparent detection chamber
- **MQTT Communication**: Real-time streaming to cloud with QoS guarantees
- **OTA Updates**: Remote firmware updates via cloud management
- **Power Management**: Solar-powered deployment with battery backup

### MQTT Communication Protocol
ESP32 publishes to structured MQTT topics:
```
devices/{device_id}/sensor_data     - Photodiode readings, particle counts
devices/{device_id}/images          - Compressed JPEG images
devices/{device_id}/analysis        - Edge AI classification results
devices/{device_id}/status          - Device health, battery, connectivity
devices/{device_id}/gps             - Location data for mobile deployments
```

### Cloud Integration Commands
```
devices/{device_id}/commands/calibrate    - Remote calibration trigger
devices/{device_id}/commands/update       - OTA firmware update
devices/{device_id}/commands/config       - Runtime configuration changes
devices/{device_id}/commands/capture      - Manual image capture
```

## Cloud-Edge Integration

### Real-Time Data Pipeline
```python
# Backend MQTT handler processes device data:
@mqtt_handler("devices/+/sensor_data")
async def handle_sensor_data(device_id: str, payload: dict):
    # Store in PostgreSQL
    await store_sensor_reading(device_id, payload)
    
    # Trigger alerts if threshold exceeded
    if payload['particle_count'] > threshold:
        await send_alert(device_id, payload)
    
    # Update real-time dashboard
    await websocket_broadcast(device_id, payload)
```

### Key Integration Patterns
- **Edge-First Processing**: ESP32 performs initial analysis, cloud provides advanced analytics
- **Bidirectional Communication**: Cloud can control devices, devices stream data continuously
- **Fault Tolerance**: Local storage buffer during connectivity issues
- **Scalable Architecture**: Support for hundreds of deployed devices
- **Real-time Alerts**: Immediate notification system for pollution events

## AI/ML Architecture

### Multi-Tier ML Pipeline
The system implements ML at both edge and cloud levels:

#### Edge AI (ESP32-CAM)
- **TensorFlow Lite Models**: Optimized for ESP32 inference
- **Real-time Classification**: Particle type identification at <100ms
- **Feature Extraction**: Morphological analysis with OpenCV
- **Model Size**: <2MB compressed models for ESP32 flash storage

#### Cloud AI (FastAPI + TensorFlow)
- **Advanced Classification**: High-accuracy models with ensemble methods
- **Forecasting Models**: Time series prediction of pollution patterns
- **Satellite Data Fusion**: Integration with remote sensing imagery
- **Decision Support**: AI-driven recommendations for intervention

### Feature Engineering Pipeline
```python
# Multi-modal feature extraction
def extract_features(image: np.ndarray, sensor_data: dict) -> Dict:
    # Computer vision features
    cv_features = extract_morphological_features(image)
    
    # Sensor features
    optical_features = process_scattering_data(sensor_data)
    
    # Environmental context
    context_features = get_environmental_context(device_location, timestamp)
    
    return {**cv_features, **optical_features, **context_features}
```

### Training Pipeline Workflow
```python
# Distributed training across data sources
trainer = MLPipeline()

# Load real-world labeled data
data = trainer.load_training_data(
    sources=['field_samples', 'lab_analysis', 'satellite_imagery']
)

# Multi-objective training
models = trainer.train_ensemble({
    'classification': ['particle_type', 'size_category', 'concentration'],
    'forecasting': ['pollution_trend', 'seasonal_patterns'],
    'detection': ['anomaly_detection', 'quality_assessment']
})

# Deploy to edge devices
trainer.deploy_to_edge(models['classification'], target='esp32')
trainer.deploy_to_cloud(models, target='fastapi')
```

### Forecasting System
```python
# Pollution pattern prediction
class PollutionForecaster:
    def __init__(self):
        self.models = {
            'lstm': LSTMForecaster(),
            'transformer': TransformerForecaster(),
            'ensemble': EnsembleForecaster()
        }
    
    async def predict_pollution_spread(
        self, 
        sensor_data: List[SensorReading],
        satellite_data: SatelliteImagery,
        weather_data: WeatherForecast
    ) -> PollutionForecast:
        # Multi-modal prediction
        predictions = await asyncio.gather(*[
            model.predict(sensor_data, satellite_data, weather_data)
            for model in self.models.values()
        ])
        
        return self.ensemble_predictions(predictions)
```

## Backend API Architecture

### FastAPI Service Structure
```python
# Main API endpoints for the integrated system
@app.post("/api/devices/{device_id}/data")
async def ingest_sensor_data(device_id: str, data: SensorData):
    """Ingest real-time data from ESP32 devices"""
    
@app.get("/api/devices/{device_id}/status")
async def get_device_status(device_id: str):
    """Get current device status and health metrics"""
    
@app.post("/api/devices/{device_id}/commands/{command}")
async def send_device_command(device_id: str, command: str):
    """Send control commands to ESP32 devices"""

@app.get("/api/analytics/pollution-forecast")
async def get_pollution_forecast(river_id: str, days: int = 7):
    """Get AI-generated pollution forecasts"""
    
@app.get("/api/analytics/real-time-dashboard")
async def get_dashboard_data():
    """Real-time data for dashboard visualization"""
    
@app.post("/api/alerts/configure")
async def configure_alerts(alert_config: AlertConfig):
    """Configure SMS/email alert thresholds"""
```

### Database Schema (PostgreSQL)
```sql
-- Core tables for the integrated system
CREATE TABLE devices (
    id UUID PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    location GEOGRAPHY(POINT, 4326),
    river_id UUID REFERENCES rivers(id),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE sensor_readings (
    id UUID PRIMARY KEY,
    device_id UUID REFERENCES devices(id),
    timestamp TIMESTAMP DEFAULT NOW(),
    particle_count INTEGER,
    photodiode_value FLOAT,
    water_temperature FLOAT,
    ph_level FLOAT,
    turbidity FLOAT,
    image_url TEXT
);

CREATE TABLE particle_classifications (
    id UUID PRIMARY KEY,
    reading_id UUID REFERENCES sensor_readings(id),
    particle_type VARCHAR(50),
    confidence FLOAT,
    size_category VARCHAR(20),
    morphological_features JSONB
);

CREATE TABLE pollution_forecasts (
    id UUID PRIMARY KEY,
    river_id UUID REFERENCES rivers(id),
    forecast_date DATE,
    predicted_concentration FLOAT,
    confidence_interval JSONB,
    model_version VARCHAR(50),
    created_at TIMESTAMP DEFAULT NOW()
);
```

### MQTT Broker Integration
```python
# MQTT message processing for IoT devices
class MQTTHandler:
    def __init__(self, broker_url: str):
        self.client = mqtt.Client()
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
    
    async def on_message(self, client, userdata, msg):
        topic_parts = msg.topic.split('/')
        device_id = topic_parts[1]
        data_type = topic_parts[2]
        
        if data_type == 'sensor_data':
            await self.process_sensor_data(device_id, msg.payload)
        elif data_type == 'images':
            await self.store_image(device_id, msg.payload)
        elif data_type == 'status':
            await self.update_device_status(device_id, msg.payload)
```

## Performance Optimization

### System Performance Requirements
- **API Response Time**: <200ms for real-time endpoints
- **MQTT Throughput**: Support 1000+ devices with 1Hz data rate
- **Database Performance**: <50ms query time for dashboard data
- **Edge Processing**: <30 seconds per measurement on ESP32
- **Scalability**: Horizontal scaling for 10,000+ concurrent devices

### Optimization Strategies

#### Backend Performance
- **Database Indexing**: Optimized indexes for time-series queries
- **Connection Pooling**: Async database connections with pgbouncer
- **Caching**: Redis for real-time dashboard data and API responses
- **Message Queuing**: Celery for background AI processing tasks
- **Load Balancing**: Kubernetes-based auto-scaling

#### Edge Processing Optimization
- **TensorFlow Lite Quantization**: INT8 quantization for ESP32 deployment
- **OpenCV Pipeline**: Gaussian blur → CLAHE → Otsu thresholding
- **Memory Management**: Circular buffer for image processing
- **Power Management**: Sleep modes between measurements

## Frontend Dashboard Architecture

### React Application Structure
```typescript
// Main dashboard components
src/
├── components/
│   ├── Dashboard/
│   │   ├── RealTimeCharts.tsx     // Recharts for live data
│   │   ├── DeviceMap.tsx          // Leaflet GIS integration
│   │   └── AlertPanel.tsx         // Real-time alerts
│   ├── Analytics/
│   │   ├── ForecastingView.tsx    // AI prediction visualization
│   │   ├── TrendAnalysis.tsx      // Historical data analysis
│   │   └── SatelliteOverlay.tsx   // Satellite data integration
│   └── DeviceManagement/
│       ├── DeviceList.tsx         // ESP32 device management
│       ├── DeviceConfig.tsx       // Remote device configuration
│       └── OTAUpdates.tsx         // Firmware update management
```

### Key Frontend Features
- **Real-time Data Visualization**: WebSocket-based live updates
- **GIS Mapping**: Interactive maps with device locations and pollution levels
- **Responsive Design**: Mobile-first approach with Tailwind CSS
- **Progressive Web App**: Offline capability for field operations
- **Role-based Access**: Admin, operator, and viewer permission levels

### Dashboard Data Flow
```typescript
// Real-time data subscription
const useSensorData = (deviceId: string) => {
  const [data, setData] = useState<SensorReading[]>([]);
  
  useEffect(() => {
    const ws = new WebSocket(`ws://api/devices/${deviceId}/stream`);
    ws.onmessage = (event) => {
      const reading = JSON.parse(event.data);
      setData(prev => [...prev.slice(-100), reading]); // Keep last 100 readings
    };
    return () => ws.close();
  }, [deviceId]);
  
  return data;
};
```

## Deployment Architecture

### Container Orchestration (Docker + Kubernetes)
```yaml
# docker-compose.yml for development
version: '3.8'
services:
  backend:
    build: ./backend
    ports: ["8000:8000"]
    environment:
      - DATABASE_URL=postgresql://user:pass@postgres:5432/microplastics
      - MQTT_BROKER_URL=mqtt://mqtt-broker:1883
    depends_on: [postgres, mqtt-broker, redis]
  
  frontend:
    build: ./frontend
    ports: ["3000:3000"]
    environment:
      - REACT_APP_API_URL=http://backend:8000
  
  postgres:
    image: postgis/postgis:13-3.1
    environment:
      - POSTGRES_DB=microplastics
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=pass
    volumes: ["postgres_data:/var/lib/postgresql/data"]
  
  mqtt-broker:
    image: eclipse-mosquitto:2.0
    ports: ["1883:1883", "9001:9001"]
    volumes: ["./mqtt-config:/mosquitto/config"]
  
  redis:
    image: redis:7-alpine
    ports: ["6379:6379"]
  
  minio:
    image: minio/minio:latest
    command: server /data --console-address ":9001"
    ports: ["9000:9000", "9001:9001"]
    environment:
      - MINIO_ROOT_USER=admin
      - MINIO_ROOT_PASSWORD=password123
    volumes: ["minio_data:/data"]
```

### Production Deployment (Kubernetes)
```yaml
# k8s/backend-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: backend
spec:
  replicas: 3
  selector:
    matchLabels:
      app: backend
  template:
    metadata:
      labels:
        app: backend
    spec:
      containers:
      - name: backend
        image: microplastics/backend:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: db-secret
              key: url
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
```

### CI/CD Pipeline
```yaml
# .github/workflows/deploy.yml
name: Deploy to Production
on:
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Run Backend Tests
      run: |
        cd backend
        pip install -r requirements.txt
        pytest tests/
    - name: Run Frontend Tests
      run: |
        cd frontend
        npm install
        npm test -- --coverage
  
  deploy:
    needs: test
    runs-on: ubuntu-latest
    steps:
    - name: Deploy to Kubernetes
      run: |
        kubectl apply -f k8s/
        kubectl rollout restart deployment/backend
        kubectl rollout restart deployment/frontend
```

## Troubleshooting

### ESP32 Hardware Issues
- **Firmware Upload Failures**: Use PlatformIO with proper ESP32-CAM board configuration
- **MQTT Connection Issues**: Verify broker URL, credentials, and network connectivity
- **TensorFlow Lite Model Loading**: Ensure model size <2MB and proper quantization
- **Camera Quality Problems**: Check power supply stability and lens focus
- **OTA Update Failures**: Verify sufficient flash memory and stable network

### Backend Service Issues
- **Database Connection Errors**: Check PostgreSQL configuration and connection pooling
- **High API Latency**: Monitor database query performance and implement caching
- **MQTT Message Processing**: Ensure proper message queue scaling with Celery workers
- **Memory Usage**: Monitor container memory limits and optimize database queries
- **Authentication Problems**: Verify JWT token configuration and user permissions

### Frontend Dashboard Issues
- **Real-time Data Not Updating**: Check WebSocket connections and CORS configuration
- **Map Rendering Problems**: Verify Leaflet/Mapbox API keys and geographic data format
- **Chart Performance**: Optimize data sampling for large time-series datasets
- **Mobile Responsiveness**: Test Tailwind CSS breakpoints on various devices
- **Build Failures**: Ensure Node.js version compatibility and dependency resolution

### System Integration Issues
- **Satellite Data Integration**: Check external API rate limits and data format compatibility
- **AI Model Performance**: Monitor inference latency and model accuracy metrics
- **Alert System**: Test SMS/email notification delivery and rate limiting
- **Data Synchronization**: Verify time zone handling and data consistency across services
- **Scalability Problems**: Monitor Kubernetes resource usage and auto-scaling configuration

### Production Deployment Issues
```bash
# Common debugging commands
# Check service health
kubectl get pods -l app=backend
kubectl logs -f deployment/backend

# Database connection testing
psql -h postgres-service -U user -d microplastics -c "SELECT COUNT(*) FROM sensor_readings;"

# MQTT broker status
mosquitto_sub -h mqtt-broker -t "devices/+/sensor_data" -v

# API endpoint testing
curl -X GET "http://api/health" -H "Authorization: Bearer $JWT_TOKEN"

# Container resource monitoring
docker stats --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}"
```

## System Architecture Overview

### Complete Project Structure
```
ESP32-Microplastic-Detection-SIH/
├── firmware/                    # ESP32-CAM edge computing
│   ├── src/
│   │   ├── main.cpp            # PlatformIO main firmware
│   │   ├── camera_handler.cpp  # Camera and image processing
│   │   ├── mqtt_client.cpp     # MQTT communication
│   │   ├── tflite_inference.cpp # Edge AI inference
│   │   └── ota_updater.cpp     # Over-the-air updates
│   ├── lib/                    # ESP32 libraries
│   └── platformio.ini          # Build configuration
├── backend/                     # FastAPI cloud services
│   ├── app/
│   │   ├── api/                # REST API endpoints
│   │   ├── core/               # Database and auth
│   │   ├── ml/                 # AI/ML pipeline
│   │   ├── mqtt/               # MQTT message handling
│   │   └── tasks/              # Background jobs (Celery)
│   ├── alembic/                # Database migrations
│   ├── tests/                  # Backend tests
│   └── requirements.txt        # Python dependencies
├── frontend/                    # React dashboard
│   ├── src/
│   │   ├── components/         # React components
│   │   ├── hooks/              # Custom React hooks
│   │   ├── services/           # API integration
│   │   ├── stores/             # State management (Zustand)
│   │   └── utils/              # Helper functions
│   ├── public/                 # Static assets
│   ├── package.json            # Node.js dependencies
│   └── tailwind.config.js      # Tailwind CSS configuration
├── ml/                         # AI/ML training pipeline
│   ├── data/                   # Training datasets
│   ├── models/                 # Trained models
│   ├── notebooks/              # Jupyter analysis notebooks
│   ├── training/               # Model training scripts
│   └── evaluation/             # Model evaluation tools
├── k8s/                        # Kubernetes deployment
│   ├── backend-deployment.yaml
│   ├── frontend-deployment.yaml
│   ├── postgres-deployment.yaml
│   ├── mqtt-deployment.yaml
│   └── ingress.yaml
├── docs/                       # Documentation
│   ├── api/                    # API documentation
│   ├── deployment/             # Deployment guides
│   └── hardware/               # Hardware setup guides
├── scripts/                    # Utility scripts
│   ├── setup_dev_env.sh       # Development environment setup
│   ├── deploy_production.sh   # Production deployment
│   └── backup_database.py     # Database backup utilities
├── docker-compose.yml          # Development environment
├── docker-compose.prod.yml     # Production environment
├── .github/workflows/          # CI/CD pipelines
└── README.md                   # Project overview
```

### Key Design Principles

#### Edge-First Architecture
- **Local Processing**: ESP32-CAM performs real-time analysis without cloud dependency
- **Smart Buffering**: Local data storage during network outages
- **Progressive Enhancement**: Cloud services provide advanced analytics and forecasting
- **Fault Tolerance**: System continues operating even with partial component failures

#### Scalable Cloud Backend
- **Microservices**: Loosely coupled services for different functionalities
- **Event-Driven**: MQTT and message queues for asynchronous processing
- **Horizontal Scaling**: Kubernetes-based auto-scaling for high load
- **Data Pipeline**: Efficient ingestion and processing of IoT sensor streams

#### Production-Ready Features
- **Security**: JWT authentication, API rate limiting, secure MQTT connections
- **Monitoring**: Comprehensive logging, metrics, and health checks
- **Deployment**: Infrastructure as Code with Docker and Kubernetes
- **Testing**: Unit, integration, and end-to-end test coverage

## Technology Integration Points

### ESP32 ↔ Cloud Communication
```cpp
// ESP32 firmware MQTT publishing
void publishSensorData() {
    DynamicJsonDocument doc(1024);
    doc["device_id"] = device_id;
    doc["timestamp"] = getUnixTimestamp();
    doc["particle_count"] = particle_count;
    doc["photodiode_value"] = photodiode_reading;
    doc["classification"] = tflite_result;
    doc["image_hash"] = image_md5_hash;
    
    char buffer[1024];
    serializeJson(doc, buffer);
    mqtt_client.publish("devices/" + device_id + "/sensor_data", buffer);
}
```

### Cloud ↔ Frontend Data Flow
```typescript
// React real-time data subscription
const useRealTimeData = () => {
  const [data, setData] = useState<SensorData[]>([]);
  
  useEffect(() => {
    const eventSource = new EventSource('/api/stream/sensor-data');
    eventSource.onmessage = (event) => {
      const newData = JSON.parse(event.data);
      setData(prev => updateTimeSeriesData(prev, newData));
    };
    return () => eventSource.close();
  }, []);
  
  return data;
};
```

This **Integrated Microplastic Detection and Forecasting System** represents a complete evolution from a hackathon prototype to a **production-ready IoT platform** capable of **large-scale environmental monitoring**, **AI-driven analytics**, and **decision support** for water quality management.
