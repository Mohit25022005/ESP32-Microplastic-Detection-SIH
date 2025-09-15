from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import uvicorn
import asyncio
from datetime import datetime
from typing import List, Dict, Any
import json
import random

# Initialize FastAPI app
app = FastAPI(
    title="Integrated Microplastic Detection API",
    description="Real-time microplastic detection and forecasting system",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://frontend:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for demo (replace with PostgreSQL in production)
devices_data = {}
sensor_readings = []

# Models for demo
class SensorData:
    def __init__(self, device_id: str, particle_count: int, photodiode_value: float):
        self.device_id = device_id
        self.particle_count = particle_count
        self.photodiode_value = photodiode_value
        self.timestamp = datetime.now().isoformat()
        self.location = {"lat": 28.6139 + random.uniform(-0.1, 0.1), "lon": 77.2090 + random.uniform(-0.1, 0.1)}

@app.get("/")
async def root():
    return {
        "message": "Integrated Microplastic Detection and Forecasting System",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "devices": "/api/devices",
            "dashboard": "/api/dashboard"
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "database": "connected",
            "mqtt": "connected", 
            "redis": "connected"
        }
    }

@app.get("/api/devices")
async def get_devices():
    """Get list of all registered ESP32 devices"""
    demo_devices = [
        {
            "id": "esp32_001",
            "name": "Yamuna River Sensor 1",
            "location": {"lat": 28.6139, "lon": 77.2090},
            "status": "online",
            "last_seen": datetime.now().isoformat()
        },
        {
            "id": "esp32_002", 
            "name": "Ganga River Sensor 1",
            "location": {"lat": 25.3176, "lon": 82.9739},
            "status": "online",
            "last_seen": datetime.now().isoformat()
        }
    ]
    return {"devices": demo_devices}

@app.get("/api/devices/{device_id}/status")
async def get_device_status(device_id: str):
    """Get current device status and health metrics"""
    return {
        "device_id": device_id,
        "status": "online",
        "battery_level": random.randint(60, 100),
        "signal_strength": random.randint(-70, -40),
        "last_measurement": datetime.now().isoformat(),
        "daily_measurements": random.randint(100, 500),
        "alerts": []
    }

@app.post("/api/devices/{device_id}/data")
async def ingest_sensor_data(device_id: str, data: dict):
    """Ingest real-time data from ESP32 devices"""
    # Store sensor data
    sensor_data = SensorData(
        device_id=device_id,
        particle_count=data.get("particle_count", 0),
        photodiode_value=data.get("photodiode_value", 0.0)
    )
    sensor_readings.append(sensor_data.__dict__)
    
    # Keep only last 1000 readings for demo
    if len(sensor_readings) > 1000:
        sensor_readings.pop(0)
    
    return {
        "status": "received",
        "device_id": device_id,
        "timestamp": sensor_data.timestamp
    }

@app.get("/api/analytics/real-time-dashboard")
async def get_dashboard_data():
    """Real-time data for dashboard visualization"""
    # Generate demo data
    current_readings = []
    for i in range(10):
        reading = {
            "timestamp": datetime.now().isoformat(),
            "device_id": f"esp32_{random.choice(['001', '002'])}",
            "particle_count": random.randint(0, 50),
            "photodiode_value": random.uniform(100, 4000),
            "water_temperature": random.uniform(15, 35),
            "ph_level": random.uniform(6.5, 8.5),
            "location": {
                "lat": 28.6139 + random.uniform(-0.1, 0.1),
                "lon": 77.2090 + random.uniform(-0.1, 0.1)
            }
        }
        current_readings.append(reading)
    
    return {
        "real_time_data": current_readings,
        "summary": {
            "total_devices": 2,
            "online_devices": 2,
            "avg_particle_count": sum(r["particle_count"] for r in current_readings) / len(current_readings),
            "pollution_level": "moderate",
            "last_updated": datetime.now().isoformat()
        }
    }

@app.get("/api/analytics/pollution-forecast")
async def get_pollution_forecast(river_id: str = "yamuna", days: int = 7):
    """Get AI-generated pollution forecasts"""
    # Generate demo forecast data
    forecast_data = []
    for i in range(days):
        forecast = {
            "date": (datetime.now().date()).isoformat(),
            "predicted_concentration": random.uniform(10, 100),
            "confidence_interval": {
                "lower": random.uniform(5, 80),
                "upper": random.uniform(20, 120)
            },
            "risk_level": random.choice(["low", "moderate", "high"]),
            "factors": ["rainfall", "industrial_discharge", "seasonal_variation"]
        }
        forecast_data.append(forecast)
    
    return {
        "river_id": river_id,
        "forecast_period": f"{days} days",
        "forecasts": forecast_data,
        "model_version": "v1.0",
        "generated_at": datetime.now().isoformat()
    }

@app.post("/api/alerts/configure")
async def configure_alerts(alert_config: dict):
    """Configure SMS/email alert thresholds"""
    return {
        "status": "configured",
        "alert_config": alert_config,
        "updated_at": datetime.now().isoformat()
    }

@app.get("/demo")
async def demo_page():
    """Simple demo page to test the system"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Microplastic Detection System - Demo</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
            .container { max-width: 1200px; margin: 0 auto; }
            .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                     color: white; padding: 20px; border-radius: 10px; text-align: center; margin-bottom: 20px; }
            .card { background: white; padding: 20px; margin: 10px 0; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
            .btn { background: #4CAF50; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; }
            .btn:hover { background: #45a049; }
            #chartContainer { height: 400px; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🔬 Microplastic Detection System</h1>
                <p>Real-time Environmental Monitoring Dashboard</p>
            </div>
            
            <div class="grid">
                <div class="card">
                    <h3>📊 Live Data</h3>
                    <div id="liveData">Loading...</div>
                    <button class="btn" onclick="refreshData()">🔄 Refresh Data</button>
                </div>
                
                <div class="card">
                    <h3>📍 Device Status</h3>
                    <div id="deviceStatus">Loading...</div>
                    <button class="btn" onclick="checkDevices()">📱 Check Devices</button>
                </div>
            </div>
            
            <div class="card">
                <h3>📈 Pollution Trends</h3>
                <canvas id="trendChart" width="400" height="200"></canvas>
            </div>
            
            <div class="card">
                <h3>🔮 AI Forecasting</h3>
                <div id="forecastData">Loading...</div>
                <button class="btn" onclick="getForecast()">🤖 Get AI Forecast</button>
            </div>
        </div>
        
        <script>
            let chart;
            
            async function refreshData() {
                try {
                    const response = await fetch('/api/analytics/real-time-dashboard');
                    const data = await response.json();
                    document.getElementById('liveData').innerHTML = `
                        <p><strong>Active Devices:</strong> ${data.summary.online_devices}</p>
                        <p><strong>Avg Particle Count:</strong> ${data.summary.avg_particle_count.toFixed(1)}</p>
                        <p><strong>Pollution Level:</strong> ${data.summary.pollution_level}</p>
                        <p><strong>Last Updated:</strong> ${new Date(data.summary.last_updated).toLocaleTimeString()}</p>
                    `;
                    updateChart(data.real_time_data);
                } catch (error) {
                    document.getElementById('liveData').innerHTML = 'Error loading data';
                }
            }
            
            async function checkDevices() {
                try {
                    const response = await fetch('/api/devices');
                    const data = await response.json();
                    document.getElementById('deviceStatus').innerHTML = data.devices.map(device => `
                        <div style="margin: 10px 0; padding: 10px; background: #f0f8ff; border-radius: 5px;">
                            <strong>${device.name}</strong><br>
                            Status: <span style="color: green;">●</span> ${device.status}<br>
                            Location: ${device.location.lat.toFixed(4)}, ${device.location.lon.toFixed(4)}
                        </div>
                    `).join('');
                } catch (error) {
                    document.getElementById('deviceStatus').innerHTML = 'Error loading devices';
                }
            }
            
            async function getForecast() {
                try {
                    const response = await fetch('/api/analytics/pollution-forecast');
                    const data = await response.json();
                    document.getElementById('forecastData').innerHTML = `
                        <p><strong>River:</strong> ${data.river_id}</p>
                        <p><strong>Forecast Period:</strong> ${data.forecast_period}</p>
                        <p><strong>Model Version:</strong> ${data.model_version}</p>
                        <div style="margin-top: 10px;">
                            ${data.forecasts.slice(0, 3).map(f => `
                                <div style="margin: 5px 0; padding: 5px; background: #f9f9f9; border-radius: 3px;">
                                    Date: ${f.date} | Risk: <strong>${f.risk_level}</strong> | Concentration: ${f.predicted_concentration.toFixed(1)}
                                </div>
                            `).join('')}
                        </div>
                    `;
                } catch (error) {
                    document.getElementById('forecastData').innerHTML = 'Error loading forecast';
                }
            }
            
            function updateChart(data) {
                const ctx = document.getElementById('trendChart').getContext('2d');
                
                if (chart) {
                    chart.destroy();
                }
                
                chart = new Chart(ctx, {
                    type: 'line',
                    data: {
                        labels: data.map((_, i) => `Point ${i + 1}`),
                        datasets: [{
                            label: 'Particle Count',
                            data: data.map(d => d.particle_count),
                            borderColor: 'rgb(75, 192, 192)',
                            backgroundColor: 'rgba(75, 192, 192, 0.1)',
                            tension: 0.1
                        }]
                    },
                    options: {
                        responsive: true,
                        scales: {
                            y: {
                                beginAtZero: true
                            }
                        }
                    }
                });
            }
            
            // Auto-refresh every 10 seconds
            setInterval(refreshData, 10000);
            
            // Initial load
            refreshData();
            checkDevices();
            getForecast();
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)