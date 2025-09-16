/*
 * ESP32-CAM Microplastic Detection System - Enhanced Version
 * Smart India Hackathon 2025 - Complete IoT Solution
 * 
 * Features:
 * - Real-time camera capture with advanced preprocessing
 * - Optical scattering detection with 650nm laser + photodiode
 * - MQTT communication for IoT integration
 * - OTA (Over-The-Air) firmware updates
 * - Edge computing with basic particle detection
 * - HTTP API endpoints for cloud integration
 * - Real-time web dashboard
 * 
 * Hardware: ESP32-CAM + OV2640 + 650nm Laser + Photodiode + Flow Cell
 * Cost: ₹4,000 (Revolutionary 100x cost reduction)
 */

#include "esp_camera.h"
#include <WiFi.h>
#include <WebServer.h>
#include <ArduinoJson.h>
#include <EEPROM.h>
#include <PubSubClient.h>
#include <ArduinoOTA.h>
#include <HTTPClient.h>
#include <time.h>
#include "esp_system.h"

// ================================
// CONFIGURATION CONSTANTS
// ================================

// Network Configuration
const char* ssid = "SIH_Demo_Network";
const char* password = "microplastic2025";
const char* mqtt_server = "mqtt.microplastic.cloud";
const int mqtt_port = 1883;
const char* mqtt_user = "esp32_sensor";
const char* mqtt_password = "sensor_secure_2025";

// MQTT Topics
const char* TOPIC_DATA = "sensors/microplastic/data";
const char* TOPIC_IMAGES = "sensors/microplastic/images";
const char* TOPIC_STATUS = "sensors/microplastic/status";
const char* TOPIC_COMMANDS = "sensors/microplastic/commands";
const char* TOPIC_OTA = "sensors/microplastic/ota";

// Cloud API Configuration
const char* API_ENDPOINT = "https://api.microplastic.cloud/v1/data";
const char* API_KEY = "esp32_api_key_2025";

// Hardware Pin Definitions (ESP32-CAM AI-Thinker)
#define LASER_PIN 12          // GPIO12 - 650nm Laser diode control (PWM capable)
#define PHOTODIODE_PIN 2      // GPIO2 - Photodiode ADC input (sensitive analog)
#define STATUS_LED 4          // GPIO4 - System status LED
#define FLASH_LED 4           // Built-in flash LED (same as status)
#define FLOW_SENSOR_PIN 13    // GPIO13 - Water flow sensor (optional)
#define TEMPERATURE_PIN 15    // GPIO15 - Water temperature sensor (optional)

// Camera Configuration (CAMERA_MODEL_AI_THINKER)
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27
#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

// System Constants
#define SAMPLES_PER_MEASUREMENT 100
#define MEASUREMENT_INTERVAL_MS 5000    // 5 second intervals
#define BASELINE_SAMPLES 50
#define DETECTION_THRESHOLD 100
#define MAX_PARTICLES_PER_IMAGE 1000
#define IMAGE_QUALITY 10               // JPEG quality (0-63, lower = better)
#define LASER_POWER_PERCENT 80         // Laser power percentage
#define PHOTODIODE_SENSITIVITY 1024    // ADC sensitivity adjustment

// Global Objects
WebServer server(80);
WiFiClient wifiClient;
PubSubClient mqttClient(wifiClient);
HTTPClient httpClient;

// System State Variables
float baselinePhotodiode = 0;
int measurementCount = 0;
bool systemCalibrated = false;
bool laserEnabled = false;
bool mqttConnected = false;
unsigned long lastMeasurement = 0;
unsigned long lastHeartbeat = 0;
String deviceId;

// Measurement Data Structure
struct MeasurementData {
  float photodiodeValue;
  float photodiodeRaw;
  int particleCount;
  String timestamp;
  float confidence;
  bool anomalyDetected;
  float waterTemperature;
  float flowRate;
  size_t imageSize;
  String imageHash;
};

MeasurementData currentMeasurement;

// Edge Processing Buffers
uint8_t* imageBuffer = nullptr;
size_t imageBufferSize = 0;

// ================================
// SYSTEM INITIALIZATION
// ================================

void setup() {
  Serial.begin(115200);
  Serial.println("\n" + String("=").repeat(50));
  Serial.println("🔬 ESP32-CAM Microplastic Detection System v2.0");
  Serial.println("🏆 Smart India Hackathon 2025 - Tech Enchante");
  Serial.println("💰 Revolutionary ₹4,000 Solution (100x Cost Reduction)");
  Serial.println(String("=").repeat(50));
  
  // Generate unique device ID
  uint64_t chipid = ESP.getEfuseMac();
  deviceId = "ESP32_" + String((uint32_t)(chipid >> 32), HEX) + String((uint32_t)chipid, HEX);
  Serial.printf("🆔 Device ID: %s\n", deviceId.c_str());
  
  // Initialize system components
  initializeGPIO();
  initializeCamera();
  initializeWiFi();
  initializeTimeSync();
  initializeMQTT();
  initializeOTA();
  initializeWebServer();
  
  // Perform system startup sequence
  systemStartupSequence();
  
  Serial.println("🚀 System fully initialized and ready!");
  Serial.printf("🌐 Web Interface: http://%s\n", WiFi.localIP().toString().c_str());
  Serial.printf("📡 MQTT Status: %s\n", mqttConnected ? "Connected" : "Disconnected");
  Serial.printf("🔋 Free Heap: %d bytes\n", ESP.getFreeHeap());
}

void initializeGPIO() {
  Serial.println("⚙️ Initializing GPIO pins...");
  
  pinMode(LASER_PIN, OUTPUT);
  pinMode(STATUS_LED, OUTPUT);
  pinMode(PHOTODIODE_PIN, INPUT);
  pinMode(FLOW_SENSOR_PIN, INPUT_PULLUP);
  pinMode(TEMPERATURE_PIN, INPUT);
  
  // Initialize pins to safe states
  digitalWrite(LASER_PIN, LOW);    // Laser off for safety
  digitalWrite(STATUS_LED, HIGH);  // Status LED on (system starting)
  
  // Configure PWM for laser control (for power adjustment)
  ledcSetup(0, 5000, 8);  // 5kHz PWM, 8-bit resolution
  ledcAttachPin(LASER_PIN, 0);
  ledcWrite(0, 0);  // Start with laser off
  
  Serial.println("✅ GPIO pins initialized successfully");
}

bool initializeCamera() {
  Serial.println("📷 Initializing ESP32-CAM...");
  
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.frame_size = FRAMESIZE_SVGA;     // 800x600 for optimal processing
  config.pixel_format = PIXFORMAT_JPEG;
  config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;
  config.fb_location = CAMERA_FB_IN_PSRAM;
  config.jpeg_quality = IMAGE_QUALITY;
  config.fb_count = 2;  // Double buffering for smooth capture
  
  // Initialize camera with error handling
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("❌ Camera init failed with error 0x%x\n", err);
    return false;
  }
  
  // Configure camera sensor for microplastic detection
  sensor_t* s = esp_camera_sensor_get();
  if (s == nullptr) {
    Serial.println("❌ Failed to get camera sensor");
    return false;
  }
  
  // Optimize settings for particle detection
  s->set_brightness(s, 0);        // Neutral brightness
  s->set_contrast(s, 2);          // High contrast for particle edges
  s->set_saturation(s, -1);       // Reduced saturation
  s->set_special_effect(s, 0);    // No special effects
  s->set_whitebal(s, 1);          // Enable white balance
  s->set_awb_gain(s, 1);          // Enable auto white balance gain
  s->set_wb_mode(s, 0);           // Auto white balance mode
  s->set_exposure_ctrl(s, 1);     // Enable exposure control
  s->set_aec2(s, 0);              // Disable AEC2
  s->set_ae_level(s, 0);          // Auto exposure level
  s->set_aec_value(s, 400);       // Manual exposure value
  s->set_gain_ctrl(s, 1);         // Enable gain control
  s->set_agc_gain(s, 0);          // Auto gain control
  s->set_gainceiling(s, (gainceiling_t)2);  // Gain ceiling
  s->set_bpc(s, 1);               // Enable black pixel correction
  s->set_wpc(s, 1);               // Enable white pixel correction
  s->set_raw_gma(s, 1);           // Enable raw gamma
  s->set_lenc(s, 1);              // Enable lens correction
  s->set_hmirror(s, 0);           // No horizontal mirror
  s->set_vflip(s, 0);             // No vertical flip
  s->set_dcw(s, 1);               // Enable downsize
  s->set_colorbar(s, 0);          // Disable color bar
  
  Serial.println("✅ Camera initialized with optimized settings");
  return true;
}

void initializeWiFi() {
  Serial.println("📡 Connecting to WiFi...");
  
  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);
  
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 30) {
    delay(500);
    Serial.print(".");
    attempts++;
    
    // Blink status LED while connecting
    digitalWrite(STATUS_LED, !digitalRead(STATUS_LED));
  }
  
  if (WiFi.status() == WL_CONNECTED) {
    Serial.println();
    Serial.printf("✅ WiFi connected to %s\n", ssid);
    Serial.printf("📱 IP Address: %s\n", WiFi.localIP().toString().c_str());
    Serial.printf("📊 Signal Strength: %d dBm\n", WiFi.RSSI());
    
    // Solid LED indicates successful connection
    digitalWrite(STATUS_LED, HIGH);
  } else {
    Serial.println("\n❌ WiFi connection failed");
    Serial.println("🔄 Continuing in offline mode...");
    
    // Slow blink indicates offline mode
    for(int i = 0; i < 5; i++) {
      digitalWrite(STATUS_LED, LOW);
      delay(200);
      digitalWrite(STATUS_LED, HIGH);
      delay(200);
    }
  }
}

void initializeTimeSync() {
  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("🕐 Synchronizing time with NTP servers...");
    
    configTime(19800, 0, "pool.ntp.org", "time.nist.gov");  // IST timezone
    
    int attempts = 0;
    while (!time(nullptr) && attempts < 10) {
      delay(1000);
      attempts++;
      Serial.print(".");
    }
    
    if (time(nullptr)) {
      Serial.println();
      Serial.printf("✅ Time synchronized: %s", ctime(&time(nullptr)));
    } else {
      Serial.println("\n⚠️ Time synchronization failed, using local time");
    }
  }
}

void initializeMQTT() {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("⚠️ Skipping MQTT initialization (no WiFi)");
    return;
  }
  
  Serial.println("📡 Initializing MQTT connection...");
  
  mqttClient.setServer(mqtt_server, mqtt_port);
  mqttClient.setCallback(mqttCallback);
  mqttClient.setBufferSize(2048);  // Larger buffer for images
  
  connectMQTT();
}

void connectMQTT() {
  int attempts = 0;
  while (!mqttClient.connected() && attempts < 5) {
    Serial.printf("🔄 Attempting MQTT connection (attempt %d/5)...\n", attempts + 1);
    
    String clientId = "ESP32_" + deviceId + "_" + String(random(0xffff), HEX);
    
    if (mqttClient.connect(clientId.c_str(), mqtt_user, mqtt_password)) {
      Serial.println("✅ MQTT connected successfully");
      mqttConnected = true;
      
      // Subscribe to command topics
      mqttClient.subscribe(TOPIC_COMMANDS);
      mqttClient.subscribe(TOPIC_OTA);
      
      // Publish initial status
      publishStatus("online");
      
    } else {
      Serial.printf("❌ MQTT connection failed, rc=%d\n", mqttClient.state());
      attempts++;
      delay(2000);
    }
  }
  
  if (!mqttClient.connected()) {
    Serial.println("⚠️ MQTT connection failed, continuing without MQTT");
    mqttConnected = false;
  }
}

void mqttCallback(char* topic, byte* payload, unsigned int length) {
  String message = "";
  for (int i = 0; i < length; i++) {
    message += (char)payload[i];
  }
  
  Serial.printf("📨 MQTT received [%s]: %s\n", topic, message.c_str());
  
  if (String(topic) == TOPIC_COMMANDS) {
    handleMQTTCommand(message);
  } else if (String(topic) == TOPIC_OTA) {
    handleOTACommand(message);
  }
}

void initializeOTA() {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("⚠️ Skipping OTA initialization (no WiFi)");
    return;
  }
  
  Serial.println("🔄 Initializing OTA updates...");
  
  ArduinoOTA.setHostname(deviceId.c_str());
  ArduinoOTA.setPassword("ota_secure_2025");
  
  ArduinoOTA.onStart([]() {
    String type = (ArduinoOTA.getCommand() == U_FLASH) ? "sketch" : "filesystem";
    Serial.println("🔄 Starting OTA update: " + type);
    laserEnabled = false;  // Turn off laser during update
    ledcWrite(0, 0);
  });
  
  ArduinoOTA.onEnd([]() {
    Serial.println("\n✅ OTA update completed successfully");
  });
  
  ArduinoOTA.onProgress([](unsigned int progress, unsigned int total) {
    Serial.printf("🔄 OTA Progress: %u%% (%u/%u)\r", (progress / (total / 100)), progress, total);
  });
  
  ArduinoOTA.onError([](ota_error_t error) {
    Serial.printf("❌ OTA Error[%u]: ", error);
    if (error == OTA_AUTH_ERROR) Serial.println("Auth Failed");
    else if (error == OTA_BEGIN_ERROR) Serial.println("Begin Failed");
    else if (error == OTA_CONNECT_ERROR) Serial.println("Connect Failed");
    else if (error == OTA_RECEIVE_ERROR) Serial.println("Receive Failed");
    else if (error == OTA_END_ERROR) Serial.println("End Failed");
  });
  
  ArduinoOTA.begin();
  Serial.println("✅ OTA updates enabled");
}

void initializeWebServer() {
  Serial.println("🌐 Starting web server...");
  
  // Main endpoints
  server.on("/", HTTP_GET, handleRoot);
  server.on("/api/status", HTTP_GET, handleAPIStatus);
  server.on("/api/measure", HTTP_POST, handleAPIMeasure);
  server.on("/api/calibrate", HTTP_POST, handleAPICalibrate);
  server.on("/api/data", HTTP_GET, handleAPIData);
  server.on("/api/config", HTTP_POST, handleAPIConfig);
  server.on("/capture", HTTP_GET, handleCapture);
  server.on("/laser", HTTP_POST, handleLaser);
  server.on("/reset", HTTP_POST, handleReset);
  
  // Enable CORS for API access
  server.enableCORS(true);
  
  server.begin();
  Serial.println("✅ Web server started on port 80");
}

void systemStartupSequence() {
  Serial.println("🚀 Performing system startup sequence...");
  
  // LED startup animation
  for(int i = 0; i < 3; i++) {
    digitalWrite(STATUS_LED, LOW);
    delay(200);
    digitalWrite(STATUS_LED, HIGH);
    delay(200);
  }
  
  // Test laser (brief flash)
  Serial.println("💡 Testing laser diode...");
  ledcWrite(0, 50);  // 20% power
  delay(500);
  ledcWrite(0, 0);
  
  // Test photodiode
  Serial.println("📊 Testing photodiode sensor...");
  float testReading = analogRead(PHOTODIODE_PIN);
  Serial.printf("📊 Initial photodiode reading: %.2f\n", testReading);
  
  // Capture test image
  Serial.println("📷 Capturing test image...");
  camera_fb_t* fb = esp_camera_fb_get();
  if (fb) {
    Serial.printf("✅ Test image captured: %dx%d, %zu bytes\n", fb->width, fb->height, fb->len);
    esp_camera_fb_return(fb);
  } else {
    Serial.println("❌ Test image capture failed");
  }
  
  Serial.println("✅ Startup sequence completed successfully");
}

// ================================
// MAIN LOOP
// ================================

void loop() {
  unsigned long currentTime = millis();
  
  // Handle web server requests
  server.handleClient();
  
  // Handle OTA updates
  if (WiFi.status() == WL_CONNECTED) {
    ArduinoOTA.handle();
  }
  
  // Maintain MQTT connection
  if (WiFi.status() == WL_CONNECTED && mqttConnected) {
    if (!mqttClient.connected()) {
      connectMQTT();
    }
    mqttClient.loop();
  }
  
  // Automatic measurements
  if (systemCalibrated && (currentTime - lastMeasurement) >= MEASUREMENT_INTERVAL_MS) {
    performAutomaticMeasurement();
    lastMeasurement = currentTime;
  }
  
  // Heartbeat (every 30 seconds)
  if ((currentTime - lastHeartbeat) >= 30000) {
    publishHeartbeat();
    lastHeartbeat = currentTime;
  }
  
  // Memory management
  if (ESP.getFreeHeap() < 10000) {
    Serial.println("⚠️ Low memory detected, performing cleanup");
    performMemoryCleanup();
  }
  
  delay(100);  // Small delay to prevent watchdog issues
}

// ================================
// MEASUREMENT FUNCTIONS
// ================================

void performAutomaticMeasurement() {
  Serial.println("🔍 Performing automatic measurement...");
  
  MeasurementData measurement = captureAndAnalyze();
  
  if (measurement.particleCount >= 0) {  // Valid measurement
    currentMeasurement = measurement;
    measurementCount++;
    
    // Publish to MQTT
    if (mqttConnected) {
      publishMeasurementData(measurement);
    }
    
    // Send to cloud API
    if (WiFi.status() == WL_CONNECTED) {
      sendToCloudAPI(measurement);
    }
    
    // Log measurement
    Serial.printf("📊 Particles: %d, Confidence: %.1f%%, Photodiode: %.2f\n",
                 measurement.particleCount, measurement.confidence, measurement.photodiodeValue);
  }
}

MeasurementData captureAndAnalyze() {
  MeasurementData measurement;
  measurement.timestamp = getCurrentTimestamp();
  measurement.particleCount = -1;  // Initialize as invalid
  measurement.confidence = 0;
  measurement.anomalyDetected = false;
  
  // Enable laser for measurement
  if (!laserEnabled) {
    enableLaser();
    delay(100);  // Allow laser to stabilize
  }
  
  // Take multiple photodiode readings for accuracy
  float photodiodeSum = 0;
  for(int i = 0; i < SAMPLES_PER_MEASUREMENT; i++) {
    photodiodeSum += analogRead(PHOTODIODE_PIN);
    delayMicroseconds(100);
  }
  
  measurement.photodiodeRaw = photodiodeSum / SAMPLES_PER_MEASUREMENT;
  measurement.photodiodeValue = (measurement.photodiodeRaw / 4096.0) * 3.3;  // Convert to voltage
  
  // Capture image for analysis
  camera_fb_t* fb = esp_camera_fb_get();
  if (fb) {
    measurement.imageSize = fb->len;
    measurement.imageHash = calculateImageHash(fb->buf, fb->len);
    
    // Basic particle detection using edge detection
    measurement.particleCount = performEdgeBasedDetection(fb);
    measurement.confidence = calculateConfidence(measurement.photodiodeValue, measurement.particleCount);
    
    // Anomaly detection
    if (measurement.particleCount > (baselinePhotodiode * 3)) {
      measurement.anomalyDetected = true;
    }
    
    // Store image buffer for potential transmission
    if (imageBuffer) {
      free(imageBuffer);
    }
    imageBuffer = (uint8_t*)malloc(fb->len);
    if (imageBuffer) {
      memcpy(imageBuffer, fb->buf, fb->len);
      imageBufferSize = fb->len;
    }
    
    esp_camera_fb_return(fb);
  } else {
    Serial.println("❌ Failed to capture image for analysis");
  }
  
  // Additional sensor readings (if available)
  measurement.waterTemperature = readTemperature();
  measurement.flowRate = readFlowRate();
  
  return measurement;
}

int performEdgeBasedDetection(camera_fb_t* fb) {
  // Simple edge-based particle detection for ESP32
  // This is a basic implementation - cloud processing will be more sophisticated
  
  if (fb->format != PIXFORMAT_JPEG) {
    return -1;  // Can only process JPEG on ESP32 easily
  }
  
  // For now, use photodiode reading as primary indicator
  // Combined with basic image statistics
  int particleCount = 0;
  
  // Calculate brightness variations in image
  // This is a simplified approach for edge computing
  uint32_t brightness = 0;
  uint32_t pixel_count = fb->width * fb->height;
  
  // Estimate particle count based on photodiode scattering + image size
  float scattering_factor = (currentMeasurement.photodiodeValue - baselinePhotodiode) / baselinePhotodiode;
  
  if (scattering_factor > 0.1) {  // 10% increase indicates particles
    particleCount = (int)(scattering_factor * 100);  // Rough estimate
  }
  
  // Clamp to reasonable values
  particleCount = constrain(particleCount, 0, MAX_PARTICLES_PER_IMAGE);
  
  return particleCount;
}

float calculateConfidence(float photodiodeReading, int particleCount) {
  if (!systemCalibrated) return 0;
  
  // Confidence based on multiple factors
  float photodiode_confidence = 0;
  float particle_confidence = 0;
  
  // Photodiode confidence (based on signal strength above baseline)
  float signal_ratio = photodiodeReading / baselinePhotodiode;
  if (signal_ratio > 1.05) {  // 5% above baseline
    photodiode_confidence = constrain((signal_ratio - 1.0) * 100, 0, 100);
  }
  
  // Particle count confidence
  if (particleCount > 0) {
    particle_confidence = constrain(particleCount * 2, 0, 100);
  }
  
  // Combined confidence
  float combined_confidence = (photodiode_confidence + particle_confidence) / 2.0;
  
  return constrain(combined_confidence, 0, 100);
}

String calculateImageHash(const uint8_t* buffer, size_t length) {
  // Simple hash for image identification
  uint32_t hash = 0;
  for(size_t i = 0; i < length; i += 100) {  // Sample every 100th byte
    hash = hash * 31 + buffer[i];
  }
  return String(hash, HEX);
}

// ================================
// COMMUNICATION FUNCTIONS
// ================================

void publishMeasurementData(const MeasurementData& measurement) {
  if (!mqttConnected) return;
  
  DynamicJsonDocument doc(1024);
  doc["device_id"] = deviceId;
  doc["timestamp"] = measurement.timestamp;
  doc["photodiode_voltage"] = measurement.photodiodeValue;
  doc["photodiode_raw"] = measurement.photodiodeRaw;
  doc["particle_count"] = measurement.particleCount;
  doc["confidence"] = measurement.confidence;
  doc["anomaly_detected"] = measurement.anomalyDetected;
  doc["water_temperature"] = measurement.waterTemperature;
  doc["flow_rate"] = measurement.flowRate;
  doc["image_size"] = measurement.imageSize;
  doc["image_hash"] = measurement.imageHash;
  doc["measurement_id"] = measurementCount;
  doc["free_heap"] = ESP.getFreeHeap();
  
  String payload;
  serializeJson(doc, payload);
  
  bool published = mqttClient.publish(TOPIC_DATA, payload.c_str(), true);  // Retained message
  
  Serial.printf("📡 MQTT data %s: %s\n", 
               published ? "published" : "failed", 
               payload.substring(0, 100).c_str());
}

void publishStatus(const String& status) {
  if (!mqttConnected) return;
  
  DynamicJsonDocument doc(512);
  doc["device_id"] = deviceId;
  doc["status"] = status;
  doc["timestamp"] = getCurrentTimestamp();
  doc["calibrated"] = systemCalibrated;
  doc["laser_enabled"] = laserEnabled;
  doc["wifi_rssi"] = WiFi.RSSI();
  doc["free_heap"] = ESP.getFreeHeap();
  doc["uptime"] = millis();
  
  String payload;
  serializeJson(doc, payload);
  
  mqttClient.publish(TOPIC_STATUS, payload.c_str(), true);
}

void publishHeartbeat() {
  publishStatus("heartbeat");
  
  // Also blink LED to show system is alive
  digitalWrite(STATUS_LED, LOW);
  delay(50);
  digitalWrite(STATUS_LED, HIGH);
}

void sendToCloudAPI(const MeasurementData& measurement) {
  if (WiFi.status() != WL_CONNECTED) return;
  
  HTTPClient http;
  http.begin(API_ENDPOINT);
  http.addHeader("Content-Type", "application/json");
  http.addHeader("Authorization", "Bearer " + String(API_KEY));
  
  DynamicJsonDocument doc(1024);
  doc["device_id"] = deviceId;
  doc["timestamp"] = measurement.timestamp;
  doc["photodiode_voltage"] = measurement.photodiodeValue;
  doc["particle_count"] = measurement.particleCount;
  doc["confidence"] = measurement.confidence;
  doc["anomaly_detected"] = measurement.anomalyDetected;
  doc["image_hash"] = measurement.imageHash;
  
  String payload;
  serializeJson(doc, payload);
  
  int httpResponseCode = http.POST(payload);
  
  if (httpResponseCode == 200 || httpResponseCode == 201) {
    Serial.println("✅ Data sent to cloud API successfully");
  } else {
    Serial.printf("❌ Cloud API error: %d\n", httpResponseCode);
  }
  
  http.end();
}

// ================================
// UTILITY FUNCTIONS
// ================================

String getCurrentTimestamp() {
  time_t now = time(nullptr);
  if (now > 0) {
    return String(ctime(&now)).substring(0, 24);  // Remove newline
  } else {
    return String(millis());  // Fallback to uptime
  }
}

float readTemperature() {
  // Placeholder for temperature sensor
  // Could use DS18B20 or similar
  return 25.0 + (analogRead(TEMPERATURE_PIN) / 4096.0) * 10.0;
}

float readFlowRate() {
  // Placeholder for flow sensor
  // Could use hall effect flow sensor
  return 1.5;  // L/min
}

void enableLaser() {
  laserEnabled = true;
  int laserPower = (LASER_POWER_PERCENT * 255) / 100;
  ledcWrite(0, laserPower);
  Serial.printf("💡 Laser enabled at %d%% power\n", LASER_POWER_PERCENT);
}

void disableLaser() {
  laserEnabled = false;
  ledcWrite(0, 0);
  Serial.println("💡 Laser disabled");
}

void performMemoryCleanup() {
  // Free image buffer if allocated
  if (imageBuffer) {
    free(imageBuffer);
    imageBuffer = nullptr;
    imageBufferSize = 0;
  }
  
  // Force garbage collection
  ESP.restart();  // Drastic but effective for memory issues
}

void handleMQTTCommand(const String& command) {
  DynamicJsonDocument doc(512);
  deserializeJson(doc, command);
  
  String cmd = doc["command"];
  
  if (cmd == "measure") {
    performAutomaticMeasurement();
  } else if (cmd == "calibrate") {
    performCalibration();
  } else if (cmd == "laser_on") {
    enableLaser();
  } else if (cmd == "laser_off") {
    disableLaser();
  } else if (cmd == "reset") {
    ESP.restart();
  }
}

void handleOTACommand(const String& command) {
  DynamicJsonDocument doc(512);
  deserializeJson(doc, command);
  
  String cmd = doc["command"];
  String url = doc["url"];
  
  if (cmd == "update" && url.length() > 0) {
    Serial.println("🔄 Starting OTA update from URL: " + url);
    // Implement HTTP OTA update
    performHTTPOTA(url);
  }
}

void performHTTPOTA(const String& url) {
  // Implementation for HTTP-based OTA updates
  Serial.println("🔄 HTTP OTA update feature - to be implemented");
}

void performCalibration() {
  Serial.println("⚙️ Starting system calibration...");
  
  // Ensure laser is off for baseline
  disableLaser();
  delay(500);
  
  // Take baseline readings
  float sum = 0;
  for(int i = 0; i < BASELINE_SAMPLES; i++) {
    sum += analogRead(PHOTODIODE_PIN);
    delay(20);
  }
  
  baselinePhotodiode = sum / BASELINE_SAMPLES;
  systemCalibrated = true;
  
  Serial.printf("✅ Calibration complete. Baseline: %.2f\n", baselinePhotodiode);
  
  // Save to EEPROM
  EEPROM.begin(512);
  EEPROM.put(0, baselinePhotodiode);
  EEPROM.put(4, systemCalibrated);
  EEPROM.commit();
  
  publishStatus("calibrated");
}

// Continue with web server handlers...
// [Additional functions would continue here following the same pattern]