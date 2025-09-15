# Assembly Guide - ESP32 Microplastic Detection System

## 🛠️ Complete Assembly Instructions

### Prerequisites
- Basic soldering skills
- Multimeter for testing
- 3D printer access (or outsourced printing)
- Computer for firmware upload

## 📋 Pre-Assembly Checklist

### Tools Required
- [ ] Soldering iron (25-40W)
- [ ] Solder (rosin core, 0.8mm)
- [ ] Wire strippers
- [ ] Small screwdrivers (Phillips, flathead)
- [ ] Multimeter
- [ ] Hot glue gun
- [ ] Drill with small bits

### Components Verification
- [ ] ESP32-CAM module
- [ ] 650nm laser diode
- [ ] Photodiode sensor
- [ ] Sample chamber
- [ ] Power supply
- [ ] Connecting wires
- [ ] Enclosure parts
- [ ] Mounting hardware

## 🔌 Wiring Diagram

```
ESP32-CAM Pin Configuration:
                     ┌─────────────────┐
                     │    ESP32-CAM    │
                     │                 │
    5V Power    ────►│ 5V          GND │◄──── Ground
    Laser Control───►│ GPIO12      3V3 │◄──── 3.3V Out
    Photodiode  ────►│ GPIO2 (ADC) IO0 │
                     │             IO1 │
    WiFi LED    ────►│ GPIO4       IO3 │
                     │                 │
                     └─────────────────┘
```

### Detailed Connections

| ESP32-CAM Pin | Component | Wire Color | Function |
|---------------|-----------|------------|----------|
| 5V | Power Supply + | Red | Main power input |
| GND | Power Supply - | Black | Ground reference |
| GPIO12 | Laser Diode + | Yellow | Laser control |
| 3V3 | Laser Diode + | Red | 3.3V power |
| GPIO2 (ADC) | Photodiode Signal | Blue | Analog input |
| 3V3 | Photodiode VCC | Red | Sensor power |
| GND | Photodiode GND | Black | Sensor ground |
| GPIO4 | Status LED | Green | System indicator |

## 🏗️ Step-by-Step Assembly

### Phase 1: Prepare the Base Platform

#### 1.1 3D Print Base Components
```
Files to print:
- base_platform.stl
- camera_mount.stl
- laser_mount.stl
- photodiode_mount.stl
- sample_chamber_holder.stl
```

**Print Settings:**
- Layer Height: 0.2mm
- Infill: 20%
- Material: PLA or ABS
- Supports: Yes for overhangs >45°

#### 1.2 Post-Processing
1. Remove support material carefully
2. Sand critical mounting surfaces
3. Test fit all components before assembly
4. Drill pilot holes if needed

### Phase 2: Electronics Preparation

#### 2.1 ESP32-CAM Setup
1. **Initial Testing:**
   ```cpp
   // Test code - upload before assembly
   void setup() {
     Serial.begin(115200);
     Serial.println("ESP32-CAM Test");
   }
   
   void loop() {
     Serial.println("System OK");
     delay(1000);
   }
   ```

2. **Pin Header Installation:**
   - Solder pin headers to ESP32-CAM
   - Test continuity with multimeter
   - Verify power supply connections

#### 2.2 Sensor Preparation
1. **Photodiode Circuit:**
   ```
   Photodiode → 10kΩ resistor → GPIO2 (ADC)
                     ↓
                   3.3V
   ```

2. **Laser Diode Circuit:**
   ```
   GPIO12 → 220Ω resistor → Laser Diode +
   3.3V ─────────────────── Laser Diode +
   GND  ─────────────────── Laser Diode -
   ```

### Phase 3: Optical System Assembly

#### 3.1 Laser Mount Installation
1. Secure laser diode in 3D printed mount
2. Align laser beam horizontally
3. Test beam path with white paper target
4. Lock position with set screws

#### 3.2 Sample Chamber Positioning
1. Place chamber in center of optical path
2. Ensure laser beam passes through chamber
3. Mark entry and exit points
4. Secure chamber holder to base

#### 3.3 Photodiode Alignment
1. Position photodiode 90° to laser beam
2. Optimize for maximum scattered light collection
3. Shield from direct laser light
4. Verify signal response with test particles

### Phase 4: Camera Integration

#### 4.1 Camera Mount Setup
1. Install ESP32-CAM in adjustable mount
2. Position for optimal sample chamber viewing
3. Ensure adequate working distance (30mm)
4. Test focus with calibration target

#### 4.2 Lighting Optimization
1. Add diffused LED illumination if needed
2. Minimize reflections from chamber walls
3. Test image quality with various samples
4. Adjust camera settings for best contrast

### Phase 5: Electronics Integration

#### 5.1 Main Board Connection
Create a simple PCB or breadboard layout:

```
Breadboard Layout:
    +5V Rail: ESP32-CAM, Laser power
    +3.3V Rail: Photodiode, sensors
    GND Rail: All grounds
    
Signal Connections:
    GPIO12 → Laser control
    GPIO2  → Photodiode signal
    GPIO4  → Status LED
```

#### 5.2 Power System
1. Connect 5V power supply to breadboard
2. Add power switch for system control
3. Install power indicator LED
4. Test all voltage levels with multimeter

### Phase 6: Software Installation

#### 6.1 Development Environment Setup
1. Install Arduino IDE
2. Add ESP32 board package
3. Install required libraries:
   ```
   - WiFi
   - WebServer
   - ArduinoJson
   - ESP32-Camera
   ```

#### 6.2 Firmware Upload
1. Connect ESP32-CAM to computer via USB-TTL
2. Put ESP32 in programming mode (GPIO0 to GND)
3. Upload the main firmware
4. Test basic functionality

### Phase 7: Calibration & Testing

#### 7.1 System Calibration
1. **Baseline Measurement:**
   - Use clean distilled water
   - Record photodiode and camera readings
   - Store as reference values

2. **Sensitivity Test:**
   - Add known microplastic samples
   - Record response at different concentrations
   - Create calibration curve

#### 7.2 Performance Validation
1. **Detection Accuracy:**
   - Test with various particle sizes
   - Compare with reference measurements
   - Validate against known standards

2. **Repeatability Test:**
   - Multiple measurements of same sample
   - Calculate standard deviation
   - Ensure <5% variation

## 🔧 Troubleshooting Guide

### Common Issues

#### Power Problems
- **Symptom**: ESP32 doesn't boot
- **Check**: 5V supply voltage and current capacity
- **Fix**: Use 2A+ power supply

#### Optical Misalignment
- **Symptom**: No photodiode signal
- **Check**: Laser beam path through chamber
- **Fix**: Realign laser and photodiode

#### WiFi Connection Issues
- **Symptom**: Can't connect to network
- **Check**: Antenna connection and credentials
- **Fix**: Use external antenna if needed

#### Camera Focus Problems
- **Symptom**: Blurry images
- **Check**: Working distance and lens focus
- **Fix**: Adjust mount position

### Diagnostic Tools

#### Built-in Diagnostics
```cpp
void diagnostics() {
  Serial.println("=== System Diagnostics ===");
  Serial.print("WiFi Status: ");
  Serial.println(WiFi.status());
  Serial.print("Free RAM: ");
  Serial.println(ESP.getFreeHeap());
  Serial.print("Laser Status: ");
  Serial.println(digitalRead(LASER_PIN));
  Serial.print("Photodiode: ");
  Serial.println(analogRead(PHOTODIODE_PIN));
}
```

## 📏 Quality Control Checklist

### Pre-Operation Verification
- [ ] All connections secure and insulated
- [ ] Optical alignment verified
- [ ] Power supply stable (±5%)
- [ ] WiFi connection established
- [ ] Camera focus optimized
- [ ] Calibration completed

### Safety Checks
- [ ] Laser power <5mW (eye-safe)
- [ ] Electrical insulation adequate
- [ ] No sharp edges exposed
- [ ] Stable mechanical mounting
- [ ] Emergency stop accessible

## 📈 Performance Specifications

### Expected Performance After Assembly
- **Detection Range**: 1-1000 μm particles
- **Sensitivity**: 10 particles/ml minimum
- **Accuracy**: ±15% for reference samples
- **Response Time**: <30 seconds
- **Power Consumption**: <3W
- **Operating Time**: 3+ hours on battery

### Acceptance Criteria
✅ System boots and connects to WiFi  
✅ Camera captures clear images  
✅ Photodiode shows signal variation  
✅ Laser operates safely  
✅ Web interface accessible  
✅ Data logging functional  

## 🎯 SIH Demo Preparation

### Quick Setup for Demonstration
1. **15-minute setup:**
   - Power on system
   - Connect to demo WiFi
   - Load test samples
   - Start measurement cycle

2. **Demo Script:**
   - Show clean water baseline
   - Add microplastic sample
   - Display real-time detection
   - Show data on mobile interface

3. **Backup Plans:**
   - Pre-recorded data if network fails
   - Manual operation mode
   - Printed results for reference

**Assembly time estimate: 4-6 hours for first build, 2-3 hours for subsequent units**

This assembly guide ensures **reliable construction** of the revolutionary ₹4,000 microplastic detection system, ready for **SIH hackathon demonstration** and **field deployment**.