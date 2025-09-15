# Hardware Components List - ESP32 Microplastic Detection System

## 📦 Complete Bill of Materials (BOM)

### Core Components

| Component | Specification | Quantity | Unit Cost (₹) | Total (₹) | Supplier |
|-----------|---------------|----------|---------------|-----------|-----------|
| **ESP32-CAM Module** | AI Thinker ESP32-CAM with OV2640 | 1 | 800-1,200 | 1,000 | Amazon/Local Electronics |
| **Laser Diode Module** | 650nm Red Laser, 5mW, 3-5V | 1 | 200-400 | 300 | Amazon/Electronics Store |
| **Photodiode Sensor** | BPW21R or similar, UV-enhanced | 1 | 300-600 | 450 | Digikey/Amazon |
| **Water Sample Chamber** | Acrylic/Glass cuvette, 10ml capacity | 1 | 500-800 | 650 | Custom fabrication |
| **Power Supply** | 5V 2A USB adapter | 1 | 300-500 | 400 | Local electronics |
| **Connecting Wires** | Jumper wires, breadboard | 1 set | 200-400 | 300 | Amazon/Electronics |
| **Enclosure** | Waterproof plastic case | 1 | 400-800 | 600 | 3D printed/Custom |
| **Mounting Hardware** | Screws, standoffs, mounts | 1 set | 200-400 | 300 | Hardware store |

**Total System Cost: ₹4,000 (approximately $48)**

## 🔧 Detailed Component Specifications

### 1. ESP32-CAM Module
- **Model**: AI-Thinker ESP32-CAM
- **Camera**: OV2640 2MP sensor
- **Processing**: Dual-core 240MHz, 520KB RAM
- **WiFi**: 802.11 b/g/n
- **Flash**: 4MB (minimum)
- **GPIO**: Multiple pins for sensors
- **Power**: 5V input, 3.3V logic

### 2. Optical Detection System
- **Laser Source**: 650nm wavelength (optimal for plastic scattering)
- **Power**: 5mW (eye-safe, sufficient intensity)
- **Beam Diameter**: ~3mm
- **Photodiode**: Silicon photodiode with UV enhancement
- **Sensitivity**: 0.4-1.1 μm wavelength range
- **Response Time**: <100ns

### 3. Sample Chamber Design
- **Material**: Optical-grade acrylic or glass
- **Volume**: 10ml working volume
- **Dimensions**: 20mm x 20mm x 25mm
- **Windows**: Anti-reflective coating (optional)
- **Inlet/Outlet**: 6mm tubing connections

### 4. Mechanical Assembly
- **Base Platform**: 3D printed ABS/PLA
- **Optical Alignment**: Precision mounts for laser/photodiode
- **Camera Mount**: Adjustable position for optimal viewing
- **Vibration Damping**: Rubber isolators

## 🏗️ Assembly Instructions

### Step 1: Prepare the Base Platform
1. 3D print or fabricate the main base platform
2. Install mounting points for ESP32-CAM
3. Create optical bench for laser/photodiode alignment

### Step 2: Install Optical Components
1. Mount laser diode with collimating lens
2. Position photodiode at 90° to laser beam
3. Align sample chamber between laser and photodiode
4. Test optical path alignment

### Step 3: Electronics Integration
1. Connect ESP32-CAM to power supply
2. Wire photodiode to ESP32 ADC pin
3. Connect laser control to GPIO pin
4. Install WiFi antenna (external if needed)

### Step 4: Software Configuration
1. Upload firmware to ESP32-CAM
2. Configure WiFi credentials
3. Calibrate photodiode readings
4. Test camera capture and processing

### Step 5: System Calibration
1. Use known microplastic samples
2. Record baseline readings with clean water
3. Create calibration curves
4. Validate detection accuracy

## 🔬 Optical Design Principles

### Light Scattering Detection
- **Mie Scattering**: For particles 1-10 μm
- **Rayleigh Scattering**: For particles <1 μm
- **Geometric Scattering**: For particles >10 μm

### Detection Geometry
- **Forward Scattering**: 15-45° for large particles
- **Side Scattering**: 90° for size determination
- **Back Scattering**: 135-165° for composition

### Camera-Based Detection
- **Microscopic Imaging**: Direct particle visualization
- **Focus Stacking**: Extended depth of field
- **Image Processing**: Edge detection and morphology

## ⚡ Power Requirements

| Component | Voltage | Current | Power |
|-----------|---------|---------|-------|
| ESP32-CAM | 5V | 300-500mA | 1.5-2.5W |
| Laser Diode | 3.3V | 50-100mA | 0.15-0.33W |
| Photodiode | 3.3V | 1-10mA | 0.003-0.033W |
| **Total** | **5V** | **~600mA** | **~3W** |

### Battery Option
- **Li-Ion 18650**: 3.7V, 3000mAh
- **Runtime**: 3-4 hours continuous operation
- **Charging**: USB-C with protection circuit

## 📐 Mechanical Drawings

### System Dimensions
- **Overall Size**: 150mm x 100mm x 80mm
- **Weight**: <500g including battery
- **Portability**: Hand-held, ruggedized design

### Critical Alignments
- **Laser-Sample Distance**: 50mm ±2mm
- **Photodiode-Sample Distance**: 50mm ±2mm
- **Camera-Sample Distance**: 30mm ±1mm
- **Optical Axis Height**: 40mm from base

## 🌡️ Environmental Specifications

| Parameter | Range | Notes |
|-----------|-------|-------|
| **Operating Temperature** | 0°C to 50°C | Extended range possible |
| **Humidity** | 10% to 90% RH | Non-condensing |
| **IP Rating** | IP54 | Splash resistant |
| **Vibration** | 2g @ 10-500Hz | Portable use |
| **EMI/EMC** | FCC Class B | WiFi compliance |

## 🔧 Maintenance & Calibration

### Daily Checks
- Clean sample chamber windows
- Check laser alignment
- Verify WiFi connectivity
- Battery level monitoring

### Weekly Calibration
- Reference sample measurement
- Photodiode dark current check
- Camera focus verification
- System sensitivity test

### Monthly Maintenance
- Deep clean optical components
- Firmware update check
- Mechanical alignment verification
- Performance validation

## 📈 Upgrade Pathways

### Version 2.0 Enhancements
- **Multi-wavelength laser**: Better particle identification
- **Polarization optics**: Enhanced scattering analysis
- **Flow cell design**: Continuous monitoring
- **GPS module**: Location tagging
- **LoRaWAN**: Extended range communication

### Advanced Features
- **Fluorescence detection**: Chemical identification
- **Multiple size bins**: Particle size distribution
- **Temperature compensation**: Environmental correction
- **Auto-calibration**: Reference standards

## 💡 Design Rationale

### Cost Optimization
- Standard components from reliable suppliers
- 3D printed parts to reduce tooling costs
- Open-source design for community development
- Modular architecture for easy upgrades

### Performance Trade-offs
- Balance between cost and accuracy
- Portable vs. lab-grade precision
- Real-time processing vs. detailed analysis
- Power consumption vs. performance

This hardware design achieves the **revolutionary cost target of ₹4,000** while providing **research-grade detection capabilities** suitable for environmental monitoring and SIH hackathon demonstration.