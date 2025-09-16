# ESP32-CAM Microplastic Detection System - Circuit Diagrams

## 🔧 Complete Hardware Wiring Guide

### System Overview
The microplastic detection system combines:
- ESP32-CAM module for image capture and processing
- 650nm laser diode for particle illumination
- Photodiode sensor for scattered light detection
- Transparent flow cell for controlled water sampling
- Power management and safety circuits

## 📋 Component Specifications

### Primary Components

| Component | Specification | Quantity | Purpose |
|-----------|---------------|----------|---------|
| ESP32-CAM (AI-Thinker) | OV2640 camera, WiFi, Bluetooth | 1 | Main controller & image capture |
| 650nm Red Laser Diode | 5mW, 3-5V, Focused beam | 1 | Particle illumination |
| Photodiode (BPW21) | Silicon PIN photodiode, 400-1100nm | 1 | Light scattering detection |
| Op-Amp (LM358) | Dual operational amplifier | 1 | Signal amplification |
| Laser Driver (LD1117V33) | 3.3V regulator with current limiting | 1 | Laser power control |
| Flow Cell | Quartz glass, 10mm path length | 1 | Water sample containment |
| Power Supply | 5V 2A DC adapter | 1 | System power |

### Supporting Components

| Component | Value/Type | Quantity | Purpose |
|-----------|------------|----------|---------|
| Resistors | 220Ω, 1kΩ, 10kΩ, 100kΩ | 4 each | Current limiting, voltage dividers |
| Capacitors | 100nF ceramic, 100µF electrolytic | 2, 2 | Power filtering, decoupling |
| LEDs | Red (status), Green (power) | 1 each | System status indicators |
| Transistor | 2N2222 NPN | 1 | Laser switching |
| Connectors | JST-PH 2.0mm, Screw terminals | Various | Modular connections |
| PCB | FR4 double-layer, 50x70mm | 1 | Component mounting |

## 🔌 Pin Connections

### ESP32-CAM Pin Assignments

```
ESP32-CAM GPIO Assignments:
┌─────────────────┬─────────────────┬─────────────────────────────┐
│ GPIO Pin        │ Function        │ Connected Component         │
├─────────────────┼─────────────────┼─────────────────────────────┤
│ GPIO 0          │ Camera XCLK     │ OV2640 XCLK                │
│ GPIO 2          │ ADC Input       │ Photodiode Amplifier Out   │
│ GPIO 4          │ Flash/Status    │ Status LED                 │
│ GPIO 12         │ PWM Output      │ Laser Driver (PWM Control) │
│ GPIO 13         │ Digital Input   │ Flow Sensor (Optional)     │
│ GPIO 15         │ ADC Input       │ Temperature Sensor         │
│ GPIO 14         │ Digital Output  │ System Enable              │
│ GPIO 16         │ UART RX         │ Debug/Programming          │
│ GPIO 17         │ UART TX         │ Debug/Programming          │
│ 3.3V            │ Power Supply    │ System 3.3V Rail           │
│ 5V              │ Power Supply    │ Laser Driver Input         │
│ GND             │ Ground          │ Common Ground              │
└─────────────────┴─────────────────┴─────────────────────────────┘
```

### Camera Module Connections (Built-in)

```
OV2640 Camera Connections (Internal to ESP32-CAM):
GPIO 5, 18, 19, 21, 22, 23, 25, 26, 27 → Camera data/control
GPIO 32 → Camera power down
GPIO 0 → Camera clock
```

## 🔬 Optical System Layout

### Laser-Photodiode Arrangement

```
Flow Cell Optical Configuration:

    Laser Diode (650nm)          Flow Cell              Photodiode
           │                    ┌─────────┐                 │
           │ ══════════════════►│ Sample  │════════════════►│
    ┌─────────────┐              │ Water   │          ┌─────────────┐
    │   650nm     │              │         │          │   BPW21     │
    │ Laser Diode │              │ 10mm    │          │ Photodiode  │
    │   5mW       │              │ Path    │          │  Si PIN     │
    └─────────────┘              │ Length  │          └─────────────┘
           │                     └─────────┘                 │
    ┌─────────────┐                   │              ┌─────────────┐
    │ Laser       │                   │              │ Amplifier   │
    │ Driver      │                   │              │ Circuit     │
    └─────────────┘              Flow Direction     └─────────────┘
           │                          ▼                     │
      ESP32 GPIO12                                     ESP32 GPIO2
```

### Angular Configuration
- **Laser Angle**: 0° (direct illumination)
- **Photodiode Angle**: 90° (side scattering detection)
- **Flow Cell**: Perpendicular to laser beam
- **Distance**: Laser to cell = 50mm, Cell to photodiode = 30mm

## 🔌 Detailed Circuit Schematic

### Main System Circuit

```
ESP32-CAM Microplastic Detection System

                    +5V
                     │
                 ┌───┴───┐
                 │ Power │
                 │Supply │
                 │5V→3.3V│
                 └───┬───┘
                     │ +3.3V
     ┌───────────────┼───────────────────────────────────┐
     │               │                                   │
     │         ┌─────┴─────┐                             │
     │         │  ESP32-   │                             │
     │         │    CAM    │                             │
     │         │           │                             │
     │         │    GPIO2  ├─────┐                       │
     │         │           │     │                       │
     │         │   GPIO12  ├────┐│                       │
     │         │           │    ││                       │
     │         │   GPIO4   ├───┐││                       │
     │         └─────┬─────┘   │││                       │
     │               │ GND     │││                       │
     │               │         │││                       │
     └─────┬─────────┴─────────┼┼┼───────────────────────┘
           │                   │││
           │ GND               │││
           │                   │││
     ┌─────┴─────┐             │││
     │   Status  │◄────────────┘││  GPIO4 → Status LED
     │    LED    │              ││
     └───────────┘              ││
                                ││
     ┌─────────────┐             ││
     │   Laser     │◄────────────┘│  GPIO12 → Laser Driver (PWM)
     │   Driver    │              │
     │   Circuit   │              │
     └─────┬───────┘              │
           │                      │
     ┌─────┴─────┐                │
     │  650nm    │                │
     │  Laser    │                │
     │  Diode    │                │
     └─────┬─────┘                │
           │                      │
           ▼ Light Beam            │
     ┌─────────────┐               │
     │  Flow Cell  │               │
     │ (Quartz)    │               │
     │             │ Scattered     │
     │   Sample    │ Light         │
     │   Water     │ ──────────────┘
     └─────────────┘               │
                                   │
     ┌─────────────┐               │
     │ Photodiode  │◄──────────────┘  GPIO2 ← Photodiode Amplifier
     │ Amplifier   │
     │ (LM358)     │
     └─────┬───────┘
           │
     ┌─────┴─────┐
     │  BPW21    │
     │Photodiode │
     └───────────┘
```

### Laser Driver Circuit

```
Laser Driver Circuit (GPIO12 → Laser Control)

     ESP32 GPIO12                                    +5V
          │                                           │
          │ PWM Signal                                │
          ▼                                           │
     ┌─────────┐     ┌─────────┐     ┌─────────┐     │
     │   1kΩ   ├─────┤ 2N2222  ├─────┤  220Ω   ├─────┤
     │Resistor │     │   NPN   │     │Resistor │     │
     └─────────┘     │ Base    │     └─────┬───┘     │
          │          └────┬────┘           │         │
          │               │ Collector      │         │
          │               │                │         │
          │               ▼ Emitter        │         │
     ┌────┴────┐     ┌────┴────┐     ┌────┴────┐    │
     │   GND   │     │   GND   │     │ Laser   │    │
     └─────────┘     └─────────┘     │ Diode   │    │
                                     │ 650nm   │    │
                                     │ 5mW     │    │
                                     └────┬────┘    │
                                          │         │
                                          │         │
                                     ┌────┴────┐    │
                                     │  GND    │    │
                                     └─────────┘    │
                                                    │
     Current Limiting: 220Ω = (5V - 2.2V) / 12mA   │
     Laser operates at safe 12mA current            │
```

### Photodiode Amplifier Circuit

```
Photodiode Signal Amplification (Photodiode → GPIO2)

     Photodiode BPW21                        ESP32 GPIO2
          │                                       ▲
          │ Photo Current                         │
          ▼                                       │
     ┌─────────┐                                  │
     │ Cathode │                                  │
     │ (Light) │                                  │
     │         │                                  │
     │  Anode  ├──────────────┐                  │
     └─────────┘               │                  │
          │                    │                  │
          │                    ▼                  │
     ┌────┴────┐         ┌─────────────┐          │
     │   GND   │         │    LM358    │          │
     └─────────┘         │   Op-Amp    │          │
                         │             │          │
     +3.3V               │   Non-Inv   │          │
       │                 │     (+)     │          │
       │                 │             │          │ Amplified
       │                 │    Inv (-)  ├──────────┘ Signal
       │            ┌────┤             │ 0-3.3V
       │            │    │   Output    │
       │            │    └─────────────┘
       │            │
       │       ┌────┴────┐
       │       │ 100kΩ   │ Feedback Resistor
       │       │Resistor │ (Gain = 100kΩ/10kΩ = 10x)
       │       └─────────┘
       │
       │       ┌─────────┐
       ├───────┤  10kΩ   ├───── GND
       │       │Resistor │ Bias Resistor
       │       └─────────┘
       │
       │       ┌─────────┐
       └───────┤ 100nF   ├───── GND
               │   Cap   │ Decoupling
               └─────────┘
```

### Power Supply Circuit

```
Power Distribution System

     DC Jack Input                          System Power Rails
       (5V 2A)
          │
          ▼
     ┌─────────┐      ┌─────────────┐           +5V Rail
     │  Input  │      │   100µF     │              │
     │Connector├──────┤ Electrolytic├──────────────┤
     │  5V DC  │  │   │  Capacitor  │              │
     └─────┬───┘  │   └─────────────┘              │
           │      │                                │
           │      │   ┌─────────────┐              │
           │      └───┤    100nF    ├──── GND      │
           │          │   Ceramic   │              │
           │          │  Capacitor  │              │
           │          └─────────────┘              │
           │                                       │
           │          ┌─────────────┐              │
           ├──────────┤ LM1117-3.3  ├──────────────┼─── +3.3V Rail
           │          │  Regulator  │              │      │
           │          └──────┬──────┘              │      │
           │                 │                     │      │
           │                 ▼                     │      │
           │          ┌─────────────┐              │      │
           │          │   100µF     │              │      │
           │          │ Electrolytic├──────────────┼──────┘
           │          │  Capacitor  │              │
           │          └─────┬───────┘              │
           │                │                      │
           ▼                ▼                      ▼
     ┌─────────┐      ┌─────────┐            ┌─────────┐
     │   GND   │      │   GND   │            │   GND   │
     └─────────┘      └─────────┘            └─────────┘

Power Requirements:
- ESP32-CAM: 3.3V @ 200-300mA (peak during WiFi transmission)
- Laser Diode: 5V @ 15mA (current limited to 12mA for safety)
- Op-Amp: 3.3V @ 5mA
- Status LEDs: 3.3V @ 20mA total
- Total: 5V @ 500mA (with safety margin → 2A power supply)
```

## 🔧 Assembly Instructions

### Step 1: PCB Preparation
1. **PCB Layout**: Use double-layer FR4 PCB (50x70mm)
2. **Component Placement**: Follow layout diagram below
3. **Solder Components**: Start with smallest components first

### Step 2: Power Circuit Assembly
```
Power Section Assembly Order:
1. Install ceramic capacitors (100nF)
2. Install electrolytic capacitors (100µF, observe polarity!)
3. Install LM1117-3.3V regulator (observe orientation)
4. Install DC jack connector
5. Test power rails with multimeter BEFORE connecting ESP32
```

### Step 3: Signal Processing Circuit
```
Amplifier Circuit Assembly:
1. Install LM358 op-amp (observe pin 1 orientation)
2. Install resistors (color code verification essential)
3. Install photodiode (observe cathode/anode polarity)
4. Test amplifier with known input signal
```

### Step 4: Laser Driver Assembly
```
Laser Safety Protocol:
⚠️  LASER SAFETY WARNING ⚠️
- Never look directly into laser beam
- Laser power must be current-limited
- Test circuit before installing laser diode
- Use laser safety glasses during assembly

Assembly Steps:
1. Install 2N2222 transistor (observe EBC pinout)
2. Install current limiting resistor (220Ω - CRITICAL!)
3. Install base resistor (1kΩ)
4. Test switching circuit with LED before laser
5. Install laser diode LAST with proper heatsinking
```

### Step 5: ESP32-CAM Integration
```
ESP32-CAM Connection:
1. Prepare female headers for ESP32-CAM module
2. Connect power (3.3V, 5V, GND)
3. Connect signal lines (GPIO2, GPIO12, GPIO4)
4. Install pull-up resistors for stable operation
5. Test basic functionality before final assembly
```

## 📐 Mechanical Design

### Flow Cell Mounting

```
Flow Cell Assembly:

     Water Inlet                    Water Outlet
         │                               │
         ▼                               ▼
    ┌────────┐                     ┌────────┐
    │ Tube   │                     │ Tube   │
    │Fitting │                     │Fitting │
    └────┬───┘                     └───┬────┘
         │                             │
         ▼                             ▼
    ┌─────────────────────────────────────────┐
    │           Flow Cell Housing             │ ◄── 3D Printed ABS
    │  ┌─────────────────────────────────┐    │
    │  │      Quartz Glass Cell          │    │ ◄── 10mm path length
    │  │     (10mm x 10mm x 50mm)        │    │
    │  └─────────────────────────────────┘    │
    │                                         │
    │  Laser Entry ──────►    ◄────── Detector Window
    └─────────────────────────────────────────┘

Mounting Specifications:
- Flow cell positioned at 50mm from laser
- Photodiode window at 90° to laser beam
- Adjustable mounting for alignment
- Black anodized aluminum housing to minimize stray light
```

### Enclosure Design

```
System Enclosure (150mm x 100mm x 60mm):

    Top View:
    ┌─────────────────────────────────────────────┐
    │  Power    Status LEDs    Access Port         │
    │  Switch   🔴 🟢         (Programming)       │
    │                                             │
    │  ┌─────────────┐     ┌─────────────┐       │
    │  │ ESP32-CAM   │     │ Power       │       │
    │  │ Module      │     │ Supply      │       │
    │  └─────────────┘     └─────────────┘       │
    │                                             │
    │  ┌─────────────┐     ┌─────────────┐       │
    │  │ Amplifier   │     │ Laser       │       │
    │  │ Circuit     │     │ Driver      │       │
    │  └─────────────┘     └─────────────┘       │
    │                                             │
    │         Flow Cell Connector                 │
    └─────────────────────────────────────────────┘

Materials:
- Enclosure: IP65 rated plastic case
- Mounting: DIN rail compatible
- Ventilation: Passive cooling slots
- Sealing: Rubber gaskets for outdoor use
```

## 🔍 Testing and Calibration

### Initial System Tests

1. **Power Supply Test**
   ```
   Measurements Required:
   - 5V rail: 4.8V - 5.2V
   - 3.3V rail: 3.2V - 3.4V
   - Current draw: <500mA total
   - Ripple voltage: <50mV peak-peak
   ```

2. **Laser Circuit Test**
   ```
   Safety Checks:
   - Laser current: 10-12mA (NEVER exceed 15mA)
   - PWM control: 0-100% duty cycle response
   - Temperature: Laser housing <40°C
   - Beam alignment: Centered on flow cell
   ```

3. **Photodiode Circuit Test**
   ```
   Signal Verification:
   - Dark current: <1mV
   - Light response: Linear with intensity
   - Amplifier gain: 10x (verified with function generator)
   - Noise floor: <10mV RMS
   ```

### Calibration Procedure

1. **Baseline Measurement**
   - Fill flow cell with distilled water
   - Enable laser at 80% power
   - Record photodiode reading for 30 seconds
   - Average = baseline value

2. **Sensitivity Calibration**
   - Use known concentration microplastic solutions
   - Measure photodiode response vs. concentration
   - Generate calibration curve
   - Store calibration constants in EEPROM

## ⚠️ Safety Considerations

### Laser Safety
- **Class**: 3R laser system (5mW, 650nm)
- **Safety**: Never look directly at beam
- **Interlocks**: Automatic shutdown on case opening
- **Labeling**: Proper laser hazard labels required
- **Training**: Operators must be laser safety trained

### Electrical Safety
- **Power**: SELV (Safety Extra Low Voltage) design
- **Grounding**: Proper earth grounding required
- **Fusing**: 1A fast-blow fuse on 5V input
- **Isolation**: Optical isolation on communication ports

### Environmental Protection
- **Ingress Protection**: IP65 rating for outdoor use
- **Temperature Range**: -10°C to +50°C operation
- **Humidity**: Up to 95% RH non-condensing
- **Vibration**: IEC 60068-2-6 compliant

## 📝 Bill of Materials (Detailed in separate file)

See `hardware/bom/components_list.md` for complete component specifications, supplier information, and current pricing.

---

**Document Version**: v1.0
**Last Updated**: September 2025
**Author**: SIH AquaGuard Team
**Review Required**: Electrical Engineer, Laser Safety Officer