
---

# Intelligent Matrix Headlights Simulator

Here is a full operational simulator of an intelligent matrix headlight system for automotive applications. It processes live video (or YouTube streams) to render real-time adaptive beam patterns, based on both environmental conditions and detected objects on the road.

## System Overview

* **Input:** Real or streamed driving video.
* **Core Components:**

  * **Object Detection:** YOLOv5x, custom-trained, recognizes cars, trucks, signs, etc.
  * **Semantic Segmentation:** U-Net with ResNet-34 backbone, segments road, lane lines, and background.
  * **Adverse Weather Detection:** Four separate ResNet-18 classifiers for rain, fog, snow, and night, each triggering weather-specific preprocessing and dynamic changes in lighting logic.
  * **Beam Simulation:** Grid-based, 3D effect beams rendered per-frame, with intensity, occlusion, and glow all modulated by both detected obstacles and weather.
  * **Bird's-Eye View (BEV):** Top-down visualization of lane/road and detected objects for debugging and presentation.

## Key Features

* **Dynamic Beam Modulation:**
  The beam pattern automatically adapts to detected cars and traffic signs, dimming or blocking parts of the light to mimic real adaptive headlights.
* **Weather-Responsive Processing:**
  The system switches between high and low beams depending on real-time predictions of rain, fog, snow, or night. Specialized preprocessing (dehazing, highlight suppression, gamma correction) is applied before segmentation in tough conditions.
* **Real-Time Multi-Modal Analysis:**
  Each video frame is analyzed for object detection, weather state, semantic segmentation, and then combined into a single output. Overlay visualizations include segmentation, beam mesh, glow, BEV, and annotated probabilities.
* **Modular and Extensible:**
  Every component (detection, segmentation, weather, beam rendering) is decoupled, making it easy to upgrade models or swap in new techniques as research evolves.

## Processing Pipeline

1. **Weather Classification:**
   Each frame is first run through four parallel classifiers (rain, fog, snow, night/day). If a weather event exceeds a threshold, both the lighting simulation and preprocessing adapt accordingly.
2. **Preprocessing (if needed):**

   * Fog: Fast dark-channel dehazing.
   * Rain: Suppression of specular highlights.
   * Snow: Gamma correction + local contrast equalization.
3. **Semantic Segmentation:**
   The preprocessed (or original) frame is segmented by a U-Net, producing masks for road, lanes, and background.
4. **Object Detection:**
   YOLOv5x identifies vehicles and signs. Their positions and confidences are used to modulate the beams.
5. **Beam Mesh Rendering:**
   A 3D mesh simulates real headlight beams, divided into cells. Cells with vehicles are turned off, those with signs are dimmed, and all others rendered at appropriate opacity. Glow halos applied with Gaussian blurring.
6. **Bird's-Eye View (BEV):**
   Segmented output and object detections are projected to a top-down view for diagnostics.
7. **Annotation and Display:**
   The current weather and day/night probabilities are overlaid. All effects are composited with the original video and BEV, and streamed to display (and optionally saved).

---

## Example Outputs

<table>
  <tr>
    <td align="center"><b>Fog Detected</b><br><img src="./demo/fog_det_1.png" width="320" alt="Fog Detection"/></td>
    <td align="center"><b>Rain Detected</b><br><img src="./demo/rain_det_1.png" width="320" alt="Rain Detection"/></td>
  </tr>
  <tr>
    <td align="center"><b>Traffic Sign Detected</b><br><img src="./demo/sign_det_1.png" width="320" alt="Sign Detection"/></td>
    <td align="center"><b>Snow Detected</b><br><img src="./demo/snow_det_1.png" width="320" alt="Snow Detection"/></td>
  </tr>
  <tr>
    <td align="center"><b>Car Detected</b><br><img src="./demo/car_det.png" width="320" alt="Car Detection"/></td>
    <td align="center"><b>Truck Detected</b><br><img src="./demo/truck_det.png" width="320" alt="Truck Detection"/></td>
  </tr>
</table>

*The above images show real-time detection of various objects and adverse weather conditions. Adaptive beam simulation and BEV overlays are visible in each sample.*

---

## Future Implementation

<img src="./demo/magisterka_workflow_new (1).png" width="640" height="360" alt="Future Implementation Diagram">

The diagram above illustrates a **proposed hardware-software architecture** for a fully integrated intelligent matrix headlight system, extending beyond pure software simulation into a real automotive environment.

**Key Elements:**

* **Multi-camera input:** Multiple cameras enable redundancy or different fields of view.
* **Modular Perception:** Each camera feeds three modules:

  * **Weather Condition Detector**
  * **Sign/Car Recognizer**
  * **Road Segmenter**
* **Automotive Integration:**
  All perception results are sent via a **CANBus Interface** to a central **Beam Pattern Manager**.
* **Real-Time Beam Adaptation:**
  The Beam Pattern Manager fuses all perception signals and computes the optimal beam pattern, considering detected obstacles, weather, and road shape.
* **Actuation Pipeline:**
  The calculated beam pattern is passed to the **Matrix Display Driver** and **LED Driver IC**, directly controlling the adaptive headlight LEDs.

*Processing is distributed, context is fused for dynamic safety-aware decisions, and hardware integration points (CANBus, SPI) are shown, underlining readiness for embedded deployment.*

---
