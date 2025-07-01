
---

## Road & Lane Segmentation Benchmark

This folder implements and compares a range of semantic-segmentation approaches on a custom dash-cam datasets collected by me or used ready ones.
Three classes: Driving area, Lane markings, other areas.

---

### Tested Models

#### **SegFormer**

* **Dataset:** Makassar City dash-cam set (374 images, 2560×1600px, 4 classes: Background, Road, Lane-solid, Lane-dashed)
* **Approach:** Lightweight transformer encoder with MLP decoder
* **Performance:** 97.8% pixel accuracy, 72.2% mIoU (train), 96.8% pixel accuracy, 74.0% mIoU (validation)

---

#### **DeepLabV3 + ResNet-50**

* **Dataset:** Custom, compiled from public night-driving videos:

  * [Paul Maloney (20 min)](https://youtu.be/XTzppDgByC4?si=2aH5rI1DXK4q_b0r): City and suburb at night
  * [jdmriding (4 min)](https://youtu.be/JJzWJOLrsp4?si=-TtQPn5KAsc2crMn): Night highway, multiple passing vehicles
  * [GitHub: Night lane detection w/ Kalman](https://github.com/diptamath/Lane-detection-in-Night-Enviroment-using-Kalman-Filter): Passing/overtaking scenes on night highways
* **Samples:** 3,003 (train), 883 (validation)
* **Approach:** Atrous convolutions, ASPP module
* **Why:** Best lane IoU and real-time viability in night scenarios

---

#### **U-Net (Basic & Expanded)**

* **Dataset:** Same as DeepLabV3
* **Variants:**

  * **Basic:** Scratch U-Net (\~88% road IoU, \~30% lane IoU, \~58% mean IoU)
  * **Expanded:** ResNet-34 encoder + Albumentations (\~83% road, \~38% lane, \~60% mean IoU in adverse conditions)

---

#### **YOLOP**

* **Dataset:** [BDD100K](https://dl.cv.ethz.ch/bdd100k/data/) (public driving dataset with labels for cars, people, traffic signs)
* **Approach:** End-to-end multitask (object detection, drivable area segmentation, lane marking)
* **Performance:** Real-time efficient (\~0.26s per frame)

---

#### **OpenCV Pipeline**

* **Dataset:** Used on above sets for baseline comparison
* **Approach:** Classical CV — gamma correction, Canny edge, Hough transform, curve fitting, Kalman smoothing
* **Notes:** Fast and lightweight, but less robust with faint/missing markings

---

### Performance Metrics

|                              Model | Conditions             | Lane IoU | Road IoU | Mean IoU                     |
| ---------------------------------: | :--------------------- | :------: | :------: | :--------------------------- |
|                          SegFormer | Day & Night            |     ―    |     ―    | 0.7403                       |
|              DeepLabV3 (ResNet-50) | Day & Night            |   > 0.5  |   > 0.5  | > 0.5                        |
|                      U-Net (basic) | Day & Night            |   0.30   |   0.88   | 0.58                         |
| U-Net (ResNet-34 + Albumentations) | Night, adverse weather |   0.38   |   0.83   | 0.60                         |
|                              YOLOP | Real‑time automotive   |     ―    |     ―    | Real‑time efficient          |
|                             OpenCV | Various conditions     | Moderate | Moderate | Moderate, resource‑efficient |

---

<table>
  <tr>
    <td align="center"><b>OpenCV Lane/Road Detection</b><br><img src="./demo/opencv_det.png" width="320" alt="OpenCV Lane & Road"/></td>
    <td align="center"><b>Road IoU vs Training Step</b><br><img src="./demo/Road_iou_vs_step_less (1).png" width="320" alt="Road IoU Plot"/></td>
  </tr>
  <tr>
    <td align="center"><b>Mean IoU vs Training Step</b><br><img src="./demo/Mean_iou_vs_step_less (1).png" width="320" alt="Mean IoU Plot"/></td>
    <td align="center"><b>Lane IoU vs Training Step</b><br><img src="./demo/Lane_iou_vs_step_less (1).png" width="320" alt="Lane IoU Plot"/></td>
  </tr>
  <tr>
    <td align="center" colspan="2"><b>Composite Results Table</b><br><img src="./demo/composite_table_new.png" width="350" alt="Results Table"/></td>
  </tr>
</table>

---

### Final Choice

* **U-Net34 (ResNet-34 + Albumentations)**
  Offers the best trade-off: 60% mean IoU on mixed conditions, \~9.5h training, lightweight deployment.

---

