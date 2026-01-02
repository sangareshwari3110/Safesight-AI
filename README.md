# 🦺 SafeSight AI - PPE Detection

This project detects whether a person is **wearing required PPE** such as a **safety vest, helmet, and safety shoes** using computer vision and deep learning.
It uses a **YOLOv8 pretrained model** for person detection and **custom-trained YOLO models** for PPE detection.

The system verifies PPE compliance by checking whether the detected PPE bounding boxes **overlap with person bounding boxes**.

---

## 🚀 Features

* Detects **persons** using YOLOv8 pretrained model
* Detects **safety vest, helmet, and safety shoes** using custom-trained YOLO models
* Determines **PPE compliance** using bounding box IoU logic
* Labels output as:

  * ✅ *Wearing PPE*
  * ❌ *Missing PPE*
* Works on **images**
* Easily extendable to **video or CCTV streams**

---

## 📁 Project Structure

```
SafeSight-AI/
│
├── safety_gear.py          # Main PPE detection script
├── vest.pt                 # YOLO model for safety vest detection
├── helmet.pt               # YOLO model for helmet detection
├── safety_shoes.pt         # YOLO model for safety shoes detection
├── yolov8n.pt              # Pretrained YOLOv8 model for person detection
├── safe.png                # Sample image (person wearing PPE)
├── unsafe.png              # Sample image (person not wearing PPE)
├── outputs/                # Folder containing output images
│   ├── safe.png
│   └── unsafe.png
└── README.md
```

---

## 🧠 How It Works

1. **Person Detection**

   * Uses `yolov8n.pt` to detect all persons in the image

2. **PPE Detection**

   * Uses `vest.pt`, `helmet.pt`, and `safety_shoes.pt` to detect each type of PPE

3. **PPE Verification Logic**

   * Checks if the detected PPE bounding boxes **overlap sufficiently** with the person bounding box → considered **compliant**

4. **Visualization**

   * Has a Label at the top of bounding box with information of whether the person is wearing the safety gear or not.

---

## 🛠️ Requirements

Install the required dependencies using pip:

```bash
pip install ultralytics opencv-python numpy
```

---

## ▶️ How to Run

1. Clone the repository or download the project files
2. Place your input image in the project directory
3. Update the image path in `safety_gear.py` if needed:

```python
IMAGE_PATH = 'unsafe.png'
```

4. Run the script:

```bash
python safety_gear.py
```

5. The output image will be displayed and saved inside the `outputs` folder

---

## 🖼️ Output Explanation

* **Green Bounding Box** → Person detected
* **Red Text** → Missing PPE items for that person

Each detected person is labeled with:

```
Vest:True/False Helmet:True/False Shoes:True/False
```

---

## 🧪 Use Cases

* Industrial and factory safety monitoring
* Construction site PPE compliance
* Warehouse and logistics safety checks
* Automated safety audits
* Workplace safety compliance tracking

---

## 🔮 Future Enhancements

* Real-time webcam and CCTV support
* Multi-person PPE compliance tracking
* Alert system for PPE violations
* Web deployment using Flask / FastAPI
* Integration with safety dashboards

---

## 👨‍💻 Author

**Sangareshwari A**
AI Engineer | Computer Vision & Deep Learning
