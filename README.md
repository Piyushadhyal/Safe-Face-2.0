# 🎯 SAFE FACE – Facial Recognition Attendance System

A real-time facial recognition attendance system with intruder detection and liveness verification.

---

## 🚀 Features

* 🎥 Real-time face detection (InsightFace)
* 🧠 Face recognition with database matching
* 🚨 Intruder detection + photo logging
* 📊 CSV logging (arrivals + intruders)
* 🔐 Liveness detection (anti-spoof)
* 🖥 GUI built with Tkinter

---

## 📦 Tech Stack

* Python
* OpenCV
* InsightFace (ONNX)
* TensorFlow (liveness model)
* MongoDB
* Tkinter GUI

---

## 📁 Project Structure

```
safe-face/
│── INF.py
│── requirements.txt
│── README.md
│
├── known_faces/
├── logs/
├── intruder_photos/
├── models/
```

---

## ⚙️ Setup Instructions

### 1. Clone repo

```bash
git clone https://github.com/your-username/safe-face.git
cd safe-face
```

---

### 2. Create virtual environment

```bash
python -m venv face_env
face_env\Scripts\activate   # Windows
```

---

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Setup MongoDB

Make sure MongoDB is running locally:

```bash
mongodb://localhost:27017/
```

---

### 5. Run the app

```bash
python INF.py
```

---

## 📸 How it works

1. Camera starts scanning faces
2. Face is detected and encoded
3. Compared with stored database
4. If matched → attendance logged
5. If unknown → intruder photo saved

---

## ⚠️ Notes

* Do NOT upload logs or intruder images to GitHub
* Liveness model (`liveness_model.keras`) train your model

---

## 🔮 Future Improvements

* GPU acceleration
* Better anti-spoofing
* Face tracking (no re-detection flicker)
* Web dashboard

---

## 👨‍💻 Author

Built for learning + real-world deployment.

---
