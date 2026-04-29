# 🎯 Safe Face 2.0 – Facial Recognition Attendance System

A real-time facial recognition attendance system with intruder detection and optional liveness verification.

---

## 🚀 Features

* 🎥 Real-time face detection (InsightFace)
* 🧠 Face recognition with database matching
* 🚨 Intruder detection with photo capture
* 📊 Attendance logging (CSV)
* 🔐 Optional liveness detection (anti-spoof)
* 🖥 Tkinter-based GUI

---

## 📦 Tech Stack

* Python
* OpenCV
* InsightFace (ONNX)
* TensorFlow (liveness model)
* MongoDB
* Tkinter

---

## ⚙️ Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/your-username/safe-face-2.0.git
cd safe-face-2.0
```

---

### 2. Create virtual environment

```bash
python -m venv face_env
face_env\Scripts\activate
```

---

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Start MongoDB

Ensure MongoDB is running locally:

```
mongodb://localhost:27017/
```

---

### 5. Run the application

```bash
python maincode.py
```

---

## 📂 Auto-Created Folders

When you run the application, these folders are automatically created:

* `logs/` → stores:

  * `arrivals.csv`
  * `intruders.csv`

* `intruder_photos/` → stores captured images of unknown faces

---

## 📸 How It Works

1. Camera starts scanning
2. Face is detected using InsightFace
3. Face encoding is generated
4. Compared with database
5. If matched → attendance logged
6. If unknown → intruder image saved

---

## 🧠 Liveness Detection Model

### 📌 Purpose

Helps prevent spoofing using:

* Photos
* Screens
* Printed images

---

### 🏋️ Train Your Own Model

Prepare dataset:

```
dataset/
  real/
  fake/
```

Train using TensorFlow:

```python
model.fit(train_data, epochs=10)
model.save("liveness_model.keras")
```

---

### 💾 Using the Model

* Place the model file in the project root:

  ```
  liveness_model.keras
  ```

* The application loads it automatically on startup

---

### ⚠️ If Model is Missing

* System still runs normally
* Liveness detection is disabled

---

## 📊 Logs

### Arrival Log

```
logs/arrivals.csv
```

Contains:

* Log ID
* Name
* Date
* Time

---

### Intruder Log

```
logs/intruders.csv
```

Contains:

* Date
* Time
* Photo path

---

## 🔮 Future Improvements

* ⚡ GPU acceleration
* 🎯 Improved liveness detection
* 🧠 Better face tracking (reduce flicker)
* 🌐 Web dashboard

---

## 👨‍💻 Author

Safe Face 2.0 – Built for real-time security and attendance tracking.
