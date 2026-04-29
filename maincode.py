"""
Facial Recognition Attendance System
InsightFace edition — drop-in replacement for face_recognition module
Requirements: opencv-python, insightface, pillow, numpy, onnxruntime, pymongo, tensorflow
Install: pip install opencv-python insightface pillow numpy onnxruntime pymongo tensorflow
"""

import cv2
import os
import csv
import numpy as np
import time
import threading
import sys
import uuid
import tkinter as tk
from tkinter import simpledialog
from tkinter import messagebox
from tkinter import filedialog
from pymongo import MongoClient
from PIL import Image, ImageTk
from datetime import datetime, timedelta
import subprocess
import tensorflow as tf
import insightface
from insightface.app import FaceAnalysis




# ══════════════════════════════════════════════════════
#  INSIGHTFACE INITIALISATION  (replaces face_recognition)
# ══════════════════════════════════════════════════════

# Initialise once at module level (reused everywhere)
_fa = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
_fa.prepare(ctx_id=0, det_size=(640, 640))


def get_face_locations_and_encodings(rgb_frame: np.ndarray):
    """
    Drop-in replacement for:
        face_recognition.face_locations(...)
        face_recognition.face_encodings(...)

    Returns:
        locs  : list of (top, right, bottom, left)  — same order as face_recognition
        encs  : list of np.ndarray (512-d normalised)
    """
    faces = _fa.get(rgb_frame)
    locs, encs = [], []
    for f in faces:
        x1, y1, x2, y2 = f.bbox.astype(int)
        locs.append((y1, x2, y2, x1))          # top, right, bottom, left
        emb = f.embedding / np.linalg.norm(f.embedding)
        encs.append(emb.astype(np.float32))
    return locs, encs


def load_face_encoding_from_file(image_path: str):
    """
    Drop-in replacement for:
        img = face_recognition.load_image_file(path)
        enc = face_recognition.face_encodings(img)[0]

    Returns the first face encoding (normalised) or None if no face found.
    """
    bgr = cv2.imread(image_path)
    if bgr is None:
        return None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    faces = _fa.get(rgb)
    if not faces:
        return None
    emb = faces[0].embedding
    return (emb / np.linalg.norm(emb)).astype(np.float32)


def face_distance(known_encodings: list, enc: np.ndarray) -> np.ndarray:
    """
    Drop-in replacement for face_recognition.face_distance().
    Uses cosine distance  (1 - cosine_similarity).
    Lower = more similar (same semantics as face_recognition).
    """
    if not known_encodings:
        return np.array([])
    known = np.array(known_encodings)
    # both sides already normalised → dot product = cosine similarity
    sims = known @ enc
    return (1.0 - sims).astype(np.float32)


# ══════════════════════════════════════════════════════
#  LIVENESS MODEL
# ══════════════════════════════════════════════════════

def is_real_face(face_img, model, history_buffer, IMG_SIZE=128):
    try:
        if face_img is None or face_img.size == 0:
            return False

        h, w = face_img.shape[:2]
        pad = int(0.1 * min(h, w))
        face_img = face_img[pad:h - pad, pad:w - pad]

        img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img.astype("float32") / 255.0
        img = np.expand_dims(img, axis=0)

        # 🔥 REAL prediction
        pred = model.predict(img, verbose=0)[0][0]

        # Smooth prediction using history
        history_buffer.append(pred)
        if len(history_buffer) > 5:
            history_buffer.pop(0)

        avg_pred = np.mean(history_buffer)

        return avg_pred > 0.5   # threshold (tune if needed)

    except Exception as e:
        print("Liveness error:", e)
        return False


# ══════════════════════════════════════════════════════
#  RECOGNITION ENGINE
# ══════════════════════════════════════════════════════

class RecognitionEngine:
    KNOWN_DIR    = "known_faces"
    LOGS_DIR     = "logs"
    INTRUDER_DIR = "intruder_photos"

    def __init__(self):
        for d in [self.KNOWN_DIR, self.LOGS_DIR, self.INTRUDER_DIR]:
            os.makedirs(d, exist_ok=True)

        self.client = MongoClient("mongodb://localhost:27017/")
        self.db = self.client["face_db"]
        self.collection = self.db["students"]

        self.arrival_log  = os.path.join(self.LOGS_DIR, "arrivals.csv")
        self.intruder_log = os.path.join(self.LOGS_DIR, "intruders.csv")

        self._ensure_csv(self.arrival_log,  ["Log#", "Name", "Date", "Time"])
        self._ensure_csv(self.intruder_log, ["Date", "Time", "Photo Path"])

        self.avg_encodings = []
        self.all_encodings = []
        self.names = []
        self._load_from_db()

        self.mark_expired_students()
        self.delete_old_expired(30)
        self.start_expiry_monitor()

        self.logger = None

    def _ensure_csv(self, path: str, headers: list) -> None:
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            with open(path, "w", newline="") as f:
                csv.writer(f).writerow(headers)

    def log(self, msg, tag="info"):
        if self.logger:
            self.logger(msg, tag)
        else:
            print(msg)

    def _next_id(self) -> int:
        try:
            with open(self.arrival_log, "r", newline="") as f:
                rows = list(csv.reader(f))
            ids = [int(r[0]) for r in rows[1:] if r and r[0].isdigit()]
            return max(ids) + 1 if ids else 1
        except Exception:
            return 1

    def _load_from_db(self):
        self.avg_encodings.clear()
        self.all_encodings.clear()
        self.names.clear()

        for doc in self.collection.find({"status": "active"}):
            name = doc["roll_number"]
            avg  = np.array(doc["avg_encoding"], dtype=np.float32)
            all_encs = [np.array(e, dtype=np.float32) for e in doc["encodings"]]

            self.names.append(name)
            self.avg_encodings.append(avg)
            self.all_encodings.append(all_encs)

        print(f"[DB] Loaded {len(self.names)} students")

    def mark_expired_students(self):
        now = datetime.now()
        for student in self.collection.find({"expiry_date": {"$lt": now}, "status": "active"}):
            roll = student["roll_number"]
            self.collection.update_one({"roll_number": roll}, {"$set": {"status": "expired"}})
            self.log(f"⚠ Student {roll} marked as expired", "warn")

    def delete_old_expired(self, days=30):
        cutoff = datetime.now() - timedelta(days=days)
        for student in self.collection.find({"status": "expired", "expiry_date": {"$lt": cutoff}}):
            roll = student["roll_number"]
            self.collection.delete_one({"roll_number": roll})
            self.log(f"🗑 Deleted expired student {roll}", "intruder")

    def start_expiry_monitor(self):
        def loop():
            while True:
                self.mark_expired_students()
                self.delete_old_expired(30)
                time.sleep(86400)
        threading.Thread(target=loop, daemon=True).start()

    def reload_faces(self) -> int:
        self._load_from_db()
        return len(self.names)

    def log_arrival(self, name: str) -> None:
        now = datetime.now()
        with open(self.arrival_log, "a", newline="") as f:
            csv.writer(f).writerow([self._next_id(), name,
                                    now.strftime("%Y-%m-%d"),
                                    now.strftime("%H:%M:%S")])

    def log_intruder(self, frame: np.ndarray, current_face_id=None) -> str:
        try:
            now = datetime.now()
            ts = now.strftime("%Y%m%d_%H%M%S_%f")[:-3]

            os.makedirs(self.INTRUDER_DIR, exist_ok=True)

            photo = os.path.join(self.INTRUDER_DIR, f"intruder_{ts}.jpg")

            success = cv2.imwrite(photo, frame)

            if not success:
                print("❌ Failed to save intruder image")
                return None

            with open(self.intruder_log, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    now.strftime("%Y-%m-%d"),
                    now.strftime("%H:%M:%S"),
                    photo
                ])
                f.flush()

            print("✅ Intruder logged:", photo)
            return photo

        except Exception as e:
            print("❌ Intruder log error:", e)
            return None

    def reset_logs(self) -> None:
        with open(self.arrival_log, "w", newline="") as f:
            csv.writer(f).writerow(["Log#", "Name", "Date", "Time"])
        with open(self.intruder_log, "w", newline="") as f:
            csv.writer(f).writerow(["Date", "Time", "Photo Path"])

    def cleanup_old_data(self, days: int = 14):
        cutoff = datetime.now() - timedelta(days=days)

        if os.path.exists(self.INTRUDER_DIR):
            for file in os.listdir(self.INTRUDER_DIR):
                path = os.path.join(self.INTRUDER_DIR, file)
                try:
                    if datetime.fromtimestamp(os.path.getmtime(path)) < cutoff:
                        os.remove(path)
                except:
                    pass

        def clean_csv(file_path):
            if not os.path.exists(file_path):
                return
            with open(file_path, "r", newline="") as f:
                rows = list(csv.reader(f))
            if not rows:
                return
            header = rows[0]
            new_rows = [header]
            for row in rows[1:]:
                try:
                    if len(row) > 2:
                        log_date = datetime.strptime(row[2], "%Y-%m-%d")
                        if log_date >= cutoff:
                            new_rows.append(row)
                except:
                    pass
            with open(file_path, "w", newline="") as f:
                csv.writer(f).writerows(new_rows)

        clean_csv(self.arrival_log)
        clean_csv(self.intruder_log)


# ══════════════════════════════════════════════════════
#  AUTO REGISTER FUNCTION
# ══════════════════════════════════════════════════════

def auto_register(folder_path: str, passout_year: int, log_callback=None):
    """Register a student from a folder containing face images"""
    roll_number = os.path.basename(folder_path)

    if not roll_number.isdigit():
        if log_callback:
            log_callback(f"❌ Invalid roll number: {roll_number}", "intruder")
        return False

    # Get all image files
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = [f for f in os.listdir(folder_path)
                   if f.lower().endswith(image_extensions)]

    if not image_files:
        if log_callback:
            log_callback(f"❌ No images found in {folder_path}", "intruder")
        return False

    encodings = []
    for img_file in image_files:
        img_path = os.path.join(folder_path, img_file)
        encoding = load_face_encoding_from_file(img_path)
        if encoding is not None:
            encodings.append(encoding.tolist())

    if not encodings:
        if log_callback:
            log_callback(f"❌ No faces detected in {folder_path}", "intruder")
        return False

    # Calculate average encoding
    avg_encoding = np.mean(encodings, axis=0).tolist()

    # Calculate expiry date (4 years from passout year)
    expiry_date = datetime(passout_year + 4, 12, 31)

    # Update or insert into database
    client = MongoClient("mongodb://localhost:27017/")
    db = client["face_db"]
    collection = db["students"]

    collection.update_one(
        {"roll_number": roll_number},
        {"$set": {
            "roll_number": roll_number,
            "encodings": encodings,
            "avg_encoding": avg_encoding,
            "passout_year": passout_year,
            "expiry_date": expiry_date,
            "status": "active"
        }},
        upsert=True
    )

    if log_callback:
        log_callback(f"✅ Registered {roll_number} with {len(encodings)} face(s)", "known")

    return True


# ══════════════════════════════════════════════════════
#  GUI APPLICATION
# ══════════════════════════════════════════════════════

class App(tk.Tk):
    COOLDOWN    = 5
    INTRUDER_CD = 10

    def __init__(self):
        super().__init__()

        # Load liveness model if exists
        try:
            self.liveness_model = tf.keras.models.load_model("liveness_model.keras")
        except:
            print("Warning: liveness_model.keras not found. Liveness detection disabled.")
            self.liveness_model = None

        self._add_win = None
        self.title("SAFE FACE")
        self.geometry("1100x660")
        self.resizable(True, True)
        self.configure(bg="#1e1e2e")

        self.engine = RecognitionEngine()
        self.engine.logger = self._log

        self.student_win = None
        self._delete_win = None
        self.engine.cleanup_old_data(14)

        # ─── INTRUDER TRACKING ───
        self._current_intruder_id = None  # Track current unknown face
        self._last_intruder_log_time = 0  # Cooldown for logging same intruder
        self._intruder_cooldown_seconds = 10  # Don't log same intruder more than once every 10 seconds

        # ── UNKNOWN FACE TRACKING (ADD THIS) ──
        self._active_unknown_face = None
        self._unknown_last_seen = 0
        self._unknown_timeout = 3  # seconds

        # ─── LIVENESS SYSTEM ───
        self._liveness_history = []
        self._final_history = []

        # blink
        self.blink_detected = False

        # motion
        self.prev_face = None

        # cooldown
        self._last_liveness_check = 0
        self.LIVENESS_COOLDOWN = 0.5
        self.running = False
        self._thread = None

        self.COOLDOWN = 8
        self.INTRUDER_COOLDOWN = 5

        self._seen = {}
        self._face_history = {}
        self._last_seen_time = {}
        self.COOLDOWN_SECONDS = 300
        self._last_intruder_time = 0

        self._locked_name = None
        self._lock_frames = 0

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self._session_id = uuid.uuid4()

        # Initialize canvas elements
        self._off_bg = None

        # Show initial camera off screen
        self.after(100, self._show_camera_off_screen)

    @staticmethod
    def eye_aspect_ratio(eye):
        import numpy as np
        A = np.linalg.norm(eye[1] - eye[5])
        B = np.linalg.norm(eye[2] - eye[4])
        C = np.linalg.norm(eye[0] - eye[3])
        return (A + B) / (2.0 * C)

    def check_motion(self, face_img):
        try:
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (128, 128))

            if not hasattr(self, "prev_face") or self.prev_face is None:
                self.prev_face = gray
                return False

            prev = cv2.resize(self.prev_face, (128, 128))
            diff = cv2.absdiff(prev, gray)
            score = np.mean(diff)
            self.prev_face = gray
            return score > 2.5

        except Exception as e:
            print("Motion error:", e)
            return False

    # ── UI ────────────────────────────────────

    def _build_ui(self) -> None:
        top = tk.Frame(self, bg="#11111b", pady=8)
        top.pack(fill=tk.X)
        tk.Label(top, text="SAFE FACE", font=("Courier", 14, "bold"),
                 bg="#11111b", fg="#cdd6f4").pack()

        main = tk.Frame(self, bg="#1e1e2e")
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)

        left = tk.Frame(main, bg="#1e1e2e")
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(left, bg="#000000",
                                highlightbackground="#45475a", highlightthickness=1)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        right = tk.Frame(main, bg="#1e1e2e", width=280)
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=(8, 0))
        right.pack_propagate(False)

        canvas = tk.Canvas(right, bg="#1e1e2e", highlightthickness=0)
        scrollbar = tk.Scrollbar(right, orient="vertical", command=canvas.yview)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.configure(yscrollcommand=scrollbar.set)

        self.scroll_frame = tk.Frame(canvas, bg="#1e1e2e")
        canvas_window = canvas.create_window((0, 0), window=self.scroll_frame, anchor="nw")

        def update_scroll_region(event=None):
            canvas.configure(scrollregion=canvas.bbox("all"))
        self.scroll_frame.bind("<Configure>", update_scroll_region)

        def resize_frame(event):
            canvas.itemconfig(canvas_window, width=event.width)
        canvas.bind("<Configure>", resize_frame)

        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        self._build_status(self.scroll_frame)
        self._build_log_panel(self.scroll_frame)
        self._build_controls(self.scroll_frame)

        bot = tk.Frame(self, bg="#11111b", pady=4)
        bot.pack(fill=tk.X, side=tk.BOTTOM)
        tk.Label(bot,
                 text="Place face images in known_faces/ folder • filename = roll number",
                 font=("Courier", 8), bg="#11111b", fg="#585b70").pack()

    def _show_camera_off_screen(self):
        if not hasattr(self, 'canvas'):
            return

        self.canvas.delete("all")
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()

        if w <= 1 or h <= 1:
            w, h = 800, 600

        self._off_bg = self.canvas.create_rectangle(0, 0, w, h, fill="#0b0b0f", outline="")
        cx, cy = w // 2, h // 2 - 40

        self.canvas.create_rectangle(cx-60, cy-30, cx+60, cy+30, outline="#cdd6f4", width=2)
        self.canvas.create_oval(cx-15, cy-15, cx+15, cy+15, outline="#cdd6f4", width=2)
        self.canvas.create_rectangle(cx-25, cy-45, cx+25, cy-30, outline="#cdd6f4", width=2)
        self.canvas.create_line(cx-80, cy+60, cx+80, cy-60, fill="#f38ba8", width=4)

        self.canvas.create_text(cx, cy+90, text="CAMERA OFF",
                                fill="#cdd6f4", font=("Courier", 22, "bold"))
        self.canvas.create_text(cx, cy+130, text="Press START to begin scanning",
                                fill="#6c7086", font=("Courier", 11))

    def _section(self, parent, title: str) -> tk.LabelFrame:
        f = tk.LabelFrame(parent, text=title, font=("Courier", 9, "bold"),
                          bg="#1e1e2e", fg="#89b4fa", bd=1, relief=tk.GROOVE,
                          padx=6, pady=6)
        f.pack(fill=tk.X, pady=(0, 8))
        return f

    def _open_intruder_folder(self):
        path = os.path.abspath(self.engine.INTRUDER_DIR)
        if sys.platform == "win32":
            os.startfile(path)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])

    def _build_status(self, parent) -> None:
        sec = self._section(parent, " STATUS ")
        def row(label, attr, val, color):
            f = tk.Frame(sec, bg="#1e1e2e")
            f.pack(fill=tk.X, pady=1)
            tk.Label(f, text=label, font=("Courier", 9), bg="#1e1e2e", fg="#a6adc8",
                     width=12, anchor="w").pack(side=tk.LEFT)
            lbl = tk.Label(f, text=val, font=("Courier", 9, "bold"),
                           bg="#1e1e2e", fg=color, anchor="w")
            lbl.pack(side=tk.LEFT)
            setattr(self, attr, lbl)
        row("Camera :", "_lbl_cam",    "OFF",     "#f38ba8")
        row("Faces :",  "_lbl_faces",  str(len(self.engine.names)), "#cdd6f4")
        row("Status :", "_lbl_status", "Standby", "#a6e3a1")

    def _build_log_panel(self, parent) -> None:
        sec = self._section(parent, " EVENTS ")
        self._log_box = tk.Text(sec, height=14, bg="#11111b", fg="#cdd6f4",
                                font=("Courier", 8), state=tk.DISABLED,
                                relief=tk.FLAT, bd=0)
        sb = tk.Scrollbar(sec, command=self._log_box.yview)
        self._log_box.configure(yscrollcommand=sb.set)
        self._log_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self._log_box.tag_config("known",    foreground="#a6e3a1")
        self._log_box.tag_config("intruder", foreground="#f38ba8")
        self._log_box.tag_config("info",     foreground="#89dceb")
        self._log_box.tag_config("warn",     foreground="#fab387")

    def _build_controls(self, parent) -> None:
        sec = self._section(parent, " CONTROLS ")
        cfg = dict(font=("Courier", 9, "bold"), relief=tk.FLAT, cursor="hand2", padx=4, pady=5)

        self._btn_start = tk.Button(sec, text="▶ START", bg="#a6e3a1", fg="#1e1e2e",
                                    command=self._start, **cfg)
        self._btn_start.pack(fill=tk.X, pady=3)

        self._btn_stop = tk.Button(sec, text="■ STOP", bg="#f38ba8", fg="#1e1e2e",
                                   command=self._stop, state=tk.DISABLED, **cfg)
        self._btn_stop.pack(fill=tk.X, pady=3)

        tk.Button(sec, text="↺ RELOAD FACES", bg="#89b4fa", fg="#1e1e2e",
                  command=self._reload_faces, **cfg).pack(fill=tk.X, pady=3)

        tk.Button(sec, text="🗑 RESET LOGS", bg="#fab387", fg="#1e1e2e",
                  command=self._reset_logs, **cfg).pack(fill=tk.X, pady=3)

        tk.Button(sec, text="📂 OPEN LOG FOLDER", bg="#313244", fg="#cdd6f4",
                  command=self._open_logs, **cfg).pack(fill=tk.X, pady=3)

        tk.Button(sec, text="📸 VIEW INTRUDERS", bg="#f38ba8", fg="#1e1e2e",
                  command=self._open_intruder_folder, **cfg).pack(fill=tk.X, pady=3)

        tk.Button(sec, text="👨‍🎓 MANAGE STUDENTS", bg="#89dceb", fg="#1e1e2e",
                  command=self._open_student_manager,
                  font=("Courier", 9, "bold"), relief=tk.FLAT,
                  cursor="hand2", padx=4, pady=5).pack(fill=tk.X, pady=3)

    def _open_student_manager(self):
        if self.student_win is not None and tk.Toplevel.winfo_exists(self.student_win):
            self.student_win.lift()
            self.student_win.focus_force()
            return

        win = tk.Toplevel(self)
        self.student_win = win
        win.attributes("-topmost", True)
        win.after(200, lambda: win.attributes("-topmost", False))
        win.title("Manage Students")
        win.geometry("400x300")
        win.configure(bg="#1e1e2e")
        win.lift()
        win.focus_force()

        tk.Label(win, text="MANAGE STUDENTS", font=("Courier", 14, "bold"),
                 bg="#1e1e2e", fg="#cdd6f4").pack(pady=10)

        tk.Button(win, text="➕ Add Student", bg="#a6e3a1", fg="#1e1e2e",
                  command=lambda: self._add_student_window(win)).pack(fill=tk.X, padx=20, pady=8)

        tk.Button(win, text="🗑 Delete Student", bg="#f38ba8", fg="#1e1e2e",
                  command=lambda: self._delete_student_window(win)).pack(fill=tk.X, padx=20, pady=8)

        def on_close():
            self.student_win = None
            win.destroy()
        win.protocol("WM_DELETE_WINDOW", on_close)

    def _add_student_window(self, parent):
        if getattr(self, "_add_win", None) is not None:
            try:
                if self._add_win.winfo_exists():
                    self._add_win.lift()
                    self._add_win.focus_force()
                    return
            except:
                self._add_win = None

        win = tk.Toplevel(parent)
        win.transient(parent)
        win.grab_set()
        win.lift()
        win.focus_force()
        win.attributes("-topmost", True)
        win.after(200, lambda: win.attributes("-topmost", False))
        self._add_win = win
        win.title("Add Student")
        win.geometry("420x220")
        win.configure(bg="#1e1e2e")

        tk.Label(win, text="SELECT STUDENT FOLDER(S)", bg="#1e1e2e", fg="#cdd6f4",
                 font=("Courier", 12, "bold")).pack(pady=20)

        def select_folder():
            path = filedialog.askdirectory(title="Select Folder or Parent Folder")
            if not path:
                return

            passout_year = simpledialog.askinteger("Passout Year", "Enter Passout Year (e.g., 2026):", parent=win)

            if not passout_year:
                self._log("⚠ Operation cancelled (no passout year)", "warn")
                return

            try:
                self._log("📂 Scanning folders...", "info")
                self.update_idletasks()

                subfolders = [f for f in os.listdir(path)
                              if os.path.isdir(os.path.join(path, f))]
                overwrite_mode = None

                if not subfolders:
                    roll = os.path.basename(path)
                    if not roll.isdigit():
                        self._log("❌ Folder name must be numeric (roll number)", "intruder")
                        return
                    exists = self.engine.collection.find_one({"roll_number": roll})
                    if exists:
                        if not messagebox.askyesno("Duplicate Found",
                                                   f"Student {roll} already exists.\n\nOverwrite?"):
                            self._log(f"⚠ Skipped {roll}", "warn")
                            return
                    self._log(f"📂 Processing {roll}...", "info")
                    auto_register(path, passout_year, log_callback=self._log)
                    self._log(f"✅ Student {roll} added/updated", "known")
                else:
                    for folder in subfolders:
                        full_path = os.path.join(path, folder)
                        if not folder.isdigit():
                            self._log(f"⚠ Skipped invalid folder: {folder}", "warn")
                            continue
                        exists = self.engine.collection.find_one({"roll_number": folder})
                        if exists:
                            if overwrite_mode is None:
                                choice = messagebox.askyesnocancel(
                                    "Duplicate Found",
                                    "Duplicates detected.\n\nYes = Overwrite ALL\n"
                                    "No = Skip ALL\nCancel = Decide individually")
                                overwrite_mode = "individual" if choice is None else choice
                            if overwrite_mode is False:
                                self._log(f"⚠ Skipped {folder}", "warn")
                                continue
                            elif overwrite_mode == "individual":
                                if not messagebox.askyesno("Overwrite?", f"Overwrite student {folder}?"):
                                    self._log(f"⚠ Skipped {folder}", "warn")
                                    continue
                        self._log(f"📂 Processing {folder}...", "info")
                        auto_register(full_path, passout_year, log_callback=self._log)
                        self._log(f"✅ {folder} added/updated", "known")

                self.engine.reload_faces()
                self._lbl_faces.configure(text=str(len(self.engine.names)))
                self._log("🎉 All operations completed", "known")
                self._add_win = None
                win.destroy()

            except Exception as e:
                self._log(f"❌ Error: {e}", "intruder")

        tk.Button(win, text="📂 Select Folder / Batch", bg="#a6e3a1", fg="#1e1e2e",
                  font=("Courier", 11, "bold"), command=select_folder).pack(fill=tk.X, padx=40, pady=20)

        def on_close():
            self._add_win = None
            win.destroy()
        win.protocol("WM_DELETE_WINDOW", on_close)

    def _delete_student_window(self, parent):
        if hasattr(self, "_delete_win") and self._delete_win is not None:
            try:
                if self._delete_win.winfo_exists():
                    self._delete_win.lift()
                    self._delete_win.focus_force()
                    return
            except:
                self._delete_win = None

        win = tk.Toplevel(parent)
        self._delete_win = win
        win.title("Delete Student")
        win.geometry("380x220")
        win.configure(bg="#1e1e2e")
        win.transient(parent)
        win.grab_set()
        win.lift()
        win.focus_force()
        win.attributes("-topmost", True)
        win.after(200, lambda: win.attributes("-topmost", False))

        tk.Label(win, text="ENTER ROLL NUMBER", bg="#1e1e2e", fg="#cdd6f4",
                 font=("Courier", 11, "bold")).pack(pady=15)

        def validate(P):
            if P.isdigit() or P == "":
                return True
            self._log("❌ Only numbers allowed", "intruder")
            return False

        vcmd = (win.register(validate), "%P")
        entry = tk.Entry(win, font=("Courier", 12), justify="center",
                         validate="key", validatecommand=vcmd)
        entry.pack(pady=5)

        def delete():
            roll = entry.get().strip()
            if not roll:
                self._log("❌ Please enter roll number", "intruder")
                return
            if not messagebox.askyesno("Confirm Delete", f"Delete student {roll}?"):
                self._log("⚠ Delete cancelled", "warn")
                return
            try:
                result = self.engine.collection.delete_one({"roll_number": roll})
                if result.deleted_count > 0:
                    self.engine.reload_faces()
                    self._lbl_faces.configure(text=str(len(self.engine.names)))
                    self._log(f"🗑 Student {roll} deleted", "intruder")
                else:
                    self._log(f"⚠ No student found: {roll}", "warn")
            except Exception as e:
                self._log(f"❌ Delete error: {e}", "intruder")
            finally:
                win.destroy()

        tk.Button(win, text="DELETE", bg="#f38ba8", fg="#1e1e2e",
                  font=("Courier", 11, "bold"), command=delete).pack(pady=15, fill=tk.X, padx=25)

        def on_close():
            self._delete_win = None
            win.destroy()
        win.protocol("WM_DELETE_WINDOW", on_close)

    # ── logging ───────────────────────────────

    def _log(self, msg: str, tag: str = "info") -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        self._log_box.configure(state=tk.NORMAL)
        self._log_box.insert(tk.END, f"[{ts}] {msg}\n", tag)
        self._log_box.see(tk.END)
        self._log_box.configure(state=tk.DISABLED)

    # ── button handlers ───────────────────────

    def _start(self) -> None:
        if self.running:
            self._session_id = uuid.uuid4()
            return
        self.running = True
        self._seen = {}
        self._current_intruder_id = None  # Reset intruder tracking
        self._last_intruder_log_time = 0
        self._thread = threading.Thread(target=self._video_loop, daemon=True)
        self._thread.start()
        self._btn_start.configure(state=tk.DISABLED)
        self._btn_stop.configure(state=tk.NORMAL)
        self._lbl_cam.configure(text="ON", fg="#a6e3a1")
        self._lbl_status.configure(text="Running", fg="#a6e3a1")
        self._log("System started.", "info")

    def _stop(self):
        self._session_id = uuid.uuid4()
        self.running = False
        self._btn_start.configure(state=tk.NORMAL)
        self._btn_stop.configure(state=tk.DISABLED)
        self._lbl_cam.configure(text="OFF", fg="#f38ba8")
        self._lbl_status.configure(text="Standby", fg="#a6e3a1")
        self._log("System stopped.", "info")
        self.canvas.delete("all")
        self.after(100, lambda: [self._show_camera_off_screen(), self._pulse_off_screen()])

    def _pulse_off_screen(self):
        if self.running or not hasattr(self, '_off_bg') or self._off_bg is None:
            return
        try:
            current = self.canvas.itemcget(self._off_bg, "fill")
            new_color = "#0b0b0f" if current == "#11111b" else "#11111b"
            self.canvas.itemconfig(self._off_bg, fill=new_color)
            self.after(700, self._pulse_off_screen)
        except:
            pass

    def _reload_faces(self) -> None:
        n = self.engine.reload_faces()
        self._lbl_faces.configure(text=str(n))
        self._log(f"Reloaded – {n} face(s) loaded.", "info")

    def _reset_logs(self) -> None:
        if messagebox.askyesno("Reset Logs", "Clear all arrival and intruder logs?"):
            self.engine.reset_logs()
            self._log("Logs reset.", "warn")

    def _open_logs(self) -> None:
        path = os.path.abspath(self.engine.LOGS_DIR)
        if sys.platform == "win32":
            os.startfile(path)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])

    # ── video loop (background thread) ────────

    def _video_loop(self):
        cap = cv2.VideoCapture(0)
        local_session = self._session_id
        frame_count = 0

        if not cap.isOpened():
            self.after(0, lambda: self._log("ERROR: Cannot open webcam."))
            self.after(0, self._stop)
            return

        process_this_frame = True
        last_locs = []
        last_encs = []

        SCALE = 1 / 0.35
        last_processed_face_id = None

        while True:
            frame_count += 1

            if not self.running or local_session != self._session_id:
                cap.release()
                break

            ret, frame = cap.read()
            if not ret:
                break

            display = frame.copy()
            now = time.time()

            # ── resize for speed ──
            small = cv2.resize(frame, (0, 0), fx=0.35, fy=0.35)
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

            # ── detect every 2nd frame ──
            if process_this_frame:
                last_locs, last_encs = get_face_locations_and_encodings(rgb)
            process_this_frame = not process_this_frame

            has_unknown = False

            for idx, ((top, right, bottom, left), enc) in enumerate(zip(last_locs, last_encs)):

                # scale back
                top = int(top * SCALE)
                right = int(right * SCALE)
                bottom = int(bottom * SCALE)
                left = int(left * SCALE)

                face_roi = frame[top:bottom, left:right]
                if face_roi.size == 0:
                    continue

                # ───────────────
                # 🔒 FACE LOCK
                # ───────────────
                if self._locked_name and self._lock_frames > 0:
                    name = self._locked_name
                    is_known = True
                    confidence = 100.0
                    self._lock_frames -= 1
                else:
                    name = "Unknown"
                    confidence = 0.0
                    is_known = False

                    if self.engine.avg_encodings:
                        avg_dists = face_distance(self.engine.avg_encodings, enc)
                        best_idx = np.argmin(avg_dists)
                        best_avg_dist = avg_dists[best_idx]

                        if best_avg_dist < 0.6:
                            candidate_encs = self.engine.all_encodings[best_idx]
                            full_dists = face_distance(candidate_encs, enc)
                            best_full_dist = np.min(full_dists)

                            if best_full_dist < 0.5:
                                name = self.engine.names[best_idx]
                                is_known = True
                                confidence = (1.0 - best_full_dist) * 100

                                # 🔥 lock face
                                self._locked_name = name
                                self._lock_frames = 15

                # ───────────────
                # 🧠 LIVENESS
                # ───────────────
                if self._locked_name:
                    is_real = True
                elif frame_count % 5 == 0 and self.liveness_model is not None:
                    is_real = is_real_face(face_roi)
                else:
                    is_real = True

                if not is_real:
                    cv2.rectangle(display, (left, top), (right, bottom), (0, 0, 255), 2)
                    cv2.putText(display, "FAKE FACE", (left, top - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                    if now - self._last_intruder_time > self.INTRUDER_COOLDOWN:
                        self._last_intruder_time = now
                        self.engine.log_intruder(frame, "fake")
                        self.after(0, lambda: self._log("❌ Fake face detected!", "intruder"))
                    continue

                # ───────────────
                # 🎯 DRAW BOX
                # ───────────────
                box_color = (0, 255, 120) if is_known else (0, 80, 255)
                cv2.rectangle(display, (left, top), (right, bottom), box_color, 2)

                label = f"{name} | {confidence:.1f}%" if is_known else "UNKNOWN"
                cv2.putText(display, label, (left, top - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

                # ───────────────
                # 🚨 INTRUDER CONTROL
                # ───────────────
                if not is_known:
                    has_unknown = True

                    face_id = f"{idx}_{top}_{left}"

                    if face_id != last_processed_face_id:
                        last_processed_face_id = face_id

                        if now - self._last_intruder_log_time > self._intruder_cooldown_seconds:
                            self._last_intruder_log_time = now
                            photo = self.engine.log_intruder(frame, face_id)

                            if photo:
                                self.after(0, lambda p=photo:
                                self._log(f"🚨 Intruder saved: {os.path.basename(p)}", "intruder"))

                # ───────────────
                # ✅ ARRIVAL LOG
                # ───────────────
                if is_known:
                    self._face_history[name] = self._face_history.get(name, 0) + 1

                    if self._face_history[name] >= 5:
                        last_time = self._last_seen_time.get(name, 0)

                        if now - last_time > self.COOLDOWN_SECONDS:
                            self._last_seen_time[name] = now
                            self.engine.log_arrival(name)
                            self.after(0, lambda n=name:
                            self._log(f"✓ {n} logged.", "known"))

            # ───────────────
            # 🔄 RESET STATES
            # ───────────────
            if not last_locs:
                self._face_history.clear()
                self._locked_name = None
                self._lock_frames = 0

            if not has_unknown:
                last_processed_face_id = None

            # ───────────────
            # 🕒 TIMESTAMP
            # ───────────────
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(display, ts, (10, display.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

            # ───────────────
            # 🖥 DISPLAY SAFE
            # ───────────────
            rgb_disp = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb_disp)

            cw = self.canvas.winfo_width()
            ch = self.canvas.winfo_height()

            if cw > 1 and ch > 1:
                img = img.resize((cw, ch), Image.LANCZOS)

            imgtk = ImageTk.PhotoImage(image=img)

            # 🔴 CRITICAL STOP CHECK
            if not self.running or local_session != self._session_id:
                cap.release()
                break

            self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
            self.canvas.image = imgtk

            time.sleep(0.03)

        cap.release()

    def _on_close(self) -> None:
        self.running = False
        time.sleep(0.1)
        self.destroy()


# ══════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════

if __name__ == "__main__":
    app = App()
    app.mainloop()