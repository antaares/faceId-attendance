from tkinter import Tk, Label, Button, messagebox
import tkinter as tk
import cv2
import dlib
from PIL import Image, ImageTk
import numpy as np
import sqlite3
import io
import face_recognition

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./models/shape_predictor_68_face_landmarks.dat')
face_rec_model = dlib.face_recognition_model_v1('./models/dlib_face_recognition_resnet_model_v1.dat')

class FaceRecognitionApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Face ID Attendance System")
        self.video_capture = None
        self.current_frame = None
        self.video_running = False
        self.known_face_encodings = []
        self.known_face_names = []

        self.face_detector = dlib.get_frontal_face_detector()
        self.init_db()  # Bazani boshlash

        self.show_main_menu()

    def init_db(self):
        self.conn = sqlite3.connect('face_recognition.db')
        self.cursor = self.conn.cursor()
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS faces (
                id INTEGER PRIMARY KEY,
                name TEXT,
                surname TEXT,
                phone TEXT,
                encoding BLOB,
                image BLOB,
                desc varchar(500)
            )
        ''')
        self.conn.commit()

    def save_face_to_db(self, name, surname, phone, encoding, image, desc):
        encoding_blob = sqlite3.Binary(np.array(encoding).tobytes())
        image_blob = sqlite3.Binary(image)
        self.cursor.execute('INSERT INTO faces (name, surname, phone, encoding, image, desc) VALUES (?, ?, ?, ?, ?, ?)', (name, surname, phone, encoding_blob, image_blob, desc))
        self.conn.commit()

    def load_known_faces(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.cursor.execute('SELECT name, encoding FROM faces')
        for row in self.cursor.fetchall():
            name, encoding_blob = row
            encoding = np.frombuffer(encoding_blob, dtype=np.float64)
            self.known_face_encodings.append(encoding)
            self.known_face_names.append(name)

    def show_main_menu(self):
        self.clear_window()

        self.btn_register_face = Button(self.master, text="Register Face", command=self.show_student_details_form)
        self.btn_register_face.pack(pady=10)

        self.btn_control_students = Button(self.master, text="Control Students", command=self.show_control_students)
        self.btn_control_students.pack(pady=10)

        self.btn_turn_on_system = Button(self.master, text="Turn On System", command=self.system_turned_on)
        self.btn_turn_on_system.pack(pady=10)

    def clear_window(self):
        for widget in self.master.winfo_children():
            widget.destroy()

    def show_student_details_form(self):
        self.clear_window()

        Label(self.master, text="Name:").grid(row=0, column=0)
        Label(self.master, text="Surname:").grid(row=1, column=0)
        Label(self.master, text="Phone:").grid(row=2, column=0)

        self.name_entry = tk.Entry(self.master)
        self.name_entry.grid(row=0, column=1)

        self.surname_entry = tk.Entry(self.master)
        self.surname_entry.grid(row=1, column=1)

        self.phone_entry = tk.Entry(self.master)
        self.phone_entry.grid(row=2, column=1)

        Button(self.master, text="Submit", command=self.show_register_face).grid(row=3, column=0, columnspan=2)

        self.btn_back = Button(self.master, text="Back", command=self.show_main_menu)
        self.btn_back.grid(row=4, column=0, columnspan=2)

    def show_register_face(self):
        self.name = self.name_entry.get()
        self.surname = self.surname_entry.get()
        self.phone = self.phone_entry.get()

        if not self.name or not self.surname or not self.phone:
            messagebox.showerror("Error", "All fields are required")
            return

        self.clear_window()

        self.master.geometry(f"{int(self.master.winfo_screenwidth() * 0.7)}x{int(self.master.winfo_screenheight() * 0.7)}")

        self.video_capture = cv2.VideoCapture(0)
        self.canvas = tk.Canvas(self.master, width=300, height=450)
        self.canvas.pack(pady=10)

        self.video_running = True
        self.update_video()

        self.btn_snapshot = Button(self.master, text="Snapshot", command=self.take_snapshot)
        self.btn_snapshot.pack(pady=5)

        self.btn_back = Button(self.master, text="Back", command=self.show_main_menu)
        self.btn_back.pack(pady=5)

    def update_video(self):
        if not self.video_running:
            return
        try:
            ret, frame = self.video_capture.read()
            if ret:
                height, width, _ = frame.shape
                center_x, center_y = width // 2, height // 2
                x1, y1 = center_x - 150, center_y - 225
                x2, y2 = center_x + 150, center_y + 225
                cropped_frame = frame[y1:y2, x1:x2]

                self.current_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
                self.photo = ImageTk.PhotoImage(image=Image.fromarray(self.current_frame))
                self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
                self.master.after(10, self.update_video)
        except Exception as ex:
            print("ERROR: 149")

    def take_snapshot(self):
        if self.current_frame is not None:
            face_locations = face_recognition.face_locations(self.current_frame)
            if len(face_locations) == 0:
                messagebox.showwarning("No Face Detected", "No face detected. Please try again.")
                return
            face_encodings = face_recognition.face_encodings(self.current_frame, face_locations)
            try:
                pil_image = Image.fromarray(self.current_frame)
                byte_arr = io.BytesIO()
                pil_image.save(byte_arr, format='PNG')
                image_bytes = byte_arr.getvalue()
                # Yuzni bazaga saqlash
                self.save_face_to_db(self.name, self.surname, self.phone, face_encodings[0], image_bytes, "Description")
                messagebox.showinfo("Success", "Face registered successfully!")
                self.show_main_menu()
            except Exception as e:
                pass

    def bytes_to_image(self, image_bytes):
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img

    def get_face_encodings(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        encodings = []
        for face in faces:
            landmarks = predictor(gray, face)
            encoding = np.array(face_rec_model.compute_face_descriptor(frame, landmarks))
            encodings.append(encoding)
        return encodings

    def compare_faces(self, known_encodings, face_encoding):
        distances = np.linalg.norm(known_encodings - face_encoding, axis=1)
        return distances

    def show_control_students(self):
        self.clear_window()
        self.master.geometry("800x600")

        self.canvas = tk.Canvas(self.master)
        self.scroll_y = tk.Scrollbar(self.master, orient="vertical", command=self.canvas.yview)

        self.frame = tk.Frame(self.canvas)
        self.frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))

        self.canvas.create_window((0, 0), window=self.frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scroll_y.set)

        self.load_student_list()

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scroll_y.pack(side="right", fill="y")

        self.btn_back = tk.Button(self.master, text="Back", command=self.show_main_menu)
        self.btn_back.pack(pady=5)

    def load_student_list(self):
        self.cursor.execute('SELECT id, name, surname, phone, image FROM faces')
        students = self.cursor.fetchall()

        for i, student in enumerate(students):
            student_id, name, surname, phone, image_blob = student

            image = Image.open(io.BytesIO(image_blob))
            image = image.resize((60, 90))
            img = ImageTk.PhotoImage(image)

            tk.Label(self.frame, text=f"Name: {name}").grid(row=i, column=0)
            tk.Label(self.frame, text=f"Surname: {surname}").grid(row=i, column=1)
            tk.Label(self.frame, text=f"Phone: {phone}").grid(row=i, column=2)

            label_img = tk.Label(self.frame, image=img)
            label_img.grid(row=i, column=3)
            label_img.image = img

            tk.Button(self.frame, text="Delete", command=lambda id=student_id: self.delete_student(id)).grid(row=i, column=4)

    def delete_student(self, student_id):
        self.cursor.execute('DELETE FROM faces WHERE id=?', (student_id,))
        self.conn.commit()
        self.show_control_students()

    def system_turned_on(self):
        self.clear_window()
        self.master.geometry("800x600")
        self.load_known_faces()

        self.video_capture = cv2.VideoCapture(0)
        self.video_running = True

        self.canvas = tk.Canvas(self.master, width=300, height=450)
        self.canvas.pack(pady=10)

        self.btn_back = Button(self.master, text="Back", command=self.show_main_menu)
        self.btn_back.pack(pady=5)

        while self.video_running:
            ret, frame = self.video_capture.read()
            if ret:
                height, width, _ = frame.shape
                center_x, center_y = width // 2, height // 2
                x1, y1 = center_x - 150, center_y - 225
                x2, y2 = center_x + 150, center_y + 225
                cropped_frame = frame[y1:y2, x1:x2]

                self.current_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
                self.photo = ImageTk.PhotoImage(image=Image.fromarray(self.current_frame))
                self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
                self.master.after(10, self.update_video)
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)
            face_encodings = self.get_face_encodings(frame=frame)

            for idx, face_encoding in enumerate(face_encodings):
                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                name = "Unknown"

                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = self.known_face_names[best_match_index]

                face = faces[idx]
                top_left = (face.left(), face.top())
                bottom_right = (face.right(), face.bottom())
                cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
                cv2.putText(frame, name, (face.left() + 6, face.top() - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            cv2.imshow('Video', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.video_running = False
                break




        self.video_capture.release()
        cv2.destroyAllWindows()

root = Tk()
app = FaceRecognitionApp(root)
root.mainloop()
