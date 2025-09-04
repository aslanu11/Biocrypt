import os

import sys

import uuid

import hashlib

import pandas as pd

import cv2

import face_recognition

import numpy as np

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,

                             QHBoxLayout, QPushButton, QLabel, QLineEdit,

                             QMessageBox, QInputDialog)

from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QFont

from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal

 

class CameraThread(QThread):

    frame_signal = pyqtSignal(np.ndarray)

 

    def __init__(self, camera_index=0):

        super().__init__()

        self.camera_index = camera_index

        self.running = True

 

    def run(self):

        cap = cv2.VideoCapture(self.camera_index)

        while self.running:

            ret, frame = cap.read()

            if ret:

                self.frame_signal.emit(frame)

        cap.release()

 

    def stop(self):

        self.running = False

 

class BiocryptApp(QMainWindow):

    def __init__(self):

        super().__init__()

        self.setWindowTitle("Biocrypt")

        self.setGeometry(100, 100, 800, 600)

 

        # Central widget and layout

        central_widget = QWidget()

        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout()

        central_widget.setLayout(main_layout)

 

        # Left sidebar for navigation

        sidebar = QWidget()

        sidebar_layout = QVBoxLayout()

        sidebar.setLayout(sidebar_layout)

        sidebar.setFixedWidth(200)

 

        # Navigation buttons

        buttons = [

            ("Generate Key", self.generate_key),

            ("Encrypt Data", self.encrypt_data),

            ("Delete Data", self.delete_data)

        ]

 

        for label, method in buttons:

            btn = QPushButton(label)

            btn.clicked.connect(method)

            sidebar_layout.addWidget(btn)

 

        # Camera preview area

        self.camera_label = QLabel("Camera Preview")

        self.camera_label.setFixedSize(500, 400)

        self.camera_label.setAlignment(Qt.AlignCenter)

 

        # Camera thread setup

        self.camera_thread = CameraThread()

        self.camera_thread.frame_signal.connect(self.update_camera_preview)

        self.camera_thread.start()

 

        # Main layout setup

        main_layout.addWidget(sidebar)

        main_layout.addWidget(self.camera_label)

 

    def create_user_database(self):

        """Create an Excel spreadsheet to store user data if it doesn't already exist."""

        filename = 'user_database.xlsx'

       

        if not os.path.exists(filename):

            df = pd.DataFrame(columns=['face_id', 'encryption_key', 'password', 'image_path'])

            df.to_excel(filename, index=False)

            QMessageBox.information(self, "Database", f"New user database created: {filename}")

        else:

            QMessageBox.information(self, "Database", f"Existing user database found: {filename}")

       

        return filename

 

    def capture_and_save(self, filename):

        """Capture an image from the camera"""

        cap = cv2.VideoCapture(0)  # Using camera index 2

        if not cap.isOpened():

            QMessageBox.warning(self, "Camera Error", f"Could not open camera index 2.")

            return False

 

        # Capture a single frame

        ret, frame = cap.read()

        if not ret:

            QMessageBox.warning(self, "Capture Error", "Could not read frame from camera.")

            cap.release()

            return False

        else:

            # Save the captured frame to the specified filename

            cv2.imwrite(filename, frame)

            QMessageBox.information(self, "Image Saved", f"Image saved as {filename}.")

 

        cap.release()

        return True

 

    def generate_unique_face_id(self):

        """Generate a unique face ID"""

        # Generate a random unique face ID

        unique_string = str(uuid.uuid4())

       

        # Create a hash and take first 6 characters

        face_id = hashlib.sha256(unique_string.encode()).hexdigest()[:6]

       

        # Create translation map with matched length

        translation_map = str.maketrans(

            '012345',

            '>^<v+-'

        )

        face_id = face_id.translate(translation_map)

        return face_id

 

    def generate_key(self):

        """Process for generating a key with face ID and password"""

        # Generate unique face ID

        face_id = self.generate_unique_face_id()

 

        # Take and save face image

        image_filename = f"face_{face_id}.jpg"

        if not self.capture_and_save(image_filename):

            return

 

        # Password input

        password, ok = QInputDialog.getText(self, "Password",

                                            "Enter a password for this face ID:",

                                            QLineEdit.Password)

        if not ok or not password:

            return

 

        # Encryption Key Generation

        encryption_key, ok = QInputDialog.getText(self, "Encryption Key",

                                                  "Enter an encryption key (or cancel for random):")

        if not ok:

            encryption_key = str(uuid.uuid4())

 

        # Load the existing database

        df = pd.read_excel('user_database.xlsx')

 

        # Store data in Excel

        new_entry = pd.DataFrame({

            'face_id': [face_id],

            'encryption_key': [encryption_key],

            'password': [password],

            'image_path': [image_filename]

        })

 

        # Append to existing database

        df = pd.concat([df, new_entry], ignore_index=True)

        df.to_excel('user_database.xlsx', index=False)

 

        QMessageBox.information(self, "Key Generation",

                                f"Key generation complete!\nFace ID: {face_id}")

 

    def compare_faces_multi(self, current_image_path, tolerance=0.6):

        """Compare the current face against all saved faces in the database"""

        try:

            # Load the existing database

            df = pd.read_excel('user_database.xlsx')

 

            # Load the current image

            current_image = face_recognition.load_image_file(current_image_path)

            current_encodings = face_recognition.face_encodings(current_image)

 

            if len(current_encodings) == 0:

                QMessageBox.warning(self, "Face Error", f"No face found in {current_image_path}!")

                return None

 

            current_encoding = current_encodings[0]

 

            # Compare against all saved faces

            for index, row in df.iterrows():

                # Load saved image

                try:

                    saved_image = face_recognition.load_image_file(row['image_path'])

                    saved_encodings = face_recognition.face_encodings(saved_image)

 

                    if len(saved_encodings) == 0:

                        continue  # Skip if no face found in saved image

 

                    saved_encoding = saved_encodings[0]

 

                    # Compare faces

                    matches = face_recognition.compare_faces(

                        [saved_encoding],

                        current_encoding,

                        tolerance=tolerance

                    )

 

                    if matches[0]:

                        # Calculate face distance

                        distance = face_recognition.face_distance([saved_encoding], current_encoding)[0]

                        QMessageBox.information(self, "Face Match",

                                                f"Face match found for Face ID {row['face_id']}!\n"

                                                f"Match confidence: {1 - distance:.2f}")

                       

                        # Return the matched user's details

                        return row.to_dict()

 

                except Exception as img_error:

                    QMessageBox.warning(self, "Image Error",

                                        f"Error processing saved image {row['image_path']}: {img_error}")

 

            # No match found

            QMessageBox.warning(self, "No Match", "No matching face found in the database.")

            return None

 

        except Exception as e:

            QMessageBox.warning(self, "Comparison Error", f"Error in face comparison: {e}")

            return None

 

    def symbolic_caesar_encrypt(self, text, key):

        """Custom encryption using symbolic Caesar shift based on face ID"""

        # Mapping of symbolic characters to shift values

        shift_map = {

            '>': 3,   # Shift forward 3

            '^': 5,   # Shift forward 5

            '<': -3,  # Shift backward 3

            'v': -5,  # Shift backward 5

            '+': 2,   # Shift forward 2

            '-': -2   # Shift backward 2

        }

       

        # Calculate cumulative shift

        total_shift = sum(shift_map.get(char, 0) for char in key)

       

        # Apply Caesar cipher

        ciphertext = ''

        for char in text:

            if char.isalpha():

                # Determine base (uppercase or lowercase)

                is_upper = char.isupper()

                char = char.lower()

               

                # Apply shift

                shifted = chr((ord(char) - ord('a') + total_shift) % 26 + ord('a'))

               

                # Restore original case

                shifted = shifted.upper() if is_upper else shifted

                ciphertext += shifted

            else:

                # Non-alphabetic characters remain unchanged

                ciphertext += char

       

        return ciphertext

 

    def encrypt_data(self):

        """Process for encrypting data with face authentication"""

        # Capture current face

        current_image_filename = "current_face.jpg"

        if not self.capture_and_save(current_image_filename):

            return

 

        # Compare current face against all saved faces

        matched_user = self.compare_faces_multi(current_image_filename)

        if not matched_user:

            return

 

        # Password verification

        attempts = 3

        while attempts > 0:

            password, ok = QInputDialog.getText(self, "Password Verification",

                                                "Enter your password:",

                                                QLineEdit.Password)

            if not ok:

                return

 

            if password == matched_user['password']:

                break

            else:

                attempts -= 1

                QMessageBox.warning(self, "Incorrect Password",

                                    f"Incorrect password. {attempts} attempts remaining.")

                if attempts == 0:

                    QMessageBox.warning(self, "Too Many Attempts",

                                        "Too many incorrect attempts.")

                    return

 

        # Encryption process

        plaintext, ok = QInputDialog.getText(self, "Encryption",

                                             "Enter the text to encrypt:")

        if not ok:

            return

 

        # Custom encryption using face ID as key

        ciphertext = self.symbolic_caesar_encrypt(plaintext, matched_user['face_id'])

 

        QMessageBox.information(self, "Encryption Complete",

                                f"Plaintext:  {plaintext}\n"

                                f"Ciphertext: {ciphertext}\n"

                                f"Used Face ID: {matched_user['face_id']}")

 

    def delete_data(self):

        """Process for deleting user data with face and password authentication"""

        # Capture current face

        current_image_filename = "current_face.jpg"

        if not self.capture_and_save(current_image_filename):

            return

 

        # Load the existing database

        df = pd.read_excel('user_database.xlsx')

 

        # Compare current face against all saved faces

        matched_user = self.compare_faces_multi(current_image_filename)

        if not matched_user:

            return

 

        # Password verification

        attempts = 3

        while attempts > 0:

            password, ok = QInputDialog.getText(self, "Password Verification",

                                                "Enter your password:",

                                                QLineEdit.Password)

            if not ok:

                return

 

            if password == matched_user['password']:

                break

            else:

                attempts -= 1

                QMessageBox.warning(self, "Incorrect Password",

                                    f"Incorrect password. {attempts} attempts remaining.")

                if attempts == 0:

                    QMessageBox.warning(self, "Too Many Attempts",

                                        "Too many incorrect attempts.")

                    return

 

        # Confirm deletion

        reply = QMessageBox.question(self, 'Confirm Deletion',

                                     "Are you sure you want to delete all data associated with this Face ID?",

                                     QMessageBox.Yes | QMessageBox.No)

 

        if reply == QMessageBox.Yes:

            # Remove the row corresponding to the face ID

            df = df[df['face_id'] != matched_user['face_id']]

            df.to_excel('user_database.xlsx', index=False)

           

            # Optional: Delete the associated image file

            try:

                os.remove(matched_user['image_path'])

                QMessageBox.information(self, "Deletion",

                                        f"Deleted image file: {matched_user['image_path']}")

            except FileNotFoundError:

                QMessageBox.warning(self, "File Not Found",

                                    f"Image file {matched_user['image_path']} not found.")

            except Exception as e:

                QMessageBox.warning(self, "Deletion Error",

                                    f"Error deleting image file: {e}")

           

            QMessageBox.information(self, "Deletion", "User data has been deleted from the database.")

        else:

            QMessageBox.information(self, "Cancelled", "Deletion cancelled.")

 

    def update_camera_preview(self, frame):

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        h, w, ch = rgb_frame.shape

        bytes_per_line = ch * w

        q_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)

        pixmap = QPixmap.fromImage(q_img)

        self.camera_label.setPixmap(pixmap.scaled(self.camera_label.size(),

                                                  Qt.KeepAspectRatio,

                                                  Qt.SmoothTransformation))

 

    def closeEvent(self, event):

        self.camera_thread.stop()

        self.camera_thread.wait()

        event.accept()

 

def main():

    # Ensure necessary libraries are installed

    try:

        import cv2

        import face_recognition

    except ImportError:

        print("Error: Required libraries not found.")

        print("Please install OpenCV (cv2) and face_recognition libraries.")

        return

   

    app = QApplication(sys.argv)

    biocrypt_app = BiocryptApp()

    biocrypt_app.create_user_database()

    biocrypt_app.show()

    sys.exit(app.exec_())

 

if __name__ == "__main__":

    main()