import sys
import pandas as pd
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from PyQt5.QtCore import Qt, QThread, QTimer
from PyQt5 import QtCore
from PyQt5.QtGui import QPixmap, QImage, QMovie
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog, QLabel, QMessageBox, QTableWidget,
    QDialog, QTableWidgetItem, QHBoxLayout, QHeaderView, QFrame, QProgressBar, QLineEdit, QFormLayout, QGroupBox,
    QSplashScreen
)
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, accuracy_score
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Classification System - Iris Dataset")
        self.setGeometry(100, 100, 800, 600)
        
        self.model_results = [] 
        
        self.webcam_timer = QTimer()
        self.webcam_timer.timeout.connect(self.update_webcam_frame)
        self.cap = None

        # Main horizontal layout
        main_layout = QHBoxLayout()
        
        leftbutton_style = "background-color: #ADE814;"
        rightbutton_style = "background-color: #B7EDF7;"

        # Left layout for buttons
        left_layout = QVBoxLayout()
        self.btn_load_csv = QPushButton("Load Iris CSV Dataset")
        self.btn_load_csv.setStyleSheet(leftbutton_style)
        self.btn_load_csv.clicked.connect(self.load_csv_dataset)
        left_layout.addWidget(self.btn_load_csv)
        
        # Iris loading progress
        self.iris_progress_bar = QProgressBar()
        self.iris_progress_bar.setMaximumWidth(200) 
        self.iris_progress_bar.setValue(0)
        self.iris_progress_bar.setTextVisible(True)
        left_layout.addWidget(self.iris_progress_bar)

        #self.btn_train_model = QPushButton("Train SVM Model")
        #self.btn_train_model.clicked.connect(self.train_model)
        #left_layout.addWidget(self.btn_train_model)
        
        self.btn_load_rose = QPushButton("Load Rose Image Folder")
        self.btn_load_rose.setStyleSheet(leftbutton_style)
        self.btn_load_rose.clicked.connect(self.load_rose_dataset)
        left_layout.addWidget(self.btn_load_rose)
        
        # Iris loading progress
        self.rose_progress_bar = QProgressBar()
        self.rose_progress_bar.setMaximumWidth(200) 
        self.rose_progress_bar.setValue(0)
        self.rose_progress_bar.setTextVisible(True)
        left_layout.addWidget(self.rose_progress_bar)

        #self.btn_train_rose = QPushButton("Train Rose Model")
        #self.btn_train_rose.clicked.connect(self.train_rose_model)
        #left_layout.addWidget(self.btn_train_rose)
        
        self.btn_train_both = QPushButton("Train Both Models")
        self.btn_train_both.setStyleSheet(leftbutton_style)
        self.btn_train_both.clicked.connect(self.train_both_models)
        left_layout.addWidget(self.btn_train_both)
        
        self.btn_load_image = QPushButton("Load Image")
        self.btn_load_image.setStyleSheet(leftbutton_style)
        self.btn_load_image.clicked.connect(self.load_single_image)
        left_layout.addWidget(self.btn_load_image)
        
        self.btn_optimize_model = QPushButton("Optimize SVM Model")
        self.btn_optimize_model.setStyleSheet(leftbutton_style)
        self.btn_optimize_model.clicked.connect(self.optimize_model)
        left_layout.addWidget(self.btn_optimize_model)
        
        self.btn_webcam_predict = QPushButton("Webcam: Predict Rose Color")
        self.btn_webcam_predict.setStyleSheet("background-color: #F9C0AF;")
        self.btn_webcam_predict.clicked.connect(self.predict_from_webcam)
        left_layout.addWidget(self.btn_webcam_predict)
        
        self.btn_stop_webcam = QPushButton("Stop Webcam")
        self.btn_stop_webcam.setStyleSheet("background-color: #FFAAAA;")
        self.btn_stop_webcam.clicked.connect(self.stop_webcam)
        self.btn_stop_webcam.setEnabled(False)
        left_layout.addWidget(self.btn_stop_webcam)

        # Confidence Bar Charts
        self.iris_confidence_label = QLabel("Iris Confidence:")
        self.rose_confidence_label = QLabel("Rose Confidence:")

        self.iris_conf_canvas = QLabel()
        self.rose_conf_canvas = QLabel()

        left_layout.addWidget(self.iris_confidence_label)
        left_layout.addWidget(self.iris_conf_canvas)
        left_layout.addWidget(self.rose_confidence_label)
        left_layout.addWidget(self.rose_conf_canvas)
        
        #self.btn_verify_rose = QPushButton("Verify Rose Image")
        #self.btn_verify_rose.clicked.connect(self.verify_rose_image)
        #left_layout.addWidget(self.btn_verify_rose)

        left_layout.addStretch()  # Push buttons to the top
        
        self.btn_help = QPushButton("Info / Help")
        self.btn_help.setStyleSheet("background-color: #FEC107;")  # Optional: yellow button
        self.btn_help.clicked.connect(self.show_help_popup)
        left_layout.addWidget(self.btn_help)

        # Right layout for label, tables, and canvas
        self.right_layout = QVBoxLayout()

        self.label = QLabel("Welcome to the Classification System")
        self.right_layout.addWidget(self.label)
        
        # Horizontal layout to hold image and form side by side
        image_form_row = QHBoxLayout()

        # Image display
        self.image_label = QLabel("No image loaded")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFixedSize(256, 256)  # Optional: constrain size
        image_form_row.addWidget(self.image_label)

        # Iris form
        iris_group = QGroupBox("Iris")
        iris_form_layout = QFormLayout()

        self.sepal_length_input = QLineEdit()
        self.sepal_width_input = QLineEdit()
        self.petal_length_input = QLineEdit()
        self.petal_width_input = QLineEdit()

        iris_form_layout.addRow("Sepal Length:", self.sepal_length_input)
        self.sepal_length_input.setStyleSheet("background-color: #FEDCDB;")
        iris_form_layout.addRow("Sepal Width:", self.sepal_width_input)
        self.sepal_width_input.setStyleSheet("background-color: #FDF1C9;")
        iris_form_layout.addRow("Petal Length:", self.petal_length_input)
        self.petal_length_input.setStyleSheet("background-color: #F2D9EF;")
        iris_form_layout.addRow("Petal Width:", self.petal_width_input)
        self.petal_width_input.setStyleSheet("background-color: #D8E5F7;")

        iris_group.setLayout(iris_form_layout)
        iris_group.setFixedWidth(200)  # Optional: control layout size

        image_form_row.addWidget(iris_group)

        # Add the combined layout to the right panel
        self.right_layout.addLayout(image_form_row)
        
        prediction_bttn = QHBoxLayout()

        self.btn_verify_rose = QPushButton("Verify Rose Image")
        self.btn_verify_rose.setStyleSheet(rightbutton_style)
        self.btn_verify_rose.clicked.connect(self.verify_rose_image)
        
        # Predict button
        self.btn_predict_iris = QPushButton("Predict Iris Class")
        self.btn_predict_iris.setStyleSheet(rightbutton_style)
        self.btn_predict_iris.clicked.connect(self.predict_iris_class)
        
        prediction_bttn.addWidget(self.btn_verify_rose)
        prediction_bttn.addWidget(self.btn_predict_iris)
        
        self.right_layout.addLayout(prediction_bttn)

        # Horizontal layout for prediction results
        prediction_row = QHBoxLayout()

        self.iris_prediction_label = QLabel("Iris prediction: N/A")
        self.rose_prediction_label = QLabel("Rose color prediction: N/A")
        
        prediction_row.addWidget(self.rose_prediction_label)
        prediction_row.addWidget(self.iris_prediction_label)

        self.right_layout.addLayout(prediction_row)

        self.metrics_table = QTableWidget()
        self.right_layout.addWidget(self.metrics_table)
        
        self.accuracy_label = QLabel("Accuracy: N/A")

        # Wrap left layout in a QFrame
        left_frame = QFrame()
        left_frame.setLayout(left_layout)
        left_frame.setFrameShape(QFrame.StyledPanel)  # or Box, Panel, etc.
        left_frame.setStyleSheet("""
            QFrame {
                border: 2px solid #444;
                padding: 10px;
                background-color: #B7EDF7;
            }
        """)

        # Wrap right layout in a QFrame
        right_frame = QFrame()
        right_frame.setLayout(self.right_layout)
        right_frame.setFrameShape(QFrame.StyledPanel)
        right_frame.setStyleSheet("""
            QFrame {
                border: 2px solid #444;
                padding: 10px;
            }
        """)

        main_layout.addWidget(left_frame, stretch=1)   # Less space
        main_layout.addWidget(right_frame, stretch=3)  # More space

        self.setLayout(main_layout)
        self.setStyleSheet("background-color: #E1F7E1;")

    def load_csv_dataset(self):
        self.iris_progress_bar.setValue(0)
        self.iris_progress_bar.setFormat("Loading... %p%")
        QApplication.processEvents()

        file_path, _ = QFileDialog.getOpenFileName(self, "Open CSV", "", "CSV Files (*.csv)")
        if file_path:
            try:
                # Simulated loading steps (just visual effect)
                for i in range(1, 101, 20):
                    self.iris_progress_bar.setValue(i)
                    QApplication.processEvents()
                    QThread.msleep(50)  # Simulate delay for animation

                df = pd.read_csv(file_path)
                # Drop ID column if it exists
                if "Id" in df.columns:
                    df = df.drop("Id", axis=1)

                # Or use the actual feature names
                X = df[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]].values
                y = df["Species"].values

                self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )

                self.iris_progress_bar.setValue(100)
                self.iris_progress_bar.setFormat("Load complete")
            except Exception as e:
                self.iris_progress_bar.setValue(100)
                self.iris_progress_bar.setFormat("Failed to load")
        else:
            self.iris_progress_bar.setFormat("Load cancelled")

    def load_rose_dataset(self):
        self.rose_progress_bar.setValue(0)
        self.rose_progress_bar.setFormat("Loading... %p%")
        QApplication.processEvents()

        data_path = QFileDialog.getExistingDirectory(self, "Select Rose Image Folder")
        if not data_path:
            self.rose_progress_bar.setFormat("Load cancelled")
            return

        X, y = [], []
        files = [f for f in os.listdir(data_path) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

        if not files:
            self.rose_progress_bar.setFormat("No valid images found")
            return

        total = len(files)
        for idx, file_name in enumerate(files):
            img_path = os.path.join(data_path, file_name)
            img = cv2.imread(img_path)
            if img is not None:
                img_resized = cv2.resize(img, (64, 64))
                gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
                features = gray.flatten()
                X.append(features)

                # Extract color label from filename
                label = file_name.lower().split()[0]
                y.append(label)

            # Update progress bar
            progress = int((idx + 1) / total * 80)
            self.rose_progress_bar.setValue(progress)
            QApplication.processEvents()

        if not X:
            self.rose_progress_bar.setFormat("No valid images found")
            return

        X = np.array(X)
        y = np.array(y)

        dummy_negatives = np.random.rand(len(X), X.shape[1]) * 255
        X = np.vstack((X, dummy_negatives))
        y = np.concatenate((y, np.zeros(len(dummy_negatives), dtype=int)))

        # Final loading step
        self.rose_X_train, self.rose_X_test, self.rose_y_train, self.rose_y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Finish progress
        self.rose_progress_bar.setValue(100)
        self.rose_progress_bar.setFormat("Rose dataset loaded")


    def train_model(self):
        if hasattr(self, 'X_train') and hasattr(self, 'y_train'):
            self.model = SVC(kernel='linear')
            self.model.fit(self.X_train, self.y_train)
            y_pred = self.model.predict(self.X_test)

            self.model_results.append({
                'name': 'Iris',
                'y_true': self.y_test,
                'y_pred': y_pred
            })
            
    def train_rose_model(self):
        if hasattr(self, 'rose_X_train') and hasattr(self, 'rose_y_train'):
            # Encode string labels (e.g., "red", "pink", "yellow") to integers
            self.rose_label_encoder = LabelEncoder()
            y_train_encoded = self.rose_label_encoder.fit_transform(self.rose_y_train)
            y_test_encoded = self.rose_label_encoder.transform(self.rose_y_test)

            # Train the model
            self.model = SVC(kernel='linear')
            self.model.fit(self.rose_X_train, y_train_encoded)

            # Predict on test set
            y_pred = self.model.predict(self.rose_X_test)

            # Store results
            self.model_results.append({
                'name': 'Rose',
                'y_true': y_test_encoded,
                'y_pred': y_pred
            })
            
    def train_both_models(self):
       # Train Iris
        if hasattr(self, 'X_train') and hasattr(self, 'y_train'):
            # Encode target labels
            self.iris_label_encoder = LabelEncoder()
            y_train_encoded = self.iris_label_encoder.fit_transform(self.y_train)
            y_test_encoded = self.iris_label_encoder.transform(self.y_test)

            self.iris_model = SVC(kernel='linear', probability=True)
            self.iris_model.fit(self.X_train, y_train_encoded)

            y_pred = self.iris_model.predict(self.X_test)

            self.model_results.append({
                'name': 'Iris',
                'y_true': y_test_encoded,
                'y_pred': y_pred
            })

        # Train Rose
        if hasattr(self, 'rose_X_train') and hasattr(self, 'rose_y_train'):
            self.rose_label_encoder = LabelEncoder()
            y_train_encoded = self.rose_label_encoder.fit_transform(self.rose_y_train)
            y_test_encoded = self.rose_label_encoder.transform(self.rose_y_test)

            self.rose_model = SVC(kernel='linear', probability=True)
            self.rose_model.fit(self.rose_X_train, y_train_encoded)

            y_pred = self.rose_model.predict(self.rose_X_test)

            self.model_results.append({
                'name': 'Rose',
                'y_true': y_test_encoded,
                'y_pred': y_pred
            })
            
        self.evaluate_model()

    def evaluate_model(self):
        # Clear old plots if they exist
        if hasattr(self, 'last_plot_widget') and self.last_plot_widget:
            self.right_layout.removeWidget(self.last_plot_widget)
            self.last_plot_widget.deleteLater()
            self.last_plot_widget = None

        if not hasattr(self, 'model_results') or not self.model_results:
            QMessageBox.warning(self, "Error", "No model results to evaluate.")
            return

        # We'll show the metrics only for the last model
        latest_result = self.model_results[-1]
        y_true = latest_result['y_true']
        y_pred = latest_result['y_pred']
        label_names = list(np.unique(y_true))

        # Fill metrics table (top of layout)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )

        self.metrics_table.setRowCount(len(label_names))
        self.metrics_table.setColumnCount(4)
        self.metrics_table.setHorizontalHeaderLabels(["Precision", "Recall", "F1-Score", "Support"])
        self.metrics_table.setVerticalHeaderLabels([str(label) for label in label_names])

        for i in range(len(label_names)):
            self.metrics_table.setItem(i, 0, QTableWidgetItem(f"{precision[i]:.2f}"))
            self.metrics_table.setItem(i, 1, QTableWidgetItem(f"{recall[i]:.2f}"))
            self.metrics_table.setItem(i, 2, QTableWidgetItem(f"{f1[i]:.2f}"))
            self.metrics_table.setItem(i, 3, QTableWidgetItem(f"{support[i]}"))

        self.metrics_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        # --- SIDE BY SIDE MATRICES AND PRECISION-RECALL ---
        h_layout = QHBoxLayout()

        for result in self.model_results:
            y_true = result['y_true']
            y_pred = result['y_pred']
            name = result['name']

            # Encode string labels if necessary
            if isinstance(y_true[0], str):
                encoder = LabelEncoder()
                y_true = encoder.fit_transform(y_true)
                y_pred = encoder.transform(y_pred)

            labels = sorted(np.unique(np.concatenate((y_true, y_pred))))
            cmx = confusion_matrix(y_true, y_pred, labels=labels)
            accuracy = accuracy_score(y_true, y_pred)

            # Metrics
            precision, recall, f1, support = precision_recall_fscore_support(
                y_true, y_pred, average=None, zero_division=0
            )

            # --- Combined Plot: Confusion Matrix + PR Points ---
            fig, axes = plt.subplots(1, 2, figsize=(9, 4))

            # Confusion Matrix
            cax = axes[0].matshow(cmx, cmap='Blues')
            fig.colorbar(cax, ax=axes[0])
            axes[0].set_xticks(range(len(labels)))
            axes[0].set_yticks(range(len(labels)))
            axes[0].set_xticklabels(labels, rotation=45, ha='left')
            axes[0].set_yticklabels(labels)
            axes[0].set_xlabel('Prediction')
            axes[0].set_ylabel('Label')
            axes[0].set_title(f"{name} Accuracy: {accuracy:.2%}")

            # Precision-Recall Points
            for i, label in enumerate(labels):
                axes[1].plot(recall[i], precision[i], marker='o', linestyle='-', label=f"Class {label}")
            axes[1].set_xlim(0, 1.05)
            axes[1].set_ylim(0, 1.05)
            axes[1].set_xlabel("Recall")
            axes[1].set_ylabel("Precision")
            axes[1].set_title("Precision-Recall Points")
            axes[1].legend()
            axes[1].grid(True)
            fig.tight_layout(pad=5.0)

            canvas = FigureCanvas(fig)
            canvas.setFixedSize(400, 250)
            h_layout.addWidget(canvas)

        wrapper = QWidget()
        wrapper.setLayout(h_layout)
        self.right_layout.addWidget(wrapper)
        self.last_plot_widget = wrapper
        
    def load_single_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.jpeg)")
        if file_path:
            image = cv2.imread(file_path)
            if image is None:
                QMessageBox.warning(self, "Error", "Failed to load image.")
                return

            # Convert to RGB and resize for display
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_rgb = cv2.resize(image_rgb, (256, 256))  # Resize for consistent display

            # Convert to QImage
            height, width, channel = image_rgb.shape
            bytes_per_line = 3 * width
            q_img = QImage(image_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
            
            pixmap = QPixmap.fromImage(q_img)
            self.image_label.setPixmap(pixmap)
            
            self.loaded_image_path = file_path  # Store it for later verification
            
    def verify_rose_image(self):
        if not hasattr(self, 'rose_model') or not hasattr(self, 'rose_label_encoder'):
            self.rose_prediction_label.setText("Error: Train the Rose model first.")
            return

        if not hasattr(self, 'loaded_image_path') or not os.path.exists(self.loaded_image_path):
            QMessageBox.warning(self, "Error", "No image has been loaded.")
            return

        img = cv2.imread(self.loaded_image_path)
        if img is None:
            QMessageBox.warning(self, "Error", "Loaded image is invalid.")
            return

        img_resized = cv2.resize(img, (64, 64))
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        features = gray.flatten().reshape(1, -1)

        predicted_class = self.rose_model .predict(features)[0]
        predicted_label = self.rose_label_encoder.inverse_transform([predicted_class])[0]

        self.rose_prediction_label.setText(f"Rose color prediction: {predicted_label.capitalize()}")

        # Show confidence bar
        probs = self.rose_model.predict_proba(features)[0]
        class_names = self.rose_label_encoder.classes_
        self.plot_confidence_bar(probs, class_names, self.rose_conf_canvas)
        
    def predict_iris_class(self):
        try:
            # Get values from inputs
            sl = float(self.sepal_length_input.text())
            sw = float(self.sepal_width_input.text())
            pl = float(self.petal_length_input.text())
            pw = float(self.petal_width_input.text())

            features = np.array([[sl, sw, pl, pw]])

            # Check if model is trained
            if not hasattr(self, 'iris_model') or not hasattr(self, 'iris_label_encoder'):
                self.iris_prediction_label.setText("Error: Train the Iris model first.")
                return

            # Predict and decode class name
            predicted_class = self.iris_model.predict(features)[0]
            predicted_label = self.iris_label_encoder.inverse_transform([predicted_class])[0]

            self.iris_prediction_label.setText(f"Iris prediction: {predicted_label}")

            # Show confidence bar
            probs = self.iris_model.predict_proba(features)[0]
            class_names = self.iris_label_encoder.classes_
            self.plot_confidence_bar(probs, class_names, self.iris_conf_canvas)
        except Exception as e:
            self.iris_prediction_label.setText(f"Invalid input: {e}")
            
    def show_help_popup(self):
        QMessageBox.information(
            self,
            "Help / Info",
            "📌 Instructions:\n\n"
            "- Load Iris dataset (CSV) to start training.\n"
            "- Load Rose images for flower classification.\n"
            "- Train both models to enable predictions.\n"
            "- Use 'Load Image' to test rose classification.\n"
            "- Enter values in the Iris form to predict flower type.\n\n"
            "📊 You'll see metrics and graphs after training.\n"
            "📞 Contact: Pandai Pandailah cari"
        )

    def plot_confidence_bar(self, class_probs, class_names, canvas_label):
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(3.2, 1 + 0.3 * len(class_probs)))  # Dynamic height
        bars = ax.barh(class_names, class_probs, color=['#4CAF50', '#2196F3', '#FFD7EE', '#F5A7A6', '#F3F5A9'][:len(class_probs)])
        
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + 0.02, bar.get_y() + bar.get_height() / 2,
                    f"{class_probs[i] * 100:.1f}%", va='center', fontsize=8)

        ax.set_xlim(0, 1.0)
        ax.set_xlabel("Confidence", fontsize=9)
        ax.tick_params(axis='y', labelsize=9)
        ax.set_title("")
        ax.invert_yaxis()
        fig.tight_layout(pad=2.0)

        canvas = FigureCanvas(fig)
        canvas.setFixedHeight(int(30 + 25 * len(class_probs)))  # Set height based on class count
        canvas.setFixedWidth(250)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(canvas)

        wrapper = QWidget()
        wrapper.setLayout(layout)

        # Replace old widget contents
        old_layout = canvas_label.layout()
        if old_layout:
            QWidget().setLayout(old_layout)  # clear previous layout
        canvas_label.setPixmap(QPixmap())  # remove old pixmap
        canvas_label.setFixedSize(250, int(30 + 25 * len(class_probs)))  # Resize display area
        canvas_label.setLayout(layout)
        
    def optimize_model(self):
        # Determine which dataset to use
        if hasattr(self, 'rose_X_train') and hasattr(self, 'rose_y_train'):
            X_train = self.rose_X_train
            y_train = self.rose_y_train
            X_test = self.rose_X_test
            y_test = self.rose_y_test
            dataset_name = "Rose"
        elif hasattr(self, 'X_train') and hasattr(self, 'y_train'):
            X_train = self.X_train
            y_train = self.y_train
            X_test = self.X_test
            y_test = self.y_test
            dataset_name = "Iris"
        else:
            QMessageBox.warning(self, "Error", "Please load the Iris or Rose dataset first.")
            return

        # Define hyperparameter grid
        param_grid = {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf', 'poly'],
            'gamma': ['scale', 'auto']
        }

        # Perform Grid Search
        grid_search = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy', verbose=0)
        grid_search.fit(X_train, y_train)

        # Store the best model
        self.model = grid_search.best_estimator_

        # Predict and evaluate
        y_pred = self.model.predict(X_test)
        self.model_results.append({
            'name': f"{dataset_name} (Optimized)",
            'y_true': y_test,
            'y_pred': y_pred
        })

        # Inform the user
        QMessageBox.information(
            self,
            "Optimization Complete",
            f"Best parameters for {dataset_name}:\n{grid_search.best_params_}\n"
            f"CV Accuracy: {grid_search.best_score_:.2%}"
        )

        self.evaluate_model()
        
    def predict_from_webcam(self):
        if not hasattr(self, 'rose_model') or not hasattr(self, 'rose_label_encoder'):
            QMessageBox.warning(self, "Error", "Please train the Rose model first.")
            return

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            QMessageBox.warning(self, "Error", "Could not open webcam.")
            return

        self.btn_webcam_predict.setEnabled(False)
        self.btn_stop_webcam.setEnabled(True)
        self.webcam_timer.start(30)  # ~33 fps
        
    def update_webcam_frame(self):
        if self.cap is None or not self.cap.isOpened():
            return

        ret, frame = self.cap.read()
        if not ret:
            return

        display_frame = cv2.resize(frame, (256, 256))
        pred_frame = cv2.resize(frame, (64, 64))

        gray = cv2.cvtColor(pred_frame, cv2.COLOR_BGR2GRAY)
        features = gray.flatten().reshape(1, -1)

        pred_class = self.rose_model.predict(features)[0]
        probs = self.rose_model.predict_proba(features)[0]

        try:
            pred_label = self.rose_label_encoder.inverse_transform([pred_class])[0]
            is_rose = str(pred_label).lower() != '0'
        except Exception:
            pred_label = "Unknown"
            is_rose = False

        cv2.putText(display_frame, f"Prediction: {pred_label.capitalize()}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        if is_rose:
            h, w, _ = display_frame.shape
            x1, y1 = w//2 - 80, h//2 - 80
            x2, y2 = x1 + 160, y1 + 160
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_img)
        self.image_label.setPixmap(pixmap)

        class_names = self.rose_label_encoder.classes_
        self.plot_confidence_bar(probs, class_names, self.rose_conf_canvas)

    def stop_webcam(self):
        if self.webcam_timer.isActive():
            self.webcam_timer.stop()
        if self.cap and self.cap.isOpened():
            self.cap.release()
            self.cap = None
        self.btn_webcam_predict.setEnabled(True)
        self.btn_stop_webcam.setEnabled(False)
        self.image_label.setPixmap(QPixmap())  # clear frame

if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Splash screen using QLabel
    splash = QLabel()
    splash.setWindowFlags(Qt.SplashScreen | Qt.FramelessWindowHint)
    splash.setAttribute(Qt.WA_TranslucentBackground)

    movie = QMovie("roboti.gif")
    movie.setScaledSize(QtCore.QSize(600, 350))  
    splash.setMovie(movie)
    movie.start()

    splash.resize(movie.frameRect().size())
    splash.show()

    def start_main():
        global main_window
        main_window = MainWindow()
        main_window.show()

    QTimer.singleShot(3000, splash.close)
    QTimer.singleShot(3000, start_main)

    sys.exit(app.exec_())


