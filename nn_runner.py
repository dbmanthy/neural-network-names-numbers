import sys
from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QWidget, QHBoxLayout
from PyQt5.QtGui import QImage, QPainter, QPen, QColor
from PyQt5.QtCore import Qt, QPoint
from PIL import Image, ImageDraw
import numpy as np

class DrawingApp(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Digit Classifier with PyQt5")
        self.setGeometry(100, 100, 400, 300)

        # Create a layout
        self.layout = QVBoxLayout()

        # # Add a label for displaying results
        # self.result_label = QLabel("Draw a digit and click 'Check'", self)
        # self.layout.addWidget(self.result_label)

        # Add the canvas
        self.canvas = QLabel(self)
        self.layout.addWidget(self.canvas)

        # Add buttons
        self.button_layout = QHBoxLayout()

        self.check_button = QPushButton("Check", self)
        self.check_button.clicked.connect(self.check_digit)
        self.button_layout.addWidget(self.check_button)

        self.clear_button = QPushButton("Clear", self)
        self.clear_button.clicked.connect(self.clear_canvas)
        self.button_layout.addWidget(self.clear_button)

        self.layout.addLayout(self.button_layout)

        # Set the layout
        self.setLayout(self.layout)

        # Set up the drawing variables
        self.image = QImage(self.size(), QImage.Format_RGB32)
        self.image.fill(Qt.darkGray)  # Dark gray background
        self.drawing = False
        self.last_point = QPoint()

    def paintEvent(self, event):
        # This method will handle drawing on the widget
        canvas_painter = QPainter(self)
        canvas_painter.drawImage(self.rect(), self.image, self.image.rect())

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.last_point = event.pos()

    def mouseMoveEvent(self, event):
        if self.drawing:
            painter = QPainter(self.image)
            pen = QPen(Qt.white, 10, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)  # White drawing color
            painter.setPen(pen)
            painter.drawLine(self.last_point, event.pos())
            self.last_point = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = False

    def clear_canvas(self):
        # Clear the canvas
        self.image.fill(Qt.darkGray)  # Reset to dark gray background
        # self.result_label.setText("Draw a digit and click 'Check'")
        self.update()

    def check_digit(self):
        # Placeholder for digit recognition
        # Here we would integrate the neural network in the future

        # Convert the QImage to a PIL image and process it
        buffer = self.image.bits().asstring(self.image.byteCount())
        pil_image = Image.frombytes("RGB", self.image.size(), buffer)
        pil_image = pil_image.convert('L')  # Convert to grayscale

        # Resize for classification (28x28 for MNIST-like model)
        img_array = np.array(pil_image.resize((28, 28))) / 255.0

        # Placeholder output for now
        # self.result_label.setText("NN will classify this digit soon")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DrawingApp()
    window.show()
    sys.exit(app.exec_())
