import sys
from PyQt5.QtWidgets import QApplication, QPushButton, QVBoxLayout, QWidget, QHBoxLayout
from PyQt5.QtGui import QImage, QPainter, QPen, QColor
from PyQt5.QtCore import Qt, QPoint, QRect
from PIL import Image, ImageDraw
import numpy as np

class DrawingApp(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Digit Classifier with PyQt5")
        self.setGeometry(100, 100, 400, 300)

        # Create a layout
        self.layout = QVBoxLayout()

        # Add the canvas (light gray background)
        self.canvas = QWidget(self)
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
        self.image = QImage(400, 300, QImage.Format_RGB32)  # Fixed base image size
        self.image.fill(QColor(50, 50, 50))  # Dark gray background
        self.drawing_area_color = QColor(100, 100, 100)  # Light gray drawing area
        self.drawing = False
        self.last_point = QPoint()

        # Fill the canvas with a light gray area
        self.fill_drawing_area()

    def fill_drawing_area(self):
        painter = QPainter(self.image)
        painter.setBrush(self.drawing_area_color)
        painter.drawRect(20, 20, 360, 260)  # Drawing area coordinates
        self.update()

    def resizeEvent(self, event):
        self.update()

    def paintEvent(self, event):
        # This method will handle drawing on the widget
        canvas_painter = QPainter(self)
        scaled_image = self.image.scaled(self.size(), Qt.KeepAspectRatio)
        canvas_painter.drawImage(self.rect(), scaled_image)

    def scale_mouse_position(self, pos):
        """Scales the mouse position according to the current size of the widget."""
        widget_width, widget_height = self.size().width(), self.size().height()
        original_width, original_height = self.image.width(), self.image.height()

        # Calculate scaling factors
        scale_x = widget_width / original_width
        scale_y = widget_height / original_height

        # Apply scaling to the mouse position
        return QPoint(int(pos.x() / scale_x), int(pos.y() / scale_y))

    def mousePressEvent(self, event):
        scaled_pos = self.scale_mouse_position(event.pos())
        if event.button() == Qt.LeftButton and self.is_in_drawing_area(scaled_pos):
            self.drawing = True
            self.last_point = scaled_pos

    def mouseMoveEvent(self, event):
        scaled_pos = self.scale_mouse_position(event.pos())
        if self.drawing and self.is_in_drawing_area(scaled_pos):
            painter = QPainter(self.image)
            pen = QPen(QColor(230, 230, 230), 10, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)  # Black drawing color
            painter.setPen(pen)
            painter.drawLine(self.last_point, scaled_pos)
            self.last_point = scaled_pos
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = False

    def is_in_drawing_area(self, point):
        # Check if the point is within the light gray drawing area
        return 20 <= point.x() <= 380 and 20 <= point.y() <= 280

    def clear_canvas(self):
        # Clear the canvas
        self.image.fill(QColor(50, 50, 50))  # Reset to dark gray background
        self.fill_drawing_area()
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

        # Placeholder output for now (in the future, you would return the digit and confidence)
        print("NN will classify this digit soon")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DrawingApp()
    window.show()
    sys.exit(app.exec_())
