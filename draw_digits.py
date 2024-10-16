import sys
import pickle
from PyQt5.QtWidgets import QApplication, QPushButton, QVBoxLayout, QWidget, QHBoxLayout, QSizePolicy
from PyQt5.QtGui import QImage, QPainter, QPen, QColor
from PyQt5.QtCore import Qt, QPoint
from PIL import Image
import numpy as np

from nn.dense import Dense
from nn.activations import Tanh
from nn.network import train, predict

class DrawingApp(QWidget):
    def __init__(self, model):
        super().__init__()
        self.model = model

        self.setWindowTitle("Digit Classifier with PyQt5")
        self.canvas_width, self.canvas_height = 280, 280  # You can change this to other multiples of 28
        self.setGeometry(100, 100, self.canvas_width + 100, self.canvas_height + 100)

        # Create a layout
        self.layout = QVBoxLayout()

        # Add the canvas (light gray background)
        self.canvas = QWidget(self)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.layout.addWidget(self.canvas)

        # Add buttons
        self.button_layout = QHBoxLayout()

        self.check_button = QPushButton("Check", self)
        self.check_button.clicked.connect(self.check_digit)
        self.button_layout.addWidget(self.check_button)

        self.clear_button = QPushButton("Clear", self)
        self.clear_button.clicked.connect(self.clear_canvas)
        self.button_layout.addWidget(self.clear_button)

        # Add buttons at the bottom
        self.layout.addLayout(self.button_layout)

        # Set the layout
        self.setLayout(self.layout)

        # Set up the drawing variables
        self.image = QImage(self.canvas_width, self.canvas_height, QImage.Format_RGB32)  # Canvas size is flexible
        self.image.fill(QColor(0, 0, 0))  # Pure black background
        self.drawing_area_color = QColor(0, 0, 0)
        self.drawing = False
        self.last_point = QPoint()

        # Initial canvas color fill
        self.clear_canvas()

    def resizeEvent(self, event):
        self.update()

    def paintEvent(self, event):
        # This method will handle drawing on the widget
        canvas_painter = QPainter(self)
        canvas_painter.drawImage(self.rect().adjusted(20, 20, -20, -20), self.image)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.last_point = self.map_to_canvas(event.pos())

    def mouseMoveEvent(self, event):
        if self.drawing:
            painter = QPainter(self.image)
            # Set brush size according to the canvas size (increase the brush size slightly)
            pen_size = max(self.canvas_width // 14, 2)  # Slightly larger brush
            pen = QPen(QColor(255, 255, 255), pen_size, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
            painter.setPen(pen)
            painter.drawLine(self.last_point, self.map_to_canvas(event.pos()))
            self.last_point = self.map_to_canvas(event.pos())
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = False

    def clear_canvas(self):
        # Clear the canvas
        self.image.fill(QColor(0, 0, 0))  # Reset to pure black background
        self.update()

    def check_digit(self):
        # Convert the QImage to a PIL image and process it
        buffer = self.image.bits().asstring(self.image.byteCount())
        pil_image = Image.frombytes("RGBA", (self.image.width(), self.image.height()), buffer)

        # Convert to grayscale
        pil_image = pil_image.convert('L')

        # Resize for classification (28x28 for MNIST-like model)
        img_array = np.array(pil_image.resize((28, 28))) / 255.0

         # Flatten the image array to pass into the model (from 28x28 to 784)
        img_array = img_array.flatten().reshape(784, 1)  # Flatten and reshape to (784, 1)


        # Placeholder: here you'd pass img_array to the neural network for classification
        output = predict(model, img_array)
        print('pred:', np.argmax(output))

        # For now, save the processed image as a reference
        pil_image_resized = Image.fromarray((img_array * 255).astype(np.uint8))
        pil_image_resized.save("processed_digit.png")

    def map_to_canvas(self, pos):
        """Map the mouse position to the canvas, considering margins."""
        widget_width = self.rect().width()
        widget_height = self.rect().height()
        canvas_x = int((pos.x() - 20) * self.canvas_width / (widget_width - 40))
        canvas_y = int((pos.y() - 20) * self.canvas_height / (widget_height - 40))
        return QPoint(canvas_x, canvas_y)

def load_model(filename='model.pkl'):
    # Load the saved weights and biases
    with open(filename, 'rb') as f:
        params = pickle.load(f)

    # Rebuild the network by inferring the architecture from the parameter shapes
    network = []
    for weights, biases in params:
        input_size = weights.shape[1]  # Infer input size from the weight matrix
        output_size = weights.shape[0]  # Infer output size from the weight matrix
        layer = Dense(input_size, output_size)

        # Assign loaded weights and biases to the layer
        layer.weights, layer.biases = weights, biases
        network.append(layer)

        # Add activation function (assuming each Dense layer is followed by Tanh)
        network.append(Tanh())

    print(f"Model loaded from {filename}")
    return network

if __name__ == "__main__":
    model = load_model()
    app = QApplication(sys.argv)
    window = DrawingApp(model)
    window.show()
    sys.exit(app.exec_())
