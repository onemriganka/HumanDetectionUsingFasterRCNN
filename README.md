# Human Detection & Counting with Faster R-CNN

This project provides a real-time human detection and counting solution using a pre-trained Faster R-CNN model in PyTorch. The application features a graphical user interface (GUI) built with Tkinter, allowing users to perform detection on images, videos, or a live camera feed.



---

## 🚀 Features

* **Real-time Detection**: Performs human detection in real-time on video streams and camera feeds.
* **Multiple Input Sources**: Supports detection from:
    * Image files
    * Video files
    * Live camera feed
* **Accurate Detection**: Utilizes a pre-trained Faster R-CNN model with a ResNet-50 backbone for high accuracy.
* **User-Friendly Interface**: A simple and intuitive GUI built with Tkinter for easy operation.
* **Bounding Box and Annotation**: Draws bounding boxes around detected humans, along with their center coordinates and a confidence score.
* **Person Counting**: Counts the number of detected individuals in the frame.

---

## ⚙️ How It Works

The application leverages the power of deep learning for object detection. Here's a breakdown of the core components:

* **Model**: A pre-trained **Faster R-CNN** model with a **ResNet-50** backbone from the `torchvision` library is used for detecting humans. This model has been trained on the COCO dataset, which includes a 'person' class.
* **GUI**: The graphical user interface is created using **Tkinter**, providing a straightforward way to interact with the detection functionalities.
* **Image Processing**: **OpenCV** is used for handling image and video processing tasks, such as reading frames from a video or camera, drawing annotations, and displaying the output.

---
