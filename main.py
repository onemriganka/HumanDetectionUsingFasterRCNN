# Real Time Human Detection & Counting

# imported necessary libraries
import cv2
import argparse
import tkinter as tk
import tkinter.messagebox as mbox
from tkinter import filedialog
from PIL import ImageTk, Image
from persondetection_pytorch import DetectorAPI
from fpdf import FPDF
import matplotlib.pyplot as plt


def argsParser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--video", default=None, help="path to Video File")
    parser.add_argument("-i", "--image", default=None, help="path to Image File")
    parser.add_argument("-c", "--camera", action='store_true', help="Use camera for detection")
    parser.add_argument("-o", "--output", type=str, help="path to optional output video file")
    return vars(parser.parse_args())


class HumanDetectorApp:
    def __init__(self):
        self.odapi = DetectorAPI()
        self.threshold = 0.7
        self.init_start_window()

    def init_start_window(self):
        self.start_win = tk.Tk()
        self.start_win.title("Real Time Human Detection & Counting")
        self.start_win.geometry('1000x700')
        tk.Label(self.start_win, text="REAL-TIME-HUMAN\nDETECTION & COUNTING",
                 font=("Arial", 50, "underline"), fg="magenta").place(x=70, y=10)
        tk.Button(self.start_win, text="▶ START", command=self.start_win.destroy,
                  font=("Arial", 25), bg="orange", fg="blue").place(x=130, y=570)
        tk.Button(self.start_win, text="❌ EXIT", command=self.exit_app,
                  font=("Arial", 25), bg="red", fg="blue").place(x=680, y=570)
        self.start_win.protocol("WM_DELETE_WINDOW", self.exit_app)
        self.start_win.mainloop()

        if not getattr(self, 'exit_flag', False):
            self.init_main_window()

    def exit_app(self):
        if mbox.askokcancel("Exit", "Do you want to exit?"):
            self.exit_flag = True
            self.start_win.destroy()

    def init_main_window(self):
        self.win = tk.Tk()
        self.win.title("Real Time Human Detection & Counting")
        self.win.geometry('1000x700')
        tk.Label(self.win, text="OPTIONS", font=("Arial", 50, "underline"), fg="brown").place(x=340, y=20)

        # Buttons
        tk.Button(self.win, text="DETECT FROM IMAGE ➡", command=self.image_option,
                  font=("Arial", 30), bg="light green", fg="blue").place(x=350, y=150)
        tk.Button(self.win, text="DETECT FROM VIDEO ➡", command=self.video_option,
                  font=("Arial", 30), bg="light blue", fg="blue").place(x=110, y=300)
        tk.Button(self.win, text="DETECT FROM CAMERA ➡", command=self.camera_option,
                  font=("Arial", 30), bg="light green", fg="blue").place(x=350, y=450)
        tk.Button(self.win, text="❌ EXIT", command=self.win.destroy,
                  font=("Arial", 25), bg="red", fg="blue").place(x=440, y=600)
        self.win.protocol("WM_DELETE_WINDOW", self.win.destroy)
        self.win.mainloop()

    def draw_annotations(self, img, box, person_idx, score):
        # Bounding box
        x1, y1, x2, y2 = box[1], box[0], box[3], box[2]
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        # Center point
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.circle(img, (cx, cy), 3, (0, 255, 0), -1)
        cv2.putText(img, f'({cx},{cy})', (cx + 5, cy - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        # Label
        cv2.putText(img, f'P{person_idx, round(score,2)}',
                    (x1 - 30, y1 - 8), cv2.FONT_HERSHEY_COMPLEX, 0.5,
                    (0, 0, 255), 1)

    def process_and_display(self, img):
        boxes, scores, classes, _ = self.odapi.processFrame(img)
        person_count = 0
        acc_sum = 0
        for i, cls in enumerate(classes):
            if cls == 1 and scores[i] > self.threshold:
                person_count += 1
                self.draw_annotations(img, boxes[i], person_count, scores[i])
                acc_sum += scores[i]
        return img, person_count, acc_sum

    def image_option(self):
        win = tk.Tk()
        win.title("Human Detection from Image")
        win.geometry('1000x700')
        path_var = tk.StringVar()

        def select(): path_var.set(filedialog.askopenfilename(parent=win))
        def preview():
            img = cv2.imread(path_var.get())
            cv2.imshow("Preview", img)

        def detect():
            img = cv2.imread(path_var.get())
            annotated, count, acc = self.process_and_display(img)
            cv2.imshow("Detection", annotated)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        tk.Label(win, text="Selected Image", font=("Arial", 30), fg="green").place(x=80, y=200)
        tk.Entry(win, textvariable=path_var, font=("Arial", 20), width=30).place(x=80, y=260)
        tk.Button(win, text="SELECT", command=select).place(x=220, y=350)
        tk.Button(win, text="PREVIEW", command=preview).place(x=410, y=350)
        tk.Button(win, text="DETECT", command=detect).place(x=620, y=350)
        win.mainloop()

    def video_option(self):
        win = tk.Tk()
        win.title("Human Detection from Video")
        win.geometry('1000x700')
        path_var = tk.StringVar()

        def select(): path_var.set(filedialog.askopenfilename(parent=win))
        def detect():
            cap = cv2.VideoCapture(path_var.get())
            args = argsParser()
            writer = None
            if args.get('output'):
                writer = cv2.VideoWriter(args['output'], cv2.VideoWriter_fourcc(*'MJPG'), 10,
                                         (int(cap.get(3)), int(cap.get(4))))
            while True:
                ret, frame = cap.read()
                if not ret: break
                annotated, count, acc = self.process_and_display(frame)
                if writer: writer.write(annotated)
                cv2.imshow("Video Detection", annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'): break
            cap.release()
            if writer: writer.release()
            cv2.destroyAllWindows()

        tk.Label(win, text="Selected Video", font=("Arial", 30), fg="green").place(x=80, y=200)
        tk.Entry(win, textvariable=path_var, font=("Arial", 20), width=30).place(x=80, y=260)
        tk.Button(win, text="SELECT", command=select).place(x=220, y=350)
        tk.Button(win, text="DETECT", command=detect).place(x=620, y=350)
        win.mainloop()

    def camera_option(self):
        cap = cv2.VideoCapture(0)
        args = argsParser()
        writer = None
        if args.get('output'):
            writer = cv2.VideoWriter(args['output'], cv2.VideoWriter_fourcc(*'MJPG'), 10,
                                     (int(cap.get(3)), int(cap.get(4))))
        while True:
            ret, frame = cap.read()
            if not ret: break
            annotated, count, acc = self.process_and_display(frame)
            if writer: writer.write(annotated)
            cv2.imshow("Camera Detection", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
        cap.release()
        if writer: writer.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    HumanDetectorApp()
