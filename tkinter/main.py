import tkinter as tk
import cv2
from PIL import Image, ImageTk

import util


class App:
    def __init__(self):
        self.main_window = tk.Tk()
        self.main_window.geometry("1200x520+350+100")

        self.login_button_main_window = util.get_button(self.main_window, 'login', 'green', self.login)
        self.login_button_main_window.place(x=750, y=200)

        self.logout_button_main_window = util.get_button(self.main_window, 'logout', 'red', self.logout)
        self.logout_button_main_window.place(x=750, y=300)

        self.register_new_user_button_main_window = util.get_button(self.main_window, 'register new user', 'gray',
                                                                    self.register_new_user, fg='black')
        self.register_new_user_button_main_window.place(x=750, y=400)

        self.webcam_label = util.get_img_label(self.main_window)
        self.webcam_label.place(x=10, y=0, width=700, height=500)

        self.add_webcam(self.webcam_label)

    def add_webcam(self, label):
        if 'cap' not in self.__dict__:
            self.cap = cv2.VideoCapture(0)

        self._label = label
        self.process_webcam()

    def process_webcam(self):
        ret, frame = self.cap.read()

        self.most_recent_capture_arr = frame
        img_ = cv2.cvtColor(self.most_recent_capture_arr, cv2.COLOR_BGR2RGB)
        self.most_recent_capture_pil = Image.fromarray(img_)
        imgtk = ImageTk.PhotoImage(image=self.most_recent_capture_pil)
        self._label.imgtk = imgtk
        self._label.configure(image=imgtk)

        self._label.after(20, self.process_webcam)

    def login(self):
        pass

    def register_new_user(self):
        pass

    def logout(self):
        pass

    def start(self):
        self.main_window.mainloop()


if __name__ == "__main__":
    app = App()
    app.start()

# vid = cv2.VideoCapture(0)
# vid.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
# vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
# window = tk.Tk()
# window.geometry("1200x720+350+100")
# window.bind('<Escape>', lambda e: window.quit())
# lbl = tk.Canvas(window, width=800, height=600)
# lbl.pack()
#
#
# def clicked():
#     _, frame = vid.read()
#     opencv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
#     captured_image = Image.fromarray(opencv_image)
#     photo_image = ImageTk.PhotoImage(image=captured_image)
#     lbl.photo_image = photo_image
#     lbl.create_image(0, 0, image=photo_image, anchor=tk.NW)
#     window.after(15, clicked)
#
#
# btn = tk.Button(window, text="turn on the camera",
#                 fg="red", command=clicked)
#
# btn.place(x=500, y=650)
#
#
#
# window.mainloop()
# cap = cv2.VideoCapture(0)
# while True:
#     ret,frame=cap.read()
#     key=cv2.waitKey(1)
#     if key==ord('q'):
#         break
# cap.release()
# cv2.destroyAllWindows()
