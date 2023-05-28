import tkinter as tk
import cv2
from PIL import Image, ImageTk

vid = cv2.VideoCapture(0)
vid.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
window = tk.Tk()
window.geometry("1200x720+350+100")
window.bind('<Escape>', lambda e: window.quit())
lbl = tk.Canvas(window, width=800, height=600)
lbl.pack()


def clicked():
    _, frame = vid.read()
    opencv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    captured_image = Image.fromarray(opencv_image)
    photo_image = ImageTk.PhotoImage(image=captured_image)
    lbl.photo_image = photo_image
    lbl.create_image(0, 0, image=photo_image, anchor=tk.NW)
    window.after(15, clicked)


btn = tk.Button(window, text="turn on the camera",
                fg="red", command=clicked)

btn.place(x=500, y=650)

# set Button grid

window.mainloop()
# cap = cv2.VideoCapture(0)
# while True:
#     ret,frame=cap.read()
#     key=cv2.waitKey(1)
#     if key==ord('q'):
#         break
# cap.release()
# cv2.destroyAllWindows()
