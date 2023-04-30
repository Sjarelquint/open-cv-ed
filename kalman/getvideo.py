import cv2

cap = cv2.VideoCapture(0)
w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('kalman.mp4', fourcc, 20.0, (int(w), int(h)))
while True:
    ret, frame = cap.read()
    if ret is False:
        break
    cv2.imshow('frame', frame)
    out.write(frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
out.release()