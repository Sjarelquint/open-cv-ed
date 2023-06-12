import cv2
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

# print(insightface.__version__)
app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0, det_size=(640, 640))
# img = ins_get_image('t1')
# # plt.imshow(img[:,:,::-1])
# # plt.show()
# faces = app.get(img)
swapper = insightface.model_zoo.get_model(
    'C:/Users/azpow/.conda/envs/swapface/Lib/site-packages/insightface/model_zoo\inswapper_128.onnx',
    download=False,
    download_zip=False)
# source_face = faces[0]
# # bbox=source_face['bbox']
# # bbox=[int(b) for b in bbox]
# # plt.imshow(img[bbox[1]:bbox[3],bbox[0]:bbox[2],::-1])
# # plt.show()
# res = img.copy()
# for face in faces:
#     res = swapper.get(res, face, source_face, paste_back=True)
# plt.imshow(res[:, :, ::-1])
# plt.show()

# two images in two photos

# img1 = cv2.imread('sinin.jpg')
# img2 = cv2.imread('bred-pitt9.jpg')
# fig, ax = plt.subplots(1, 2, figsize=(15, 10))
# ax[0].imshow(img1[:, :, ::-1])
# ax[0].axis('off')
# ax[1].imshow(img2[:, :, ::-1])
# ax[1].axis('off')
# plt.show()
# face1 = app.get(img1)[0]
# face2 = app.get(img2)[0]
# img1 = img1.copy()
# img2 = img2.copy()
# img1 = swapper.get(img1, face1, face2, paste_back=True)
# img2 = swapper.get(img2, face2, face1, paste_back=True)
# cv2.imwrite('img1.jpg',img1)
# cv2.imwrite('img2.jpg',img2)
# fig, ax = plt.subplots(1, 2, figsize=(15, 10))
# ax[0].imshow(img1[:, :, ::-1])
# ax[0].axis('off')
# ax[1].imshow(img2[:, :, ::-1])
# ax[1].axis('off')
# plt.show()

# two images in one photo

img = cv2.imread('img10.jpg')
# plt.imshow(img[:, :, ::-1])
# plt.axis('off')
# plt.show()
face1 = app.get(img)[0]
face2 = app.get(img)[1]
img = img.copy()
img = swapper.get(img, face1, face2, paste_back=True)
img = swapper.get(img, face2, face1, paste_back=True)
plt.imshow(img[:, :, ::-1])
plt.axis('off')
plt.show()
