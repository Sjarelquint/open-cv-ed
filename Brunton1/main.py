import matplotlib.pyplot as plt
from matplotlib.image import imread
import cv2
import numpy as np

plt.rcParams['figure.figsize'] = [16, 8]

X = cv2.imread('frame_saved.jpg',0)
plt.imshow(X,cmap='gray')
plt.axis('off')
plt.show()
U, S, VT = np.linalg.svd(X)
S = np.diag(S)
r=10
Xapprox = U[:, :r] @ S[0:r, :r] @ VT[:r, :]
plt.imshow(Xapprox,cmap='gray')
plt.axis('off')
plt.show()


# A = imread('frame_saved.jpg')
# X = np.mean(A, -1);  # Convert RGB to grayscale
#
# img = plt.imshow(X,cmap='gray')
#
# plt.axis('off')
# plt.show()
# U, S, VT = np.linalg.svd(X, full_matrices=False)
# print(S.shape)
# S = np.diag(S)
# print(S.shape)
# r = 5
# Xapprox = U[:, :r] @ S[0:r, :r] @ VT[:r, :]
# plt.imshow(Xapprox,cmap='gray')
# plt.axis('off')
# plt.show()
