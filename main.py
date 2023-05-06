import cv2
import matplotlib.pyplot as plt
from skimage import measure
import numpy as np

# image = cv2.imread('cells2.jpg')
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# blurred = cv2.GaussianBlur(gray, (5, 5), 0)
# cells = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]

cells = cv2.imread('cells.png', 0)
# cells = cv2.imread(thresh, 0)

ret, thresh = cv2.threshold(cells, 20, 255, cv2.THRESH_BINARY_INV)


labels = measure.label(thresh, background=0)
bg_label = labels[0, 0]
labels[labels == bg_label] = 0    # Assign background label to 0

props = measure.regionprops(labels)

fig, ax = plt.subplots(1, 1)
plt.axis('off')
ax.imshow(cells, cmap='gray')
centroids = np.zeros(shape=(len(np.unique(labels)), 2))  # Access the coordinates of centroids
for i, prop in enumerate(props):
    my_centroid = prop.centroid
    centroids[i, :] = my_centroid
    ax.plot(my_centroid[1], my_centroid[0], 'r.')

# print(centroids)
# fig.savefig('out.png', bbox_inches='tight', pad_inches=0)
plt.show()