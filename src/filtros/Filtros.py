import cv2
import matplotlib.pyplot as plt
import numpy as np

image=cv2.imread('deltoide2.jpg')
# Aplicar el filtro de desenfoque
blurred_image=cv2.blur(image,(50,50))

plt.title('blurred image')
plt.imshow(blurred_image)
plt.axis('off')
plt.show()

cv2.imwrite('blurred_image.jpg',blurred_image)

# Aplicar el filtro de desenfoque gaussiano

gaussian_blurred_image=cv2.GaussianBlur(image,(51,51),0)

plt.title('gaussian blurred image')
plt.imshow(gaussian_blurred_image)
plt.axis('off')
plt.show()

cv2.imwrite('gaussian_blurred_image.jpg',gaussian_blurred_image)

#filtro de nitidez

# Crear un kernel para nítidez
sharpen_kernel = np.array([[-1, -1, -1],
                           [-1,  9, -1],
                           [-1, -1, -1]])

# Aplicar el filtro de nítidez el segundo parametro de profundidad de la imagen
# de salida ddepth

sharpen_image = cv2.filter2D(image, 0, sharpen_kernel)

plt.title('sharpen image')
plt.imshow(sharpen_image)
plt.axis('off')
plt.show()


cv2.imwrite('sharpen_image.jpg',sharpen_image)

# Aplicar el filtro de detección de bordes (Canny)
edges = cv2.Canny(image, 100, 200)

plt.title('deteccion de bordes cany')
plt.imshow(edges)
plt.axis('off')
plt.show()
cv2.imwrite('edges.jpg',edges)

#Filtro de Emboss (Relieve)

emboss_kernel = np.array([[-2, -1, 0],
                          [-1,  1, 1],
                          [ 0,  1, 2]])

#aplicar filtro
emboss_image = cv2.filter2D(image, -1, emboss_kernel)

plt.title('emboss image')
plt.imshow(emboss_image)
plt.axis('off')
plt.show()

cv2.imwrite('emboss_image.jpg',emboss_image)

#Filtro de Escala de Grises

# Convertir la imagen a escala de grises

gray_image=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

plt.title('gray image')
plt.imshow(gray_image)
plt.axis('off')
plt.show()
cv2.imwrite('gray_image.jpg',gray_image)


##Filtro de Dilatación

_, binary_image = cv2.threshold(gray_image, 180, 255, cv2.THRESH_BINARY)

# Crear un kernel para la dilatación
kernel = np.ones((5, 5), np.uint8)

dilation = cv2.dilate(binary_image, kernel, iterations=1)

plt.title('dilated image')
plt.imshow(dilation)
plt.axis('off')
plt.show()

cv2.imwrite('dilate_image.jpg',dilation)