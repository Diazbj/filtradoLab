import cv2
import matplotlib.pyplot as plt

img = cv2.imread('deltoide2.jpg')

#Convertir imagen bgr a rgb
img_rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)


#guardar imagen
cv2.imwrite('deltoideRGB.jpg',img)

#Imagen en escala de grises

img_gray=cv2.cvtColor(img_rgb,cv2.COLOR_BGR2GRAY)

cv2.imwrite('deltoideGray.jpg',img_gray)

#imagen en blanco y negro (binario)
_, img_binary=cv2.threshold(img_gray,180,255,cv2.THRESH_BINARY)

# imgGray=cv2.imread('deltoideGray.jpg')

cv2.imwrite('deltoideBinary.jpg',img_binary)

# Mostrar las imágenes utilizando matplotlib
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Escala de grises
axes[0].imshow(img_gray, cmap='gray')
axes[0].set_title("Escala de Grises")
axes[0].axis('off')

# Blanco y negro (binaria)
axes[1].imshow(img_binary, cmap='gray')
axes[1].set_title("Blanco y Negro")
axes[1].axis('off')

# Imagen RGB
axes[2].imshow(cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB))
axes[2].set_title("RGB")
axes[2].axis('off')

plt.show()


