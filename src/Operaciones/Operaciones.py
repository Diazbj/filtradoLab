import cv2
import matplotlib.pyplot as plt
from numpy.ma.core import resize

from src.basico.basico import img_gray

img = cv2.imread('deltoide2.jpg')

#convertir rgb
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#convertir escala de grises

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

resize_img = cv2.resize(gray_img,(90,90))

# #imagen redimensionada
# plt.imshow(resize_img)
# plt.axis('off')
# plt.show()

#guardar imagen
cv2.imwrite('resize_img.jpg', resize_img)

#obtener las dimensiones
(h,w)=img.shape[:2]

#centro de la imagen
center=(w//2,h//2)

#matriz de rotacion
M = cv2.getRotationMatrix2D(center,45,1)

#Rotar imagen
rotated_img = cv2.warpAffine(img,M,(w,h))

# #mostrar imagen
# plt.imshow(rotated_img)
# plt.axis('off')
# plt.show()

cv2.imwrite('rotated_img.jpg', rotated_img)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# imagen original
axes[0].imshow(img_rgb, cmap='gray')
axes[0].set_title("Deltoide")
axes[0].axis('off')

#imagen redimencionadas
axes[1].imshow(resize_img, cmap='gray')
axes[1].set_title("redimencion")
axes[1].axis('off')

#imagen rotada
axes[2].imshow(rotated_img, cmap='gray')
axes[2].set_title("rotacion")
axes[2].axis('off')

plt.show()


##Operaciones aritmeticas

img2 = cv2.imread('elipse2.jpg')

img_elipse = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

#suma

sum_img=cv2.add(img_elipse,img_rgb)
cv2.imwrite('sum_img.jpg', sum_img)

#resta
diff_img=cv2.subtract(img_elipse,img_rgb)
cv2.imwrite('diff_img.jpg', diff_img)

#Multiplicacion
mult_img=cv2.multiply(sum_img,diff_img)
cv2.imwrite('mult_img.jpg', mult_img)

#Division de imagenes
div_img=cv2.divide(sum_img,diff_img)
cv2.imwrite('div_img.jpg', div_img)


fig, axes = plt.subplots(1, 4, figsize=(15, 5))

axes[0].imshow(sum_img, cmap='gray')
axes[0].set_title("suma")
axes[0].axis('off')

axes[1].imshow(diff_img, cmap='gray')
axes[1].set_title("resta")
axes[1].axis('off')

axes[2].imshow(mult_img, cmap='gray')
axes[2].set_title("multiplicacion")
axes[2].axis('off')

axes[3].imshow(div_img, cmap='gray')
axes[3].set_title("division")
axes[3].axis('off')
plt.show()


## Operaciones Logicas

img1 = cv2.imread('deltoide2.jpg',cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('elipse2.jpg',cv2.IMREAD_GRAYSCALE)

plt.imshow(img1, cmap='gray')
plt.show()
plt.imshow(img2, cmap='gray')
plt.show()

_, img1_bin=cv2.threshold(img1,180,255,cv2.THRESH_BINARY)
_, img2_bin=cv2.threshold(img2,180,255,cv2.THRESH_BINARY)
plt.imshow(img1_bin, cmap='gray')
plt.show()
plt.imshow(img2_bin, cmap='gray')
plt.show()
#Operacion AND

and_img=cv2.bitwise_and(img1_bin,img2_bin)

#Operacion OR
OR_img=cv2.bitwise_or(img1_bin,img2_bin)

#Operacion NOT
not_img1=cv2.bitwise_not(img1_bin)
not_img2=cv2.bitwise_not(img2_bin)

cv2.imwrite('and_img.jpg', and_img)
cv2.imwrite('OR_img.jpg', OR_img)
cv2.imwrite('not_img1.jpg', not_img1)
cv2.imwrite('not_img2.jpg', not_img2)

# Mostrar las imágenes utilizando matplotlib
fig, axes = plt.subplots(2,2 , figsize=(15, 5))

axes[0][0].imshow(and_img, cmap='gray')
axes[0][0].set_title("AND")
axes[0][0].axis('off')

axes[0][1].imshow(OR_img, cmap='gray')
axes[0][1].set_title("OR")
axes[0][1].axis('off')

axes[1][0].imshow(not_img1, cmap='gray')
axes[1][0].set_title("NOT IMAGEN UNO")
axes[1][0].axis('off')

axes[1][1].imshow(not_img2, cmap='gray')
axes[1][1].set_title("NOT IMAGEN DOS")
axes[1][1].axis('off')

plt.show()

##Interpolacion

resized_bilinear =cv2.resize(img_rgb,(200,200),interpolation=cv2.INTER_LINEAR)
resized_nearest=cv2.resize(img_rgb,(200,200),interpolation=cv2.INTER_NEAREST)

cv2.imwrite('resized_bilinear.jpg', resized_bilinear)
cv2.imwrite('resized_nearest.jpg', resized_nearest)


#mostrar imagenes

fig, axes = plt.subplots(2, 1, figsize=(15, 5))

#escala de grises
axes[0].imshow(resized_bilinear, cmap='gray')
axes[0].set_title("Resized Bilinear")
axes[0].axis('off')

axes[1].imshow(resized_nearest, cmap='gray')
axes[1].set_title("Resized Nearest")
axes[1].axis('off')
plt.show()

