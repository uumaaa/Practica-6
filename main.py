import cv2
import matplotlib.pyplot as plt
from Thresholding import otsu
from Boundaries import boundaries_detection
from Borders import border_detection
from Color import space_color
if __name__ == "__main__":
    pacman = cv2.imread('images/pacman.png')
    daltonism = cv2.imread('images/daltonismo.png')
    #to rgb
    pacman = cv2.cvtColor(pacman, cv2.COLOR_BGR2RGB)
    daltonism = cv2.cvtColor(daltonism, cv2.COLOR_BGR2RGB)
    #grayscale
    pacman_gray = space_color.convert_gray_scale(pacman)
    daltonism_gray = space_color.convert_gray_scale(daltonism)
    #thresholding
    pacman_bi = otsu.otsu_image(pacman_gray)
    daltonism_bi = daltonism_gray
    #canny
    pacman_borders = border_detection.canny_bordering(pacman_bi)
    daltonism_borders = border_detection.canny_bordering(daltonism_bi)
    #moore
    pacman_boundary = boundaries_detection.moore_boundary_detection(pacman_borders)
    daltonism_boundary = boundaries_detection.moore_boundary_detection(daltonism_borders)
    plt.figure(1)
    #og images
    plt.subplot(2,4,1)
    plt.imshow(pacman)
    plt.subplot(2,4,5)
    plt.imshow(daltonism)
    #gs images
    plt.subplot(2,4,2)
    plt.imshow(pacman_gray,cmap="gray")
    plt.subplot(2,4,6)
    plt.imshow(daltonism_gray,cmap="gray")
    #borders
    plt.subplot(2,4,3)  
    plt.imshow(pacman_borders,cmap="gray")
    plt.subplot(2,4,7)  
    plt.imshow(daltonism_borders,cmap="gray")
    #moore
    plt.subplot(2,4,4)
    plt.imshow(pacman_boundary,cmap="nipy_spectral")
    plt.subplot(2,4,8)
    plt.imshow(daltonism_boundary,cmap="nipy_spectral")
    plt.show()

    #Se guardan las imagenes en la carpeta de resultados
    cv2.imwrite('Results/pacman.png',pacman_boundary)
    cv2.imwrite('Results/daltonismo.png',daltonism_boundary)
