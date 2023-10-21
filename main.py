import cv2
import matplotlib.pyplot as plt
from Thresholding import otsu
from Boundaries import boundaries_detection
from Borders import border_detection
from Color import space_color
from Boundaries import freeman_chain_code 
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

    #Pruebas del algoritmo de cadenas de Freeman
    # Carga la imagen binaria de bordes
    image_path = "Results/daltonismo.png"
    binary_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    

    chain_code = freeman_chain_code.calculate_chain_code(binary_image)
    chain1, first_difference1, menor_magnitud1 = freeman_chain_code.imprimir_cadena(chain_code.copy())
    result_image = freeman_chain_code.dibujar_borde(binary_image, chain_code)
    cv2.imshow("Image with Freeman Chain Code", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    #Se rota la imagen 90 grados para comparar
    rotated_image = cv2.rotate(binary_image, cv2.ROTATE_90_CLOCKWISE)
    chain_code = freeman_chain_code.calculate_chain_code(rotated_image)
    chain2, first_difference2, menor_magnitud2 = freeman_chain_code.imprimir_cadena(chain_code.copy())
    result_image = freeman_chain_code.dibujar_borde(rotated_image, chain_code)
    cv2.imshow("Image with Freeman Chain Code", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #Se comparan los resultados
    print("Comparación de resultados:")
    print("Menor magnitud:", menor_magnitud1 == menor_magnitud2)
