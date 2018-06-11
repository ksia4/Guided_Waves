import numpy as np
import cv2
import pygame
from GUI import text
from MES_dir import config, MES, dispersion_curves

def display_image(image, title):

    img = cv2.imread(image, 1)
    cv2.imshow(title, img)
    while True:
        k = cv2.waitKey(100) # change the value from the original 0 (wait forever) to something appropriate
        if k == 27:
            print('ESC')
            cv2.destroyAllWindows()
            break
        if cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
            break
    cv2.destroyAllWindows()

def set_parameters(screen, BACKGROUND, screen_width, screen_height, BUTTONBACKGROUND):

    screen.fill(BACKGROUND)
    disp_text = text.Text('Obliczanie nowych krzywych dyspersji', screen_width/2, screen_height/2)
    disp_text.render_text(screen)

    # parametry preta
    length = 3
    radius = 10
    num_of_circles = 6
    num_of_points_at_c1 = 6

    # wektor liczby falowej
    config.kvect_min = 1e-10
    config.kvect_max = np.pi / 4
    config.kvect_no_of_points = 101

    # rysowanie wykresow
    config.show_plane = True
    config.show_bar = False
    config.show_elements = False

    pygame.display.update()

    MES.mes(length, radius, num_of_circles, num_of_points_at_c1)
    dispersion_curves.draw_dispercion_curves('../eig', True)

# display_image('dis_curves.png', 'Krzywe dyspersji')
