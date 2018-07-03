import numpy as np
import cv2
import pygame
from GUI import text
from MES_dir import config, MES, dispersion_curves
from GUI import radioButton as rb

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
    disp_text = text.Text('Obliczanie nowych krzywych dyspersji', screen_width/2, 100)

    mesh_type_text = text.Text('Rodzaj siatki', 150, 150, size=30)

    tetroidy_text = text.Text('Siatka czworościenna', 175, 200, size=20)
    heksoidy_text = text.Text('Siatka sześciościenna', 175, 230, size=20)


    disp_text.render_text(screen)
    mesh_type_text.render_text(screen)
    rect_tetroidy_text =  tetroidy_text.render_text(screen)
    rect_heksoidy_text = heksoidy_text.render_text(screen)


    tetroidy = rb.RadioButton(50, 200)
    tetroidy.changeState(True)
    heksoidy = rb.RadioButton(50, 230)

    rect_tetroidy = tetroidy.draw(screen)
    rect_heksoidy = heksoidy.draw(screen)
    while True: # pętla nieskończona
        for event in pygame.event.get(): #z pord wszystkich eventow
            if event.type == pygame.QUIT: #jesli pojawi się zamknij
                pygame.quit() #to zamknij pygame
                exit() # i zamknij system
            if event.type == pygame.MOUSEBUTTONDOWN:
                if rect_tetroidy_text.collidepoint(event.pos) or rect_tetroidy.collidepoint(event.pos): #Wygeneruj nowe mesowe dane
                    tetroidy.changeState(True)
                    heksoidy.changeState(False)
                    tetroidy.draw(screen)
                    heksoidy.draw(screen)
                elif rect_heksoidy_text.collidepoint(event.pos) or rect_heksoidy.collidepoint(event.pos):
                    tetroidy.changeState(False)
                    heksoidy.changeState(True)
                    tetroidy.draw(screen)
                    heksoidy.draw(screen)
        pygame.display.update()



    # # parametry preta
    # length = 3
    # radius = 10
    # num_of_circles = 6
    # num_of_points_at_c1 = 6
    #
    # # wektor liczby falowej
    # config.kvect_min = 1e-10
    # config.kvect_max = np.pi / 4
    # config.kvect_no_of_points = 101
    #
    # # rysowanie wykresow
    # config.show_plane = True
    # config.show_bar = False
    # config.show_elements = False
    #
    pygame.display.update()
    #
    # MES.mes(length, radius, num_of_circles, num_of_points_at_c1)
    # dispersion_curves.draw_dispercion_curves('../eig', True)

# display_image('dis_curves.png', 'Krzywe dyspersji')
