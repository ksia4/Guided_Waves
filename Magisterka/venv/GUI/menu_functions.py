import numpy as np
import cv2
import pygame
from GUI import text
from MES_dir import config, MES, dispersion_curves
from GUI import radioButton as rb, inputBox as ib, button

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
    write_wavenumber_min = False
    write_wavenumber_max = False
    write_wavenumber_len = False
    write_bar_heksoid_radius_box = False
    write_bar_tetroid_num_of_points_at_c1_box = False
    write_bar_tetroid_radius_box = False
    write_bar_tetroid_num_of_circles_box = False
    write_bar_heksoid_numberOfPlanes_box = False
    write_bar_heksoid_firstCircle_box = False
    write_bar_heksoid_circles_box = False
    write_bar_heksoid_addNodes_box = False

    disp_text = text.Text('Obliczanie nowych krzywych dyspersji', screen_width/2, 70)

    mesh_type_text = text.Text('Rodzaj siatki', 150, 150, size=30)

    tetroidy_text = text.Text('Siatka czworościenna', 175, 200, size=20)
    heksoidy_text = text.Text('Siatka sześciościenna', 175, 230, size=20)

    wavenumber_text = text.Text('Parametry wektora liczb falowych', 600, 150, size=30)
    wavenumber_min_text = text.Text('min', 500, 200, size=20)
    wavenumber_max_text = text.Text('max', 500, 230, size=20)
    wavenumber_len_text = text.Text('ilość punktów', 500, 260, size=20)

    bar_text = text.Text('Parametry pręta', screen_width/2, 320, size=30)
    bar_tetroid_text = text.Text('Dla siatki czworościennej', screen_width/4, 350, size=25)
    bar_heksoid_text = text.Text('Dla siatki sześciościennej', 3*screen_width/4, 350, size=25)
    bar_tetroid_radius_text = text.Text('Promień', screen_width/8, 380, size=20)
    bar_tetroid_num_of_circles_text = text.Text('Liczba okręgów', screen_width/8, 410, size=20)
    bar_tetroid_num_of_points_at_c1_text = text.Text('Liczba punktów w 1.', screen_width/8, 440, size=20)

    bar_heksoid_radius_text = text.Text('Promień', 5*screen_width/8, 380, size=20)
    bar_heksoid_numberOfPlanes_text = text.Text('Liczba płaszczyzn', 5*screen_width/8, 410, size=20)
    bar_heksoid_firstCircle_text = text.Text('Pierwszy okrąg', 5*screen_width/8, 440, size=20)
    bar_heksoid_addNodes_text = text.Text('Ilość dodatkowych węzłów', 5*screen_width/8, 470, size=20)
    bar_heksoid_circles_text = text.Text('okręgi', 5*screen_width/8, 500, size=20)

    show_plane_text = text.Text('Pokaż płaszczyznę', screen_width/4, 600, size=20)
    show_bar_text = text.Text('Pokaż pręt', 2*screen_width/4, 600, size=20)
    show_elements_text = text.Text('Pokaż elementy', 3*screen_width/4, 600, size=20)


    disp_text.render_text(screen)
    mesh_type_text.render_text(screen)
    wavenumber_text.render_text(screen)
    wavenumber_len_text.render_text(screen)
    wavenumber_max_text.render_text(screen)
    wavenumber_min_text.render_text(screen)
    bar_text.render_text(screen)
    bar_heksoid_text.render_text(screen)
    bar_tetroid_text.render_text(screen)
    bar_tetroid_num_of_circles_text.render_text(screen)
    bar_tetroid_num_of_points_at_c1_text.render_text(screen)
    bar_tetroid_radius_text.render_text(screen)
    bar_heksoid_addNodes_text.render_text(screen)
    bar_heksoid_circles_text.render_text(screen)
    bar_heksoid_firstCircle_text.render_text(screen)
    bar_heksoid_numberOfPlanes_text.render_text(screen)
    bar_heksoid_radius_text.render_text(screen)
    rect_tetroidy_text = tetroidy_text.render_text(screen)
    rect_heksoidy_text = heksoidy_text.render_text(screen)
    rect_show_plane_text = show_plane_text.render_text(screen)
    rect_show_bar_text = show_bar_text.render_text(screen)
    rect_show_elements_text = show_elements_text.render_text(screen)



    tetroidy = rb.RadioButton(50, 200)
    tetroidy.changeState(True)
    heksoidy = rb.RadioButton(50, 230)

    show_elements = rb.RadioButton(int(3*screen_width/4 - 85), 600)
    show_bar = rb.RadioButton(int(2*screen_width/4 - 60), 600)
    show_plane = rb. RadioButton(int(screen_width/4 - 100), 600)

    wavenumber_min = ib.InputBox(1e-10, 600, 200, 70, size=20)
    wavenumber_max = ib.InputBox(np.pi / 4, 600, 230, 70, size=20)
    wavenumber_len = ib.InputBox(51, 600, 260, 70, size=20)

    wavenumber_max.text = str(round(wavenumber_max.value, 5))

    bar_tetroid_radius_box = ib.InputBox(10, 3*screen_width/8, 380, size=20)
    bar_tetroid_num_of_circles_box = ib.InputBox(4, 3*screen_width/8, 410, size=20)
    bar_tetroid_num_of_points_at_c1_box = ib.InputBox(4, 3*screen_width/8, 440, size=20)

    bar_heksoid_radius_box = ib.InputBox(10, 7*screen_width/8, 380, size=20)
    bar_heksoid_numberOfPlanes_box = ib.InputBox(3, 7*screen_width/8, 410, size=20)
    bar_heksoid_firstCircle_box = ib.InputBox(16, 7*screen_width/8, 440, size=20)
    bar_heksoid_addNodes_box = ib.InputBox(0, 7*screen_width/8, 470, size=20)
    bar_heksoid_circles_box = ib.InputBox(1, 7*screen_width/8, 500, size=20)

    rect_tetroidy = tetroidy.draw(screen)
    rect_heksoidy = heksoidy.draw(screen)

    rect_show_elements = show_elements.draw(screen)
    rect_show_bar = show_bar.draw(screen)
    rect_show_plane = show_plane.draw(screen)

    wavenumber_min.drawBox(screen, plus_width=20)
    wavenumber_max.drawBox(screen, plus_width=9)
    wavenumber_len.drawBox(screen, plus_width=47)

    bar_heksoid_addNodes_box.drawBox(screen, plus_width=55)
    bar_heksoid_circles_box.drawBox(screen, plus_width=55)
    bar_heksoid_firstCircle_box.drawBox(screen, plus_width=45)
    bar_heksoid_numberOfPlanes_box.drawBox(screen, plus_width=55)
    bar_tetroid_num_of_circles_box.drawBox(screen, plus_width=55)
    bar_tetroid_radius_box.drawBox(screen, plus_width=45)
    bar_tetroid_num_of_points_at_c1_box.drawBox(screen, plus_width=55)
    bar_heksoid_radius_box.drawBox(screen, plus_width=45)

    rect_wavenumber_max = wavenumber_max.get_rect()
    rect_wavenumber_min = wavenumber_min.get_rect()
    rect_wavenumber_len = wavenumber_len.get_rect()

    rect_bar_tetroid_radius_box = bar_tetroid_radius_box.get_rect()
    rect_bar_tetroid_num_of_circles_box = bar_tetroid_num_of_circles_box.get_rect()
    rect_bar_tetroid_num_of_points_at_c1_box = bar_tetroid_num_of_points_at_c1_box.get_rect()

    rect_bar_heksoid_radius_box = bar_heksoid_radius_box.get_rect()
    rect_bar_heksoid_numberOfPlanes_box = bar_heksoid_numberOfPlanes_box.get_rect()
    rect_bar_heksoid_firstCircle_box = bar_heksoid_firstCircle_box.get_rect()
    rect_bar_heksoid_addNodes_box = bar_heksoid_addNodes_box.get_rect()
    rect_bar_heksoid_circles_box = bar_heksoid_circles_box.get_rect()

    calculate = button.Button("Oblicz!", screen_width/2, 650)
    calculate.draw(screen)
    rect_calculate = calculate.get_rect()
    end_while = False
    while True: # pętla nieskończona
        for event in pygame.event.get(): #z pord wszystkich eventow
            if event.type == pygame.QUIT: #jesli pojawi się zamknij
                pygame.quit() #to zamknij pygame
                exit() # i zamknij system
            if event.type == pygame.MOUSEBUTTONDOWN:
                if rect_tetroidy_text.collidepoint(event.pos) or rect_tetroidy.collidepoint(event.pos):
                    tetroidy.changeState(True)
                    heksoidy.changeState(False)
                    tetroidy.draw(screen)
                    heksoidy.draw(screen)
                elif rect_heksoidy_text.collidepoint(event.pos) or rect_heksoidy.collidepoint(event.pos):
                    tetroidy.changeState(False)
                    heksoidy.changeState(True)
                    tetroidy.draw(screen)
                    heksoidy.draw(screen)

                if rect_show_elements.collidepoint(event.pos) or rect_show_elements_text.collidepoint(event.pos):
                    show_elements.changeState(not show_elements.state)
                    show_elements.draw(screen)
                elif rect_show_bar.collidepoint(event.pos) or rect_show_bar_text.collidepoint(event.pos):
                    show_bar.changeState(not show_bar.state)
                    show_bar.draw(screen)
                elif rect_show_plane.collidepoint(event.pos) or rect_show_plane_text.collidepoint(event.pos):
                    show_plane.changeState(not show_plane.state)
                    show_plane.draw(screen)
                if rect_wavenumber_min.collidepoint(event.pos):
                    write_wavenumber_min = True
                    write_wavenumber_max = False
                    write_wavenumber_len = False
                    write_bar_tetroid_radius_box = False
                    write_bar_tetroid_num_of_circles_box = False
                    write_bar_tetroid_num_of_points_at_c1_box = False

                    write_bar_heksoid_radius_box = False
                    write_bar_heksoid_numberOfPlanes_box = False
                    write_bar_heksoid_firstCircle_box = False
                    write_bar_heksoid_addNodes_box = False
                    write_bar_heksoid_circles_box = False

                if rect_wavenumber_max.collidepoint(event.pos):
                    write_wavenumber_min = False
                    write_wavenumber_max = True
                    write_wavenumber_len = False
                    write_bar_tetroid_radius_box = False
                    write_bar_tetroid_num_of_circles_box = False
                    write_bar_tetroid_num_of_points_at_c1_box = False

                    write_bar_heksoid_radius_box = False
                    write_bar_heksoid_numberOfPlanes_box = False
                    write_bar_heksoid_firstCircle_box = False
                    write_bar_heksoid_addNodes_box = False
                    write_bar_heksoid_circles_box = False

                if rect_wavenumber_len.collidepoint(event.pos):
                    write_wavenumber_min = False
                    write_wavenumber_max = False
                    write_wavenumber_len = True
                    write_bar_tetroid_radius_box = False
                    write_bar_tetroid_num_of_circles_box = False
                    write_bar_tetroid_num_of_points_at_c1_box = False

                    write_bar_heksoid_radius_box = False
                    write_bar_heksoid_numberOfPlanes_box = False
                    write_bar_heksoid_firstCircle_box = False
                    write_bar_heksoid_addNodes_box = False
                    write_bar_heksoid_circles_box = False

                if rect_bar_tetroid_radius_box.collidepoint(event.pos):
                    write_wavenumber_min = False
                    write_wavenumber_max = False
                    write_wavenumber_len = False
                    write_bar_tetroid_radius_box = True
                    write_bar_tetroid_num_of_circles_box = False
                    write_bar_tetroid_num_of_points_at_c1_box = False

                    write_bar_heksoid_radius_box = False
                    write_bar_heksoid_numberOfPlanes_box = False
                    write_bar_heksoid_firstCircle_box = False
                    write_bar_heksoid_addNodes_box = False
                    write_bar_heksoid_circles_box = False

                if rect_bar_tetroid_num_of_circles_box.collidepoint(event.pos):
                    write_wavenumber_min = False
                    write_wavenumber_max = False
                    write_wavenumber_len = False
                    write_bar_tetroid_radius_box = False
                    write_bar_tetroid_num_of_circles_box = True
                    write_bar_tetroid_num_of_points_at_c1_box = False

                    write_bar_heksoid_radius_box = False
                    write_bar_heksoid_numberOfPlanes_box = False
                    write_bar_heksoid_firstCircle_box = False
                    write_bar_heksoid_addNodes_box = False
                    write_bar_heksoid_circles_box = False

                if rect_bar_tetroid_num_of_points_at_c1_box.collidepoint(event.pos):
                    write_wavenumber_min = False
                    write_wavenumber_max = False
                    write_wavenumber_len = False
                    write_bar_tetroid_radius_box = False
                    write_bar_tetroid_num_of_circles_box = False
                    write_bar_tetroid_num_of_points_at_c1_box = True

                    write_bar_heksoid_radius_box = False
                    write_bar_heksoid_numberOfPlanes_box = False
                    write_bar_heksoid_firstCircle_box = False
                    write_bar_heksoid_addNodes_box = False
                    write_bar_heksoid_circles_box = False

                if rect_bar_heksoid_radius_box.collidepoint(event.pos):
                    write_wavenumber_min = False
                    write_wavenumber_max = False
                    write_wavenumber_len = False
                    write_bar_tetroid_radius_box = False
                    write_bar_tetroid_num_of_circles_box = False
                    write_bar_tetroid_num_of_points_at_c1_box = False

                    write_bar_heksoid_radius_box = True
                    write_bar_heksoid_numberOfPlanes_box = False
                    write_bar_heksoid_firstCircle_box = False
                    write_bar_heksoid_addNodes_box = False
                    write_bar_heksoid_circles_box = False

                if rect_bar_heksoid_numberOfPlanes_box.collidepoint(event.pos):
                    write_wavenumber_min = False
                    write_wavenumber_max = False
                    write_wavenumber_len = False
                    write_bar_tetroid_radius_box = False
                    write_bar_tetroid_num_of_circles_box = False
                    write_bar_tetroid_num_of_points_at_c1_box = False

                    write_bar_heksoid_radius_box = False
                    write_bar_heksoid_numberOfPlanes_box = True
                    write_bar_heksoid_firstCircle_box = False
                    write_bar_heksoid_addNodes_box = False
                    write_bar_heksoid_circles_box = False

                if rect_bar_heksoid_firstCircle_box.collidepoint(event.pos):
                    write_wavenumber_min = False
                    write_wavenumber_max = False
                    write_wavenumber_len = False
                    write_bar_tetroid_radius_box = False
                    write_bar_tetroid_num_of_circles_box = False
                    write_bar_tetroid_num_of_points_at_c1_box = False

                    write_bar_heksoid_radius_box = False
                    write_bar_heksoid_numberOfPlanes_box = False
                    write_bar_heksoid_firstCircle_box = True
                    write_bar_heksoid_addNodes_box = False
                    write_bar_heksoid_circles_box = False

                if rect_bar_heksoid_addNodes_box.collidepoint(event.pos):
                    write_wavenumber_min = False
                    write_wavenumber_max = False
                    write_wavenumber_len = False
                    write_bar_tetroid_radius_box = False
                    write_bar_tetroid_num_of_circles_box = False
                    write_bar_tetroid_num_of_points_at_c1_box = False

                    write_bar_heksoid_radius_box = False
                    write_bar_heksoid_numberOfPlanes_box = False
                    write_bar_heksoid_firstCircle_box = False
                    write_bar_heksoid_addNodes_box = True
                    write_bar_heksoid_circles_box = False

                if rect_bar_heksoid_circles_box.collidepoint(event.pos):
                    write_wavenumber_min = False
                    write_wavenumber_max = False
                    write_wavenumber_len = False
                    write_bar_tetroid_radius_box = False
                    write_bar_tetroid_num_of_circles_box = False
                    write_bar_tetroid_num_of_points_at_c1_box = False

                    write_bar_heksoid_radius_box = False
                    write_bar_heksoid_numberOfPlanes_box = False
                    write_bar_heksoid_firstCircle_box = False
                    write_bar_heksoid_addNodes_box = False
                    write_bar_heksoid_circles_box = True

                if rect_calculate.collidepoint(event.pos):
                    end_while = True
                    break


            if event.type == pygame.KEYDOWN and pygame.key.name(event.key) != "backspace":
                if write_wavenumber_min:
                    wavenumber_min.addNumber(pygame.key.name(event.key))
                    wavenumber_min.drawBox(screen)
                elif write_wavenumber_max:
                    wavenumber_max.addNumber(pygame.key.name(event.key))
                    wavenumber_max.drawBox(screen)
                elif write_wavenumber_len:
                    wavenumber_len.addNumber(pygame.key.name(event.key))
                    wavenumber_len.drawBox(screen)
                elif write_bar_tetroid_radius_box:
                    bar_tetroid_radius_box.addNumber(pygame.key.name(event.key))
                    bar_tetroid_radius_box.drawBox(screen)
                elif write_bar_tetroid_num_of_circles_box:
                    bar_tetroid_num_of_circles_box.addNumber(pygame.key.name(event.key))
                    bar_tetroid_num_of_circles_box.drawBox(screen)
                elif write_bar_tetroid_num_of_points_at_c1_box:
                    bar_tetroid_num_of_points_at_c1_box.addNumber(pygame.key.name(event.key))
                    bar_tetroid_num_of_points_at_c1_box.drawBox(screen)
                elif write_bar_heksoid_radius_box:
                    bar_heksoid_radius_box.addNumber(pygame.key.name(event.key))
                    bar_heksoid_radius_box.drawBox(screen)
                elif write_bar_heksoid_numberOfPlanes_box:
                    bar_heksoid_numberOfPlanes_box.addNumber(pygame.key.name(event.key))
                    bar_heksoid_numberOfPlanes_box.drawBox(screen)
                elif write_bar_heksoid_firstCircle_box:
                    bar_heksoid_firstCircle_box.addNumber(pygame.key.name(event.key))
                    bar_heksoid_firstCircle_box.drawBox(screen)
                elif write_bar_heksoid_addNodes_box:
                    bar_heksoid_addNodes_box.addNumber(pygame.key.name(event.key))
                    bar_heksoid_addNodes_box.drawBox(screen)
                elif write_bar_heksoid_circles_box:
                    bar_heksoid_circles_box.addNumber(pygame.key.name(event.key))
                    bar_heksoid_circles_box.drawBox(screen)

            if event.type == pygame.KEYDOWN and pygame.key.name(event.key) == "backspace":
                if write_wavenumber_min:
                    wavenumber_min.delNumber()
                    wavenumber_min.drawBox(screen)
                elif write_wavenumber_max:
                    wavenumber_max.delNumber()
                    wavenumber_max.drawBox(screen)
                elif write_wavenumber_len:
                    wavenumber_len.delNumber()
                    wavenumber_len.drawBox(screen)
                elif write_bar_tetroid_radius_box:
                    bar_tetroid_radius_box.delNumber()
                    bar_tetroid_radius_box.drawBox(screen)
                elif write_bar_tetroid_num_of_circles_box:
                    bar_tetroid_num_of_circles_box.delNumber()
                    bar_tetroid_num_of_circles_box.drawBox(screen)
                elif write_bar_tetroid_num_of_points_at_c1_box:
                    bar_tetroid_num_of_points_at_c1_box.delNumber()
                    bar_tetroid_num_of_points_at_c1_box.drawBox(screen)
                elif write_bar_heksoid_radius_box:
                    bar_heksoid_radius_box.delNumber()
                    bar_heksoid_radius_box.drawBox(screen)
                elif write_bar_heksoid_numberOfPlanes_box:
                    bar_heksoid_numberOfPlanes_box.delNumber()
                    bar_heksoid_numberOfPlanes_box.drawBox(screen)
                elif write_bar_heksoid_firstCircle_box:
                    bar_heksoid_firstCircle_box.delNumber()
                    bar_heksoid_firstCircle_box.drawBox(screen)
                elif write_bar_heksoid_addNodes_box:
                    bar_heksoid_addNodes_box.delNumber()
                    bar_heksoid_addNodes_box.drawBox(screen)
                elif write_bar_heksoid_circles_box:
                    bar_heksoid_circles_box.delNumber()
                    bar_heksoid_circles_box.drawBox(screen)
        pygame.display.update()

        if end_while:
            break

    please_wait(screen, BACKGROUND, screen_width, screen_height)
    config.show_plane = show_plane.state
    config.show_bar = show_bar.state
    config.show_elements = show_elements.state
    if tetroidy.state:
        MES.mes4(bar_tetroid_radius_box.value, bar_tetroid_num_of_circles_box.value, bar_tetroid_num_of_points_at_c1_box.value)
    else:
        MES.mes8(bar_heksoid_numberOfPlanes_box.value, bar_heksoid_radius_box.value, bar_heksoid_circles_box.value, bar_heksoid_firstCircle_box.value, bar_heksoid_addNodes_box.value)


def please_wait(screen, BACKGROUND, screen_width, screen_height):
    screen.fill(BACKGROUND)
    please_wait = text.Text('Obliczenia trwają, proszę czekać', screen_width/2, screen_height/2)
    please_wait.render_text(screen)
    pygame.display.update()
