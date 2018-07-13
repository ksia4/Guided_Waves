import pygame, time
from GUI import button, text, menu_functions
from MES_dir import dispersion_curves
from Propagation import selectMode as sm, compensation_disp
from Animation import Anim_dyspersji as ad
import matplotlib.pyplot as plt


WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
RED = (220, 0, 0)
GREEN = (0, 200, 0)
BUTTON_BACKGROUND = (220, 220, 220)
HEAD = (0, 255, 0)
PURPLE = (147, 112, 219)
YELLOW = (255, 255, 0)
DARK_GREEN = (0, 150, 0)
GREY = (100, 100, 100)
BACKGROUND = (135, 206, 235)

class Game_Window(object):
    def __init__(self):
        pygame.init()
        pygame.display.set_caption('GW - GUI')

        self.width = 1200
        self.height = 800
        self.screen = pygame.display.set_mode((self.width, self.height), 0, 32) #inicjuje ekran na wymiary 800x800 32 bitowe kolory
        self.clock = pygame.time.Clock() #inicjuje zegar
        self.mode_number = 1 # 1 - licz nowe dane, 2 - ostatnio wyliczone dane 3 - dane z podanej ścieżki
        self.to_do = 0
        self.font = pygame.font.SysFont('Arial', 40) #font jest czcionką o rozmiarze 40 i Arial
        self.header_font = pygame.font.SysFont('Arial', 100)
        self.number_of_curves = 60
        self.path = '../eig'
        self.button = button.Button('tekst', self.width / 2, self.height / 2, BLACK, 40, 'Arial')
        self.text = text.Text('tekst', self.width/2, self.height/2, BLACK, 40, 'Arial')
        signal_array, time_x_freq = ad.get_chirp()
        self.signal_to_propagate = [time_x_freq[0], signal_array[3]]

    def run(self): #funkcja run jest głowną funkcją obsługującą GUI
        while True: # pętla nieskończona
            for event in pygame.event.get(): #z pord wszystkich eventow
                if event.type == pygame.QUIT: #jesli pojawi się zamknij
                    pygame.quit() #to zamknij pygame
                    exit() # i zamknij system
            if self.to_do == 0:
                print(self.to_do)
                self.start_menu()
            if self.to_do == 1:
                print(self.to_do)
                if self.mode_number == 1:
                    print("Będziemy liczyć nowe dane")
                    self.MES_menu()
                    self.to_do =2
                elif self.mode_number == 2:
                    print("Wczytamy ostatnio wygenerowane dane")
                    self.load_menu()
                elif self.mode_number == 3:
                    print("Wczytamy dane z podanej ścieżki")
                else:
                    print(self.mode_number)
                    print("Coś nie pykło... nie ma takiego numerka")
            if self.to_do == 2:
                print("todo2!")
                self.main_menu()
            if self.to_do == 3:
                print(self.to_do)
                self.gen_signal()
            if self.to_do == 4:
                print(self.to_do)
                self.sec_met()
            if self.to_do == 5:
                print(self.to_do)
                self.last_method()


    def load_menu(self):

        disp_text = self.header_font.render('Ładowanie danych', True, BLACK)
        self.screen.fill(BACKGROUND)

        rect_text = disp_text.get_rect()
        rect_text.center = (self.width/2, self.height/2)

        self.screen.blit(disp_text, rect_text)
        pygame.display.update()

        self.disp_curves = sm.SelectedMode('../eig/kvect', '../eig/omega')
        self.disp_curves.selectMode()

        self.screen.fill(BACKGROUND)
        disp_text = self.header_font.render('Krzywe dyspersji', True, BLACK)
        rect_text = disp_text.get_rect()
        rect_text.center = (self.width/2, self.height/2)
        self.screen.blit(disp_text, rect_text)
        pygame.display.update()

        self.disp_curves.plot_modes(100)
        self.to_do = 2

    def start_menu(self):
        font_start = pygame.font.SysFont("Arial", 100) #ustawiam czcionke na ekran startowy na 100
        start_surface = font_start.render('GUIDED WAVES!', True, BLACK) #renderuje napis czcionką startową w kolorze białym
        press_spacebar = self.font.render('Skąd chcesz wziąć dane?', True, BLACK)
        src_last = self.font.render("Dane z ostatnio wygenerowanych plików", True, BLACK)
        src_new = self.font.render('Wygeneruj nowe dane przy pomocy MES', True, BLACK)

        self.screen.fill(BACKGROUND)
        rect_surface = start_surface.get_rect()
        rect_spacebar = press_spacebar.get_rect()
        rect_last = src_last.get_rect()
        rect_new = src_new.get_rect()

        rect_surface.center = (self.width/2, 200)
        rect_spacebar.center = (self.width / 2, 300)
        rect_last.center = (self.width/2, 400)
        rect_new.center = (self.width/2, 550)

        rect_button_last = rect_last.inflate(15, 30)
        rect_button_new = rect_new.inflate(15, 30)

        pygame.draw.rect(self.screen, BUTTON_BACKGROUND, rect_button_last)
        pygame.draw.rect(self.screen, BUTTON_BACKGROUND, rect_button_new)

        self.screen.blit(src_new, rect_new)
        self.screen.blit(src_last, rect_last)
        self.screen.blit(start_surface, rect_surface)
        self.screen.blit(press_spacebar, rect_spacebar)

        wait = True
        while wait: # pętla nieskończona
            for event in pygame.event.get(): #z pord wszystkich eventow
                if event.type == pygame.QUIT: #jesli pojawi się zamknij
                    pygame.quit() #to zamknij pygame
                    exit() # i zamknij system
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if rect_button_new.collidepoint(event.pos): #Wygeneruj nowe mesowe dane
                        self.mode_number = 1
                        self.to_do = 1
                        wait = False
                        break

                    if rect_button_last.collidepoint(event.pos): #Czyli użyj ostatnio wygenerowanych danych
                        self.mode_number = 2
                        self.to_do = 1
                        wait = False
                        break

            pygame.display.update()
            pygame.time.Clock().tick(30)

    def MES_menu(self):
        menu_functions.set_parameters(self.screen, BACKGROUND, self.width, self.height, BUTTON_BACKGROUND)
        self.disp_curves = sm.SelectedMode('../eig/kvect', '../eig/omega')
        self.disp_curves.selectMode()

    def gen_signal(self):
        [dist, indexes] = menu_functions.gen_signal_param(self.screen, BACKGROUND, self.width, self.height, BUTTON_BACKGROUND)
        menu_functions.please_wait(self.screen, BACKGROUND, self.width, self.height)
        self.signal_after_propagation = compensation_disp.wave_length_propagation(self.signal_to_propagate, indexes, self.disp_curves, dist, True, 100)
        self.compensated_signal = compensation_disp.mapping_from_time_to_distance(self.signal_after_propagation, self.disp_curves, indexes)
        plt.plot(self.compensated_signal[0], self.compensated_signal[1])
        plt.title("Skompensowany sygnał")
        plt.xlabel("Odległość [m]")
        plt.ylabel("Amplituda")
        plt.show()
        self.to_do = 2

    def last_method(self):
        [dist, indexes] = menu_functions.gen_signal_to_propagate(self.screen, BACKGROUND, self.width, self.height, BUTTON_BACKGROUND)
        menu_functions.please_wait(self.screen, BACKGROUND, self.width, self.height)
        self.signal_to_compensate = compensation_disp.time_reverse_compensation(self.signal_to_propagate, dist, indexes, self.disp_curves)
        plt.plot(self.signal_to_compensate[0], self.signal_to_compensate[1])
        plt.title("Wygenerowany sygnał")
        plt.xlabel("Odległość [m]")
        plt.ylabel("Amplituda")
        plt.show()
        [dist2, indexes2] = menu_functions.gen_comp_signal_after_propagate(self.screen, BACKGROUND, self.width, self.height, BUTTON_BACKGROUND, indexes)
        menu_functions.please_wait(self.screen, BACKGROUND, self.width, self.height)
        self.signal_after_propagation = compensation_disp.wave_length_propagation(self.signal_to_compensate, indexes2, self.disp_curves, dist2, True, 10)
        plt.plot(self.signal_after_propagation[0], self.signal_after_propagation[1])
        plt.title("Przepropagowany sygnał")
        plt.xlabel("Odległość [m]")
        plt.ylabel("Amplituda")
        plt.show()

        self.to_do = 2

    def sec_met(self):
        [dist, indexes] = menu_functions.gen_signal_param(self.screen, BACKGROUND, self.width, self.height, BUTTON_BACKGROUND)
        menu_functions.please_wait(self.screen, BACKGROUND, self.width, self.height)
        self.signal_after_propagation = compensation_disp.wave_length_propagation(self.signal_to_propagate, indexes, self.disp_curves, dist, True, 100)
        self.compensated_signal = compensation_disp.linear_mapping_compensation(self.signal_after_propagation, indexes[0], self.disp_curves)

        plt.plot(self.compensated_signal[0], self.compensated_signal[1])
        plt.title("Skompensowany sygnał")
        plt.xlabel("czas[s]")
        plt.ylabel("Amplituda")
        plt.show()
        self.to_do = 2

    def main_menu(self):
        print("main menu")
        self.screen.fill(BACKGROUND)
        self.button.set_text('Wyświetl krzywe dyspersji')
        self.button.set_center_y(100)
        self.button.draw(self.screen, 800)
        rect_dysp = self.button.get_rect()
        print(rect_dysp)


        self.button.set_text('Wygeneruj nowe dane')
        self.button.set_center_y(200)
        self.button.draw(self.screen, 850)
        rect_new_dysp = self.button.get_rect()
        print(rect_new_dysp)

        self.button.set_text('Symulacja kompensacji metodą mapowania na dziedzinę odległości')
        self.button.size = 20
        self.button.set_center_y(300)
        self.button.draw(self.screen, 160)
        self.button.size = 40
        rect_excit = self.button.get_rect()
        print(rect_excit)

        self.button.set_text('Symulacja komensacji metodą mapowania liniowego')
        self.button.set_center_y(400)
        self.button.draw(self.screen, 300)
        rect_prop = self.button.get_rect()
        print(rect_prop)

        self.button.set_text('Symulacja sygnału kompensującego się na zadanej odległości')
        self.button.set_center_y(500)
        self.button.draw(self.screen, 350)
        rect_chirp = self.button.get_rect()
        print(rect_chirp)

        wait = True
        while wait: # pętla nieskończona
            for event in pygame.event.get(): #z pord wszystkich eventow
                if event.type == pygame.QUIT: #jesli pojawi się zamknij
                    pygame.quit() #to zamknij pygame
                    exit() # i zamknij system

                if event.type == pygame.MOUSEBUTTONDOWN:
                    if rect_dysp.collidepoint(event.pos):
                        self.disp_curves.plot_modes(100)

                    if rect_new_dysp.collidepoint(event.pos):
                        wait = False
                        self.to_do = 1
                        self.mode_number = 1

                    if rect_excit.collidepoint(event.pos):
                        wait = False
                        self.to_do = 3

                    if rect_prop.collidepoint(event.pos):
                        wait = False
                        self.to_do = 4

                    if rect_chirp.collidepoint(event.pos):
                        wait = False
                        self.to_do = 5

            pygame.display.update()




if __name__ == '__main__':
    app = Game_Window()
    app.run()
