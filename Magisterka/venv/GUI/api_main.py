import pygame, time
from GUI import button, text, input_box, menu_functions
from MES_dir import dispersion


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

    def run(self): #funkcja run jest głowną funkcją obsługującą GUI
        while True: # pętla nieskończona
            for event in pygame.event.get(): #z pord wszystkich eventow
                if event.type == pygame.QUIT: #jesli pojawi się zamknij
                    pygame.quit() #to zamknij pygame
                    exit() # i zamknij system
            if self.to_do == 0:
                self.start_menu()
            if self.to_do == 1:
                if self.mode_number == 1:
                    print("Będziemy liczyć nowe dane")
                    self.MES_menu()
                elif self.mode_number == 2:
                    print("Wczytamy ostatnio wygenerowane dane")
                    self.load_menu()
                elif self.mode_number == 3:
                    print("Wczytamy dane z podanej ścieżki")
                else:
                    print(self.mode_number)
                    print("Coś nie pykło... nie ma takiego numerka")
            if self.to_do == 2:
                self.main_menu()
            #print(self.mode_number)
            #exit(0)
            # self.mode()
            # self.choose_snake()
            # self.game()
            # self.game_over()

    def load_menu(self):

        disp_text = self.header_font.render('Ładowanie danych', True, BLACK)
        self.screen.fill(BACKGROUND)

        rect_text = disp_text.get_rect()
        rect_text.center = (self.width/2, self.height/2)

        self.screen.blit(disp_text, rect_text)
        wait = True
        first = True
        path = self.path
        while wait: # pętla nieskończona
            pygame.display.update()
            for event in pygame.event.get(): #z pord wszystkich eventow
                if event.type == pygame.QUIT: #jesli pojawi się zamknij
                    pygame.quit() #to zamknij pygame
                    exit() # i zamknij system

            if first:
                self.screen.fill(BACKGROUND)
                self.text.set_text('Wygenerowane z pliku krzywe dyspersji')
                self.text.set_center(self.width/2, 100)
                self.text.render_text(self.screen)
                pygame.display.update()
                dispersion.draw_dispercion_curves_from_file(path, self.number_of_curves, True)
                # first = False
                self.to_do = 2
                # wait = True
                break

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


    def main_menu(self):
        print("main menu")
        self.screen.fill(BACKGROUND)
        self.button.set_text('Wyświetl krzywe dyspersji')
        self.button.set_center_y(100)
        self.button.draw(self.screen, 229)
        rect_dysp = self.button.get_rect()
        print(rect_dysp)

        self.button.set_text('Wczytaj inne dane')
        self.button.set_center_y(200)
        self.button.draw(self.screen, 366)
        rect_next_dysp = self.button.get_rect()
        print(rect_next_dysp)


        self.button.set_text('Wygeneruj nowe dane')
        self.button.set_center_y(300)
        self.button.draw(self.screen, 299)
        rect_new_dysp = self.button.get_rect()
        print(rect_new_dysp)

        self.button.set_text('Wylicz krzywe wzbudzalności')
        self.button.set_center_y(400)
        self.button.draw(self.screen, 173)
        rect_excit = self.button.get_rect()
        print(rect_excit)

        self.button.set_text('Symulacja propagacji fali prowadzonej')
        self.button.set_center_y(500)
        self.button.draw(self.screen)
        rect_prop = self.button.get_rect()
        print(rect_prop)

        self.button.set_text('Symulacja propagacji chirpa')
        self.button.set_center_y(600)
        self.button.draw(self.screen, 190)
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
                        menu_functions.display_image('dis_curves.png', 'Krzywe dyspersji')

            pygame.display.update()




if __name__ == '__main__':
    app = Game_Window()
    app.run()
