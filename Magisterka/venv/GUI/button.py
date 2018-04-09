import pygame
BLACK = (0, 0, 0)
BUTTON_BACKGROUND = (220, 220, 220)


class Button:
    def __init__(self, text, center_x, center_y, color=BLACK, size=40, style='Arial'):
        self.style = style
        self.size = size
        self.text = text
        self.center_x = center_x
        self.center_y = center_y
        self.color = color
        self.set_font()

    def set_center_x(self, new_center_x):
        self.center_x = new_center_x

    def set_center_y(self, new_center_y):
        self.center_y = new_center_y

    def set_center(self, new_center_x, new_center_y):
        self.set_center_x(new_center_x)
        self.set_center_y(new_center_y)

    def set_text(self, text):
        self.text = text

    def set_text_color(self, color):
        self.color = color
        self.set_font()

    def set_text_size(self, size):
        self.size = size
        self.set_font()

    def set_font(self):
        self.font = pygame.font.SysFont(self.style, self.size)

    def draw(self, screen, plus_width=15, plus_height=30, color=BUTTON_BACKGROUND):
        render_text = self.font.render(self.text, True, self.color)
        rect_text = render_text.get_rect()
        rect_text.center = (self.center_x, self.center_y)
        rect_button_text = rect_text.inflate(plus_width, plus_height)
        self.rect = rect_button_text
        pygame.draw.rect(screen, color, rect_button_text)
        screen.blit(render_text, rect_text)

    def get_rect(self):
        return self.rect
