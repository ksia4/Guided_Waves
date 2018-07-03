import pygame

BLACK = (0, 0, 0)


class Text:
    def __init__(self, text, center_x, center_y, color=BLACK, size=40, style='Arial'):
        self.text = text
        self.color = color
        self.font = pygame.font.SysFont(style, size)
        self.center_x = center_x
        self.center_y = center_y
        self.size = size
        self.style = style

    def set_text(self, text):
        self.text = text

    def set_color(self, color):
        self.color = color

    def set_center(self, center_x, center_y):
        self.center_y = center_y
        self.center_x = center_x

    def set_font(self, size, style):
        self.size = size
        self.style = style
        self.font = pygame.font.SysFont(self.style, self.size)

    def render_text(self, screen):
        render_text = self.font.render(self.text, True, self.color)
        rect_render_text = render_text.get_rect()
        rect_render_text.center = (self.center_x, self.center_y)
        screen.blit(render_text, rect_render_text)
        return rect_render_text

