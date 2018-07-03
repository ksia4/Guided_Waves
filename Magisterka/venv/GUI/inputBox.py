import pygame as pg

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREY = (50, 50, 50)

class InputBox:
    def __init__(self, x_center=0, y_center=0, style='Arial', size=40, textColor=BLACK, pipeColor=GREY, backgroundColor=WHITE):
        self.text = ""
        self.center_x = x_center
        self.center_y = y_center
        self.style = style
        self.size = size
        self.textColot = textColor
        self.pipeColor = pipeColor
        self.backgroundColor = backgroundColor
        self.set_font()


    def set_font(self):
        self.font = pg.font.SysFont(self.style, self.size)
