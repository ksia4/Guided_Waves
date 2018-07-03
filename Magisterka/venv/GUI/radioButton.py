import pygame as pg

GREEN = (0, 200, 0)
GREY = (100, 100, 100)

class RadioButton():
    def __init__(self, center_x=0,center_y=0):
        self.radio = 10
        self.state = False
        self.colorOn = GREEN
        self.colorOff = GREY
        self.actualColor = self.colorOff
        self.center_x = center_x
        self.center_y = center_y

    def set_center_x(self, new_center_x):
        self.center_x = new_center_x

    def set_center_y(self, new_center_y):
        self.center_y = new_center_y

    def set_center(self, new_center_x, new_center_y):
        self.set_center_x(new_center_x)
        self.set_center_y(new_center_y)

    def draw(self, screen):
        return pg.draw.circle(screen, self.actualColor, [self.center_x, self.center_y], self.radio)

    def changeState(self, state):
        self.state = state
        if state:
            self.actualColor = self.colorOn
        else:
            self.actualColor = self.colorOff



