import pygame as pg

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREY = (50, 50, 50)

class InputBox:
    def __init__(self, value, x_center=0, y_center=0, width=70, style='Arial', size=40, textColor=BLACK, pipeColor=GREY, backgroundColor=WHITE):
        self.center_x = x_center
        self.center_y = y_center
        self.style = style
        self.size = size
        self.textColor = textColor
        self.pipeColor = pipeColor
        self.backgroundColor = backgroundColor
        self.set_font()
        self.width = width
        self.value = value
        self.text = str(self.value)
        self.table = []
        self.table.append(self.value)
        self.is_table = False


    def set_font(self):
        self.font = pg.font.SysFont(self.style, self.size)

    def drawBox(self, screen, plus_width=5, plus_height=5):
        render_text = self.font.render(self.text, True, self.textColor)
        rect_text = render_text.get_rect()
        rect_text.center = (self.center_x, self.center_y)
        rect_button_text = rect_text.inflate(plus_width, plus_height)
        self.rect = rect_button_text
        pg.draw.rect(screen, self.backgroundColor, rect_button_text)
        screen.blit(render_text, rect_text)

    def make_table(self):
        self.table = []
        tab = self.text.split(",")
        for num in tab:
            if len(num) > 0:
                self.table.append(int(num))
        print(self.table)

    def addNumber(self, cypher):
        if self.text == "0":
            self.text = cypher
        else:
            self.text = self.text + cypher
        if cypher == ".":
            self.value = float(self.text[0:(len(self.text)-1)])
        elif cypher == ",":
            self.is_table = True
        elif cypher == " ":
            self.text = self.text
        else:
            if self.is_table:
                self.make_table()
            else:
                self.value = float(self.text)
        print("Ustawiono wartość")
        print(self.value)

    def delNumber(self):
        self.text = self.text[0:(len(self.text)-1)]
        if len(self.text) == 0:
            self.text = "0"
        elif self.text[-1] == ".":
            self.value = float(self.text[0:len(self.text)-1])
        elif self.is_table:
            self.value = self.value
            self.make_table()
        else:
            self.value = float(self.text)

    def get_rect(self):
        return self.rect
