import threading, time
import sys
from MES_dir import dispersion


class Drawer(threading.Thread):

    def __init__(self, number_of_curves_to_display):
        threading.Thread.__init__(self)
        self.number_of_curves = number_of_curves_to_display
        print("Jestem wątkiem")

    def run(self):
        dispersion.draw_dispercion_curves_from_file('../eig', self.number_of_curves)
        time.sleep(1)

    def _stop(self):
        sys.exit()

# class Plot_Shower(threading.Thread):
#
#     def __init__(self, plot_to_show):
#         threading.Thread.__init__(self)
#         print("Jestem wątkiem do rysowania")
#         self.plot_to_show = plot_to_show
#
#     def run(self):
#         self.plot_to_show.show()
#         time.sleep(100)
