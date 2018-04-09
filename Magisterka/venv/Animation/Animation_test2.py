import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from multiprocessing import Process
import time # optional for testing only
import matplotlib.animation as animation

# A. First we define some useful tools:

def wait_fig():
    # Block the execution of the code until the figure is closed.
    # This works even with multiprocessing.
    if matplotlib.pyplot.isinteractive():
        matplotlib.pyplot.ioff() # this is necessary in mutliprocessing
        matplotlib.pyplot.show(block=True)
        matplotlib.pyplot.ion() # restitute the interractive state
    else:
        matplotlib.pyplot.show(block=True)

    return

def wait_anim(anim_flag, refresh_rate = 0.1):
    #This will be used in synergy with the animation class in the example
    #below, whenever the user want the figure to close automatically just
    #after the animation has ended.
    #Note: this function uses the controversial event_loop of Matplotlib, but
    #I see no other way to obtain the desired result.

    while anim_flag[0]: #next code extracted from plt.pause(...)
        backend = plt.rcParams['backend']
        if backend in plt._interactive_bk:
            figManager = plt._pylab_helpers.Gcf.get_active()
            if figManager is not None:
                figManager.canvas.start_event_loop(refresh_rate)

def draw_fig(fig = None):
    #Draw the artists of a figure immediately.
    #Note: if you are using this function inside a loop, it should be less time
    #consuming to set the interactive mode "on" using matplotlib.pyplot.ion()
    #before the loop, event if restituting the previous state after the loop.

    if matplotlib.pyplot.isinteractive():
        if fig is None:
            matplotlib.pyplot.draw()
        else:
            fig.canvas.draw()
    else:
        matplotlib.pyplot.ion()
        if fig is None:
            matplotlib.pyplot.draw()
        else:
            fig.canvas.draw()
        matplotlib.pyplot.ioff()  # restitute the interactive state

    matplotlib.pyplot.show(block=False)
    return

def pause_anim(t):  # This is taken from plt.pause(...), but without unnecessary

    # stuff. Note that the time module should be previously imported.
    # Again, this use the controversial event_loop of Matplotlib.
    backend = matplotlib.pyplot.rcParams['backend']
    if backend in matplotlib.pyplot._interactive_bk:
        figManager = matplotlib.pyplot._pylab_helpers.Gcf.get_active()
        if figManager is not None:
            figManager.canvas.start_event_loop(t)
            return
    else: time.sleep(t)

def f(x, y):
    return np.sin(x) + np.cos(y)

def plot_graph():
    fig = plt.figure()
    x = np.linspace(0, 2 * np.pi, 120)
    y = np.linspace(0, 2 * np.pi, 100).reshape(-1, 1)
    im = fig.gca().imshow(f(x, y))
    draw_fig(fig)
    n_frames = 50

        # ==============================================
        # First method - direct animation: This use the start_event_loop, so is
        # somewhat controversial according to the Matplotlib doc.
        # Uncomment and put the "Second method" below into comments to test.

    for i in range(n_frames): # n_frames iterations
        x += np.pi / 15.
        y += np.pi / 20.
        im.set_array(f(x, y))
        draw_fig(fig)  
        pause_anim(0.015) # plt.pause(0.015) can also be used, but is slower

    wait_fig() # simply suppress this command if you want the figure to close 
               # automatically just after the animation has ended     

    #================================================
    #Second method: this uses the Matplotlib prefered animation class.
    #Put the "first method" above in comments to test it.

    '''def updatefig(i, fig, im, x, y, anim_flag, n_frames):
        x = x + i * np.pi / 15.
        y = y + i * np.pi / 20.
        im.set_array(f(x, y))

        if i == n_frames - 1:
            anim_flag[0] = False

    anim_flag = [True]
    animation.FuncAnimation(fig, updatefig, repeat=False, frames=n_frames,
                            interval=50, fargs=(fig, im, x, y, anim_flag, n_frames), blit=False)
    # Unfortunately, blit=True seems to causes problems

    wait_fig()'''
    # wait_anim(anim_flag) #replace the previous command by this one if you want the
    # figure to close automatically just after the animation
    # has ended
    # ================================================
    return


# C. Using multiprocessing to obtain the desired effects. I believe this
# method also works with the "threading" module, but I haven't test that.

def main(): # it is important that ALL the code be typed inside
           # this function, otherwise the program will do weird
           # things with the Ipython or even the Python console.
           # Outside of this condition, type nothing but import
           # clauses and function/class definitions.
    if __name__ != '__main__': return
    p = Process(target=plot_graph)
    p.start()
    print('hello', flush = True) #just to have something printed here
    # p.join() # suppress this command if you want the animation be executed in
             # parallel with the subsequent code
    for i in range(3): # This allows to see if execution takes place after the
                       #process above, as should be the case because of p.join().
        print('world', flush = True)
        time.sleep(1)

main()