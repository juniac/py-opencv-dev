import sdl2
import sdl2.ext
import cv2

class Display(object):
    def __init__(self, width, height):
        sdl2.ext.init()
        self.width, self.height = width, height
        self.window = sdl2.ext.Window(
            "Py-OPENCV", size=(width, height), position=(0, 500))
        self.window.show()

    def paint(self, image):

        cv2.imshow('image', image)
        ''' 
        events = sdl2.ext.get_events()
        for event in events:
            if event.type == sdl2.SDL_QUIT:
                exit(0)

        # print(dir(window))
        surf = sdl2.ext.pixels2d(self.window.get_surface())
        # surf = sdl2.ext.pixels3d(self.window.get_surface())
        surf[:, :] = image.swapaxes(0, 1)
 '''
        self.window.refresh()
