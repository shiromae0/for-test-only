import pygame
import sys
def get_path(name = ""):
    if name == "main_menu":
        return ""
    if name == "play_background":
        return ""
    if name == "conveyor belt":
        return ""
def ss():
    return 1
class main_menu():
    def __init__(self, pos, size,path = get_path("main_menu")):
        self.path = path
        self.pos = pos
        self.size = size
        self.image = pygame.image.load(self.path)
        self.image = pygame.transform.scale(self.image, self.size)
    def handle_events(self):
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN and self.current_scene == 0:
                if event.key == pygame.K_RETURN:
                    back = pygame.image.load(scene.getpath("play_ground"))
        #screen.blit(back,(0,0))
        screen.fill((0,0,0))
        pygame.display.flip()

class main_play():
    def __init__(self, pos, size, path="images/Background.jpg"):
        super().__init__(pos, size, path)
        self.path = path
        self.pos = pos
        self.size = size
        self.image = pygame.image.load(self.path)
        self.image = pygame.transform.scale(self.image, self.size)


