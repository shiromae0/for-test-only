import sys
import pygame
import scene
class main_menu:
    def __init__(self):
        self.back = ""
        self.screen = pygame.display.set_mode((1280, 800))
    def start(self):
        self.screen.fill((0, 255, 255))
        pygame.display.flip()
    def handle_events(self):
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    self.screen.fill((0, 0, 0))
                    #self.back = pygame.image.load(scene.get_path("play_ground"))
                    break
        #screen.blit(self.back,(0,0))
        #self.screen.fill((0,0,0))
        pygame.display.flip()
class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((1280,800))
        pygame.display.set_caption("shapez")
        self.clock = pygame.time.Clock()
        self.running = True
        self.current_scene = 0
        #self.grid = Grid(5,9)
    def run(self):
        Main = main_menu()
        Main.start()
        while self.running:
            Main.handle_events()
            self.clock.tick(60)
        pygame.quit()

game  = Game()
game.run()