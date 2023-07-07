"""
Driving Game Module

This module contains the main class `Game` that represents a driving game. The game involves cars navigating through a
track, hitting checkpoints, and reaching the finish line. The module utilizes the Pygame library for graphics and
user input.

The `Game` class allows you to run the game, update frames, draw checkpoints, update cars, draw cars, draw obstacles,
draw the mouse position, and draw game information. The game can be run with or without AI control.

Usage:
    To use this module, create an instance of the `Game` class with optional settings for the game. Then, call the
    `run_game` method to start the game loop and run the driving game.

Classes:
    Game: Class for a driving game.

Dependencies:
    - math
    - struct
    - sys
    - concurrent.futures.ThreadPoolExecutor
    - multiprocessing
    - pathlib.Path
    - pygame
    - pygame.time.Clock
    - util
    - car.Car
    - config.Config, config.get_config

Attributes:
    All attributes of the `Game` class are listed in the class docstring.

Methods:
    All methods of the `Game` class are listed in the class docstring.

Note:
    This module requires the `pygame` library and the following asset files in the "assets" directory:
    - track-border.png: Image of the track border.
    - grass.jpg: Image of the grass background.
    - checkpoint.png: Image of the checkpoint.
    - finish.png: Image of the finish line.
"""

import math
import struct
import sys
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
from pathlib import Path

import pygame
from pygame.time import Clock

import util
from car import Car
from config import Config, get_config


class Game:
    """
    Class for a driving game.

    Args:
        staggered_start (bool): Whether to use a staggered start for cars.
        run_game (bool): Whether to run the game automatically after initialization.
        AI_enabled (bool): Whether AI is enabled.
        AI_training (bool): Wether AI is training.
        total_genomes (int): Total number of genomes.
        config (Config): Configuration object for the game.

    Attributes:
        logger: Logger object for logging game information.
        screen: Pygame screen object.
        cars (list): List of cars in the game.
        cars_alive (list): List of currently alive cars.
        cars_pending (list): List of pending cars waiting to be spawned.

    Methods:
        run_game(self) -> None:
            Run the game loop.
        update_frames(self, clock: Clock) -> None:
            Update frame information.
        draw_checkpoints(self) -> None:
            Draw the checkpoints on the screen.
        update_cars(self) -> None:
            Update the cars in the game.
        draw_cars(self, best_cars: List[Car]) -> None:
            Draw the cars on the screen.
        draw_mouse_pos(self) -> None:
            Draw the mouse position on the screen.
        draw_game_info(self, best_car: Car) -> None:
            Draw the game information on the screen.
        check_dead_cars(self) -> None:
            Check and handle dead cars.
    """
    def __init__(
            self,
            staggered_start: bool = False,
            run_game: bool = True,
            AI_enabled: bool = False,
            AI_training: bool = False,
            total_genomes: int = 0,
            config: Config = get_config(),
            hide_gui: bool = False,
    ):
        self.logger = util.get_logger(config)
        self.log_death_data = config.reporting.log_death_data

        self.hide_gui = hide_gui
        self.AI_enabled = AI_enabled
        self.AI_training = AI_training
        self.total_genomes = total_genomes
        self.threads = multiprocessing.cpu_count() - 1
        self.draw_car_limit = config.game.draw_car_limit
        self.car_spawned_limit = config.game.car_spawned_limit
        self.show_checkpoints = config.game.show_checkpoints
        self.show_all_checkpoints = config.game.show_all_checkpoints
        self.show_track_border = config.game.show_track_border
        self.show_map = config.game.show_map
        self.show_mouse_pos = config.game.show_mouse_pos
        self.time_limit = config.game.time_limit
        self.draw_ai_car_beams = config.ai.draw_ai_car_beams
        self.draw_ai_car_beam_intersections = config.ai.draw_ai_car_beam_intersections

        # Load generation number from file
        file_path = Path(util.get_path("generation.bin"))
        try:
            with file_path.open("rb") as f:
                self.generation = struct.unpack("i", f.read(4))[0]
        except Exception as error:
            self.logger.error(error)
            raise error

        # Initialize Pygame
        pygame.init()

        # Create track border mask
        track_border = pygame.image.load(util.get_path(["assets", "track-border.png"]))
        track_border_mask = pygame.mask.from_surface(track_border)
        track_border_mask_fx = pygame.mask.from_surface(pygame.transform.flip(track_border, True, False))
        track_border_mask_fy = pygame.mask.from_surface(pygame.transform.flip(track_border, False, True))
        track_border_mask_fxy = pygame.mask.from_surface(pygame.transform.flip(track_border, True, True))
        self.flipped_masks = [[track_border_mask, track_border_mask_fy], [track_border_mask_fx, track_border_mask_fxy]]

        # Create the game window
        self.screen = pygame.display.set_mode((900, 900))
        pygame.display.set_caption("My AI Racing Game")

        # Create cars
        cars = []
        if staggered_start:
            cars.append(Car(43, 180, self.flipped_masks, config))
            cars.append(Car(68, 190, self.flipped_masks, config))
            cars.append(Car(93, 200, self.flipped_masks, config))
            cars.append(Car(43, 220, self.flipped_masks, config))
            cars.append(Car(68, 230, self.flipped_masks, config))
            cars.append(Car(93, 240, self.flipped_masks, config))
        else:
            cars.append(Car(68, 190, self.flipped_masks, config))
        self.cars = cars
        self.cars_alive = []
        self.cars_pending = []

        if not self.AI_training:
            # Create track
            if config.game.show_track_border:
                self.track = pygame.image.load(util.get_path(["assets", "track-border.png"]))
                self.track = pygame.mask.from_surface(self.track).to_surface()
            if config.game.show_map:
                self.track = pygame.image.load(util.get_path(["assets", "track.png"]))

            # Create grass
            grass = pygame.image.load(util.get_path(["assets", "grass.jpg"]))
            self.grass = pygame.transform.scale(grass, (900, 900))

        # Create Checkpoints
        checkpoint_image = pygame.image.load(util.get_path(["assets", "checkpoint.png"]))
        self.checkpoint_image = pygame.transform.scale(checkpoint_image, (80, 20))
        self.highest_checkpoint = 0

        self.checkpoint_locations = util.get_checkpoint_locations()

        # Create finish line
        finish_line = pygame.image.load(util.get_path(["assets", "finish.png"]))
        self.finish_line = pygame.transform.scale(finish_line, (80, 20))
        self.finish_line_mask = pygame.mask.from_surface(self.finish_line)
        self.finish_position = (28, 140)

        self.frame_count: int = 0
        self.fps: float = 0
        self.seconds_frames: int = 0
        self.seconds_real_time: int = 0
        self.start_time: int = 0

        if run_game:
            self.run_game()

    def run_game(self) -> None:
        """
        Run the game loop.

        This method executes the main game loop. It updates frame information, checks car collision with checkpoints and
        the finish line, clears the screen, draws the track, checkpoints, and cars, updates and draws game information,
        removes dead cars and adds cars from the waiting list, checks if the game should continue running, and handles
        quitting the game.

        Returns:
            None
        """
        clock = pygame.time.Clock()
        self.start_time = pygame.time.get_ticks()

        # Load cars
        for car in self.cars:
            if len(self.cars_alive) < self.car_spawned_limit:
                self.cars_alive.append(car)
            else:
                self.cars_pending.append(car)

        # Main game loop
        running = True
        while running:
            # Enforce 60 fps
            clock.tick(60)

            # Update frame information
            self.update_frames(clock)

            if self.hide_gui:
                # Clear the console line
                print('\033[2K', end='')
                # Move the cursor to the beginning of the line
                print('\r', end='')
                # Print the loading percentage
                print(f"Cars:{len(self.cars_alive)} FPS:{self.fps}", end='')

            # Check car collision with checkpoint / finish line
            for car in self.cars_alive:
                car.check_checkpoints(self.checkpoint_image, self.checkpoint_locations)
                car.check_finish_line(self.finish_line_mask, self.finish_position)

            if not self.hide_gui:

                # Clear the screen
                self.screen.fill((0, 0, 0))

                # Draw grass
                if self.show_map and not self.AI_training:
                    self.screen.blit(self.grass, (0, 0))

                if (self.show_track_border or self.show_map) and not self.AI_training:
                    # Draw track
                    self.screen.blit(self.track, (0, 0))
                    # Draw finish line
                    self.screen.blit(self.finish_line, self.finish_position)
                    # Draw checkpoints
                    self.draw_checkpoints()

            # Update the cars !Threading!
            self.update_cars()

            # Get best cars
            best_cars = sorted(self.cars_alive, key=lambda x: x.score, reverse=True)
            best_car = best_cars[0]

            if not self.hide_gui:
                # Draw cars; pass sorted_best cars for filtering car limit
                self.draw_cars(best_cars)

            # Set highest checkpoint reached
            self.highest_checkpoint = best_car.checkpoints_hit

            if not self.hide_gui:
                # Draw mouse pos
                self.draw_mouse_pos()

                # Draw game info
                self.draw_game_info(best_car)

                # Update the display
                pygame.display.flip()

            # Remove dead cars and add cars from waiting list
            self.check_dead_cars()

            # Quit if no cars alive and AI_enabled
            running = self.cars_alive  # or not self.AI_enabled

            if not self.hide_gui:
                # Check if pygame quit
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        sys.exit(0)

        # Clean up
        pygame.quit()

    def update_frames(self, clock: Clock) -> None:
        """
        Update frame information.

        This method updates the frame count, calculates the frames per second (FPS), the seconds passed in frame time,
        and the seconds passed in real time.

        Args:
            clock (Clock): The Pygame Clock object.

        Returns:
            None
        """
        self.frame_count += 1
        self.fps = math.floor(clock.get_fps())
        self.seconds_frames = math.floor(self.frame_count / 60)
        elapsed_time = pygame.time.get_ticks() - self.start_time
        self.seconds_real_time = math.floor(elapsed_time / 1000)

    def draw_checkpoints(self) -> None:
        """
        Draw the checkpoints on the screen.

        This method draws the checkpoints on the screen based on the current settings. It determines which checkpoints
        to show and their positions based on the highest checkpoint reached. It also displays the checkpoint numbers.

        Returns:
            None
        """
        if self.show_all_checkpoints:
            first = 0
            last = len(self.checkpoint_locations)
        else:
            first = max(self.highest_checkpoint - 1, 0)
            last = self.highest_checkpoint + 1

        for i, (location, rotate) in enumerate(self.checkpoint_locations[first:last]):
            i += self.highest_checkpoint - 1 if self.highest_checkpoint and not self.show_all_checkpoints else 0
            rotated_checkpoint = \
                pygame.transform.rotate(self.checkpoint_image, rotate) if rotate else self.checkpoint_image
            self.screen.blit(rotated_checkpoint, location)
            # Draw checkpoint number
            font = pygame.font.Font(None, 14)
            text = font.render(f"{i+2:2}", True, (255, 255, 255))
            text_rect = text.get_rect()
            checkpoint_rect = rotated_checkpoint.get_rect()
            text_rect.center = (checkpoint_rect.centerx + location[0], checkpoint_rect.centery + location[1])
            self.screen.blit(text, text_rect)

    def update_cars(self) -> None:
        """
        Update the cars in the game.

        This method updates the cars in the game. If there is only one car, it updates the car sequentially.
        If there are multiple cars, it updates the cars using multiple threads for parallel processing.

        Returns:
            None
        """
        # Update car function
        def update_car(_car: Car):
            POIs = _car.update(self.screen) if self.draw_ai_car_beams else _car.update()
            if self.draw_ai_car_beam_intersections:
                for POI in POIs:
                    pygame.draw.circle(self.screen, (255, 0, 0), POI, radius=2)

        # Unthreaded update cars
        if len(self.cars_alive) <= 1:
            for car in self.cars_alive:
                update_car(car)

        # threaded updated cars
        else:
            with ThreadPoolExecutor(max_workers=self.threads) as executor:
                executor.map(update_car, self.cars_alive)

    def draw_cars(self, best_cars: list[Car]) -> None:
        """
        Draw the cars on the screen.

        This method draws the cars on the screen. It takes a list
        of best cars to highlight and limits the number of cars
        to be drawn based on the `draw_car_limit` setting.

        Args:
            best_cars (list[Car]): The list of best cars to highlight.

        Returns:
            None
        """
        if self.draw_car_limit:
            best_cars = best_cars[:self.draw_car_limit]
        for car in best_cars:
            car.draw(self.screen)

    def draw_mouse_pos(self) -> None:
        """
        Draw the mouse position on the screen.

        If the `show_mouse_pos` flag is True, this method gets the current mouse position
        using `pygame.mouse.get_pos()`, creates a font object, and renders the mouse position
        text with the font. The text is then displayed on the screen at an offset position
        relative to the mouse position.

        Returns:
            None
        """
        if self.show_mouse_pos:
            mouse_pos = pygame.mouse.get_pos()
            font = pygame.font.Font(None, 36)
            font = font.render(str(mouse_pos), True, (200, 200, 200))
            self.screen.blit(font, (mouse_pos[0]+15, mouse_pos[1]+15))

    def draw_game_info(self, best_car: Car) -> None:
        """
        Draw the game information on the screen.

        This method displays various game information on the screen, such as frames per second (FPS), frame time (FT),
        real time (RT), number of cars, score of the best car, generation and batch numbers, and total number of cars.
        The information is rendered using different font objects and displayed at specific positions on the screen.

        Args:
            best_car (Car): The best car object.

        Returns:
            None
        """
        font = pygame.font.Font(None, 24)
        if self.AI_enabled:
            s1 = f"FPS: {self.fps :2} | FT: {self.seconds_frames:3} | RT: {self.seconds_real_time}"
            s3 = f"Cars: {len(self.cars_alive)+len(self.cars_pending):2} | Score: {best_car.score:2}"
            s2 = f"Generation: {self.generation} | " \
                 f"Total Cars: {self.total_genomes}"
            text1 = font.render(s1, True, (255, 255, 153))
            text2 = font.render(s2, True, (255, 255, 153))
            text3 = font.render(s3, True, (255, 255, 153))
            text1_rect = text1.get_rect()
            text2_rect = text2.get_rect()
            text3_rect = text3.get_rect()
            text1_rect.midleft = (0, 12)
            text2_rect.center = (450, 12)
            text3_rect.midright = (900, 12)
            self.screen.blit(text1, text1_rect)
            self.screen.blit(text2, text2_rect)
            self.screen.blit(text3, text3_rect)
        else:
            s = f"Frame Time: {self.seconds_frames} Real Time: {self.seconds_real_time} Score: {best_car.score} " \
                f"Lap: {best_car.lap} Checkpoint: {best_car.checkpoints_hit} FPS: {self.fps}"
            text = font.render(s, True, (255, 255, 153))
            text_rect = text.get_rect()
            text_rect.center = (450, 12)
            self.screen.blit(text, text_rect)

    def check_dead_cars(self) -> None:
        """
        Check for dead cars and handle their removal.

        This method checks if any cars in the `cars_alive` list have exceeded the time limit or failed to reach a
        certain number of checkpoints. If a car is considered dead, it is marked as not alive, timed out, and removed
        from the `cars_alive` list. If there are pending cars, the next car in the queue is spawned and added to
        the `cars_alive` list. If `log_death_data` is True, the method logs the death information of the car.

        Returns:
            None
        """
        for car in self.cars_alive:
            time_alive = (self.frame_count - car.spawn_frame) / 60
            if time_alive >= self.time_limit or (time_alive >= 15 and car.total_checkpoints_hit <= 35):
                car.alive = False
                car.timed_out = True

            if not car.alive:
                self.cars_alive.remove(car)
                car.time_alive = time_alive
                if self.log_death_data:
                    self.logger.info(car.death_str())
                if self.cars_pending:
                    next_car = self.cars_pending.pop()
                    next_car.spawn_frame = self.frame_count
                    self.cars_alive.append(next_car)
