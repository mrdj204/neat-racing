"""
This module defines a Car class for a game where cars need to navigate a course.

Classes:
    Car: Represents a car in the game. It has attributes such as position,
         speed, health points, and can perform various operations such as
         moving, detecting collisions, drawing itself, etc.

Dependencies:
    configparser: For reading configuration files.
    math: For mathematical operations.
    os: For file and directory operations.
    random: For generating random values.
    pygame: Library for creating video games.
    neat: For the neural network functionality.
    AI: For the AI functionality.
    models: For Beam class.

Attributes:
    All attributes of the `Game` class are listed in the class docstring.

Methods:
    All methods of the `Game` class are listed in the class docstring.

Note:
    This code is meant to be used within a larger framework, where the game logic is handled.
"""

import math
import os
import random
from typing import Union

import pygame
from neat.nn import FeedForwardNetwork
from pygame.mask import Mask
from pygame.surface import Surface

import util
from config import Config, get_config
from util import Beam


class Car:
    """
    Represents a car in the game.

    Attributes:
        config: A ConfigParser object containing the configurations for the car.
        spawn_frame: A frame counter keeping track of when the car was spawned.
        time_alive: Counter for how long the car has been alive.
        x, y: The x and y coordinates of the car on the screen.
        width, height: The dimensions of the car.
        hp: The car's hit points.
        alive: Boolean indicating whether the car is alive.
        one_hit_ko: Boolean indicating if the car should be destroyed in one hit.
        timed_out: Boolean indicating if the car is timed out.
        checkpoints_hit: Counter for how many checkpoints the car has hit.
        total_checkpoints_hit: Total checkpoints hit by the car.
        total_checkpoints: Total checkpoints present in the race.
        lap: Counter for how many laps the car has completed.
        score: The car's score.
        angle: The car's current angle of rotation.
        angle_offset: Offset angle used in calculations.
        acceleration: The car's acceleration.
        max_speed: The car's maximum speed.
        base_speed: The car's base speed.
        speed: The car's current speed.
        reverse_max_speed: The car's maximum reverse speed.
        deceleration: The car's deceleration.
        speed_list: List of speeds maintained for the car.
        image: The car's image.
        beam_surface: A pygame Surface object for the car's beams.
        net: The neural network used by the car if controlled by AI.
        flipped_masks: List of masks used for collision detection.
        genome_key: Key for the car's genome in the NEAT algorithm.

    Methods:
        __lt__(other):
            Compares two car objects based on their scores.
        is_clicked(mouse_pos):
            Checks if car is under mouse_pos
        update(screen: Union[Surface, None] = None) -> list[tuple[int, int]]:
            Updates the state of the car.
        draw(screen: Surface):
            Draws the car on the given surface.
        future_collision() -> Union[tuple[int, int], None]:
            Detects a future collision.
        check_checkpoints(checkpoint_image: Surface, locations: list[tuple[tuple[int, int], int]]):
            Checks if the car has passed through a checkpoint.
        check_finish_line(finish_line_mask: Mask, finish_position: tuple[int, int]):
            Checks if the car has crossed the finish line.
        get_distances_to_walls(screen: Union[Surface, None] = None) -> tuple[list[tuple[int, int]], list[float]]:
            Gets distances to walls using beams.
        get_beams() -> list[Beam]:
            Generates beams for distance measurement.
        draw_beam(beam: Beam, surface: Union[Surface, None] = None) -> tuple[tuple[int, int], float]:
            Draws a beam and returns collision information.
        update_score() -> int:
            Updates and returns the car's score.
        deal_damage(damage: float):
            Deal damage to the car and check if it's still alive.
        death_str() -> str:
            Returns the cars death string
    """

    def __init__(self, x: int, y: int, flipped_masks: list[list[Mask]], config: Config = get_config(),
                 color: str = None, net: FeedForwardNetwork = None, genome_key: int = None):
        """
        Initialize the Car object.

        Args:
            x (int): Initial x coordinate of the car.
            y (int): Initial y coordinate of the car.
            flipped_masks (list[list[Mask]]): List of flipped masks for collision detection.
            config (Config): Configuration parser object.
            color (str, optional): Color of the car. Defaults to None.
            net (FeedForwardNetwork, optional): Neural network for AI control. Defaults to None.
            genome_key (int, optional): The genome key associated with the car. Defaults to None.
        """

        self.spawn_frame = 0
        self.time_alive = 0

        self.x = x
        self.y = y
        self.width = 25 / 3 * 2
        self.height = 49 / 3 * 2

        self.keys = {}

        self.hp = 100
        self.alive = True
        self.one_hit_ko = True
        self.timed_out = False

        self.checkpoints_hit = 0
        self.total_checkpoints_hit = 0
        self.total_checkpoints = 0
        self.lap = -1
        self.score = 0

        self.angle = 0
        self.angle_offset = 90

        self.acceleration = .025
        self.max_speed = 3
        self.base_speed = 0  # config.ai.base_speed if net else 0
        self.speed = self.base_speed
        self.reverse_max_speed = self.max_speed / -2
        self.deceleration = .01
        self.speed_list = []

        if color:
            color = util.get_path(["assets", "cars", f"{color}.png"])
        if not color or not os.path.isfile(color):
            directory = "assets/cars/"
            files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)) and "pink" not in f]
            random_file = random.choice(files)
            color = util.get_path(["assets", "cars", random_file])
        image = pygame.image.load(color)
        self.image = pygame.transform.scale(image, (self.width, self.height))

        self.beam_surface = pygame.Surface((900, 900), pygame.SRCALPHA)

        self.net = net
        self.flipped_masks = flipped_masks
        self.genome_key = genome_key

    def debug(self):
        if self.keys:
            UP = self.keys[pygame.K_UP] or self.keys[pygame.K_w]
            SPACE = self.keys[pygame.K_SPACE]
            LEFT = self.keys[pygame.K_LEFT] or self.keys[pygame.K_a]
            RIGHT = self.keys[pygame.K_RIGHT] or self.keys[pygame.K_d]
            return f"{'UP' if UP else '':^2}", \
                   f"{'SPACE' if SPACE else '':^5}", \
                   f"{'LEFT' if LEFT else '':^4}", \
                   f"{'RIGHT' if RIGHT else '':^5}", \
                   float(f"{round(self.speed, 5)}"), \
                   f"{self.score}"
        return "", "", "", "", 0, ""

    def print_debug(self):
        UP = self.keys[pygame.K_UP] or self.keys[pygame.K_w]
        SPACE = self.keys[pygame.K_SPACE]
        LEFT = self.keys[pygame.K_LEFT] or self.keys[pygame.K_a]
        RIGHT = self.keys[pygame.K_RIGHT] or self.keys[pygame.K_d]

        output = f"{'UP' if UP else '':^2} | " \
                 f"{'SPACE' if SPACE else '':^5} | " \
                 f"{'LEFT' if LEFT else '':^4} | " \
                 f"{'RIGHT' if RIGHT else '':^5} | " \
                 f"{self.speed:.3f} | " \
                 f"{self.score}"
        print('\033[2K\r' + output, end='')

    def __lt__(self, other):
        """
        Compare two Car objects based on their score.

        Args:
            other: The other Car object to compare.

        Returns:
            bool: True if the score of self is less than the score of other, False otherwise.
        """
        return self.score < other.score

    def is_clicked(self, mouse_pos: tuple[int, int]) -> bool:
        """
        Check if the car is clicked by the mouse.

        Args:
            mouse_pos: A tuple containing the x and y coordinates of the mouse position.

        Returns:
            bool: A boolean value indicating whether the car is clicked by the mouse.
        """
        rect = pygame.transform.rotate(self.image, self.angle).get_rect()
        rect.center = (self.x, self.y)
        return rect.collidepoint(mouse_pos)

    def update(self, screen: Union[Surface, None] = None) -> list[tuple[int, int]]:
        """
        Update the car's position and state based on user input or AI decision.

        Args:
            screen (Union[Surface, None], optional): The pygame Surface object. Defaults to None.

        Returns:
            list[tuple[int, int]]: A list of points of intersections (POIs) for collision detection.
        """

        if self.net:
            # Get key press from AI
            POIs, vision = self.get_distances_to_walls(screen)
            keys = self.net.activate((*vision, self.speed))
        else:
            # Get key press from User
            POIs = None
            keys = pygame.key.get_pressed()
        self.keys = keys

        is_moving = False
        deceleration = self.acceleration if keys[pygame.K_SPACE] else self.deceleration
        if (keys[pygame.K_UP] or keys[pygame.K_w]) and not keys[pygame.K_SPACE]:
            is_moving = True
            self.speed += self.acceleration
            self.speed = min(self.speed, self.max_speed)
            if not self.future_collision():
                self.x += math.cos(math.radians(self.angle + self.angle_offset)) * self.speed
                self.y -= math.sin(math.radians(self.angle + self.angle_offset)) * self.speed
            else:
                self.speed = self.base_speed
        elif keys[pygame.K_DOWN] or keys[pygame.K_s] and not keys[pygame.K_SPACE]:
            is_moving = True
            self.speed -= self.acceleration
            self.speed = max(self.speed, self.reverse_max_speed)
            if not self.future_collision():
                self.x += math.cos(math.radians(self.angle + self.angle_offset)) * self.speed
                self.y -= math.sin(math.radians(self.angle + self.angle_offset)) * self.speed
            else:
                self.speed = self.base_speed
        else:
            if self.speed > self.base_speed or (self.net and self.speed >= self.base_speed):
                is_moving = True
                self.speed -= deceleration
                self.speed = max(self.speed, self.base_speed)
                if not self.future_collision():
                    self.x += math.cos(math.radians(self.angle + self.angle_offset)) * self.speed
                    self.y -= math.sin(math.radians(self.angle + self.angle_offset)) * self.speed
                else:
                    self.speed = self.base_speed
            elif self.speed < self.base_speed:
                is_moving = True
                self.speed += deceleration
                self.speed = min(self.speed, self.base_speed)
                if not self.future_collision():
                    self.x += math.cos(math.radians(self.angle + self.angle_offset)) * self.speed
                    self.y -= math.sin(math.radians(self.angle + self.angle_offset)) * self.speed
                else:
                    self.speed = self.base_speed

        self.speed_list.append(self.speed)

        if is_moving:
            if keys[pygame.K_LEFT] or keys[pygame.K_a]:
                self.angle += 1.5
            if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
                self.angle -= 1.5

        return POIs

    def draw(self, screen: Surface):
        """
        Draw the car on the pygame screen.

        Args:
            screen (Surface): The pygame Surface object to draw on.
        """

        # Rotate the car based on the angle
        rotated_car = pygame.transform.rotate(self.image, self.angle)
        rotated_car_rect = rotated_car.get_rect()
        rotated_car_rect.center = (self.x, self.y)
        screen.blit(rotated_car, rotated_car_rect)

    def future_collision(self) -> tuple[int, int] | None:
        """
        Check if there will be a collision in the future based on the car's speed and angle.

        Returns:
            tuple[int, int] | None: The point of intersection (POI) if a collision will occur, None otherwise.
        """

        rotated_car = pygame.transform.rotate(self.image, self.angle)
        rotated_car_rect = rotated_car.get_rect()
        rotated_car = pygame.mask.from_surface(rotated_car)

        x = self.x - rotated_car_rect.centerx
        y = self.y - rotated_car_rect.centery
        x += math.cos(math.radians(self.angle + self.angle_offset)) * self.speed
        y -= math.sin(math.radians(self.angle + self.angle_offset)) * self.speed

        offset = (int(x), int(y))
        POI = self.flipped_masks[0][0].overlap(rotated_car, offset)

        if POI:
            x = POI[0] - self.x
            y = POI[1] - self.y
            if x > 0:
                self.x -= 1
            elif x < 0:
                self.x += 1
            if y > 0:
                self.y -= 1
            elif y < 0:
                self.y += 1
            self.deal_damage(self.speed)
        return POI

    def check_checkpoints(self, checkpoint_image: Surface, locations: list[tuple[tuple[int, int], int]]):
        """
        Check if the car has passed any checkpoints.

        Args:
            checkpoint_image (Surface): The pygame Surface object representing the checkpoint image.
            locations (list[tuple[tuple[int, int], int]]): A list of checkpoint locations and rotations.
        """
        self.total_checkpoints = len(locations)

        rotated_car = pygame.transform.rotate(self.image, self.angle)
        rotated_car_rect = rotated_car.get_rect()
        rotated_car = pygame.mask.from_surface(rotated_car)

        if self.checkpoints_hit != len(locations):
            location = locations[self.checkpoints_hit]
            x = location[0][0]
            y = location[0][1]
            rotation = location[1]
            checkpoint_mask = pygame.mask.from_surface(pygame.transform.rotate(checkpoint_image, rotation))

            x = self.x - rotated_car_rect.centerx - x
            y = self.y - rotated_car_rect.centery - y

            offset = (int(x), int(y))
            collision = checkpoint_mask.overlap(rotated_car, offset)

            if collision:
                self.checkpoints_hit += 1
                self.total_checkpoints_hit += 1
                self.update_score()

    def check_finish_line(self, finish_line_mask: Mask, finish_position: tuple[int, int]):
        """
        Check if the car has crossed the finish line.

        Args:
            finish_line_mask (Mask): The pygame Mask object representing the finish line.
            finish_position (tuple[int, int]): The position of the finish line.
        """

        rotated_car = pygame.transform.rotate(self.image, self.angle)
        rotated_car_rect = rotated_car.get_rect()
        rotated_car = pygame.mask.from_surface(rotated_car)

        x = finish_position[0]
        y = finish_position[1]

        x = self.x - rotated_car_rect.centerx - x
        y = self.y - rotated_car_rect.centery - y

        offset = (int(x), int(y))
        collision = finish_line_mask.overlap(rotated_car, offset)
        # print(("car", self.lap, self.checkpoints_hit, self.total_checkpoints))

        if collision and (self.checkpoints_hit == self.total_checkpoints or self.lap == -1):
            self.lap += 1
            self.checkpoints_hit = 0
            self.update_score()

    def get_distances_to_walls(self, screen: Union[Surface, None] = None) -> tuple[list[tuple[int, int]], list[float]]:
        """
        Calculate the distances to the track walls.

        Args:
            screen (Union[Surface, None], optional): The pygame Surface object. Defaults to None.

        Returns:
            tuple[list[tuple[int, int]], list[float]]: A tuple containing a list of points of interest (POIs)
                and a list of distances to the walls.
        """

        beams = self.get_beams()
        POIs = []
        colission_distances = []
        for beam in beams:
            POI, distance = self.draw_beam(beam, screen)
            if POI:
                POIs.append(POI)
                colission_distances.append(distance)
        return POIs, colission_distances

    def get_beams(self) -> list[Beam]:
        """
        Generate the beams for distance calculation.

        Returns:
            list[Beam]: A list of Beam objects representing the beams.
        """

        lines: list[Beam] = []

        # Removed beams for better performance
        beams = [
            (8.3, 16.3, 360),  # right
            # (-16.3, 0, 270),  # middle
            (-16.3, 8.3, 280),  # slight right
            (-16.3, -8.3, 260),  # slight left
            (8.3, -16.3, 180),  # left
            # (-8.3, -8.3, 225),  # middle left
            # (-8.3, 8.3, 315)  # middle right
        ]

        for offset_x, offset_y, rotation in beams:
            start_x = self.x + offset_x * math.cos(math.radians(self.angle + rotation)) - offset_y * math.sin(
                math.radians(self.angle + rotation))
            start_y = self.y - offset_x * math.sin(math.radians(self.angle + rotation)) - offset_y * math.cos(
                math.radians(self.angle + rotation))

            line_start = pygame.math.Vector2(start_x, start_y)
            lines.append(Beam(vector=line_start, rotation=rotation))

        return lines

    def draw_beam(self, beam: Beam, surface: Union[Surface, None] = None) -> tuple[tuple[int, int], float]:
        """
        Draw a single beam and calculate the point of intersection (POI) with the wall.

        Args:
            beam (Beam): The Beam object representing the beam to be drawn.
            surface (Union[Surface, None], optional): The pygame Surface object. Defaults to None.

        Returns:
            tuple[tuple[int, int], float]: A tuple containing the point of intersection (POI) and the distance to it.
        """

        pos = beam.vector
        angle = beam.rotation - self.angle

        BLUE = (0, 0, 255)
        GREEN = (0, 255, 0)

        c = math.cos(math.radians(angle))
        s = math.sin(math.radians(angle))

        flip_x = c < 0
        flip_y = s < 0
        filpped_mask = self.flipped_masks[flip_x][flip_y]

        # compute beam final point
        x_dest = 900 * abs(c)
        y_dest = 900 * abs(s)

        self.beam_surface.fill((0, 0, 0, 0))

        # draw a single beam to the beam surface based on computed final point
        pygame.draw.line(self.beam_surface, BLUE, (0, 0), (x_dest, y_dest))
        beam_mask = pygame.mask.from_surface(self.beam_surface)

        # find overlap between "global mask" and current beam mask
        offset_x = 899 - pos[0] if flip_x else pos[0]
        offset_y = 899 - pos[1] if flip_y else pos[1]
        hit = filpped_mask.overlap(beam_mask, (offset_x, offset_y))
        hit_pos = (0, 0)
        if hit is not None and (hit[0] != pos[0] or hit[1] != pos[1]):
            hx = 899 - hit[0] if flip_x else hit[0]
            hy = 899 - hit[1] if flip_y else hit[1]
            hit_pos = (hx, hy)

            if isinstance(surface, pygame.Surface):
                pygame.draw.line(surface, BLUE, pos, hit_pos)
                pygame.draw.circle(surface, GREEN, hit_pos, 3)

        return hit_pos, math.dist(pos, hit_pos)

    def update_score(self):
        """
        Update the car's score based on the number of laps and checkpoints hit.

        Modifies the `score` attribute of the car.
        """
        self.score = (self.lap + 1) * self.total_checkpoints_hit

    def deal_damage(self, damage: float):
        """
        Deal damage to the car's health points (HP).

        Args:
            damage (float): The amount of damage to be dealt.
        """
        self.hp -= damage
        self.hp = max(self.hp, 0)
        if self.one_hit_ko or not self.hp:
            self.alive = False

    def death_str(self) -> str:
        """
        Generate a string representation of the car's death information.

        Returns:
            str: A string representing the car's death information.
        """
        return f"Laps: {self.lap} Checkpoints: {self.total_checkpoints_hit} Score: {self.score}"
