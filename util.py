"""
This module provides utility functions for the game.

Functions:
    get_logger(config: Config) -> logging.Logger:
        Returns a logger object configured based on the provided `Config` object.

    get_checkpoint_locations() -> list[tuple[tuple[int, int], int]]:
        Returns a list of checkpoint locations in the form of (position, angle) tuples.

Note:
    This code assumes the presence of a `Config` class from the `config` module.
"""
import logging
import sys

from pydantic import BaseModel
from pygame.math import Vector2

from config import Config


def get_logger(config: Config) -> logging.Logger:
    """
    Returns a logger object configured based on the provided `Config` object.

    Args:
        config (Config): The configuration object containing logger settings.

    Returns:
        logging.Logger: The configured logger object.
    """
    logger = logging.getLogger()
    if logger.handlers:
        return logger
    if config.reporting.stream_to_file:
        logging.basicConfig(filename='neat.log', level=logging.DEBUG, filemode='a', format='%(message)s')
        logger.addHandler(logging.StreamHandler(sys.stdout))
    else:
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format='%(message)s')
    return logger


def get_checkpoint_locations() -> list[tuple[tuple[int, int], int]]:
    """
    Returns a list of checkpoint locations in the form of (position, angle) tuples.

    Returns:
        list[tuple[tuple[int, int], int]]: The list of checkpoint locations.
    """
    checkpoint_locations = [
        ((55, 65), -45),
        ((123, 45), 90),
        ((140, 65), 45),
    ]

    for i in range(140, 380 + 20, 20):
        checkpoint_locations.append(((156, i), 0))

    checkpoint_locations.append(((178, 403), 45))
    checkpoint_locations.append(((245, 418), 90))
    checkpoint_locations.append(((261, 404), -45))

    start = 380
    step = 20
    for i in range(13):
        y = start - step * i
        checkpoint_locations.append(((272, y), 0))

    checkpoint_locations.append(((297, 67), -45))

    for i in range(370, 740 + 20, 20):
        checkpoint_locations.append(((i, 45), 90))

    checkpoint_locations.append(((770, 70), 45))

    for i in range(140, 220 + 20, 20):
        checkpoint_locations.append(((782, i), 0))

    checkpoint_locations.append(((768, 237), -45))

    start = 745
    step = 20
    for i in range(13):
        x = start - step * i
        checkpoint_locations.append(((x, 250), 90))

    checkpoint_locations.append(((432, 272), -45))
    checkpoint_locations.append(((410, 340), 0))
    checkpoint_locations.append(((433, 358), 45))

    start = 505
    step = 20
    for i in range(13):
        x = start + step * i
        checkpoint_locations.append(((x, 365), 90))

    checkpoint_locations.append(((765, 390), 45))

    for i in range(460, 720 + 20, 20):
        checkpoint_locations.append(((782, i), 0))

    checkpoint_locations.append(((770, 760), -45))
    checkpoint_locations.append(((736, 772), 90))
    checkpoint_locations.append(((650, 758), 45))

    for i in range(720, 620 - 20, -20):
        checkpoint_locations.append(((626, i), 0))

    checkpoint_locations.append(((600, 525), 45))
    checkpoint_locations.append(((549, 493), 90))
    checkpoint_locations.append(((448, 525), -45))

    for i in range(620, 720 + 20, 20):
        checkpoint_locations.append(((411, i), 0))

    checkpoint_locations.append(((398, 758), -45))
    checkpoint_locations.append(((370, 775), 90))

    x = 285.0
    y = 753.0
    for i in range(0, 16):
        checkpoint_locations.append(((round(x), round(y)), 45))
        x -= 15.2
        y -= 15.3

    # checkpoint_locations.append(((288, 755), 45))
    # checkpoint_locations.append(((212, 678), 45))
    # checkpoint_locations.append(((136, 602), 45))
    # checkpoint_locations.append(((60, 525), 45))

    start = 485
    step = 20
    for i in range(17):
        y = start - step * i
        checkpoint_locations.append(((28, y), 0))

    return checkpoint_locations


class Beam(BaseModel):
    vector: Vector2
    rotation: int

    class Config:
        arbitrary_types_allowed = True
