import json
from configparser import ConfigParser

from pydantic import BaseSettings

import util


class MainConfig(BaseSettings):
    run_player: bool
    run_ai: bool
    run_id: int
    train_ai: bool
    train_id: int


class AIConfig(BaseSettings):
    enabled: bool
    auto_close: bool
    generations: int
    play_best_genome_each_generation: bool
    play_genome_with_player: bool
    generations_per_checkpoint: int
    base_speed: int
    rerun_genomes: bool
    draw_ai_car_beams: bool
    draw_ai_car_beam_intersections: bool


class ReportingConfig(BaseSettings):
    show_species_detail: bool
    stream_to_file: bool
    log_death_data: bool


class GameConfig(BaseSettings):
    time_limit: int
    show_mouse_pos: bool
    draw_car_limit: int
    car_spawned_limit: int
    show_checkpoints: bool
    show_all_checkpoints: bool
    show_track_border: bool
    show_map: bool


class DebugConfig(BaseSettings):
    show_car_debug: bool


class Config(BaseSettings):
    main: MainConfig
    ai: AIConfig
    reporting: ReportingConfig
    game: GameConfig
    debug: DebugConfig


def get_config() -> Config:
    """
    Read the configuration settings from the `config.ini` file and return a `Config` object.

    Returns:
        Config: The configuration settings.

    Raises:
        ValueError: If the config file is invalid.
    """
    parser = ConfigParser()
    parser.read(util.get_path("config.ini"))
    try:
        return Config(
            main=MainConfig(
                run_player=parser.getboolean("main", "run_player"),
                run_ai=parser.getboolean("main", "run_ai"),
                run_id=parser.getint("main", "run_id"),
                train_ai=parser.getboolean("main", "train_ai"),
                train_id=parser.getint("main", "train_id"),
            ),
            ai=AIConfig(
                enabled=parser.getboolean("ai", "enabled"),
                auto_close=parser.getboolean("ai", "auto_close"),
                generations=parser.getint("ai", "generations"),
                play_best_genome_each_generation=parser.getboolean("ai", "play_best_genome_each_generation"),
                play_genome_with_player=parser.getboolean("ai", "play_genome_with_player"),
                generations_per_checkpoint=parser.getint("ai", "generations_per_checkpoint"),
                base_speed=parser.getint("ai", "base_speed"),
                rerun_genomes=parser.getboolean("ai", "rerun_genomes"),
                draw_ai_car_beams=parser.getboolean("ai", "draw_ai_car_beams"),
                draw_ai_car_beam_intersections=parser.getboolean("ai", "draw_ai_car_beam_intersections"),
            ),
            reporting=ReportingConfig(
                show_species_detail=parser.getboolean("reporting", "show_species_detail"),
                stream_to_file=parser.getboolean("reporting", "stream_to_file"),
                log_death_data=parser.getboolean("reporting", "log_death_data"),
            ),
            game=GameConfig(
                time_limit=parser.getint("game", "time_limit"),
                show_mouse_pos=parser.getboolean("game", "show_mouse_pos"),
                draw_car_limit=parser.getint("game", "draw_car_limit"),
                car_spawned_limit=parser.getint("game", "car_spawned_limit"),
                show_checkpoints=parser.getboolean("game", "show_checkpoints"),
                show_all_checkpoints=parser.getboolean("game", "show_all_checkpoints"),
                show_track_border=parser.getboolean("game", "show_track_border"),
                show_map=parser.getboolean("game", "show_map"),
            ),
            debug=DebugConfig(
                show_car_debug=parser.getboolean("debug", "show_car_debug"),
            ),
        )
    except ValueError:
        error = True
    if error:
        raise ValueError("Invalid Config File")


def print_config(config: Config):
    """
    Print the configuration settings in a formatted JSON format.

    Args:
        config (Config): The configuration settings.
    """
    print(json.dumps(json.loads(config.json()), indent=2))
