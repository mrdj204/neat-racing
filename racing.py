if __name__ == "__main__":
    """
    The main entry point of the program.
    """
    from ai import AI
    from config import get_config
    from game import Game

    # Retrieve the configuration settings
    config = get_config()

    if config.main.run_player:
        # Start the game with player-controlled cars
        Game(staggered_start=False, run_game=True, config=config)

    if config.main.run_ai:
        # Play a specific genome using the AI module
        ai = AI(config=config)
        ai.play_genome(ID=config.main.run_id)
        # ai.play_genome(best_network=True)

    if config.main.train_ai:
        # Start training using the NEAT algorithm
        ai = AI(config=config)
        ai.train_neat(checkpoint=config.main.train_id)
