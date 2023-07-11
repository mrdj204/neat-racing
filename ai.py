"""
Module: racing_ai

This module contains the AI class for a racing game.

Classes:
    AI: Class representing the AI for the racing game.
    CustomNet: Custom streamlined FeedForwardNetwork class.
    CustomStdOutReporter: Custom reporter class for displaying information during the NEAT algorithm run.
    CustomCheckpointer: Custom checkpointer class for saving and loading checkpoints.

Dependencies:
    Python 3.8 or higher
    neat (NeuroEvolution of Augmenting Topologies)
    pygame
    statistics
"""

import configparser
import gzip
import math
import os
import pickle
import random
import struct
import time
from statistics import mean, stdev

import neat
import pygame
from neat.graphs import feed_forward_layers
from neat.nn import FeedForwardNetwork
from neat.six_util import itervalues

import util
from car import Car
from config import Config, get_config
from game import Game


class AI:
    """
    Class representing the AI for the racing game.

    Attributes:
        config (Config): The configuration object for the AI.
        neat_config (neat.Config): The NEAT configuration object.

    Methods:
        __init__(self, config=get_config()): Initializes the AI object.
        play_genome(self, genome=None, ID=0, best_network=False): Plays the game with the specified genome.
        train_neat(self, checkpoint=0): Trains the NEAT algorithm.
        setup_population(self, checkpoint=0): Sets up the population for training.
        eval_genomes(genomes, neat_config): Evaluates the fitness of the genomes.
        make_decision(vision, speed, net): Makes a decision based on the AI's neural network.
    """
    def __init__(self, config: Config = get_config()):
        """
        Initializes the AI object.

        Args:
            config (Config, optional): The configuration object for the AI. Defaults to the global config object.
        """
        self.config = config

        neat_config_path = util.get_path('neat-config.ini')
        neat_config = configparser.ConfigParser()
        neat_config.read(neat_config_path)
        self.neat_config: neat.Config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                                    neat.DefaultSpeciesSet, neat.DefaultStagnation, neat_config_path)

    def setup_population(self, checkpoint: int = 0) -> neat.Population:
        """
        Sets up the population for training.

        Args:
            checkpoint (int, optional): The checkpoint number. Defaults to 0.

        Returns:
            neat.Population: The NEAT population object.
        """
        # Create the population or load checkpoint
        if not checkpoint:
            p = neat.Population(self.neat_config)
        elif checkpoint == -1:
            try:
                path = os.path.join(util.get_path(['best']), "checkpoint")
                p = neat.Checkpointer.restore_checkpoint(path)
            except (IndexError, FileNotFoundError):
                p = neat.Population(self.neat_config)
        else:
            path = util.get_path(["my_checkpoints", f"neat-checkpoint-{checkpoint}"])
            p = neat.Checkpointer.restore_checkpoint(path)

        # Update config file
        p.config = self.neat_config

        # Add reporters
        generation_interval = self.config.ai.generations_per_checkpoint
        p.add_reporter(CustomStdOutReporter(self.config))
        p.add_reporter(neat.StatisticsReporter())
        checkpoint_path = util.get_path("my_checkpoints")
        checkpoint_prefix = os.path.join(f"{checkpoint_path}", "neat-checkpoint-")
        p.add_reporter(neat.Checkpointer(generation_interval, filename_prefix=checkpoint_prefix))
        best_checkpoint_path = os.path.join(util.get_path("best"), "")
        p.add_reporter(CustomCheckpointer(generation_interval, filename_prefix=best_checkpoint_path))
        return p

    def play_genome(self, genome: neat.DefaultGenome = None, ID: int = 0) -> None:
        """
        Plays the game with the specified genome.

        Args:
            genome (neat.DefaultGenome, optional): The genome to play with. Defaults to None.
            ID (int, optional): The ID of the genome to play with. Defaults to 0.

        Returns:
            None
        """

        # Get genome if ID
        if not genome:
            if ID:
                try:
                    if ID == -1:
                        path = util.get_path(["best", "best.pickle"])
                        with open(path, "rb") as f:
                            genome = pickle.load(f)
                    else:
                        path = util.get_path(["genomes", f"{ID}.pickle"])
                        with open(path, "rb") as f:
                            genome = pickle.load(f)
                except FileNotFoundError:
                    pass
            # Save current gen number for AI/game to read, easier than passing this value through arguements
            path = util.get_path("generation.bin")
            with open(path, "wb") as f:
                f.write(struct.pack("i", ID))

        if not genome:
            raise Exception(f"Couldn't find genome: Genome=({genome}) id=({ID})")

        # Setup and run game
        game = Game(run_game=False, AI_enabled=True, config=self.config)
        net = CustomNet.create(genome, self.neat_config)
        if self.config.ai.play_genome_with_player:
            game.cars = [
                Car(68, 290, game.flipped_masks, self.config, "red", net),
                Car(68, 290, game.flipped_masks, self.config, "green"),
            ]
        else:
            game.cars = [
                Car(43, 180, game.flipped_masks, self.config, net=net),
                Car(68, 190, game.flipped_masks, self.config, net=net),
                Car(93, 200, game.flipped_masks, self.config, net=net),
                Car(43, 220, game.flipped_masks, self.config, net=net),
                Car(68, 230, game.flipped_masks, self.config, net=net),
                Car(93, 240, game.flipped_masks, self.config, net=net),
            ]
        game.run_game()

    def train_neat(self, checkpoint: int = 0) -> None:
        """
        Trains the NEAT algorithm.

        Args:
            checkpoint (int, optional): The checkpoint number. Defaults to 0.

        Returns:
            None
        """
        p = self.setup_population(checkpoint)
        p.config = self.neat_config

        # Use config to specify how many generations to run for
        for i in range(0, self.config.ai.generations):
            # Save current generation number to file for pygame to read
            try:
                path = util.get_path("generation.bin")
                with open(path, "wb") as f:
                    f.write(struct.pack("i", p.generation))
            except (PermissionError, OSError) as e:
                util.get_logger(self.config).exception(e)
            # Train AI
            winner = p.run(self.eval_genomes, 1)
            # Save best genome
            folder_path = util.get_path(['genomes', f"{p.generation - 1}.pickle"])
            with open(folder_path, "wb") as f:
                pickle.dump(winner, f)
            # For portfolio
            folder_path = util.get_path(['best', "best.pickle"])
            with open(folder_path, "wb") as f:
                pickle.dump(winner, f)

            # Use config to set if best genome should play each generation
            if self.config.ai.play_best_genome_each_generation:
                self.play_genome(genome=winner)

    @staticmethod
    def eval_genomes(genomes: list[tuple[int, neat.DefaultGenome]], neat_config: neat.Config) -> None:
        """
        Evaluates the fitness of the genomes.

        Args:
            genomes (list[tuple[int, neat.DefaultGenome]]): The list of genomes to evaluate.
            neat_config (configparser.ConfigParser): The NEAT configuration object.

        Returns:
            None
        """

        config = get_config()
        RERUN_GENOMES = config.ai.rerun_genomes

        genomes_to_evaluate = [] if not RERUN_GENOMES else genomes
        genome_dict = {} if not RERUN_GENOMES else {genome_key: genome for genome_key, genome in genomes}

        if not RERUN_GENOMES:
            for genome_key, genome in genomes:
                if not genome.fitness:
                    genome_dict.update({genome_key: genome})
                    genomes_to_evaluate.append((genome_key, genome))

        util.get_logger(config).info(f"Genomes: {len(genomes_to_evaluate)}/{len(genomes)}")

        # Setup game
        game = Game(
            run_game=False,
            AI_enabled=True,
            AI_training=True,
            total_genomes=len(genomes_to_evaluate),
        )
        game.cars = []
        for genome_key, genome in genomes_to_evaluate:
            if not genome.fitness or RERUN_GENOMES:
                color = "pink" if genome.fitness else "yellow"
                genome.fitness = 0
                net = CustomNet.create(genome, neat_config)
                game.cars.append(
                    Car(
                        x=68,
                        y=490,
                        net=net,
                        genome_key=genome_key,
                        flipped_masks=game.flipped_masks,
                        config=config,
                        color=color
                    )
                )

        # Run game
        game.run_game()

        # Get fitness values
        scores = []
        for car in game.cars:
            car_genome = genome_dict[car.genome_key]

            max_time = config.game.time_limit
            time_alive = round((max_time - car.time_alive), 1)
            is_alive = 1 if car.timed_out else 0
            avg_speed = mean(car.speed_list) if len(car.speed_list) > 1 else 1
            w1 = 1
            w2 = 0 if is_alive else 1
            w3 = max_time
            w4 = 2
            f1 = w1 * car.score
            f2 = w2 * time_alive
            f3 = w3 * is_alive
            f4 = math.pow(avg_speed, w4)
            car_genome.fitness = round(f1 * (f2 + f3) * f4)
            # util.get_logger(config).info(f"{car_genome.fitness} = {f1} * ({f2} + {f3}) * {f4}")
            scores.append(car_genome.fitness)

        print(f'{sorted(scores, reverse=True)}\n'
              f'[LEN: {len(scores)}] [AVG: {round(mean(scores), 1) if scores else 0}]\n')


class CustomNet(FeedForwardNetwork):
    """
    Custom streamlined FeedForwardNetwork

    Methods:
        activate(self, inputs: list[float]): Custom activation function that returns pygame keypresses
        create(genome, config): Custom create function that returns CustomeNet since it doesn't use __init__
    """
    def activate(self, inputs: list[float]) -> dict[int, bool]:
        """
        Custom activation function that returns pygame keypresses

        Args:
            inputs (list[float]): The inputs for the AI

        Returns:
            dict[int, bool]: A dictionary of keys the AI wants to press
        """
        # Get decision from super()
        decision = super().activate(inputs)

        # Set keys
        keys = {
            pygame.K_SPACE: decision[0] <= decision[1],
            pygame.K_UP: decision[0] > decision[1],
            pygame.K_DOWN: False,
            pygame.K_LEFT: decision[2] > decision[3],
            pygame.K_RIGHT: decision[2] < decision[3],
            pygame.K_w: False,
            pygame.K_s: False,
            pygame.K_a: False,
            pygame.K_d: False,
        }

        # Send action to game and return False if game is over
        return keys

    @staticmethod
    def create(genome, config):
        """ Receives a genome and returns its phenotype (my CustomNet). """

        # Gather expressed connections.
        connections = [cg.key for cg in itervalues(genome.connections) if cg.enabled]

        layers = feed_forward_layers(config.genome_config.input_keys, config.genome_config.output_keys, connections)
        node_evals = []
        for layer in layers:
            for node in layer:
                inputs = []
                node_expr = []  # currently unused
                for conn_key in connections:
                    inode, onode = conn_key
                    if onode == node:
                        cg = genome.connections[conn_key]
                        inputs.append((inode, cg.weight))
                        node_expr.append("v[{}] * {:.7e}".format(inode, cg.weight))

                ng = genome.nodes[node]
                aggregation_function = config.genome_config.aggregation_function_defs.get(ng.aggregation)
                activation_function = config.genome_config.activation_defs.get(ng.activation)
                node_evals.append((node, activation_function, aggregation_function, ng.bias, ng.response, inputs))

        return CustomNet(config.genome_config.input_keys, config.genome_config.output_keys, node_evals)


class CustomStdOutReporter(neat.reporting.BaseReporter):
    """
    Uses `print` to output information about the run; an example reporter class.
    """

    def __init__(self, config: Config):
        """
        Initialize the CustomStdOutReporter.

        Args:
            config (Config): The configuration object.
        """
        self.show_species_detail = config.reporting.show_species_detail
        self.generation = None
        self.generation_start_time = None
        self.generation_times = []
        self.num_extinctions = 0
        self.logger = util.get_logger(config)

    def start_generation(self, generation):
        """
        Called at the start of a generation.

        Args:
            generation (int): The current generation.
        """
        self.generation = generation
        self.logger.info('\n ****** Running generation {0} ****** \n'.format(generation))
        self.generation_start_time = time.time()

    def end_generation(self, config, population, species_set):
        """
        Called at the end of a generation.

        Args:
            config: The NEAT configuration object.
            population (dict): A dictionary of genomes in the population.
            species_set (neat.species.SpeciesSet): The set of species.
        """
        ng = len(population)
        ns = len(species_set.species)
        if self.show_species_detail:
            self.logger.info('Population of {0:d} members in {1:d} species:'.format(ng, ns))
            self.logger.info("   ID   age  size  fitness  adj fit  stag")
            self.logger.info("   ===  ===  ====  =======  =======  ====")
            for sid in sorted(species_set.species):
                s = species_set.species[sid]
                a = self.generation - s.created
                n = len(s.members)
                # noinspection PyTypeChecker
                f = "--" if s.fitness is None else "{:,}".format(round(s.fitness))
                af = "--" if s.adjusted_fitness is None else f"{s.adjusted_fitness:.3f}"
                st = self.generation - s.last_improved
                self.logger.info(f"  {sid:>4}  {a:>3}  {n:>4}  {f:>7}  {af:>7}  {st:>4}")
        else:
            self.logger.info('Population of {0:d} members in {1:d} species'.format(ng, ns))

        elapsed = time.time() - self.generation_start_time
        self.generation_times.append(elapsed)
        self.generation_times = self.generation_times[-10:]
        average = sum(self.generation_times) / len(self.generation_times)
        # print('Total extinctions: {0:d}'.format(self.num_extinctions))
        if len(self.generation_times) > 1:
            self.logger.info("Generation time: {0:.3f} sec ({1:.3f} average)".format(elapsed, average))
        else:
            self.logger.info("Generation time: {0:.3f} sec".format(elapsed))

    def post_evaluate(self, config, population, species, best_genome):
        """
        Called after the fitness evaluation for the population.

        Args:
            config: The NEAT configuration object.
            population (dict): A dictionary of genomes in the population.
            species (neat.species.SpeciesSet): The set of species.
            best_genome (neat.genome.DefaultGenome): The best genome in the population.
        """
        # pylint: disable=no-self-use
        fitnesses = [c.fitness for c in population.values()]
        fit_mean = mean(fitnesses)
        fit_std = stdev(fitnesses)
        best_species_id = species.get_species_id(best_genome.key)
        self.logger.info('Population\'s average fitness: {:,} stdev: {:,}'.format(round(fit_mean), round(fit_std)))
        self.logger.info('Best fitness: {:,} - size: {!r} - species {} - id {}'
                         .format(best_genome.fitness, best_genome.size(), best_species_id, best_genome.key))

    def complete_extinction(self):
        """Called when all species in the population go extinct."""
        self.num_extinctions += 1
        self.logger.info('All species extinct.')

    def found_solution(self, config, generation, best):
        """
        Called when a solution meeting the fitness threshold is found.

        Args:
            config: The NEAT configuration object.
            generation (int): The generation where the solution was found.
            best (neat.genome.DefaultGenome): The best genome that meets the fitness threshold.
        """
        self.logger.info('\nBest individual in generation {0} meets fitness threshold - complexity: {1!r}'
                         .format(self.generation, best.size()))

    def species_stagnant(self, sid, species):
        """
        Called when a species becomes stagnant (no improvement in fitness).

        Args:
            sid: The ID of the stagnant species.
            species (neat.species.Species): The stagnant species object.
        """
        if self.show_species_detail:
            self.logger.info("\nSpecies {0} with {1} members is stagnated: removing it"
                             .format(sid, len(species.members)))

    def info(self, msg):
        """
        Output an informational message.

        Args:
            msg (str): The informational message.
        """
        self.logger.info(msg)


class CustomCheckpointer(neat.Checkpointer):
    def save_checkpoint(self, config, population, species_set, generation):
        """ Save the current simulation state. """
        filename = f'{self.filename_prefix}checkpoint'
        print(f"Saving checkpoint to {filename}")

        with gzip.open(filename, 'w', compresslevel=5) as f:
            data = (generation, config, population, species_set, random.getstate())
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
