import multiprocessing
import os
import pickle

import cart_pole
import neat

def run_and_eval_pole_genome(genome, config):
    runs_per_net = 5   # <-- Note we are running multiple times as we have randomness built in..
    simulation_seconds = 60.0
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    fitnesses = []

    for runs in range(runs_per_net):
        sim = cart_pole.CartPole()
        # Run the given simulation for up to num_steps time steps.
        fitness = 0.0
        while sim.t < simulation_seconds:
            inputs = sim.get_scaled_state()
            action = net.activate(inputs)
            # Apply action to the simulated cart-pole
            force = cart_pole.discrete_actuator_force(action)
            sim.step(force)
            # Stop if the network fails to keep the cart within the position or angle limits.
            # The per-run fitness is the number of time steps the network can balance the pole without exceeding these limits.
            if abs(sim.x) >= sim.position_limit or abs(sim.theta) >= sim.angle_limit_radians:
                break
            fitness = sim.t
        fitnesses.append(fitness)

    # The genome's fitness is its worst performance across all runs.
    return min(fitnesses)