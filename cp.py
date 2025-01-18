#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 22:01:34 2024

@author: hahaha
"""
import numpy as np
import matplotlib.pyplot as plt

class SimulatedAnnealingOptimization:
    def __init__(self, n, seed, wrap=True):
        self.n = n
        self.seed = seed
        self.wrap = wrap
        self.C = self.generate_data(n, seed)
        self.current_state = (np.random.randint(n), np.random.randint(n))
        self.best_state = self.current_state
        self.current_cost = self.C[self.current_state]
        self.best_cost = self.current_cost
        self.history = []
        self.acceptance_probabilities = []

    def generate_data(self, n, seed):
        if type(seed) != str:
            raise TypeError("The seed should be the string representing your ID")
        if type(n) != int:
            raise TypeError("The dimension of the problem should be an integer value")
        if n % 2 != 0:
            raise ValueError("The dimension of the problem should be even!!")

        aggregate_counter = 0
        for char in seed:
            aggregate_counter += int(char)

        aggregate_counter = 10 * aggregate_counter
        marginal_diff = int(aggregate_counter) % n
        bsize = 12 + int(np.floor(n**(1/2)))
        if (bsize % 2) != 0:
            bsize = bsize - 1

        if (marginal_diff < bsize + 4):
            marginal_diff = bsize + 4
        if (marginal_diff > (n - bsize - 4)):
            marginal_diff = n - bsize - 4
        if ((marginal_diff > n / 2 - bsize - 5) and (marginal_diff < n / 2 + bsize + 5)):
            marginal_diff = int(n / 2 - bsize - 5)

        c_marginal_diff = n - marginal_diff

        b1 = -100 + np.random.randn(bsize, bsize)
        b2 = -50 + np.random.randn(bsize, bsize)

        f_values = np.zeros((n, n), dtype=np.float32)
        f_values = 1 + 0.05 * np.random.randn(n, n)
        f_values[marginal_diff - bsize // 2:marginal_diff + bsize // 2, c_marginal_diff - bsize // 2:c_marginal_diff + bsize // 2] = b1
        f_values[c_marginal_diff - bsize // 2:c_marginal_diff + bsize // 2, marginal_diff - bsize // 2:marginal_diff + bsize // 2] = b2
        return f_values

    def accept_with_prob(self, delta_cost, beta):
        if delta_cost <= 0:
            return True, 1.0
        if beta == np.inf:
            return False, 0.0

        prob = np.exp(-beta * delta_cost)
        return np.random.random() < prob, prob

    def propose_move(self):
        i, j = self.current_state
        if self.wrap:
            proposed_i = np.random.choice([(i - 1) % self.n, (i + 1) % self.n])
            proposed_j = np.random.choice([(j - 1) % self.n, (j + 1) % self.n])
        else:
            proposed_i = np.random.choice([max(i - 1, 0), min(i + 1, self.n - 1)])
            proposed_j = np.random.choice([max(j - 1, 0), min(j + 1, self.n - 1)])
        return (proposed_i, proposed_j)

    def compute_delta_cost(self, proposed_state):
        proposed_cost = self.C[proposed_state]
        return proposed_cost - self.current_cost

    def accept_move(self, proposed_state):
        self.current_state = proposed_state
        self.current_cost = self.C[proposed_state]

    def simann(self, beta0=0.1, beta1=10., anneal_steps=10, mcmc_steps=10, seed=None):
        if seed is not None:
            np.random.seed(seed)

        betas = np.zeros(anneal_steps)
        betas[:-1] = np.linspace(beta0, beta1, anneal_steps - 1)
        betas[-1] = np.inf

        for beta in betas:
            accepted_moves = 0
            for t in range(mcmc_steps):
                proposed_state = self.propose_move()
                delta_cost = self.compute_delta_cost(proposed_state)
                accept, prob = self.accept_with_prob(delta_cost, beta)
                self.acceptance_probabilities.append(prob)
                if accept:
                    accepted_moves += 1
                    self.accept_move(proposed_state)
                    if self.best_cost >= self.current_cost:
                        self.best_cost = self.current_cost
                        self.best_state = self.current_state
            self.history.append((beta, accepted_moves / mcmc_steps, self.current_cost, self.best_cost))
            print(f"beta={beta} accept_freq={accepted_moves/mcmc_steps} c={self.current_cost} best_c={self.best_cost}")

        print(f"Best state: {self.best_state} with cost: {self.best_cost}")
        return self.best_state, self.best_cost

    def display(self, n, wrap):
        plt.imshow(self.C, cmap='hot', interpolation='nearest')
        plt.colorbar()
        plt.scatter(self.best_state[1], self.best_state[0], color='blue')
        title = f'n = {n}, wrap = {"True" if wrap else "False"}'
        plt.title(title)
        plt.show()

    def performance_analysis(self, n_values):
        results_wrap = {}
        results_no_wrap = {}
        for n in n_values:
            # With wrapping
            self.wrap = True
            self.n = n
            self.C = self.generate_data(n, self.seed)
            self.simann()
            self.display(n, True)
            results_wrap[n] = self.best_cost

            # Without wrapping
            self.wrap = False
            self.n = n
            self.C = self.generate_data(n, self.seed)
            self.simann()
            self.display(n, False)
            results_no_wrap[n] = self.best_cost

        return results_wrap, results_no_wrap

    def plot_acceptance_probabilities(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.acceptance_probabilities, label="Acceptance Probability")
        plt.xlabel("Iteration")
        plt.ylabel("Acceptance Probability")
        plt.title("Acceptance Probability at Each Step")
        plt.legend()
        plt.show()

# Implement
seed = "3133703"
n_values = [100, 200, 500, 1000, 5000]
sim_ann = SimulatedAnnealingOptimization(n_values[0], seed)
performance_wrap, performance_no_wrap = sim_ann.performance_analysis(n_values)
print("Performance with wrapping:", performance_wrap)
print("Performance without wrapping:", performance_no_wrap)

# Plot acceptance probabilities
sim_ann.plot_acceptance_probabilities()

# Plot optimization for a fixed value of n
fixed_n = 200  # Example fixed value of n
sim_ann_fixed = SimulatedAnnealingOptimization(fixed_n, seed)
sim_ann_fixed.simann()
sim_ann_fixed.display(fixed_n, sim_ann_fixed.wrap)
