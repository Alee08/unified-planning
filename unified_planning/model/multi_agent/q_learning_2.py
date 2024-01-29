import functools
import random
from copy import copy
import numpy as np
from gymnasium.spaces import Discrete, MultiDiscrete
import pygame
from pettingzoo import ParallelEnv
from unified_planning.shortcuts import *

class Q_learning:
    def __init__(self, state_space_size, action_space_size, learning_rate, gamma):
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.q_table = np.zeros((state_space_size, action_space_size))

    def encode_state(self, state, agent, state_rm):
        # Estrai i valori dallo stato
        RM_agent = agent.get_reward_machine()
        num_state = RM_agent.numbers_state()
        #rm_current_state = RM_agent.get_current_state()
        rm_state_index = RM_agent.get_state_index(state_rm)

        valori = list(state.values())
        #print(valori, "oooo")
        #ok = valori[1].constant_value() * 10 + valori[0].constant_value()
        # Definisci i valori massimi per le dimensioni x e y
        max_x_value = 4  # Assicurati che questo valore sia adeguato
        max_y_value = 4  # Assicurati che questo valore sia adeguato
        # Calcola l'indice codificato
        indice_codificato = 0
        for i, valore in enumerate(valori):
            max_valore_per_dimensione = max_x_value if i == 0 else max_y_value
            indice_codificato += valore * (max_valore_per_dimensione ** i)
        ####
        pos = valori[1].constant_value() * max_x_value + valori[0].constant_value()
        state = pos * num_state + rm_state_index
        #print(pos, rm_state_index, "stateeeeeeeeeee")
        #print(indice_codificato.simplify(), "indice_codificato")
        ####
        # Verifica e semplifica l'indice codificato
        if isinstance(indice_codificato, FNode):
            indice_codificato = indice_codificato.simplify()
            indice_codificato = indice_codificato.constant_value()


        #print("x,y", indice_codificato)
        #print("rm_state_index", rm_state_index)
        # Calcola l'indice finale dello stato
        enc_state = indice_codificato * num_state + rm_state_index
        #print(rm_state_index)
        if enc_state >= self.state_space_size:
            raise ValueError("Indice di stato codificato supera la dimensione della Q-table")
        return state

    def update(self, state, next_state, action, reward, agent, state_rm, new_state_rm):
        #print(f"Updating Q-table with state: {state}, action: {action}")
        #breakpoint()
        enc_state = self.encode_state(state, agent, state_rm)
        enc_next_state = self.encode_state(next_state, agent, new_state_rm)

        current_q = self.q_table[enc_state, action]
        #print(action, "ooooooooooooooooooooooooooooooooo", state, "\n", self.q_table)
        max_future_q = np.max(self.q_table[enc_next_state])
        new_q = (1 - self.learning_rate) * current_q + self.learning_rate * (reward + self.gamma * max_future_q)
        #print(new_q, "new_qnew_qnew_q", reward)
        """if reward != 0:
            print(enc_state, "\n", enc_next_state)
            breakpoint()"""
        self.q_table[enc_state, action] = new_q

    def choose_action(self, state, epsilon, agent, state_rm):
        if random.uniform(0, 1) < epsilon:
            return random.randint(0, self.action_space_size - 1)
        else:
            enc_state = self.encode_state(state, agent, state_rm)
            return np.argmax(self.q_table[enc_state])
