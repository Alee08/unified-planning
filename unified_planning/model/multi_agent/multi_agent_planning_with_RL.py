import functools
import random
from copy import copy

import numpy as np
from gymnasium.spaces import Discrete, MultiDiscrete
import pygame
from time import *
from pettingzoo import ParallelEnv
from algo_rl import RL_algorithms
from q_learning_2 import Q_learning
from RewardMachine import RewardMachine
#from ma_problem import MultiAgentProblem
#from agent import Agent
#from ma_environment import MAEnvironment
from unified_planning.shortcuts import *
from unified_planning.model.multi_agent import *
from collections import namedtuple
from unified_planning.io.ma_pddl_writer import MAPDDLWriter
from agent_rl import AgentRL
import cv2
from unified_planning.engines.ma_sequential_simulator import (
        UPSequentialSimulatorMA as SequentialSimulatorMA,
    )
import cProfile
import json
import pickle
from building_RM import RM_dict, RM_dict_true, RM_dict_true_seq
from message import Message

class MAP_RL_Env(ParallelEnv, MultiAgentProblem):

    """The metadata holds environment constants.

    The "name" metadata allows the environment to be pretty printed.
    """

    metadata = {
        "name": "multi_agent_planning_with_RL",
    }

    def __init__(self):

        """The init method takes in environment arguments.

        Should define the following attributes:
        - escape x and y coordinates
        - guard x and y coordinates
        - prisoner x and y coordinates
        - timestamp
        - possible_agents

        Note: as of v1.18.1, the action_spaces and observation_spaces attributes are deprecated.
        Spaces should be defined in the action_space() and observation_space() methods.
        If these methods are not overridden, spaces will be inferred from self.observation_spaces/action_spaces, raising a warning.

        These attributes should not be changed after initialization.
        """

        #self.rl_algorithm.epsilon = self.epsilon  # Aggiorna epsilon nell'algoritmo
        # Inizializzazione di MultiAgentProblem
        MultiAgentProblem.__init__(self)



        self.walls = {}  # Inserisci le coordinate dei muri qui
        self.grid_width = 4  # 10 celle di larghezza
        self.grid_height = 4  #  10 celle di altezza
        # Reimposta epsilon all'inizio di ogni episodio
        self.epsilon_start = 1.0  # Alto valore iniziale per maggiore esplorazione
        self.epsilon_end = 0.01  # Valore finale basso per maggiore sfruttamento
        self.epsilon_decay = 0.995#0.995 #0.995 #0.99995  # Tasso di riduzione di epsilon
        self.epsilon = self.epsilon_start  # Inizializza epsilon con il valore iniziale
        self.rewards = 0
        self.current_state = None
        self.position_A = (3, 1)
        self.position_B = (8, 9)
        self.position_C = (4, 7)
        self.position_D = (8, 1)
        self.position_E = (7, 5)
        self.position_F = (1, 8)
        self.new_state = None
        self.num_rm_states = 4  # Aggiorna questo valore in base al tuo specifico caso
        self.Location = UserType("Location")
        self.l33 = Object("l33", self.Location)
        self.l34 = Object("l34", self.Location)
        self.current_state_env = None
        #self.seq_ag = SequentialSimulatorMA(self)
        #self.current_state_env = self.seq_ag.get_initial_state()
        self.initial_states = None
        #self.message_conditions = None
        self.penalty_cells = [self.position_A, self.position_B, self.position_C, self.position_D, self.position_E, self.position_F]  # Coordinate delle celle con penalità
        self.penalty_amount = -1



    def initialize_state(self):
        self.seq_ag = SequentialSimulatorMA(self)
        self.initial_states = self.seq_ag.get_initial_state()

    def reset(self, seed=None, options=None):
        """Reset set the environment to a starting point.

        It needs to initialize the following attributes:
        - agents
        - timestamp
        - Questi sono dentro agents:
            - prisoner x and y coordinates
            - guard x and y coordinates
        - observation
        - infos

        And must set up the environment so that render(), step(), and observe() can be called without issues.
        """
        # Aggiorna epsilon per il controllo dell'esplorazione
        self.rewards = {agent.name: 0 for agent in self.agents}
        self.epsilon = max(self.epsilon_end, self.epsilon_decay * self.epsilon)
        self.timestep = 0
        #self.escape_x = 4
        #self.escape_y = 7

        self.current_state = {}
        """for agent in self.agents:
            # Reset della RewardMachine dell'agente
            agent.get_reward_machine().reset_to_initial_state()
            for fluente in agent.fluents:
                chiave = (agent, fluente)
                self.current_state[chiave] = self.initial_values[Dot(agent, fluente)]"""
        #self.seq_ag = SequentialSimulatorMA(self)
        #self.current_state_env = self.seq_ag.get_initial_state()
        self.current_state_env = self.initial_states
        self.agent_states = {agent.name: {} for agent in self.agents}
        #self.message_conditions = None

        for agent in self.agents:
            agent.reset_messages()
            agent.message_sent = False
            agent.get_reward_machine().reset_to_initial_state()
            ag_state = self.current_state_env.get_dot_values(agent, only_true_values=True)
            self.agent_states[agent.name] = ag_state


        """
        for agent in self.agents:
            agent.get_reward_machine().reset_to_initial_state()
            for fluent in agent.fluents:
                fluent_key = (agent.name, fluent.name)  # Utilizza il nome dell'agente e il nome del fluente come chiave
                self.agent_states[agent.name][fluent_key] = self.current_state_env.get_value(
                    Dot(agent, fluent)).constant_value()"""




        # Reset the overall environment state
        #self.current_state_env = self.seq_ag.get_initial_state()
        observations = []
        # Get dummy infos
        infos = {agent: {} for agent in self.agents}


        return observations, infos


    def step(self, actions):
        terminations, truncations, infos = {}, {}, {}
        #import copy

        for agent in self.agents:
            current_statee = self.get_state(agent)
            breakpoint()
            action = actions[agent.name]
            self.execute_agent_action(agent, action, current_statee)
            new_state = self.get_state(agent)

            # Aggiorna la ricompensa e lo stato della Reward Machine
            state_rm = agent.reward_machine.get_current_state()
            event = self.detect_event(agent, state_rm)
            reward = agent.get_reward_machine().get_reward(event)
            self.rewards[agent.name] += reward
            new_state_rm = agent.reward_machine.get_current_state()

            #self.walls_reward(agent, new_state)

            # Aggiorna la Q-table
            #q_learning = self.agents_q_learning[agent.name]
            q_learning = agent.get_learning_algorithm()
            agent_action = agent.actions_idx(action)
            q_learning.update(current_statee, new_state, agent_action, reward, agent, state_rm, new_state_rm)

        self.timestep += 1
        terminations, truncations, infos = self.check_terminations()
        observations = self.agent_states

        return observations, self.rewards, terminations, truncations, infos

    def walls_reward(self, agent, new_state):
        # Controlla se la nuova posizione è una cella di penalità
        agent_pos = (new_state[(agent.name, 'pos_x')], new_state[(agent.name, 'pos_y')])
        if agent_pos in self.penalty_cells:
            self.rewards[agent.name] += self.penalty_amount  # Applica la penalità

    def execute_agent_action(self, agent, action, current_state):

        key_x = (agent.name, 'pos_x')
        key_y = (agent.name, 'pos_y')
        x = current_state[key_x]
        y = current_state[key_y]
        current_location = self.get_location_by_coordinates(agent, x, y)

        new_location = self.update_coordinates(agent, action.name, current_location)

        #print(agent.name, action.name, (current_location, new_location),x, y) #self.current_state_env)

        #print(current_location, (x,y))
        """if new_location == None:
            appl = False
            new_location = current_location
        else:
            appl = self.seq_ag._is_applicable(agent, self.current_state_env, action, (current_location, new_location))


        #appl = self.seq_ag.is_applicable(self.current_state_env, action)

        #appl = self.seq_ag._is_applicable(agent, self.current_state_env, action, ( self.l33,  self.l34))

        #breakpoint()

        #appl = self.seq_ag._is_applicable(agent, self.current_state_env, action, (current_location, new_location))

            #breakpoint()
        #breakpoint()
        if appl:"""

        #breakpoint()
        #self.current_state_env = self.seq_ag.apply_unsafe(agent, self.current_state_env, action)
        if new_location != None:
            state = self.seq_ag._apply(self.ma_environment, agent, self.current_state_env, action, ( current_location, new_location))
        else:
            state = None

        if state != None:
            self.current_state_env = state

        #print(self.current_state_env, "aaaaaaaa")
        """for fluent in agent.fluents:
            fluent_key = (agent.name, fluent.name)
            self.agent_states[agent.name][fluent_key] = self.current_state_env.get_value(
                Dot(agent, fluent)).constant_value()"""
        ag_state = self.current_state_env.get_dot_values(agent, only_true_values=False)
        self.agent_states[agent.name].update(ag_state)

        # Raccogli i fluenti da rimuovere
        fluents_to_remove = [fluent for fluent, value in self.agent_states[agent.name].items() if value is False]
        # Rimuovi i fluenti raccolti dal dizionario
        for fluent in fluents_to_remove:
            del self.agent_states[agent.name][fluent]
        #self.update_agent_states(agent.name, ag_state)
        #print("\n\n ag_state", ag_state)
        #print("\n\n agent_states", self.agent_states[agent.name])



        #()



        # Controlla se tutte le precondizioni sono soddisfatte
        """if all(self.evaluate_precondition(agent, precondition) for precondition in action.preconditions):
            # Applica gli effetti dell'azione
            for effect in action.effects:
                fluent = effect.fluent.fluent()
                value = effect.value
                fluent_key = (agent.name, fluent)

                current_fnode = self.current_state[fluent_key]

                # Crea un'espressione per l'effetto
                if effect.kind == EffectKind.ASSIGN:
                    new_expression = value
                elif effect.kind == EffectKind.INCREASE:
                    new_expression = self._env.expression_manager.Plus(current_fnode, value)
                elif effect.kind == EffectKind.DECREASE:
                    new_expression = self._env.expression_manager.Minus(current_fnode, value)
                else:
                    raise NotImplementedError("Effetto non supportato")

                # Semplifica l'espressione per ottenere il risultato
                new_fnode = new_expression.simplify()

                # Aggiorna lo stato corrente
                self.current_state[fluent_key] = new_fnode
                #return self.current_state
                #print(action.preconditions, "\n", effect, effect.kind, "\n", current_state, self.timestep, "\n", self.current_state, self.timestep)
                #breakpoint()
        else:
            # Le precondizioni non sono soddisfatte, l'azione non viene eseguita
            pass

    def evaluate_precondition(self, agent, precondition):
        #print("oooooooooooooo", precondition)

        # Assicurati che la precondizione abbia esattamente due argomenti
        if len(precondition.args) != 2:
            raise ValueError("La precondizione non ha due argomenti")

        # Determina se il primo o il secondo argomento è il fluento


        if precondition.args[0].is_fluent_exp():
            fluent = precondition.args[0].fluent()
            value = precondition.args[1].constant_value()  # Assumi che sia una costante
            reversed_order = False
        elif precondition.args[1].is_fluent_exp():
            fluent = precondition.args[1].fluent()
            value = precondition.args[0].constant_value()  # Assumi che sia una costante
            reversed_order = True
        else:
            #print(precondition)
            breakpoint()


        # Ottieni il valore corrente del fluento dallo stato
        current_value = self.current_state[(agent.name, fluent)].constant_value()

        # Valuta la precondizione considerando l'ordine degli argomenti
        if precondition.is_lt():
            return current_value < value if not reversed_order else value < current_value
        elif precondition.is_le():
            return current_value <= value if not reversed_order else value <= current_value
        elif precondition.is_equals():
            return current_value == value
        else:
            # Gestisci altri tipi di precondizioni se necessario
            return False"""

    def update_agent_states(self, agent_name: str, ag_state):
        # Rimuovi i fluenti che sono diventati falsi o non sono presenti in ag_state
        for fluent, value in self.agent_states[agent_name].items():  # Usa list() per creare una copia perché modificherai il dizionario durante l'iterazione
            if fluent not in ag_state:
                del self.agent_states[agent_name][fluent]
        self.agent_states[agent_name].update(ag_state)
    def check_terminations(self):
        terminations = {a.name: False for a in self.agents}
        truncations = {a.name: False for a in self.agents}

        for agente in self.agents:
            rm = agente.get_reward_machine()
            rm_state = rm.get_current_state()
            if rm_state == rm.get_final_state():
                terminations[agente.name] = True
            """if rm_state == "completed":
                terminations[agente.name] = True"""

        if self.timestep > 1000:
            for a in self.agents:
                truncations[a.name] = True


        infos = {a.name: {} for a in self.agents}  # Info aggiuntive, se necessarie

        return terminations, truncations, infos



    """def get_state(self, agent):
        # Restituisce lo stato attuale senza trasformarlo
        agent_state = {}
        for k, v in self.current_state.items():
            if k[0] == agent.name:
                agent_state[k] = v

        stato = agent_state.copy()
        return stato"""

    #@functools.lru_cache(maxsize=128)

    def get_state(self, agent):
        # Restituisce una copia dello stato corrente dell'agente per evitare modifiche accidentali
        return self.agent_states[agent.name].copy()

    """def get_rm_state_index(self, rm_state):
        # Mappa ciascuno stato della RM a un indice unico
        state_mapping = {
            "start": 0,
            "at_pos1": 1,
            "at_pos2": 2,
            "completed": 3
            # Aggiungi qui altri stati RM se necessario
        }
        return state_mapping[rm_state]"""

    # Potresti dover implementare o aggiornare questa funzione

    def check_conditions(self, current_state_rm, conditions, next_state, reward):
        if conditions:
            print(conditions)
    def detect_event(self, ag, state_rm):
        # Recupera lo stato corrente dell'agente
        current_state_ = self.get_state(ag)
        RM = ag.get_reward_machine()
        transitions = RM.get_transitions

        # Usa direttamente le chiavi fluenti per accedere ai valori nello stato dell'agente
        #pos_x = current_state_[Dot(ag, ag.fluent('pos_x'))]
        #pos_y = current_state_[Dot(ag, ag.fluent('pos_y'))]
        pos_x_key = (ag.name, 'pos_x')  # Chiave per pos_x
        pos_y_key = (ag.name, 'pos_y')  # Chiave per pos_y

        #breakpoint()
        pos_x = current_state_[pos_x_key]
        pos_y = current_state_[pos_y_key]
        #print(ag.name, pos_x, pos_y)
        # Logica per determinare gli eventi basata sulla posizione corrente dell'
        #ita_x = self.agent_states['a1'][('a1', 'pos_x')]
        #ita_y = self.agent_states['a1'][('a1', 'pos_y')]
        #cat_x = self.agent_states['a3'][('a3', 'pos_x')]
        #cat_y = self.agent_states['a3'][('a3', 'pos_y')]
        #breakpoint()


        fluent = f'pos({self.get_location_by_coordinates(ag, pos_x, pos_y)})'
        current_location_map = (fluent, True)
        #print(ag.name, pos_x, pos_y,"current_location_map:", current_location_map)
        #print(ag.name, ag.messages)
        for (state, conditions), (next_state, reward) in transitions.items():
            if state != state_rm:
                continue

            """if 'X' in next_state:
                ag.message_conditions = conditions"""

            if 'X' in state_rm:
                list_agents_mess = self.extract_agent_ids(conditions)
                #ag._send_message(list_agents_mess, ag.message_conditions)
                #ag.message_sent = True
                dic_messaggi = ag.return_messages()
                if dic_messaggi is None:
                    dic_messaggi = {}  # Inizializza a un dizionario vuoto se None
                # Qui inizia la verifica delle condizioni basata sui messaggi ricevuti
                #conditions_satisfied = True
                # Verifica se le condizioni sono soddisfatte
                if dic_messaggi != {}:
                    condizioni_verificate = self.verifica_condizioni(ag, conditions, dic_messaggi, current_location_map)
                    if condizioni_verificate:
                        return conditions
            else:
                for condition in conditions:
                    fluent, bool = condition
                    if str(fluent) == current_location_map[0] and bool == current_location_map[1]:
                        return conditions



        #print(ag.name, ag.messages)

        """if 'X' in state_rm:
            # Cerca le condizioni che portano allo stato con 'X'
            self.extract_agent_ids(transitions)
            breakpoint()
            agents_to_notify = [cond[0][0] for cond in message_conditions if
                                isinstance(cond[0], tuple) and cond[0].startswith('a')]
            for (state, conditions), (next_state, _) in transitions.items():
                if next_state == state_rm:
                    message_conditions = conditions


                    for agent_to_notify in agents_to_notify:
                        self.send_event_to_agent(agent_to_notify, message_conditions)
                    breakpoint()
                    break  # Uscita dal ciclo dopo aver trovato e gestito lo stato 'X'
        else:
            # Gestione normale degli eventi se non ci troviamo in uno stato 'X'
            for key, value in transitions.items():
                state, conditions = key
                if state == state_rm:
                    next_state, reward = value
                    for condition in conditions:
                        fluent, bool = condition
                        if str(fluent) == current_location_map[0] and bool == current_location_map[1]:
                            return conditions
        return None  # Restituisce None se nessuna condizione corrisponde"""


    def send_message(self, ag, RM_ag):
        state_rm = RM_ag.get_current_state()
        transitions = RM_ag.get_transitions

        #fluent = f'pos({self.get_location_by_coordinates(ag, pos_x, pos_y)})'
        #current_location_map = (fluent, True)
        # print(ag.name, pos_x, pos_y,"current_location_map:", current_location_map)
        # print(ag.name, ag.messages)
        for (state, conditions), (next_state, reward) in transitions.items():
            if state != state_rm:
                continue

            if 'X' in next_state:
                ag.message_conditions = conditions

            if 'X' in state_rm:
                list_agents_mess = self.extract_agent_ids(conditions)
                ag._send_message(list_agents_mess, ag.message_conditions)
                ag.message_sent = True



    def verifica_condizioni(self, ag, conditions, dic_messaggi, current_location_map):
        # Verifica le condizioni inviate dagli altri agenti
        for condition in conditions:
            if isinstance(condition[0], tuple):  # La condizione coinvolge un altro agente
                agent_condition, fluent_condition = condition[0][0], condition[0][1]
                # Utilizza messaggi_semplificati per il confronto
                messaggio_chiave = (agent_condition, fluent_condition)
                if messaggio_chiave not in dic_messaggi or dic_messaggi[messaggio_chiave] != \
                        condition[1]:
                    return False  # La condizione non è soddisfatta
            else:  # La condizione è locale per l'agente
                fluent, value = condition
                if str(fluent) != current_location_map[0] or value != current_location_map[1]:
                    return False  # La condizione locale non è soddisfatta

        # Tutte le condizioni sono soddisfatte
        return True

    def extract_agent_ids(self, conditions):
        agent_ids = set()
        for cond in conditions:
            if isinstance(cond[0], tuple) and len(cond[0]) == 2 and isinstance(cond[0][0], str):
                agent_id = cond[0][0]  # Estrai l'identificatore dell'agente
                agent_ids.add(agent_id)
        return list(agent_ids)

    def broadcast_message(self, agents, message):
        """for agent in agents:
            breakpoint()
            self.agent(agent).receive_message(message)"""
        for agent in agents:
            if agent != message.sender:
                ag = self.agent(agent)
                ag._receive_message(message)

    def init_pygame(self):
        pygame.init()
        cell_size = 100
        self.frames = []

        # Carica e scala l'immagine per le posizioni
        self.colosseo = pygame.image.load("colosseo.png")
        self.colosseo = pygame.transform.scale(self.colosseo, (90, 90))

        self.piazza = pygame.image.load("piazza.png")
        self.piazza = pygame.transform.scale(self.piazza, (90, 90))

        self.bcn = pygame.image.load("bcn.png")
        self.bcn = pygame.transform.scale(self.bcn, (95, 95))

        self.madrid = pygame.image.load("mdn.png")
        self.madrid = pygame.transform.scale(self.madrid, (90, 90))

        self.battlo = pygame.image.load("battlo.png")
        self.battlo = pygame.transform.scale(self.battlo, (90, 90))

        self.piazza_di_spagna = pygame.image.load("piazza_di_spagna2.png")
        self.piazza_di_spagna = pygame.transform.scale(self.piazza_di_spagna, (95, 95))


        self.ita_man = pygame.image.load("ita_man.png")
        self.ita_man = pygame.transform.scale(self.ita_man,
                                                     (85, 85))  # Scala l'immagine alla dimensione desiderata

        self.bcn_man = pygame.image.load("bcn_man2.png")
        self.bcn_man = pygame.transform.scale(self.bcn_man,
                                                     (80, 80))  # Scala l'immagine alla dimensione desiderata
        self.CR7 = pygame.image.load("CR7.png")
        self.CR7 = pygame.transform.scale(self.CR7,
                                              (70, 70))  # Scala l'immagine alla dimensione desiderata

        self.juve = pygame.image.load("juve.png")
        self.juve = pygame.transform.scale(self.juve,
                                          (75, 75))  # Scala l'immagine alla dimensione desiderata

        # Carica l'immagine del ponte
        self.ponte_immagine = pygame.image.load('ponte_.png')
        self.ponte_immagine = pygame.transform.scale(self.ponte_immagine, (40, 40))
        self.barca_a_remi = pygame.image.load('barca_.png')
        self.barca_a_remi = pygame.transform.scale(self.barca_a_remi, (40, 40))

        screen_width = self.grid_width * cell_size
        screen_height = self.grid_height * cell_size
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        self.clock = pygame.time.Clock()

    def get_agent_image(self, agent, small=False):
        # Funzione per selezionare e, se necessario, ridimensionare l'immagine dell'agente
        if agent.name == 'a1':
            image = self.ita_man
        elif agent.name == 'a3':
            image = self.bcn_man
        elif agent.name == 'a4':
            image = self.CR7
        elif agent.name == 'a2':
            image = self.juve
        elif agent.name == 'a5':
            image = self.juve
        # Aggiungi qui altri agenti se necessario

        if small:
            return pygame.transform.scale(image, (image.get_width() // 2, image.get_height() // 2))
        else:
            return image
    def render(self, episode, state=None):
        cell_size = 100
        # Regola la velocità di aggiornamento dello schermo
        self.clock.tick(6000 if episode < 89998 else 60)
        self.screen.fill((255, 255, 255))


        # Disegna le linee della griglia
        # Gruppi di celle con coordinate (x, y)
        gruppo_1 = [self.position_A, self.position_B, self.position_C]
        gruppo_2 = [self.position_D, self.position_E, self.position_F]

        # Colori per i gruppi
        colore_gruppo_1 = (255, 0, 0)  # Rosso
        colore_gruppo_2 = (0, 0, 255)  # Blu

        # Disegna le linee della griglia
        for x in range(0, self.grid_width * cell_size, cell_size):
            for y in range(0, self.grid_height * cell_size, cell_size):
                # Determina il colore delle linee in base al gruppo di appartenenza
                cell_x, cell_y = x // cell_size, y // cell_size
                if (cell_x, cell_y) in gruppo_1:
                    colore_linea = colore_gruppo_1
                elif (cell_x, cell_y) in gruppo_2:
                    colore_linea = colore_gruppo_2
                else:
                    colore_linea = (0, 0, 0)  # Nero per le altre celle

                # Disegna le linee della cella
                pygame.draw.line(self.screen, colore_linea, (x, y), (x, y + cell_size))  # Linea verticale
                pygame.draw.line(self.screen, colore_linea, (x, y), (x + cell_size, y))  # Linea orizzontale


        #ponte
        self.draw_connection((0, 2), (0, 3), self.ponte_immagine, 100, 92, 100, 100)
        self.draw_connection((0, 3), (1, 3), self.barca_a_remi, 100, 100, 100, 100)

        # Disegno dei muri
        for wall_x, wall_y in self.walls:
            pygame.draw.rect(self.screen, (128, 128, 128),
                             (wall_x * cell_size, wall_y * cell_size, cell_size, cell_size))

        # Dizionario per tenere traccia delle posizioni degli agenti
        agent_positions = {}

        # Raccogliere informazioni sulla posizione per tutti gli agenti
        for ag in self.agents:
            agent_state = self.get_state(ag)
            pos_x, pos_y = agent_state[(ag.name, 'pos_x')], agent_state[(ag.name, 'pos_y')]
            position = (pos_x, pos_y)
            if position not in agent_positions:
                agent_positions[position] = []
            agent_positions[position].append(ag)

        # Disegnare gli agenti
        for position, agents_at_pos in agent_positions.items():
            if len(agents_at_pos) > 1:
                # Se più di un agente condivide la stessa posizione, ridimensiona e affianca le loro immagini
                for index, ag in enumerate(agents_at_pos):
                    agent_image = self.get_agent_image(ag,
                                                       small=True)  # Funzione per ottenere l'immagine ridimensionata
                    offset = (index * cell_size // len(agents_at_pos), 0)  # Calcola l'offset per affiancare le immagini
                    self.screen.blit(agent_image,
                                     (position[0] * cell_size + offset[0], position[1] * cell_size + offset[1]))
            else:
                # Se solo un agente occupa la posizione, usa la dimensione standard
                # Calcola l'offset per centrare l'immagine nella cella


                ag = agents_at_pos[0]
                agent_image = self.get_agent_image(ag, small=False)
                image_width, image_height = agent_image.get_width(), agent_image.get_height()
                base_x = position[0] * cell_size + (cell_size - image_width) // 2
                base_y = position[1] * cell_size + (cell_size - image_height) // 2
                self.screen.blit(agent_image, (base_x, base_y))



        #pos_colors = (0, 0, 100)
        #pos_color = pos_colors[i % len(pos_colors)]  # Cicla i colori se ci sono più guardie dei colori disponibili
        #pygame.draw.rect(self.screen, pos_colors, (self.position_A[0] * cell_size, self.position_A[1] * cell_size, cell_size, cell_size))
        #pygame.draw.rect(self.screen, pos_colors, (self.position_B[0] * cell_size, self.position_B[1] * cell_size, cell_size, cell_size))
        #pygame.draw.rect(self.screen, pos_colors, (self.position_C[0] * cell_size, self.position_C[1] * cell_size, cell_size, cell_size))

        #posizioni = [self.position_A, self.position_B, self.position_C]

        #pygame.draw.rect(self.screen, pos_colors,(pos[0] * cell_size, pos[1] * cell_size, cell_size, cell_size))
        self.screen.blit(self.colosseo, (self.position_A[0] * 101, self.position_A[1] * 101))
        self.screen.blit(self.piazza, (self.position_B[0] * 101, self.position_B[1] * 101))
        self.screen.blit(self.piazza_di_spagna, (self.position_C[0] * 101, self.position_C[1] * 100.5))

        self.screen.blit(self.madrid, (self.position_D[0] * 101, self.position_D[1] * 101))
        self.screen.blit(self.battlo, (self.position_E[0] * 101, self.position_E[1] * 101))
        self.screen.blit(self.bcn, (self.position_F[0] * 101, self.position_F[1] * 100.5))

        #self.screen.blit(self.prisoner_image, (pos_x * cell_size, pos_y * cell_size))

        # Disegno della via di fuga
        """pygame.draw.rect(self.screen, (0, 255, 0),
                         (self.escape_x * cell_size, self.escape_y * cell_size, cell_size, cell_size))"""

        pygame.display.flip()

        # if episode % 5000 == 0:
        image_data = pygame.surfarray.array3d(pygame.display.get_surface())
        image_data = image_data.transpose([1, 0, 2])
        self.frames.append(image_data)
    def save_episode(self, episode):

        if self.frames:
            video_path = f"episode_{episode}.avi"
            height, width, layers = self.frames[0].shape
            video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'DIVX'), 2, (width, height))

            for frame in self.frames:
                video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            cv2.destroyAllWindows()
            video.release()
            self.frames = []  # Pulisci la lista dei frames

    # Observation space should be defined here.
    # lru_cache allows observation and action spaces to be memoized, reducing clock cycles required to get each agent's space.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        RM_agent = agent.get_reward_machine()
        num_state = RM_agent.numbers_state()
        # Ora ci sono 2 agenti
        max_pos = self.grid_width * self.grid_height - 1 * num_state
        return MultiDiscrete([max_pos] * 2)  # 5 componenti: 1 a1, 2 guardie, 1 via di fuga

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        if agent == "a1" or agent.startswith("a3") or agent.startswith("a4") or agent.startswith("a2") or agent.startswith("a5"):
            return Discrete(9)  # Quattro possibili azioni per ogni agente
        else:
            return None  # Restituisce None per agenti non riconosciuti

    def initialize_location_mapping(self, coordinates):
        # Inizializza i dizionari per il mapping includendo l'agente
        self.coord_to_location_map = {}
        self.location_to_coord_map = {}

        for agent in self.agents:
            for coord, location_obj in coordinates:
                # Usa una tupla (agente, location) come chiave
                self.coord_to_location_map[(agent.name, coord)] = location_obj
                self.location_to_coord_map[(agent.name, location_obj)] = coord

    def get_location_by_coordinates(self, agent, x, y):
        # Usa l'agente e le coordinate per ottenere la location
        return self.coord_to_location_map.get((agent.name, (x, y)))

    def get_coordinates_by_location(self, agent, location):
        # Usa l'agente e la location per ottenere le coordinate
        return self.location_to_coord_map.get((agent.name, location))

    def update_coordinates(self, agent, action_name, current_location):
        # Ottieni le coordinate correnti basate sull'agente e la current_location
        current_coords = self.get_coordinates_by_location(agent, current_location)
        if current_coords is None:
            return None

        action_to_coord_change = {
            "up": (0, -1),
            "down": (0, +1),
            "left": (-1, 0),
            "right": (1, 0),
            "cross_up": (0, -1),
            "cross_down": (0, +1),
            "cross_left": (-1, 0),
            "cross_right": (1, 0),
        }

        dx, dy = action_to_coord_change.get(action_name, (0, 0))
        new_coords = (current_coords[0] + dx, current_coords[1] + dy)


        # Ottieni la nuova location basata sull'agente e le nuove coordinate
        return self.get_location_by_coordinates(agent, *new_coords)

    def draw_connection(self, cell1, cell2, bridge_image, cell1_size_x, cell1_size_y, cell2_size_x, cell2_size_y):
        # Dimensioni dell'immagine del ponte
        cell_size = 100
        bridge_width, bridge_height = bridge_image.get_size()

        # Calcolo delle coordinate centrali per entrambe le celle
        cell1_center_x = (cell1[0] + 0.5) * cell1_size_x
        cell1_center_y = (cell1[1] + 0.5) * cell1_size_y
        cell2_center_x = (cell2[0] + 0.5) * cell2_size_x
        cell2_center_y = (cell2[1] + 0.5) * cell2_size_y

        # Calcolo della posizione del ponte
        # Se le celle sono adiacenti verticalmente
        if cell1[0] == cell2[0]:
            bridge_x = min(cell1_center_x, cell2_center_x) - bridge_width / 2
            bridge_y = (cell1_center_y + cell2_center_y) / 2 - bridge_height / 2
        # Se le celle sono adiacenti orizzontalmente
        else:
            bridge_x = (cell1_center_x + cell2_center_x) / 2 - bridge_width / 2
            bridge_y = min(cell1_center_y, cell2_center_y) - bridge_height / 2

        # Disegno del ponte sulla superficie di Pygame
        self.screen.blit(bridge_image, (bridge_x, bridge_y))


#@profile
def main():

    import wandb
    from pettingzoo.test import parallel_api_test
    import random
    NUM_EPISODES = 2001  # Numero di partite da giocare per l'apprendimento
    #wandb.init(project='maze_RL', entity='alee8')
    env = MAP_RL_Env()
    env.init_pygame()

    a1 = AgentRL('a1', env)
    a2 = AgentRL('a2', env)
    a3 = AgentRL('a3', env)
    a4 = AgentRL('a4', env)
    a5 = AgentRL('a5', env)

    Location = UserType("Location")
    max_x_value = env.grid_width
    max_y_value = env.grid_height
    # Righe/Colonne
    """l11 = Object("l11", Location)
    l12 = Object("l12", Location)
    l13 = Object("l13", Location)
    l14 = Object("l14", Location)
    l21 = Object("l21", Location)
    l22 = Object("l22", Location)
    l23 = Object("l23", Location)
    l24 = Object("l24", Location)
    l31 = Object("l31", Location)
    l32 = Object("l32", Location)
    l33 = Object("l33", Location)
    l34 = Object("l34", Location)
    l41 = Object("l41", Location)
    l42 = Object("l42", Location)
    l43 = Object("l43", Location)
    l44 = Object("l44", Location)"""

    #prova:
    l11 = Object("l11", Location)
    l12 = Object("l12", Location)
    l13 = Object("l13", Location)
    l14 = Object("l14", Location)
    l15 = Object("l15", Location)
    l16 = Object("l16", Location)
    l17 = Object("l17", Location)
    l18 = Object("l18", Location)
    l19 = Object("l19", Location)
    l21 = Object("l21", Location)
    l22 = Object("l22", Location)
    l23 = Object("l23", Location)
    l24 = Object("l24", Location)
    l25 = Object("l25", Location)
    l26 = Object("l26", Location)
    l27 = Object("l27", Location)
    l28 = Object("l28", Location)
    l29 = Object("l29", Location)
    l31 = Object("l31", Location)
    l32 = Object("l32", Location)
    l33 = Object("l33", Location)
    l34 = Object("l34", Location)
    l35 = Object("l35", Location)
    l36 = Object("l36", Location)
    l37 = Object("l37", Location)
    l38 = Object("l38", Location)
    l39 = Object("l39", Location)
    l41 = Object("l41", Location)
    l42 = Object("l42", Location)
    l43 = Object("l43", Location)
    l44 = Object("l44", Location)
    l45 = Object("l45", Location)
    l46 = Object("l46", Location)
    l47 = Object("l47", Location)
    l48 = Object("l48", Location)
    l49 = Object("l49", Location)
    l51 = Object("l51", Location)
    l52 = Object("l52", Location)
    l53 = Object("l53", Location)
    l54 = Object("l54", Location)
    l55 = Object("l55", Location)
    l56 = Object("l56", Location)
    l57 = Object("l57", Location)
    l58 = Object("l58", Location)
    l59 = Object("l59", Location)
    l61 = Object("l61", Location)
    l62 = Object("l62", Location)
    l63 = Object("l63", Location)
    l64 = Object("l64", Location)
    l65 = Object("l65", Location)
    l66 = Object("l66", Location)
    l67 = Object("l67", Location)
    l68 = Object("l68", Location)
    l69 = Object("l69", Location)
    l71 = Object("l71", Location)
    l72 = Object("l72", Location)
    l73 = Object("l73", Location)
    l74 = Object("l74", Location)
    l75 = Object("l75", Location)
    l76 = Object("l76", Location)
    l77 = Object("l77", Location)
    l78 = Object("l78", Location)
    l79 = Object("l79", Location)
    l81 = Object("l81", Location)
    l82 = Object("l82", Location)
    l83 = Object("l83", Location)
    l84 = Object("l84", Location)
    l85 = Object("l85", Location)
    l86 = Object("l86", Location)
    l87 = Object("l87", Location)
    l88 = Object("l88", Location)
    l89 = Object("l89", Location)
    l91 = Object("l91", Location)
    l92 = Object("l92", Location)
    l93 = Object("l93", Location)
    l94 = Object("l94", Location)
    l95 = Object("l95", Location)
    l96 = Object("l96", Location)
    l97 = Object("l97", Location)
    l98 = Object("l98", Location)
    l99 = Object("l99", Location)
    l101 = Object("l101", Location)
    l102 = Object("l102", Location)
    l103 = Object("l103", Location)
    l104 = Object("l104", Location)
    l105 = Object("l105", Location)
    l106 = Object("l106", Location)
    l107 = Object("l107", Location)
    l108 = Object("l108", Location)
    l109 = Object("l109", Location)


    l10x = Object("l10x", Location)
    l20x = Object("l20x", Location)
    l30x = Object("l30x", Location)
    l40x = Object("l40x", Location)
    l50x = Object("l50x", Location)
    l60x = Object("l60x", Location)
    l70x = Object("l70x", Location)
    l80x = Object("l80x", Location)
    l90x = Object("l90x", Location)
    l100x = Object("l100x", Location)



    """Locations = [l11, l12, l13, l14, l15, l16, l17, l18, l19, l21, l22, l23, l24, l25, l26, l27, l28, l29, l31, l32,
                 l33, l34, l35, l36, l37, l38, l39, l41, l42, l43, l44, l45, l46, l47, l48, l49, l51, l52, l53, l54,
                 l55, l56, l57, l58, l59, l61, l62, l63, l64, l65, l66, l67, l68, l69, l71, l72, l73, l74, l75, l76,
                 l77, l78, l79, l81, l82, l83, l84, l85, l86, l87, l88, l89, l91, l92, l93, l94, l95, l96, l97, l98, l99,
                 l101, l102, l103, l104, l105, l106, l107, l108, l109,
                 l10x, l20x, l30x, l40x, l50x, l60x, l70x, l80x, l90x, l100x]"""
    #Locations = [l11, l12, l13, l14, l21, l22, l23, l24, l31, l32, l33, l34, l41, l42, l43, l44]


    """coordinates = [
        ((0, 0), l11), ((0, 1), l12), ((0, 2), l13), ((0, 3), l14),
        ((1, 0), l21), ((1, 1), l22), ((1, 2), l23), ((1, 3), l24),
        ((2, 0), l31), ((2, 1), l32), ((2, 2), l33), ((2, 3), l34),
        ((3, 0), l41), ((3, 1), l42), ((3, 2), l43), ((3, 3), l44)
    ]

    coordinates = [
        ((0, 0), l11), ((0, 1), l12), ((0, 2), l13), ((0, 3), l14), ((0, 4), l15), ((0, 5), l16), ((0, 6), l17),
        ((0, 7), l18), ((0, 8), l19),
        ((1, 0), l21), ((1, 1), l22), ((1, 2), l23), ((1, 3), l24), ((1, 4), l25), ((1, 5), l26), ((1, 6), l27),
        ((1, 7), l28), ((1, 8), l29),
        ((2, 0), l31), ((2, 1), l32), ((2, 2), l33), ((2, 3), l34), ((2, 4), l35), ((2, 5), l36), ((2, 6), l37),
        ((2, 7), l38), ((2, 8), l39),
        ((3, 0), l41), ((3, 1), l42), ((3, 2), l43), ((3, 3), l44), ((3, 4), l45), ((3, 5), l46), ((3, 6), l47),
        ((3, 7), l48), ((3, 8), l49),
        ((4, 0), l51), ((4, 1), l52), ((4, 2), l53), ((4, 3), l54), ((4, 4), l55), ((4, 5), l56), ((4, 6), l57),
        ((4, 7), l58), ((4, 8), l59),
        ((5, 0), l61), ((5, 1), l62), ((5, 2), l63), ((5, 3), l64), ((5, 4), l65), ((5, 5), l66), ((5, 6), l67),
        ((5, 7), l68), ((5, 8), l69),
        ((6, 0), l71), ((6, 1), l72), ((6, 2), l73), ((6, 3), l74), ((6, 4), l75), ((6, 5), l76), ((6, 6), l77),
        ((6, 7), l78), ((6, 8), l79),
        ((7, 0), l81), ((7, 1), l82), ((7, 2), l83), ((7, 3), l84), ((7, 4), l85), ((7, 5), l86), ((7, 6), l87),
        ((7, 7), l88), ((7, 8), l89),
        ((8, 0), l91), ((8, 1), l92), ((8, 2), l93), ((8, 3), l94), ((8, 4), l95), ((8, 5), l96), ((8, 6), l97),
        ((8, 7), l98), ((8, 8), l99),
        ((9, 0), l101), ((9, 1), l102), ((9, 2), l103), ((9, 3), l104), ((9, 4), l105), ((9, 5), l106), ((9, 6), l107),
        ((9, 7), l108), ((9, 8), l109),
        ((0, 9), l10x), ((1, 9), l20x), ((2, 9), l30x), ((3, 9), l40x), ((4, 9), l50x), ((5, 9), l60x), ((6, 9), l70x),
        ((7, 9), l80x), ((8, 9), l90x), ((9, 9), l100x)
    ]"""

    # Ottieni l'oggetto Location date le coordinate
    #location = env.get_location_by_coordinates(0, 0)  # Dovrebbe restituire l'oggetto per "l11"
    # Ottieni le coordinate date la location
    #coordinates = env.get_coordinates_by_location(location)  # Dovrebbe restituire (0, 0)
    def generate_grid_locations_and_coordinates(grid_size):
        # Define the object type for locations
        Location = UserType("Location")

        locations = []  # List to track created locations
        coordinates = []  # List to track coordinates and corresponding locations

        # Loop through grid rows and columns to generate locations and coordinates
        for row in range(1, grid_size + 1):
            for col in range(1, grid_size + 1):
                # Create location name based on row and column
                location_name = f"l{row}{col}"

                # Create a Location object with the generated name
                location = Object(location_name, Location)

                # Add the Location object to the list of locations
                locations.append(location)

                # Add the coordinate-location pair to the list of coordinates
                coordinates.append(((row - 1, col - 1), location))

        return locations, coordinates

    # Example usage with a 4x4 grid
    grid_size = 10
    locations, coordinates = generate_grid_locations_and_coordinates(grid_size)
    env.add_objects(locations)
    # Print generated locations and coordinates
    """for location in locations:
        print(location)

    for coordinate in coordinates:
        print(coordinate)"""

    def connect_locations(locations, grid_size):
        """
        Connects locations in a grid to their neighboring locations.

        Args:
        - locations: List of location objects.
        - grid_size: Size of one dimension of the grid (assuming a square grid).

        Returns:
        - connections: List of tuples representing connected locations.
        """
        connections = []

        # Dictionary to map location names to location objects
        location_dict = {str(location): location for location in locations}

        for row in range(1, grid_size + 1):
            for col in range(1, grid_size + 1):
                current_location_name = f"l{row}{col}"
                current_location = location_dict.get(current_location_name)

                # Connect to the right
                if col < grid_size:
                    right_location_name = f"l{row}{col + 1}"
                    right_location = location_dict.get(right_location_name)
                    connections.append((current_location, right_location))

                # Connect downwards
                if row < grid_size:
                    down_location_name = f"l{row + 1}{col}"
                    down_location = location_dict.get(down_location_name)
                    connections.append((current_location, down_location))

        return connections

    connections_ = connect_locations(locations, 10)


    pos = Fluent("pos", BoolType(), pos=Location)
    pos_x = Fluent("pos_x", RealType(0, ))
    pos_y = Fluent("pos_y", RealType(0, ))
    a1.add_public_fluent(pos_x)
    a1.add_public_fluent(pos_y)
    a1.add_public_fluent(pos, default_initial_value=False)
    env.add_agent(a1)
    env.set_initial_value(Dot(a1, pos_x), 3) #83
    env.set_initial_value(Dot(a1, pos_y), 0)
    #env.set_initial_value(Dot(a1, pos(l33)), False)
    #env.set_initial_value(Dot(a1, pos(l34)), True)

    a2.add_public_fluent(pos_x)
    a2.add_public_fluent(pos_y)
    a2.add_public_fluent(pos, default_initial_value=False)
    env.add_agent(a2)
    env.set_initial_value(Dot(a2, pos_x), 2) #77
    env.set_initial_value(Dot(a2, pos_y), 2)

    a3.add_public_fluent(pos_x)
    a3.add_public_fluent(pos_y)
    a3.add_public_fluent(pos, default_initial_value=False)
    env.add_agent(a3)
    env.set_initial_value(Dot(a3, pos_x), 1) #10
    env.set_initial_value(Dot(a3, pos_y), 0)
    #env.set_initial_value(Dot(a3, pos(l34)), True)
    #env.set_initial_value(Dot(a3, pos(l33)), False)

    a4.add_public_fluent(pos_x)
    a4.add_public_fluent(pos_y)
    a4.add_public_fluent(pos, default_initial_value=False)
    env.add_agent(a4)
    env.set_initial_value(Dot(a4, pos_x), 0) #00
    env.set_initial_value(Dot(a4, pos_y), 0)

    a5.add_public_fluent(pos_x)
    a5.add_public_fluent(pos_y)
    a5.add_public_fluent(pos, default_initial_value=False)
    env.add_agent(a5)
    env.set_initial_value(Dot(a5, pos_x), 1) #98
    env.set_initial_value(Dot(a5, pos_y), 3)

    env.initialize_location_mapping(coordinates)
    # s1 = Object("s1", button)
    # s2 = Object("s2", button)
    # s3 = Object("s3", button)
    # d1 = Object("d1", door)
    # d2 = Object("d2", door)
    # d3 = Object("d3", door)
    # br1 = Object("br1", bridge)
    # br2 = Object("br2", bridge)
    # br3 = Object("br3", bridge)
    # bo1 = Object("bo1", boat)
    # bo2 = Object("bo2", boat)
    # bo3 = Object("bo3", boat)


    """connections = [
        (l11, l12), (l12, l13),  # (l13, l14), -> bridge
        (l21, l22), (l22, l23),  # (l23, l24), -> bridge
        (l31, l32), (l33, l34),  # (l32, l33), -> bridge
        (l42, l43),  # (l43, l44), ->door (l41, l42),->boat

        (l11, l21), (l21, l31), (l31, l41),
        (l12, l22), (l22, l32), (l32, l42),
        (l23, l33), (l33, l43),
        # (l14, l24), (l34, l44) -> boats
    ]"""

    connections = []


    is_connected = Fluent("is_connected", BoolType(), l1=Location, l2=Location)
    for connection in connections_:
        #print(connection)
        env.set_initial_value(is_connected(connection[0], connection[1]), True)
        env.set_initial_value(is_connected(connection[1], connection[0]), True)
    #breakpoint()
    """for loc in Locations:
        if loc != l34:
            env.set_initial_value(Dot(a1, pos(loc)), False)
            env.set_initial_value(Dot(a3, pos(loc)), False)"""

    bridge = UserType("bridge")
    br1 = Object("br1", bridge)
    br2 = Object("br2", bridge)
    br3 = Object("br3", bridge)
    env.add_object(br1)
    env.add_object(br2)
    env.add_object(br3)
    has_bridge = Fluent("has_bridge", BoolType(), connect_from=Location, connect_to=Location)
    has_boat = Fluent("has_boat", BoolType(), connect_from=Location, connect_to=Location)
    env.ma_environment.add_fluent(has_bridge, default_initial_value=False)
    env.ma_environment.add_fluent(has_boat, default_initial_value=False)

    #Setto i ponti
    env.set_initial_value(has_bridge(l13, l14), True)
    env.set_initial_value(has_bridge(l14, l13), True)
    env.set_initial_value(is_connected(l13, l14), False)
    env.set_initial_value(is_connected(l14, l13), False)

    env.set_initial_value(has_boat(l14, l24), True)
    env.set_initial_value(has_boat(l24, l24), True)
    env.set_initial_value(is_connected(l14, l24), False)
    env.set_initial_value(is_connected(l24, l14), False)

    #10x10 griglia:
    env.set_initial_value(is_connected(l14, l15), False)
    env.set_initial_value(is_connected(l15, l14), False)

    """env.set_initial_value(has_bridge(l32, l33), True)
    env.set_initial_value(has_bridge(l33, l32), True)
    env.set_initial_value(is_connected(l32, l33), False)
    env.set_initial_value(is_connected(l33, l32), False)"""


    """l99 = env.get_location_by_coordinates(a1, 8, 8)
    l910 = env.get_location_by_coordinates(a1, 8, 9)
    env.set_initial_value(has_bridge(l99, l910), True)
    env.set_initial_value(has_bridge(l910, l99), True)
    l1010 = env.get_location_by_coordinates(a1, 9, 9)
    env.set_initial_value(is_connected(l910, l1010), False)
    env.set_initial_value(is_connected(l1010, l910), False)
    l810 = env.get_location_by_coordinates(a1, 7, 9)
    env.set_initial_value(is_connected(l810, l910), False)
    env.set_initial_value(is_connected(l910, l810), False)
    env.set_initial_value(is_connected(l99, l910), False)
    env.set_initial_value(is_connected(l910, l99), False)"""




    env.ma_environment.add_fluent(is_connected, default_initial_value=False)
    # Azione move_down
    move_up = InstantaneousAction("up", l_from=Location, l_to=Location)
    l_from = move_up.parameter("l_from")
    l_to = move_up.parameter("l_to")
    move_up.add_precondition(LT(0, pos_y))  # Precondizione: pos_y > 0
    move_up.add_precondition(is_connected(l_from, l_to))
    #move_up.add_precondition(pos(l_from))
    move_up.add_decrease_effect(pos_y, 1)
    move_up.add_effect(pos(l_to), True)
    move_up.add_effect(pos(l_from), False)
    # move_down.add_effect(pos_y, Minus(pos_y, 1))  # Effetto: decrementa pos_y di 1
    a1.add_rl_action(move_up)
    a2.add_rl_action(move_up)
    a3.add_rl_action(move_up)
    a4.add_rl_action(move_up)
    a5.add_rl_action(move_up)

    # Azione move_up
    move_down = InstantaneousAction("down", l_from=Location, l_to=Location)
    move_down.add_precondition(LT(pos_y, max_y_value - 1))  # Precondizione: pos_y < max_y_value
    move_down.add_precondition(is_connected(l_from, l_to))
   # move_down.add_precondition(pos(l_from))
    move_down.add_increase_effect(pos_y, 1)
    move_down.add_effect(pos(l_to), True)
    move_down.add_effect(pos(l_from), False)
    # move_up.add_effect(pos_y, Plus(pos_y, 1))  # Effetto: incrementa pos_y di 1
    a1.add_rl_action(move_down)
    a2.add_rl_action(move_down)
    a3.add_rl_action(move_down)
    a4.add_rl_action(move_down)
    a5.add_rl_action(move_down)

    # Azione move_left
    move_left = InstantaneousAction("left", l_from=Location, l_to=Location)
    move_left.add_precondition(LT(0, pos_x))  # Precondizione: pos_x > 0
    move_left.add_precondition(is_connected(l_from, l_to))
    move_left.add_decrease_effect(pos_x, 1)
    move_left.add_precondition(pos(l_from))
    move_left.add_effect(pos(l_to), True)
    move_left.add_effect(pos(l_from), False)
    # move_left.add_effect(pos_x, Minus(pos_x, 1))  # Effetto: decrementa pos_x di 1
    a1.add_rl_action(move_left)
    a2.add_rl_action(move_left)
    a3.add_rl_action(move_left)
    a4.add_rl_action(move_left)
    a5.add_rl_action(move_left)

    # Azione move_right
    move_right = InstantaneousAction("right", l_from=Location, l_to=Location)
    move_right.add_precondition(LT(pos_x, max_x_value - 1))  # Precondizione: pos_x < max_x_value
    move_right.add_precondition(is_connected(l_from, l_to))
    move_right.add_increase_effect(pos_x, 1)
    move_right.add_precondition(pos(l_from))
    move_right.add_effect(pos(l_to), True)
    move_right.add_effect(pos(l_from), False)
    # move_right.add_effect(pos_x, Plus(pos_x, 1))  # Effetto: incrementa pos_x di 1
    a1.add_rl_action(move_right)
    a2.add_rl_action(move_right)
    a3.add_rl_action(move_right)
    a4.add_rl_action(move_right)
    a5.add_rl_action(move_right)

    cross_up = InstantaneousAction("cross_up", l_from=Location, l_to=Location)
    cross_up.add_precondition(LT(0, pos_y))
    cross_up.add_precondition(has_bridge(l_from, l_to))
    cross_up.add_decrease_effect(pos_y, 1)
    #cross_up.add_effect(has_bridge(l_from, l_to), False)
    #cross_up.add_effect(pos_x, env.get_coordinates_by_location(a1, l_to)[0], True)
    #cross_up.add_effect(pos_y, env.get_coordinates_by_location(a1, l_to)[1], True)

    cross_down = InstantaneousAction("cross_down", l_from=Location, l_to=Location)
    cross_down.add_precondition(LT(pos_y, max_y_value - 1))
    cross_down.add_precondition(has_bridge(l_from, l_to))
    cross_down.add_increase_effect(pos_y, 1)
    #cross_down.add_effect(has_bridge(l_from, l_to), False)

    cross_right = InstantaneousAction("cross_right", l_from=Location, l_to=Location)
    cross_right.add_precondition(LT(pos_x, max_x_value - 1))
    cross_right.add_precondition(has_bridge(l_from, l_to))
    cross_right.add_increase_effect(pos_x, 1)
    #cross_right.add_effect(has_bridge(l_from, l_to), False)

    cross_left = InstantaneousAction("cross_left", l_from=Location, l_to=Location)
    cross_left.add_precondition(LT(0, pos_x))
    cross_left.add_precondition(has_bridge(l_from, l_to))
    cross_left.add_decrease_effect(pos_x, 1)
    #cross_left.add_effect(has_bridge(l_from, l_to), False)

    wait = InstantaneousAction("wait", l_from=Location, l_to=Location)
    wait.add_decrease_effect(pos_x, 0)

    row_up = InstantaneousAction("row_up", l_from=Location, l_to=Location)
    row_up.add_precondition(LT(0, pos_y))
    row_up.add_precondition(has_boat(l_from, l_to))
    row_up.add_decrease_effect(pos_y, 1)

    row_down = InstantaneousAction("row_down", l_from=Location, l_to=Location)
    row_down.add_precondition(LT(pos_y, max_y_value - 1))
    row_down.add_precondition(has_boat(l_from, l_to))
    row_down.add_increase_effect(pos_y, 1)

    row_right = InstantaneousAction("row_right", l_from=Location, l_to=Location)
    row_right.add_precondition(LT(pos_x, max_x_value - 1))
    row_right.add_precondition(has_boat(l_from, l_to))
    row_right.add_increase_effect(pos_x, 1)

    row_left = InstantaneousAction("row_left", l_from=Location, l_to=Location)
    row_left.add_precondition(LT(0, pos_x))
    row_left.add_precondition(has_boat(l_from, l_to))
    row_left.add_decrease_effect(pos_x, 1)

    a1.add_rl_action(cross_up)
    a1.add_rl_action(cross_down)
    a1.add_rl_action(cross_right)
    a1.add_rl_action(cross_left)
    a1.add_rl_action(wait)
    a1.add_rl_action(row_up)
    a1.add_rl_action(row_down)
    a1.add_rl_action(row_right)
    a1.add_rl_action(row_left)

    a2.add_rl_action(cross_up)
    a2.add_rl_action(cross_down)
    a2.add_rl_action(cross_right)
    a2.add_rl_action(cross_left)
    a2.add_rl_action(wait)
    a2.add_rl_action(row_up)
    a2.add_rl_action(row_down)
    a2.add_rl_action(row_right)
    a2.add_rl_action(row_left)

    a3.add_rl_action(cross_up)
    a3.add_rl_action(cross_down)
    a3.add_rl_action(cross_right)
    a3.add_rl_action(cross_left)
    a3.add_rl_action(wait)
    a3.add_rl_action(row_up)
    a3.add_rl_action(row_down)
    a3.add_rl_action(row_right)
    a3.add_rl_action(row_left)

    a4.add_rl_action(cross_up)
    a4.add_rl_action(cross_down)
    a4.add_rl_action(cross_right)
    a4.add_rl_action(cross_left)
    a4.add_rl_action(wait)
    a4.add_rl_action(row_up)
    a4.add_rl_action(row_down)
    a4.add_rl_action(row_right)
    a4.add_rl_action(row_left)

    a5.add_rl_action(cross_up)
    a5.add_rl_action(cross_down)
    a5.add_rl_action(cross_right)
    a5.add_rl_action(cross_left)
    a5.add_rl_action(wait)
    a5.add_rl_action(row_up)
    a5.add_rl_action(row_down)
    a5.add_rl_action(row_right)
    a5.add_rl_action(row_left)
    


    # Reward Machines
    initial_state = "state1"
    reward_for_pos1 = 5
    reward_for_pos2 = 10  # Puoi scegliere ricompense diverse se necessario
    reward_for_pos3 = 30

    """prisoner_transitions = {
        ("state1", "reached_A"): ("state1X", reward_for_pos1),
        ("state1X", "reached_B"): ("state2", reward_for_pos2),
        ("state2", "reached_C"): ("completed", reward_for_pos3)
    }"""
    prisoner_transitions = {
        ("state1", ((3, 1))): ("state1X", reward_for_pos1),
        ("state1X", ((3, 1), ('a3:', (8, 9)))): ("state2", reward_for_pos2),
        ("state2", "reached_C"): ("completed", reward_for_pos3)
    }
    # RM complete
    #transitions_ag_1 = RM_dict_true['a1']
    #transitions_ag_3 = RM_dict_true['a3']
    #transitions_ag_4 = RM_dict_true['a4']

    #Sequenza di azioni concorrenti
    transitions_ag_1 = RM_dict_true_seq['a1']
    transitions_ag_2 = RM_dict_true_seq['a2']
    transitions_ag_3 = RM_dict_true_seq['a3']
    transitions_ag_4 = RM_dict_true_seq['a4']
    transitions_ag_5 = RM_dict_true_seq['a5']

    x_A, y_A = 3, 0
    x_B, y_B = 8, 9
    x_C, y_C = 4, 7
    position_A = (x_A, y_A)
    position_B = (x_B, y_B)
    position_C = (x_C, y_C)

    x_D, y_D = 8, 1
    x_E, y_E = 7, 5
    x_F, y_F = 1, 8
    position_D = (x_D, y_D)
    position_E = (x_E, y_E)
    position_F = (x_F, y_F)

    #RM_1 = RewardMachine(initial_state, prisoner_transitions)
    #RM_3 = RewardMachine(initial_state, prisoner_transitions)
    RM_1 = RewardMachine(transitions_ag_1)
    RM_2 = RewardMachine(transitions_ag_2)
    RM_3 = RewardMachine(transitions_ag_3)
    RM_4 = RewardMachine(transitions_ag_4)
    RM_5 = RewardMachine(transitions_ag_5)

    a1.set_reward_machine(RM_1)
    a2.set_reward_machine(RM_2)
    a3.set_reward_machine(RM_3)
    a4.set_reward_machine(RM_4)
    a5.set_reward_machine(RM_5)


    q_learning1 = Q_learning(
        state_space_size=env.grid_width * env.grid_height * RM_1.numbers_state(), #env.num_rm_states,
        action_space_size=13,
        max_x_value=env.grid_width,
        max_y_value=env.grid_height,
        learning_rate=0.1,
        gamma=0.9,
    )

    q_learning2 = Q_learning(
        state_space_size=env.grid_width * env.grid_height * RM_2.numbers_state(),  # env.num_rm_states,
        action_space_size=13,
        max_x_value=env.grid_width,
        max_y_value=env.grid_height,
        learning_rate=0.1,
        gamma=0.9,
    )

    q_learning3 = Q_learning(
        state_space_size=env.grid_width * env.grid_height * RM_3.numbers_state(), #env.num_rm_states,
        action_space_size=13,
        max_x_value=env.grid_width,
        max_y_value=env.grid_height,
        learning_rate=0.1,
        gamma=0.9,
    )

    q_learning4 = Q_learning(
        state_space_size=env.grid_width * env.grid_height * RM_4.numbers_state(),  # env.num_rm_states,
        action_space_size=13,
        max_x_value=env.grid_width,
        max_y_value=env.grid_height,
        learning_rate=0.1,
        gamma=0.9,
    )

    q_learning5 = Q_learning(
        state_space_size=env.grid_width * env.grid_height * RM_5.numbers_state(),  # env.num_rm_states,
        action_space_size=13,
        max_x_value=env.grid_width,
        max_y_value=env.grid_height,
        learning_rate=0.1,
        gamma=0.9,
    )

    a1.set_learning_algorithm(q_learning1)
    a2.set_learning_algorithm(q_learning2)
    a3.set_learning_algorithm(q_learning3)
    a4.set_learning_algorithm(q_learning4)
    a5.set_learning_algorithm(q_learning5)

    # from unified_planning.engines.compilers import Grounder, GrounderHelper

    # _grounder = GrounderHelper(env)

    # env.initialize_agents_q_learning()
    actions_log = {}
    q_tables = {}
    env.initialize_state()
    for episode in range(NUM_EPISODES):
        obs, infos = env.reset()
        done = {a.name: False for a in env.agents}
        rewards = {a.name: 0 for a in env.agents}  # Inizializza le ricompense episodiche
        record_episode = episode % 1000 == 0
        #record_episode = False
        if record_episode:
            env.render(episode, obs)  # Cattura frame durante l'episodio
            actions_log = {agent.name: [] for agent in env.agents}

        while not all(done.values()):
            actions = {}
            for ag in env.agents:

                current_state = env.get_state(ag)
                RM_agent = ag.get_reward_machine()
                rm_current_state = RM_agent.get_current_state()
                env.send_message(ag, RM_agent)

                # action_index = env.agents_q_learning[ag.name].choose_action(current_state, env.epsilon, ag, rm_current_state)
                action_index = ag.get_learning_algorithm().choose_action(current_state, env.epsilon, ag,
                                                                         rm_current_state)
                actions[ag.name] = ag.actions_dix()[action_index]

                # Log delle azioni nell'ultimo episodio
                if record_episode:
                    actions_log[ag.name].append(actions[ag.name].name)

                # grounded_act = _grounder.ground_action(actions[ag.name], actions[ag.name].parameters)
                # appl = seq_ag._is_applicable(ag, current_state_env, actions[ag.name], (l42, l43))
                # if appl:
                #    current_state_env = seq_ag._apply(ag, current_state_env, actions[ag.name], (l42, l43))
                # print("Action:", actions[ag.name], appl)
                # print(env.get_state(ag))
                # breakpoint()

            obs, rewards, done, truncations, infos = env.step(actions)
            # print(obs)
            # breakpoint()
            #env.render(episode, obs) #commentare x training
            if record_episode:
                env.render(episode, obs)  # Cattura frame durante l'episodio
                #breakpoint()

            if all(truncations.values()):
                break
        if record_episode:
            env.save_episode(episode)  # Salva il video solo alla fine dell'episodio

        # Salva la Q-table all'ultimo episodio
        if record_episode:
            # Usa un dizionario per raccogliere tutte le Q-table
            q_tables_dict = {}
            for ag in env.agents:
                q_table = ag.get_learning_algorithm().q_table
                q_tables_dict[f'q_table_{ag.name}'] = q_table
            # Salva tutte le Q-table in un file .npz compresso
            np.savez_compressed('q_tables.npz', **q_tables_dict)

        #wandb.log({**rewards, 'epsilon': env.epsilon, 'episode': episode, 'step': env.timestep})
        print(f"Episodio {episode + 1}: Ricompensa = {rewards}, Step: {env.timestep}, Epsilon = {env.epsilon} ")

    # Salva il log delle azioni e le Q-table in un file JSON
    with open("final_episode_log.json", "w") as f:
        json.dump({"actions_log": actions_log, "q_tables": q_tables}, f, indent=4)

if __name__ == "__main__":
    main()
    """cProfile.run('main()', 'output_filename_2.prof')
    import pstats

    # Sostituisci 'output_filename.prof' con il percorso del tuo file .prof
    p = pstats.Stats('output_filename_2.prof')
    p.strip_dirs().sort_stats('cumulative').print_stats(100)  # Stampa le prime 10 righe"""

#Carica Q_table:
#data = np.load('q_tables.npz')
#q_table_a1 = data['q_table_a1']  # Ad esempio, per accedere alla Q-table dell'agente 'a1'