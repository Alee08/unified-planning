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
        self.grid_width = 10  # 10 celle di larghezza
        self.grid_height = 10  #  10 celle di altezza
        # Reimposta epsilon all'inizio di ogni episodio
        self.epsilon_start = 1.0  # Alto valore iniziale per maggiore esplorazione
        self.epsilon_end = 0.01  # Valore finale basso per maggiore sfruttamento
        self.epsilon_decay = 0.99995  # Tasso di riduzione di epsilon
        self.epsilon = self.epsilon_start  # Inizializza epsilon con il valore iniziale
        self.rewards = 0
        self.current_state = None
        self.position_A = (3, 0)
        self.position_B = (8, 9)
        self.position_C = (4, 7)
        self.position_D = (8, 1)
        self.position_E = (7, 5)
        self.position_F = (1, 8)
        self.new_state = None
        self.num_rm_states = 4  # Aggiorna questo valore in base al tuo specifico caso


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
        for agent in self.agents:
            # Reset della RewardMachine dell'agente
            agent.get_reward_machine().reset_to_initial_state()
            for fluente in agent.fluents:
                chiave = (agent.name, fluente)
                self.current_state[chiave] = self.initial_values[Dot(agent, fluente)]

        observations = []
        # Get dummy infos
        infos = {agent: {} for agent in self.agents}


        return observations, infos

    def step(self, actions):
        terminations, truncations, infos = {}, {}, {}
        #import copy

        for agent in self.agents:
            current_statee = self.get_state(agent)
            action = actions[agent.name]
            self.execute_agent_action(agent, action)
            new_state = self.get_state(agent)

            # Aggiorna la ricompensa e lo stato della Reward Machine
            state_rm = agent.reward_machine.get_current_state()
            event = self.detect_event(agent, state_rm)
            reward = agent.get_reward_machine().get_reward(event)
            self.rewards[agent.name] += reward #-1
            new_state_rm = agent.reward_machine.get_current_state()

            # Aggiorna la Q-table
            #q_learning = self.agents_q_learning[agent.name]
            q_learning = agent.get_learning_algorithm()
            agent_action = agent.actions_idx(action)
            q_learning.update(current_statee, new_state, agent_action, reward, agent, state_rm, new_state_rm)

        terminations, truncations, infos = self.check_terminations()
        observations = self.current_state

        return observations, self.rewards, terminations, truncations, infos



    def execute_agent_action(self, agent, action):

        # Controlla se tutte le precondizioni sono soddisfatte
        if all(self.evaluate_precondition(agent, precondition) for precondition in action.preconditions):
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
        else:
            fluent = precondition.args[1].fluent()
            value = precondition.args[0].constant_value()  # Assumi che sia una costante
            reversed_order = True

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
            return False

    def check_terminations(self):
        terminations = {a.name: False for a in self.agents}
        truncations = {a.name: False for a in self.agents}

        for agente in self.agents:

            rm_state = agente.get_reward_machine().get_current_state()
            if rm_state == "completed":
                terminations[agente.name] = True

        if self.timestep > 1000:
            for a in self.agents:
                truncations[a.name] = True
        self.timestep += 1

        infos = {a.name: {} for a in self.agents}  # Info aggiuntive, se necessarie

        return terminations, truncations, infos



    def get_state(self, agent):
        # Restituisce lo stato attuale senza trasformarlo
        agent_state = {}
        for k, v in self.current_state.items():
            if k[0] == agent.name:
                agent_state[k] = v

        stato = agent_state.copy()
        return stato

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
    def detect_event(self, ag, state_rm):

        chiave_1 = (ag.name, ag.fluent('pos_x'))
        chiave_2 = (ag.name, ag.fluent('pos_y'))
        current_state_ = self.get_state(ag)
        pos_x = current_state_[chiave_1].constant_value()
        pos_y = current_state_[chiave_2].constant_value()
        #print((pos_x, pos_y), "ooooo", ag.name)
        if ag.name == "italiano":
            if state_rm == "start" and (pos_x, pos_y) == self.position_A:
                return "reached_A"
            elif state_rm == "at_pos1" and (pos_x, pos_y) == self.position_B:
                return "reached_B"
            elif state_rm == "at_pos2" and (pos_x, pos_y) == self.position_C:
                return "reached_C"
        else:
            if state_rm == "start" and (pos_x, pos_y) == self.position_D:
                return "reached_A"
            elif state_rm == "at_pos1" and (pos_x, pos_y) == self.position_E:
                return "reached_B"
            elif state_rm == "at_pos2" and (pos_x, pos_y) == self.position_F:
                return "reached_C"

    """def calculate_reward(self):
        # Verifica se il italiano è stato catturato da una delle guardie
        for i in range(self.num_guards):
            guard_x, guard_y = self.guard_x[i], self.guard_y[i]
            if self.prisoner_x == guard_x and self.prisoner_y == guard_y:
                return -1  # Ricompensa negativa per essere stato catturato

        #self.rewards -= 0.01  # Penalità per ogni step
        return self.rewards  # Nessuna ricompensa se nessuna delle condizioni è verificata"""

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
                                                     (cell_size, cell_size))  # Scala l'immagine alla dimensione desiderata

        self.bcn_man = pygame.image.load("bcn_man2.png")
        self.bcn_man = pygame.transform.scale(self.bcn_man,
                                                     (92, 92))  # Scala l'immagine alla dimensione desiderata

        screen_width = self.grid_width * cell_size
        screen_height = self.grid_height * cell_size
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        self.clock = pygame.time.Clock()

    def render(self, episode, state=None):
        cell_size = 100
        # Regola la velocità di aggiornamento dello schermo
        self.clock.tick(600 if episode < 89998 else 60)
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




        # Disegno dei muri
        for wall_x, wall_y in self.walls:
            pygame.draw.rect(self.screen, (128, 128, 128),
                             (wall_x * cell_size, wall_y * cell_size, cell_size, cell_size))



        a1 = self.agents[0]
        chiave_1 = (a1.name, a1.fluent('pos_x'))
        chiave_2 = (a1.name, a1.fluent('pos_y'))
        pos_x_a1 = self.current_state[chiave_1].constant_value()
        pos_y_a1 = self.current_state[chiave_2].constant_value()
        self.screen.blit(self.ita_man, (pos_x_a1 * cell_size, pos_y_a1 * cell_size))

        a2 = self.agents[1]
        chiave_1 = (a2.name, a2.fluent('pos_x'))
        chiave_2 = (a2.name, a2.fluent('pos_y'))
        pos_x_a2 = self.current_state[chiave_1].constant_value()
        pos_y_a2 = self.current_state[chiave_2].constant_value()
        self.screen.blit(self.bcn_man, (pos_x_a2 * 101, pos_y_a2 * 101))

        #pos_colors = (0, 0, 100)
        #pos_color = pos_colors[i % len(pos_colors)]  # Cicla i colori se ci sono più guardie dei colori disponibili
        #pygame.draw.rect(self.screen, pos_colors, (self.position_A[0] * cell_size, self.position_A[1] * cell_size, cell_size, cell_size))
        #pygame.draw.rect(self.screen, pos_colors, (self.position_B[0] * cell_size, self.position_B[1] * cell_size, cell_size, cell_size))
        #pygame.draw.rect(self.screen, pos_colors, (self.position_C[0] * cell_size, self.position_C[1] * cell_size, cell_size, cell_size))

        #posizioni = [self.position_A, self.position_B, self.position_C]

        #pygame.draw.rect(self.screen, pos_colors,(pos[0] * cell_size, pos[1] * cell_size, cell_size, cell_size))
        self.screen.blit(self.colosseo, (position_A[0] * 101, position_A[1] * 101))
        self.screen.blit(self.piazza, (position_B[0] * 101, position_B[1] * 101))
        self.screen.blit(self.piazza_di_spagna, (position_C[0] * 101, position_C[1] * 100.5))

        self.screen.blit(self.madrid, (position_D[0] * 101, position_D[1] * 101))
        self.screen.blit(self.battlo, (position_E[0] * 101, position_E[1] * 101))
        self.screen.blit(self.bcn, (position_F[0] * 101, position_F[1] * 100.5))

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
        return MultiDiscrete([max_pos] * 2)  # 5 componenti: 1 italiano, 2 guardie, 1 via di fuga

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        if agent == "italiano" or agent.startswith("catalano"):
            return Discrete(4)  # Quattro possibili azioni per ogni agente
        else:
            return None  # Restituisce None per agenti non riconosciuti



from pettingzoo.test import parallel_api_test
import random
NUM_EPISODES = 100000  # Numero di partite da giocare per l'apprendimento
if __name__ == "__main__":
    import wandb

    #wandb.init(project='maze_RL', entity='alee8')
    env = MAP_RL_Env()
    env.init_pygame()

    italiano = AgentRL('italiano', env)
    catalano = AgentRL('catalano', env)

    max_x_value = 10
    max_y_value = 10
    pos_x = Fluent("pos_x", RealType(0, ))
    pos_y = Fluent("pos_y", RealType(0, ))
    italiano.add_public_fluent(pos_x)
    italiano.add_public_fluent(pos_y)
    env.add_agent(italiano)
    env.set_initial_value(Dot(italiano, pos_x), 2)
    env.set_initial_value(Dot(italiano, pos_y), 2)


    catalano.add_public_fluent(pos_x)
    catalano.add_public_fluent(pos_y)
    env.add_agent(catalano)
    env.set_initial_value(Dot(catalano, pos_x), 5)
    env.set_initial_value(Dot(catalano, pos_y), 5)

    # Azione move_down
    move_up = InstantaneousAction("up")
    move_up.add_precondition(LT(0, pos_y))  # Precondizione: pos_y > 0
    move_up.add_decrease_effect(pos_y, 1)
    #move_down.add_effect(pos_y, Minus(pos_y, 1))  # Effetto: decrementa pos_y di 1
    italiano.add_rl_action(move_up)
    catalano.add_rl_action(move_up)

    # Azione move_up
    move_down = InstantaneousAction("down")
    move_down.add_precondition(LT(pos_y, max_y_value - 1))  # Precondizione: pos_y < max_y_value
    move_down.add_increase_effect(pos_y, 1)
    #move_up.add_effect(pos_y, Plus(pos_y, 1))  # Effetto: incrementa pos_y di 1
    italiano.add_rl_action(move_down)
    catalano.add_rl_action(move_down)

    # Azione move_left
    move_left = InstantaneousAction("left")
    move_left.add_precondition(LT(0, pos_x))  # Precondizione: pos_x > 0
    move_left.add_decrease_effect(pos_x, 1)
    #move_left.add_effect(pos_x, Minus(pos_x, 1))  # Effetto: decrementa pos_x di 1
    italiano.add_rl_action(move_left)
    catalano.add_rl_action(move_left)

    # Azione move_right
    move_right = InstantaneousAction("right")
    move_right.add_precondition(LT(pos_x, max_x_value - 1))  # Precondizione: pos_x < max_x_value
    move_right.add_increase_effect(pos_x, 1)
    #move_right.add_effect(pos_x, Plus(pos_x, 1))  # Effetto: incrementa pos_x di 1
    italiano.add_rl_action(move_right)
    catalano.add_rl_action(move_right)

    #Reward Machines
    initial_state = "start"
    reward_for_pos1 = 0
    reward_for_pos2 = 0  # Puoi scegliere ricompense diverse se necessario
    reward_for_pos3 = 1

    prisoner_transitions = {
        ("start", "reached_A"): ("at_pos1", reward_for_pos1),
        ("at_pos1", "reached_B"): ("at_pos2", reward_for_pos2),
        ("at_pos2", "reached_C"): ("completed", reward_for_pos3)
    }

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

    RM_1 = RewardMachine(initial_state, prisoner_transitions)
    RM_2 = RewardMachine(initial_state, prisoner_transitions)
    italiano.set_reward_machine(RM_1)
    catalano.set_reward_machine(RM_2)

    q_learning1 = Q_learning(
                state_space_size=env.grid_width * env.grid_height * env.num_rm_states,
                action_space_size=4,
                learning_rate=0.1,
                gamma=0.9,
            )

    q_learning2 = Q_learning(
        state_space_size=env.grid_width * env.grid_height * env.num_rm_states,
        action_space_size=4,
        learning_rate=0.1,
        gamma=0.9,
    )
    italiano.set_learning_algorithm(q_learning1)
    catalano.set_learning_algorithm(q_learning2)

    #env.initialize_agents_q_learning()
    for episode in range(NUM_EPISODES):
        obs, infos = env.reset()
        done = {a.name: False for a in env.agents}
        rewards = {a.name: 0 for a in env.agents}  # Inizializza le ricompense episodiche
        #record_episode = episode % 10000 == 0
        record_episode = False
        while not all(done.values()):
            actions = {}
            for ag in env.agents:
                current_state = env.get_state(ag)
                RM_agent = ag.get_reward_machine()
                rm_current_state = RM_agent.get_current_state()

                #action_index = env.agents_q_learning[ag.name].choose_action(current_state, env.epsilon, ag, rm_current_state)
                action_index = ag.get_learning_algorithm().choose_action(current_state, env.epsilon, ag, rm_current_state)
                actions[ag.name] = ag.actions_dix()[action_index]

            obs, rewards, done, truncations, infos = env.step(actions)

            env.render(episode, obs) #commentare x training
            if record_episode:
                env.render(episode, obs)  # Cattura frame durante l'episodio

            if all(truncations.values()):
                break
        if record_episode:
            env.save_episode(episode)  # Salva il video solo alla fine dell'episodio


        #wandb.log({**rewards, 'epsilon': env.epsilon, 'episode': episode, 'step': env.timestep})
        print(f"Episodio {episode + 1}: Ricompensa = {rewards}, Step: {env.timestep}, Epsilon = {env.epsilon} ")

