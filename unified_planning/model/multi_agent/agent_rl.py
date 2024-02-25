import unified_planning as up
from agent import Agent
from RewardMachine import RewardMachine
from typing import Optional, List, Union, Iterable
from message import Message
class AgentRL(Agent):
    """
    Estende la classe Agent per includere aspetti specifici del Reinforcement Learning,
    come la Reward Machine e azioni basate su RL.
    """

    def __init__(
            self,
            name: str,
            ma_problem,
            reward_machine: Optional["RewardMachine"] = None,
    ):
        """
        Inizializza un agente con le capacità di RL.

        :param name: Nome univoco dell'agente.
        :param ma_problem: Riferimento al problema di planning multi-agente.
        :param reward_machine: Istanza di RewardMachine specifica per questo agente.
        """
        super().__init__(name, ma_problem)
        self.reward_machine = reward_machine
        self.ma_problem = ma_problem
        self.actions_dict = {}
        self.learning_algorithm = None
        self.message_conditions = None
        self.messages = {}  # Uno stato interno per tracciare informazioni rilevanti
        self.message_sent = False

        # Potrebbe essere necessario aggiungere ulteriori attributi specifici per il RL

    def actions_dix(self):
        for idx, act in enumerate(self.actions):
            self.actions_dict[idx] = act
        return self.actions_dict

    def actions_idx(self, action):
        dict = self.actions_dix()
        # Trovare la chiave per un valore specifico
        chiave_trovata = None
        for chiave, valore in dict.items():
            if valore == action:
                chiave_trovata = chiave
                break

        return chiave_trovata

    def get_reward(self, event):
        return self.reward_machine.get_reward(event) if self.reward_machine else 0

    def set_reward_machine(self, reward_machine: RewardMachine):
        self.reward_machine = reward_machine

    def get_reward_machine(self):
        return self.reward_machine


    def add_rl_action(self, action):
        """
        Aggiunge un'azione specifica per il RL all'agente.

        :param action: Azione RL da aggiungere.
        """
        # Implementazione dipende dalla struttura delle azioni RL
        self.actions.append(action)

    # Altri metodi specifici per il RL possono essere aggiunti qui

    def set_learning_algorithm(self, algorithm):
        """
        Assegna un algoritmo di apprendimento all'agente.

        :param algorithm: Istanza dell'algoritmo di apprendimento.
        """
        self.learning_algorithm = algorithm

    def get_learning_algorithm(self):
        """
        Ritorna l'algoritmo di apprendimento dell'agente.
        """
        return self.learning_algorithm

    def _send_message(self, agents, condition):
        """
        Invia un messaggio agli altri agenti quando l'agente raggiunge uno stato "X".

        :param condition: La condizione da comunicare agli altri agenti.
        """
        message = Message(self.name, condition)
        # Logica per inviare il messaggio agli altri agenti
        # Potrebbe essere qualcosa come:
        self.ma_problem.broadcast_message(agents, message)

    def _receive_message(self, message):
        # Gestisci il messaggio ricevuto
        if isinstance(message, Message):
            # Aggiorna lo stato interno in base al messaggio
            self.process_message(message)

    def process_message(self, message):
        # Logica specifica per processare il messaggio
        # Ad esempio, aggiorna lo stato interno o cambia strategia
        chiave_messaggio = (message.sender, message.condition[0][0])  # Crea una tupla (agente, fluent)
        self.messages[chiave_messaggio] = message.condition[0][1]
        #self.messages[message.sender] = message.condition

        # Puoi implementare qui ulteriori logiche, come reagire specificamente a certi tipi di messaggi
        """if message.condition == some_specific_condition:
            self.take_specific_action()"""

    def take_specific_action(self):
        # Implementa azioni specifiche da intraprendere in risposta a certi messaggi o condizioni
        pass

    def reset_messages(self):
        self.messages = {}
        self.message_conditions = None

    def return_messages(self):
        return self.messages

    def execute_action(self, action_name, state):
        for action in self.actions:
            if action.name == action_name:
                action.execute(state, self)
                return

