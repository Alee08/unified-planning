from typing import Optional, List, Union, Iterable

class RewardMachine:
    def __init__(self, transitions):
        self.transitions = transitions  # {(current_state, event): (new_state, reward)}
        self.initial_state = self._get_start_state()  # Memorizza lo stato iniziale
        self.current_state = self.initial_state
        self.state_indices = self._generate_state_indices()

    def _generate_state_indices(self):
        # Raccogli tutti gli stati univoci (sia di partenza che di arrivo) dalle transizioni
        unique_states = set()
        for (from_state, _), (to_state, _) in self.transitions.items():
            unique_states.add(from_state)
            unique_states.add(to_state)

        # Assicurati che lo stato iniziale sia incluso e mappato a zero
        unique_states.add(self.current_state)
        sorted_states = sorted(unique_states)
        sorted_states.remove(self.current_state)
        sorted_states.insert(0, self.current_state)

        # Assegna un indice univoco a ciascuno stato
        return {state: i for i, state in enumerate(sorted_states)}

    def get_state_index(self, rm_state):
        return self.state_indices[rm_state]

    def get_reward(self, event):
        if (self.current_state, event) in self.transitions:
            new_state, reward = self.transitions[(self.current_state, event)]
            self.current_state = new_state
            return reward
        return 0

    def get_current_state(self):
        return self.current_state

    def numbers_state(self):
        states = set()
        for (from_state, _), (to_state, _) in self.transitions.items():
            states.add(from_state)
            states.add(to_state)
        return len(states)

    @property
    def get_transitions(self):
        return self.transitions

    def reset_to_initial_state(self):
        self.current_state = self.initial_state  # Assumi che 'initial_state' sia memorizzato come attributo

    def get_final_state(self):
        # Cerca l'ultimo stato di arrivo tra le transizioni
        if self.transitions:  # Assicurati che ci siano transizioni
            last_to_state = next(reversed(self.transitions.values()))[0]
            return last_to_state
        else:
            # Nessuna transizione definita, gestisci come preferisci
            return None

    def _get_start_state(self):
        # Assicurati che ci siano transizioni definite
        if not self.transitions:
            return None

        # Prendi il primo stato di partenza dalla prima transizione della lista delle transizioni
        first_transition = next(iter(self.transitions))
        start_state = first_transition[0]

        return start_state
