from typing import Optional, List, Union, Iterable

class RewardMachine:
    def __init__(self, initial_state, transitions):
        self.initial_state = initial_state  # Memorizza lo stato iniziale
        self.current_state = initial_state
        self.transitions = transitions  # {(current_state, event): (new_state, reward)}
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

    def reset_to_initial_state(self):
        self.current_state = self.initial_state  # Assumi che 'initial_state' sia memorizzato come attributo