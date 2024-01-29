class ActionRL:
    def __init__(self, name, environment, preconditions=None, effects=None):
        """
        Inizializza un'azione RL.

        :param name: Nome dell'azione.
        :param environment: Riferimento all'ambiente in cui l'azione verrà eseguita.
        :param preconditions: Lista di precondizioni per l'azione.
        :param effects: Lista di effetti dell'azione.
        """
        self.name = name
        self.environment = environment
        self.preconditions = preconditions if preconditions is not None else []
        self.effects = effects if effects is not None else []

    def is_applicable(self, state):
        """
        Verifica se l'azione è applicabile nello stato corrente.

        :param state: Stato corrente dell'ambiente.
        :return: True se l'azione è applicabile, False altrimenti.
        """
        for precondition in self.preconditions:
            if not precondition(state):
                return False
        return True

    def execute(self, state, agent):
        if self.is_applicable(state):
            for effect in self.effects:
                effect(state, agent)
