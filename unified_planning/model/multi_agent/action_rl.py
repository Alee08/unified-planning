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

    def add_effect_location(
        self,
        fluent: Union["up.model.fnode.FNode", "up.model.fluent.Fluent"],
        value: "up.model.expression.Expression",
        condition: "up.model.expression.BoolExpression" = True,
        forall: Iterable["up.model.variable.Variable"] = tuple(),
    ):
        """
        At the given time, adds the given assignment to the `action's effects`.

        :param timing: The exact time in which the assignment is applied.
        :param fluent: The `fluent` which value is modified by the assignment.
        :param value: The `value` to assign to the given `fluent`.
        :param condition: The `condition` in which this `effect` is applied; the default
            value is `True`.
        :param forall: The 'Variables' that are universally quantified in this
            effect; the default value is empty.
        """
        (
            fluent_exp,
            value_exp,
            condition_exp,
        ) = self._environment.expression_manager.auto_promote(fluent, value, condition)
        if not fluent_exp.is_fluent_exp():
            raise UPUsageError(
                "fluent field of add_effect must be a Fluent or a FluentExp"
            )
        if not self._environment.type_checker.get_type(condition_exp).is_bool_type():
            raise UPTypeError("Effect condition is not a Boolean condition!")
        if not fluent_exp.type.is_compatible(value_exp.type):
            raise UPTypeError(
                f"DurativeAction effect has an incompatible value type. Fluent type: {fluent_exp.type} // Value type: {value_exp.type}"
            )
        self._add_effect_instance(
            timing,
            up.model.effect.Effect(fluent_exp, value_exp, condition_exp, forall=forall),
        )

    def execute(self, state, agent):
        if self.is_applicable(state):
            for effect in self.effects:
                effect(state, agent)
