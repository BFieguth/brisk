import itertools

class TrainingManager:
    def __init__(self, method_config, scoring_config, methods, datasets):
        self.method_config = method_config
        self.scoring_config = scoring_config

        if self.__validate_methods(methods):
            self.configurations = list(itertools.product(datasets, methods))

    
    def __validate_methods(self, methods):
        """Check all methods are included in the method_config"""
        included_methods = self.method_config.keys()
        if set(methods).issubset(included_methods):
            return True
        else:
            invalid_methods = list(set(methods) - set(included_methods))
            raise ValueError(
                f"The following methods are not included in the configuration: {invalid_methods}"
                )
