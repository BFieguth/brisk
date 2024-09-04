class ModelWrapper:
    def __init__(self, 
                 name: str, 
                 model_class, 
                 default_params=None, 
                 hyperparam_grid=None
    ):
        self.name = name
        self.model_class = model_class
        self.default_params = default_params if default_params else {}
        self.hyperparam_grid = hyperparam_grid if hyperparam_grid else {}

    def instantiate(self):
        return self.model_class(**self.default_params)

    def get_hyperparam_grid(self):
        return self.hyperparam_grid
