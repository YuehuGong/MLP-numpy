from abc import abstractmethod
import numpy as np


class Optimizer:
    def __init__(self, init_lr, model) -> None:
        self.init_lr = init_lr
        self.model = model

    @abstractmethod
    def step(self):
        pass


class SGD(Optimizer):
    def __init__(self, init_lr, model):
        super().__init__(init_lr, model)
    
    def step(self):
        for layer in self.model.layers:
            if layer.optimizable == True:
                for key in layer.params.keys():
                    if layer.weight_decay:
                        layer.params[key] *= (1 - self.init_lr * layer.weight_decay_lambda)
                    layer.params[key] = layer.params[key] - self.init_lr * layer.grads[key]
                    # print("layer.grads[key].reshape(-1)[0:5]",layer.grads[key].reshape(-1)[0:5])
                    # if layer.grads[key].reshape(-1)[0:5][0]-0 < 1e-12:
                    #     print(key)


class MomentGD(Optimizer):
    def __init__(self, init_lr, model, mu):
        super().__init__(init_lr, model)
        self.mu = mu
        self.velocity = {}

        # Initialize velocity for each parameter
        for layer in self.model.layers:
            if layer.optimizable:
                self.velocity[layer] = {}
                for key in layer.params:
                    self.velocity[layer][key] = np.zeros_like(layer.params[key])  # Match parameter shape

    def step(self):
        for layer in self.model.layers:
            if layer.optimizable:
                for key in layer.params:
                    v_prev = self.velocity[layer][key]
                    grad = layer.grads[key]
                    
                    # Fix: Ensure shapes match for weight decay
                    if layer.weight_decay:
                        grad = grad + layer.weight_decay_lambda * layer.params[key].reshape(grad.shape)
                    
                    # Momentum update
                    v_new = self.mu * v_prev - self.init_lr * grad
                    self.velocity[layer][key] = v_new
                    layer.params[key] += v_new