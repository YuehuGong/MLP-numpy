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

        # 初始化每个参数的 velocity
        for layer in self.model.layers:
            if layer.optimizable:
                self.velocity[layer] = {}
                for key in layer.params:
                    self.velocity[layer][key] = 0 

    def step(self):
        for layer in self.model.layers:
            if layer.optimizable:
                for key in layer.params:
                    v_prev = self.velocity[layer][key]
                    grad = layer.grads[key]
                    if layer.weight_decay:
                        grad += layer.weight_decay_lambda * layer.params[key]
                    v_new = self.mu * v_prev - self.init_lr * grad
                    self.velocity[layer][key] = v_new
                    layer.params[key] += v_new
