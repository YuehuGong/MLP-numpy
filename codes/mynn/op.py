from abc import abstractmethod
import numpy as np

class Layer():
    def __init__(self) -> None:
        self.optimizable = True
    
    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def backward(self):
        pass


class Linear(Layer):
    """
    The linear layer for a neural network. You need to implement the forward function and the backward function.
    """
    # original initialize_method: np.random.normal
    def __init__(self, in_dim, out_dim, initialize_method= None, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        if initialize_method is None:
            # Xavier initialization: U(-limit, limit)
            limit = np.sqrt(6 / (in_dim + out_dim))
            W = np.random.uniform(-limit, limit, size=(in_dim, out_dim))
            b = np.zeros((1, out_dim))  # Bias 可以初始化为 0
        else:
            W = initialize_method(size=(in_dim, out_dim))
            b = initialize_method(size=(1, out_dim))
        self.grads = {'W' : None, 'b' : None}
        self.input = None # Record the input for backward process.

        self.params = {'W' : W, 'b' : b}

        self.weight_decay = weight_decay # whether using weight decay
        self.weight_decay_lambda = weight_decay_lambda # control the intensity of weight decay
            
    
    def __call__(self, X) -> np.ndarray:
        return self.forward(X)

    def forward(self, X):
        """
        input: [batch_size, in_dim]
        out: [batch_size, out_dim]
        """
        self.input = X
        output = np.matmul(X,self.params['W']) + self.params['b']
        return output

    def backward(self, grad : np.ndarray):
        """
        input: [batch_size, out_dim] the grad passed by the next layer.
        output: [batch_size, in_dim] the grad to be passed to the previous layer.
        This function also calculates the grads for W and b.
        """
        batch_size, in_dim = self.input.shape
        out_dim = self.params['W'].shape[1]
        self.grads['W'] = np.matmul(self.input.T, grad)
        self.grads['b'] = grad.sum(axis = 0)
        # print("self.grads['W']:",self.grads['W'])
        return np.matmul(grad, self.params['W'].T)
    
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}

class Conv2D(Layer):
    """
    The 2D convolutional layer. Try to implement it on your own.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8, reshape_input = False) -> None:
        limit = np.sqrt(2 / (in_channels * kernel_size * kernel_size))
        kernels = np.random.uniform(-limit, limit, 
                                  size=(out_channels, in_channels, kernel_size, kernel_size))
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size

        self.input = None # Record the input for backward process.
        self.params = {
            'kernels': kernels,
            'bias': np.zeros(out_channels)  # 添加bias
        }
        self.grads = {
            'kernels': np.zeros_like(self.params['kernels']),
            'bias': np.zeros_like(self.params['bias'])
        }
        self.weight_decay = weight_decay # whether using weight decay
        self.weight_decay_lambda = weight_decay_lambda # control the intensity of weight decay
        self.reshape_input = reshape_input
        self.optimizable = True

    def __call__(self, X) -> np.ndarray:
        return self.forward(X)

    # def forward(self, X):
    #     """
    #     input X: [batch, channels, H, W]
    #     W : [1, out, in, k, k]
    #     no padding
    #     output: [batch_size, output_channels, (d_H + p_H + s_H - k_H)/s_H,...]
    #     """
    #     batch = X.shape[0]
    #     # print("X.shape:",X.shape)
    #     if self.reshape_input:
    #         X = X.reshape(batch,1,28,28)
    #     self.input = X
    #     batch, in_c, H, W = X.shape
    #     out_c, _, kH, kW = self.params['kernels'].shape
        
    #     new_H = (H + 2*self.padding - kH) // self.stride + 1
    #     new_W = (W + 2*self.padding - kW) // self.stride + 1
        
    #     if self.padding > 0:
    #         X_pad = np.pad(X, ((0,0), (0,0), (self.padding, self.padding), 
    #                          (self.padding, self.padding)), mode='constant')
    #     else:
    #         X_pad = X
            
    #     output = np.zeros((batch, out_c, new_H, new_W))

    #     for b in range(batch):
    #         for oc in range(out_c):
    #             for i in range(new_H):
    #                 for j in range(new_W):
    #                     h_start = i * self.stride
    #                     w_start = j * self.stride
    #                     patch = X_pad[b, :, h_start:h_start+kH, w_start:w_start+kW]
    #                     output[b, oc, i, j] = np.sum(patch * self.params['kernels'][oc]) + self.params['bias'][oc]
    #     return output

    def forward(self, X):
        if self.reshape_input:
            X = X.reshape(X.shape[0], 1, 28, 28)
        self.input = X
        
        batch, in_c, H, W = X.shape
        out_c, _, kH, kW = self.params['kernels'].shape
        
        new_H = (H + 2 * self.padding - kH) // self.stride + 1
        new_W = (W + 2 * self.padding - kW) // self.stride + 1
        
        if self.padding > 0:
            X_pad = np.pad(X, ((0, 0), (0, 0), (self.padding, self.padding), 
                            (self.padding, self.padding)), mode='constant')
        else:
            X_pad = X
        
        # 使用修正后的im2col
        patches = self._im2col(X_pad, kH, kW, self.stride)
        kernels_flat = self.params['kernels'].reshape(out_c, -1)
        
        output = (kernels_flat @ patches).reshape(out_c, batch, new_H, new_W)
        output = output.transpose(1, 0, 2, 3) + self.params['bias'][None, :, None, None]
        
        return output

    def _im2col(self, X, kH, kW, stride):
        batch, in_c, H_pad, W_pad = X.shape
        new_H = (H_pad - kH) // stride + 1
        new_W = (W_pad - kW) // stride + 1

        # 使用as_strided提取图像块（无内存拷贝）
        shape = (batch, in_c, new_H, new_W, kH, kW)
        strides = (X.strides[0], X.strides[1], 
                stride * X.strides[2], stride * X.strides[3], 
                X.strides[2], X.strides[3])
        patches = np.lib.stride_tricks.as_strided(X, shape=shape, strides=strides)
        
        # 调整维度并展开
        patches = patches.transpose(1, 4, 5, 0, 2, 3).reshape(in_c * kH * kW, -1)
        return patches

        
    def backward(self, grad_output):
        X = self.input
        batch, in_c, H, W = X.shape
        out_c, _, kH, kW = self.params['kernels'].shape
        _, _, new_H, new_W = grad_output.shape

        # 初始化梯度
        grad_kernels = np.zeros_like(self.params['kernels'])
        grad_bias = np.zeros_like(self.params['bias'])
        grad_input = np.zeros_like(X)
        
        # 处理padding
        if self.padding > 0:
            X_pad = np.pad(X, ((0,0), (0,0), (self.padding, self.padding), 
                             (self.padding, self.padding)), mode='constant')
            grad_pad = np.pad(grad_input, ((0,0), (0,0), (self.padding, self.padding), 
                                 (self.padding, self.padding)), mode='constant')
        else:
            X_pad = X
            grad_pad = grad_input

        # 计算梯度
        for b in range(batch):
            for oc in range(out_c):
                grad_bias[oc] += np.sum(grad_output[b, oc])
                for i in range(new_H):
                    for j in range(new_W):
                        h_start = i * self.stride
                        w_start = j * self.stride
                        patch = X_pad[b, :, h_start:h_start+kH, w_start:w_start+kW]
                        
                        # 计算kernels梯度
                        grad_kernels[oc] += grad_output[b, oc, i, j] * patch
                        
                        # 计算input梯度（需要考虑stride）
                        grad_pad[b, :, h_start:h_start+kH, w_start:w_start+kW] += (
                            grad_output[b, oc, i, j] * self.params['kernels'][oc])
        
        # 移除padding部分的梯度
        if self.padding > 0:
            grad_input = grad_pad[:, :, self.padding:-self.padding, self.padding:-self.padding]
        else:
            grad_input = grad_pad
            
        self.grads['kernels'] = grad_kernels
        self.grads['bias'] = grad_bias
        
        return grad_input

    def clear_grad(self):
        self.grads = {
            'kernels': np.zeros_like(self.params['kernels']),
            'bias': np.zeros_like(self.params['bias'])
        }
            
class MaxPool2D(Layer):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if stride is not None else kernel_size  # 默认stride=kernel_size
        self.padding = padding
        self.input = None
        self.mask = None  # 记录最大值位置，用于反向传播
        self.optimizable = False  # Pooling层无可训练参数

    def __call__(self, X) -> np.ndarray:
        return self.forward(X)

    def forward(self, X):
        """
        input X: [batch, channels, H, W]
        output: [batch, channels, new_H, new_W]
        """
        self.input = X
        batch, channels, H, W = X.shape
        kH, kW = self.kernel_size
        
        # 计算输出尺寸
        new_H = (H + 2 * self.padding - kH) // self.stride + 1
        new_W = (W + 2 * self.padding - kW) // self.stride + 1
        
        # 添加padding
        if self.padding > 0:
            X_pad = np.pad(X, ((0,0), (0,0), (self.padding, self.padding), 
                          (self.padding, self.padding)), mode='constant')
        else:
            X_pad = X
        
        output = np.zeros((batch, channels, new_H, new_W))
        self.mask = np.zeros_like(X_pad)  # 记录最大值位置
        
        # Max Pooling
        for b in range(batch):
            for c in range(channels):
                for i in range(new_H):
                    for j in range(new_W):
                        h_start = i * self.stride
                        w_start = j * self.stride
                        window = X_pad[b, c, h_start:h_start+kH, w_start:w_start+kW]
                        output[b, c, i, j] = np.max(window)
                        
                        # 记录最大值位置（用于反向传播）
                        max_idx = np.unravel_index(np.argmax(window), window.shape)
                        self.mask[b, c, h_start+max_idx[0], w_start+max_idx[1]] = 1
        return output

    def backward(self, grad_output):
        """
        input grad_output: [batch, channels, pooled_H, pooled_W]
        output grad_input: [batch, channels, H, W]
        """
        batch, channels, pooled_H, pooled_W = grad_output.shape
        kH, kW = self.kernel_size
        
        # 初始化梯度（考虑padding）
        if self.padding > 0:
            grad_input_pad = np.zeros_like(self.mask)
        else:
            grad_input_pad = np.zeros_like(self.input)
        
        # 将梯度传播到最大值位置
        for b in range(batch):
            for c in range(channels):
                for i in range(pooled_H):
                    for j in range(pooled_W):
                        h_start = i * self.stride
                        w_start = j * self.stride
                        
                        # 只有最大值位置接收梯度
                        window_mask = self.mask[b, c, h_start:h_start+kH, w_start:w_start+kW]
                        grad_input_pad[b, c, h_start:h_start+kH, w_start:w_start+kW] += \
                            grad_output[b, c, i, j] * window_mask
        
        # 移除padding
        if self.padding > 0:
            grad_input = grad_input_pad[:, :, self.padding:-self.padding, self.padding:-self.padding]
        else:
            grad_input = grad_input_pad
            
        return grad_input

    def clear_grad(self):
        pass  

class Flatten(Layer):
    """
    Flatten层：将多维输入张量展平为二维（batch_size, -1）
    前向传播：reshape
    反向传播：恢复原始形状
    """
    def __init__(self):
        super().__init__()
        self.input_shape = None  # 保存输入形状以便反向传播
        self.optimizable = False

    def __call__(self, X) -> np.ndarray:
        return self.forward(X)


    def forward(self, X):
        self.input_shape = X.shape  # 保存输入形状
        return X.reshape(X.shape[0], -1)  # (batch_size, channels * H * W)

    def backward(self, grad_output):
        return grad_output.reshape(self.input_shape)


class ReLU(Layer):
    """
    An activation layer.
    """
    def __init__(self) -> None:
        super().__init__()
        self.input = None

        self.optimizable =False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input = X
        output = np.where(X<0, 0, X)
        return output
    
    def backward(self, grads):
        assert self.input.shape == grads.shape
        output = np.where(self.input < 0, 0, grads)
        return output

class MultiCrossEntropyLoss(Layer):
    """
    A multi-cross-entropy loss layer, with Softmax layer in it, which could be cancelled by method cancel_softmax
    """
    def __init__(self, model = None, max_classes = 10, has_softmax = True) -> None:
        self.model = model
        self.max_classes = max_classes
        self.input = {}
        self.has_softmax = has_softmax
        

    def __call__(self, predicts, labels):
        return self.forward(predicts, labels)
    
    def forward(self, predicts, labels):
        """
        predicts: [batch_size, D]
        labels : [batch_size, ]
        This function generates the loss.
        """
        # / ---- your codes here ----/
        batch_size, D = predicts.shape
        self.input["predicts"] = predicts
        self.input["labels"] = labels
        if self.has_softmax:
            predicts = softmax(predicts)
        self.input["softmax_predicts"] = predicts
        probs = np.clip(predicts[np.arange(batch_size), labels], 1e-12, 1.0)
        loss = -np.mean(np.log(probs))
        return loss
        
    
    def backward(self):
        # first compute the grads from the loss to the input
        # / ---- your codes here ----/
        # Then send the grads to model for back propagation
        # self.model.backward(self.grads)
        # has_softmax:[(1)*(batch_size, D)] * [(batch_size, D)*(batch_size, D)]
        # !has_softmax:[(1)*(batch_size, D)]
        batch_size, D = self.input["predicts"].shape

        if self.has_softmax:
            grads = self.input["softmax_predicts"].copy()
            grads[np.arange(batch_size), self.input["labels"]] -= 1
            grads /= batch_size
        else:
            probs = np.clip(self.input["predicts"], 1e-12, 1.0)
            grads = np.zeros_like(probs)
            grads[np.arange(batch_size), self.input["labels"]] = -1 / probs[np.arange(batch_size), self.input["labels"]]
            grads /= batch_size

        self.model.backward(grads)
        return grads
        

    def cancel_soft_max(self):
        self.has_softmax = False
        return self
    
class L2Regularization(Layer):
    """
    L2 Reg can act as weight decay that can be implemented in class Linear.
    """
    pass
       
def softmax(X):
    x_max = np.max(X, axis=-1, keepdims=True)
    x_exp = np.exp(X - x_max)
    partition = np.sum(x_exp, axis=1, keepdims=True)
    return x_exp / partition

class Dropout(Layer):
    def __init__(self, p=0.1):
        super().__init__()
        self.p = p  
        self.mask = None  
        self.optimizable =False

    def __call__(self, X) -> np.ndarray:
        return self.forward(X)

    def forward(self, X):
        self.mask = (np.random.rand(*X.shape) > self.p) / (1 - self.p)  # 缩放补偿
        return X * self.mask

    def backward(self, grad_output):
        return grad_output * self.mask  

class ODEFunction(Layer):
    """定义ODE的微分方程f(t,x)"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.net = Sequential(
            Linear(hidden_dim, hidden_dim),
            Tanh(),  # 推荐使用Tanh保证稳定性
            Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, t, x):
        return self.net(x)

class Tanh(Layer):
    """Hyperbolic Tangent激活函数层"""
    def __init__(self):
        super().__init__()
        self.optimizable = False  # 无参数
        self.input = None  # 缓存输入用于反向传播

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        """
        input: 任意形状的数组
        output: 相同形状的tanh激活结果
        """
        self.input = X  # 缓存输入用于反向传播
        return np.tanh(X)

    def backward(self, grad_output):
        """
        input: 上游传来的梯度，形状与forward输出相同
        output: 传递给前层的梯度
        """
        # tanh的导数是 1 - tanh^2(x)
        tanh_x = np.tanh(self.input)
        grad_input = grad_output * (1 - tanh_x ** 2)
        return grad_input

    def clear_grad(self):
        pass  # 无参数需清空梯度

class ODESolver(Layer):
    """通用ODE求解器基类"""
    def __init__(self, ode_func, method='euler', dt=0.1):
        super().__init__()
        self.ode_func = ode_func  # 微分方程函数 f(t, x)
        self.method = method      # 求解方法：'euler'/'rk4'
        self.dt = dt              # 步长
        self.optimizable = False

    def forward(self, x, t_span=(0, 1)):
        if self.method == 'euler':
            return self._euler_solve(x, t_span)
        elif self.method == 'rk4':
            return self._rk4_solve(x, t_span)
        else:
            raise ValueError("Unsupported method")

    def _euler_solve(self, x, t_span):
        t_start, t_end = t_span
        num_steps = int((t_end - t_start) / self.dt)
        for _ in range(num_steps):
            dx = self.ode_func(None, x)  # 不依赖时间t
            x = x + self.dt * dx
        return x

    def _rk4_solve(self, x, t_span):
        t_start, t_end = t_span
        num_steps = int((t_end - t_start) / self.dt)
        for _ in range(num_steps):
            k1 = self.ode_func(None, x)
            k2 = self.ode_func(None, x + 0.5*self.dt*k1)
            k3 = self.ode_func(None, x + 0.5*self.dt*k2)
            k4 = self.ode_func(None, x + self.dt*k3)
            x = x + (self.dt/6) * (k1 + 2*k2 + 2*k3 + k4)
        return x

    def backward(self, grad_output):
        # 使用伴随方法(adjoint method)计算梯度
        return grad_output  # 简化实现，实际需要更复杂的处理

class NeuralODE(Layer):
    """可学习的ODE层"""
    def __init__(self, hidden_dim, method='euler', dt=0.1):
        super().__init__()
        # 定义微分方程对应的神经网络
        self.net = Sequential(
            Linear(hidden_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, hidden_dim)
        )
        self.solver = ODESolver(self.ode_func, method, dt)
        
    def ode_func(self, t, x):
        return self.net(x)

    def forward(self, x, t_span=(0, 1)):
        return self.solver.forward(x, t_span)

    def backward(self, grad_output):
        # 这里需要实现伴随方法的反向传播
        return self.solver.backward(grad_output)
