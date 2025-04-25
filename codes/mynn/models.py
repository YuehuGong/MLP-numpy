from .op import *
import pickle

class Model_MLP(Layer):
    """
    A model with linear layers. We provied you with this example about a structure of a model.
    """
    # def __init__(self, size_list=None, act_func=None, lambda_list=None,dropout_p=0.0):
    #     self.size_list = size_list
    #     self.act_func = act_func

    #     if size_list is not None and act_func is not None:
    #         self.layers = []
    #         for i in range(len(size_list) - 1):
    #             layer = Linear(in_dim=size_list[i], out_dim=size_list[i + 1])
    #             if lambda_list is not None:
    #                 layer.weight_decay = True
    #                 layer.weight_decay_lambda = lambda_list[i]
    #             if act_func == 'Logistic':
    #                 raise NotImplementedError
    #             elif act_func == 'ReLU':
    #                 layer_f = ReLU()
    #             self.layers.append(layer)
    #             if i < len(size_list) - 2:
    #                 self.layers.append(layer_f)
    def __init__(self, size_list=None, act_func=None, lambda_list=None, dropout_p=0.0):
        self.size_list = size_list
        self.act_func = act_func
        self.dropout_p = dropout_p  # 添加dropout概率参数
        self.dropout_layers = []  # 记录哪些层是Dropout层

        if size_list is not None and act_func is not None:
            self.layers = []
            for i in range(len(size_list) - 1):
                layer = Linear(in_dim=size_list[i], out_dim=size_list[i + 1])
                if lambda_list is not None:
                    layer.weight_decay = True
                    layer.weight_decay_lambda = lambda_list[i]
                if act_func == 'Logistic':
                    raise NotImplementedError
                elif act_func == 'ReLU':
                    layer_f = ReLU()
                
                self.layers.append(layer)
                if i < len(size_list) - 2:
                    self.layers.append(layer_f)
                    # 在激活层后添加Dropout
                    if dropout_p > 0:
                        dropout_layer = Dropout(p=dropout_p)
                        self.layers.append(dropout_layer)
                        self.dropout_layers.append(dropout_layer)
    def __call__(self, X,training=False):
        # if isinstance(self.layers[2],Linear):
        #     print(f"self.layers[{2}]:",self.layers[2].W[0,0:5]) 
        return self.forward(X, training)

    def forward(self, X, training=False):
        outputs = X
        for layer in self.layers:
            if isinstance(layer, Dropout) and not training:
                continue
            else:
                outputs = layer(outputs)
        return outputs

    def backward(self, loss_grad):
        grads = loss_grad
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
        return grads

    def load_model(self, param_list):
        with open(param_list, 'rb') as f:
            param_list = pickle.load(f)
        self.size_list = param_list[0]
        self.act_func = param_list[1]

        for i in range(len(self.size_list) - 1):
            self.layers = []
            for i in range(len(self.size_list) - 1):
                layer = Linear(in_dim=self.size_list[i], out_dim=self.size_list[i + 1])
                layer.W = param_list[i + 2]['W']
                layer.b = param_list[i + 2]['b']
                layer.params['W'] = layer.W
                layer.params['b'] = layer.b
                layer.weight_decay = param_list[i + 2]['weight_decay']
                layer.weight_decay_lambda = param_list[i+2]['lambda']
                if self.act_func == 'Logistic':
                    raise NotImplemented
                elif self.act_func == 'ReLU':
                    layer_f = ReLU()
                self.layers.append(layer)
                if i < len(self.size_list) - 2:
                    self.layers.append(layer_f)
        
    def save_model(self, save_path):
        param_list = [self.size_list, self.act_func]
        for layer in self.layers:
            if layer.optimizable:
                param_list.append({'W' : layer.params['W'], 'b' : layer.params['b'], 'weight_decay' : layer.weight_decay, 'lambda' : layer.weight_decay_lambda})
        
        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)
        

# class Model_CNN(Layer):
#     """
#     A model with conv2D layers. Implement it using the operators you have written in op.py
#     """
#     def __init__(self):
#         # input&output:[inchannels,W,H,D] = [1,28,28,10]
#         self.W = 28
#         self.H = 28
#         self.D = 10
#         self.in_channels = 1

#         # cnn_layer1:[out_channels,kernel_size,stripe,padding = 0]
#         self.kernel_1 = 4
#         self.out_channels_1 = 16
#         self.stripe_1 = 2
#         self.padding_1 = 0
#         # linear_1(懒得实现pooling,使用relu的linear代替):
#         # cnn_layer2:[out_channels,kernel_size,stripe,padding = 0]
#         self.kernel_2 = 2
#         self.out_channels_2 = 32
#         self.stripe_2 = 1
#         self.padding_2 = 0      



#     def __call__(self, X):
#         return self.forward(X)

#     def forward(self, X):
#         pass

#     def backward(self, loss_grad):
#         pass
    
#     def load_model(self, param_list):
#         pass
        
#     def save_model(self, save_path):
#         pass

class Model_CNN(Layer):
    """
    CNN model with Conv2D, Pooling and Linear layers.
    Structure: Conv2D -> ReLU -> MaxPool -> Conv2D -> ReLU -> MaxPool -> Flatten -> Linear
    """
    def __init__(self, input_shape=(1, 28, 28), num_classes=10):
        super().__init__()
        self.input_shape = input_shape  # (channels, H, W)
        self.num_classes = num_classes
        
        # Layer 1: Conv2D -> ReLU -> MaxPool
        self.conv1 = Conv2D(in_channels=input_shape[0], out_channels=16, 
                          kernel_size=3, stride=1, padding=1,reshape_input = True)
        self.relu1 = ReLU()
        self.pool1 = MaxPool2D(kernel_size=2, stride=2)
        
        # Layer 2: Conv2D -> ReLU -> MaxPool
        self.conv2 = Conv2D(in_channels=16, out_channels=32, 
                          kernel_size=3, stride=1, padding=1)
        self.relu2 = ReLU()
        self.pool2 = MaxPool2D(kernel_size=2, stride=2)
        
        # 纯NumPy方式计算flatten尺寸
        self.flatten = Flatten()
        
        # 计算Flatten后的尺寸
        self.flatten_size = self._calculate_flatten_size()

        # Fully Connected Layer
        self.linear = Linear(in_dim=self.flatten_size, out_dim=num_classes)
        
        self.layers = [
            self.conv1, self.relu1, self.pool1,
            self.conv2, self.relu2, self.pool2,
            self.flatten,  
            self.linear
        ]
    def _calculate_flatten_size(self):
        """手动计算经过所有卷积和池化层后的特征图尺寸"""
        # 初始尺寸
        channels, height, width = self.input_shape
        
        # Conv1: padding=1, kernel=3, stride=1 → 尺寸不变
        # Pool1: kernel=2, stride=2 → 尺寸减半
        height = height // 2
        width = width // 2
        
        # Conv2: padding=1, kernel=3, stride=1 → 尺寸不变
        # Pool2: kernel=2, stride=2 → 尺寸减半
        height = height // 2
        width = width // 2
        
        # 最终特征图尺寸: channels * height * width
        return 32 * height * width  # 32是第二层的输出通道数
    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        # Conv Block 1
        out = self.conv1(X)
        out = self.relu1(out)
        out = self.pool1(out)
        
        # Conv Block 2
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.pool2(out)
        
        # Flatten and classify
        out = self.flatten(out)
        out = self.linear(out)
        return out

    def backward(self, loss_grad):
        grads = loss_grad
        for layer in reversed(self.layers):
            # print("grads.shape",grads.shape)
            grads = layer.backward(grads)
        return grads

    def load_model(self, param_list):
        with open(param_list, 'rb') as f:
            params = pickle.load(f)
        
        # Load conv layers
        self.conv1.params['kernels'] = params['conv1_kernels']
        self.conv1.params['bias'] = params['conv1_bias']
        self.conv2.params['kernels'] = params['conv2_kernels']
        self.conv2.params['bias'] = params['conv2_bias']
        
        # Load linear layer
        self.linear.params['W'] = params['linear_W']
        self.linear.params['b'] = params['linear_b']

    def save_model(self, save_path):
        params = {
            'conv1_kernels': self.conv1.params['kernels'],
            'conv1_bias': self.conv1.params['bias'],
            'conv2_kernels': self.conv2.params['kernels'],
            'conv2_bias': self.conv2.params['bias'],
            'linear_W': self.linear.params['W'],
            'linear_b': self.linear.params['b']
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(params, f)

    def clear_grad(self):
        for layer in self.layers:
            if hasattr(layer, 'clear_grad'):
                layer.clear_grad()

class MLP_ODE(Layer):
    def __init__(self, size_list=None, act_func=None, lambda_list=None, 
                 ode_hidden_dim=64, ode_method='euler', ode_t_span=(0,1)):
        """
        Args:
            size_list: MLP各层尺寸，如[input_dim, h1_dim, h2_dim,..., output_dim]
            act_func: 激活函数类型 ('ReLU'/'Tanh')
            lambda_list: 各层权重衰减系数
            ode_hidden_dim: ODE层的隐藏维度
            ode_method: ODE求解方法 ('euler'/'rk4')
            ode_t_span: ODE求解时间区间
        """
        self.size_list = size_list
        self.act_func = act_func
        self.lambda_list = lambda_list
        self.ode_hidden_dim = ode_hidden_dim
        self.ode_method = ode_method
        self.ode_t_span = ode_t_span
        self.training = True  # 训练/测试模式标志

        if size_list is not None and act_func is not None:
            # 初始化MLP部分
            self.layers = []
            self.linear_layers = []  # 单独记录线性层用于参数访问
            for i in range(len(size_list) - 1):
                # 添加线性层
                linear_layer = Linear(in_dim=size_list[i], out_dim=size_list[i+1])
                if lambda_list is not None:
                    linear_layer.weight_decay = True
                    linear_layer.weight_decay_lambda = lambda_list[i]
                self.layers.append(linear_layer)
                self.linear_layers.append(linear_layer)
                
                # 添加激活层(最后一层不加)
                if i < len(size_list) - 2:
                    if act_func == 'ReLU':
                        self.layers.append(ReLU())
                    elif act_func == 'Tanh':
                        self.layers.append(Tanh())
            
            # 初始化ODE部分 (插入在最后一个隐藏层和输出层之间)
            self.ode_func_layers = [
                Linear(ode_hidden_dim, ode_hidden_dim),
                Tanh(),
                Linear(ode_hidden_dim, ode_hidden_dim)
            ]
            self.ode_solver = ODESolver(self._ode_func, method=ode_method)
            
    def _ode_func(self, t, x):
        """ODE的微分方程定义"""
        for layer in self.ode_func_layers:
            x = layer(x)
        return x

    def __call__(self, X, training=False):
        self.training = training
        return self.forward(X)

    def forward(self, X):
        # MLP前向传播 (到最后一个隐藏层)
        x = X
        for layer in self.layers[:-1]:  # 跳过最后的输出层
            x = layer(x)
        
        # ODE求解
        x = self.ode_solver.forward(x, t_span=self.ode_t_span)
        
        # 最后的输出层
        output = self.layers[-1](x)
        return output

    def backward(self, loss_grad):
        # 输出层梯度
        grads = self.layers[-1].backward(loss_grad)
        
        # ODE部分反向传播 (简化版)
        grads = self.ode_solver.backward(grads)
        
        # MLP部分反向传播
        for layer in reversed(self.layers[:-1]):
            grads = layer.backward(grads)
            
        return grads
    def save_model(self, save_path):
        """保存模型参数到文件"""
        params = {
            # 保存MLP部分参数
            'mlp_params': [],
            # 保存ODE部分参数
            'ode_linear1_W': self.ode_func_layers[0].params['W'],
            'ode_linear1_b': self.ode_func_layers[0].params['b'],
            'ode_linear2_W': self.ode_func_layers[2].params['W'],
            'ode_linear2_b': self.ode_func_layers[2].params['b'],
            # 保存模型配置
            'config': {
                'size_list': self.size_list,
                'act_func': self.act_func,
                'lambda_list': self.lambda_list,
                'ode_hidden_dim': self.ode_hidden_dim,
                'ode_method': self.ode_method,
                'ode_t_span': self.ode_t_span
            }
        }
        
        # 保存MLP各层参数
        for i, layer in enumerate(self.linear_layers):
            params['mlp_params'].append({
                'W': layer.params['W'],
                'b': layer.params['b'],
                'weight_decay': layer.weight_decay,
                'weight_decay_lambda': layer.weight_decay_lambda
            })
        
        with open(save_path, 'wb') as f:
            pickle.dump(params, f)

    def load_model(self, param_list):
        """从文件加载模型参数"""
        with open(param_list, 'rb') as f:
            params = pickle.load(f)
        
        # 加载模型配置
        config = params['config']
        self.size_list = config['size_list']
        self.act_func = config['act_func']
        self.lambda_list = config['lambda_list']
        self.ode_hidden_dim = config['ode_hidden_dim']
        self.ode_method = config['ode_method']
        self.ode_t_span = config['ode_t_span']
        
        # 初始化模型结构
        self.__init__(
            size_list=config['size_list'],
            act_func=config['act_func'],
            lambda_list=config['lambda_list'],
            ode_hidden_dim=config['ode_hidden_dim'],
            ode_method=config['ode_method'],
            ode_t_span=config['ode_t_span']
        )
        
        # 加载MLP部分参数
        for i, layer in enumerate(self.linear_layers):
            layer.params['W'] = params['mlp_params'][i]['W']
            layer.params['b'] = params['mlp_params'][i]['b']
            layer.weight_decay = params['mlp_params'][i]['weight_decay']
            layer.weight_decay_lambda = params['mlp_params'][i]['weight_decay_lambda']
        
        # 加载ODE部分参数
        self.ode_func_layers[0].params['W'] = params['ode_linear1_W']
        self.ode_func_layers[0].params['b'] = params['ode_linear1_b']
        self.ode_func_layers[2].params['W'] = params['ode_linear2_W']
        self.ode_func_layers[2].params['b'] = params['ode_linear2_b']
