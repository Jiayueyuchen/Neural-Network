import numpy as np

# 神经网络模型
class ThreeLayerNN:
    def __init__(self, input_size, hidden_size, output_size, activation='relu'):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.activation = activation
        
        # 初始化参数
        self.W1 = np.random.randn(hidden_size, input_size) * np.sqrt(2.0/input_size)
        self.b1 = np.zeros((hidden_size, 1))
        self.W2 = np.random.randn(output_size, hidden_size) * np.sqrt(2.0/hidden_size)
        self.b2 = np.zeros((output_size, 1))
        
    def forward(self, X):
        self.X = X
        # 隐藏层
        self.z1 = np.dot(self.W1, X) + self.b1
        if self.activation == 'relu':
            self.a1 = np.maximum(0, self.z1)
        elif self.activation == 'sigmoid':
            self.a1 = 1/(1+np.exp(-self.z1))
        else:
            raise ValueError("Unsupported activation")
        
        # 输出层
        self.z2 = np.dot(self.W2, self.a1) + self.b2
        # Softmax
        exp_scores = np.exp(self.z2 - np.max(self.z2, axis=0, keepdims=True))
        self.probs = exp_scores / np.sum(exp_scores, axis=0, keepdims=True)
        return self.probs
    
    def compute_loss(self, y, reg_lambda):
        m = y.shape[1]
        corect_logprobs = -np.sum(y * np.log(self.probs + 1e-8)) 
        data_loss = corect_logprobs / m # 交叉熵损失
        reg_loss = 0.5*reg_lambda*(np.sum(self.W1**2) + np.sum(self.W2**2)) # L2正则化
        return data_loss + reg_loss
    
    def backward(self, y, reg_lambda):
        m = y.shape[1]
        # 输出层梯度
        dz2 = self.probs - y
        dW2 = np.dot(dz2, self.a1.T)/m + reg_lambda*self.W2
        db2 = np.sum(dz2, axis=1, keepdims=True)/m
        
        # 隐藏层梯度
        if self.activation == 'relu':
            da1 = (self.z1 > 0).astype(float)
        elif self.activation == 'sigmoid':
            da1 = self.a1*(1-self.a1)
        dz1 = np.dot(self.W2.T, dz2) * da1
        dW1 = np.dot(dz1, self.X.T)/m + reg_lambda*self.W1
        db1 = np.sum(dz1, axis=1, keepdims=True)/m
        
        return {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}
    
    def update_params(self, grads, lr):
        self.W1 -= lr * grads['dW1']
        self.b1 -= lr * grads['db1']
        self.W2 -= lr * grads['dW2']
        self.b2 -= lr * grads['db2']
    
    def save_weights(self, path):
        np.savez(path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2)
    
    def load_weights(self, path):
        weights = np.load(path)
        self.W1 = weights['W1']
        self.b1 = weights['b1']
        self.W2 = weights['W2']
        self.b2 = weights['b2']
        
    # 在ThreeLayerNN类中添加SGD优化器方法
    def update_params_sgd(self, grads, lr, momentum=0.9):
        # 如果没有速度缓存，则初始化
        if not hasattr(self, 'velocity_W1'):
            self.velocity_W1 = np.zeros_like(self.W1)
            self.velocity_b1 = np.zeros_like(self.b1)
            self.velocity_W2 = np.zeros_like(self.W2)
            self.velocity_b2 = np.zeros_like(self.b2)
        
        # 计算速度并更新参数 (SGD with momentum)
        self.velocity_W1 = momentum * self.velocity_W1 - lr * grads['dW1']
        self.W1 += self.velocity_W1
        
        self.velocity_b1 = momentum * self.velocity_b1 - lr * grads['db1']
        self.b1 += self.velocity_b1
        
        self.velocity_W2 = momentum * self.velocity_W2 - lr * grads['dW2']
        self.W2 += self.velocity_W2
        
        self.velocity_b2 = momentum * self.velocity_b2 - lr * grads['db2']
        self.b2 += self.velocity_b2
