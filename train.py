import numpy as np
import os
import pickle
from model import ThreeLayerNN

# 数据加载和预处理
def load_cifar10(data_dir):
    def unpickle(file):
        with open(file, 'rb') as fo:
            data = pickle.load(fo, encoding='bytes')
        return data
    
    # 加载训练数据
    train_data = []
    train_labels = []
    for i in range(1, 6):
        batch = unpickle(os.path.join(data_dir, f'data_batch_{i}'))
        train_data.append(batch[b'data'])
        train_labels.extend(batch[b'labels'])
    
    # 加载测试数据
    test_batch = unpickle(os.path.join(data_dir, 'test_batch'))
    
    # 合并数据并预处理
    X_train = np.concatenate(train_data).astype(np.float32) / 255.0
    y_train = np.array(train_labels)
    X_test = test_batch[b'data'].astype(np.float32) / 255.0
    y_test = np.array(test_batch[b'labels'])
    
    # 划分验证集
    val_size = 5000
    X_val = X_train[:val_size]
    y_val = y_train[:val_size]
    X_train = X_train[val_size:]
    y_train = y_train[val_size:]
    
    # 转换为one-hot编码
    def to_onehot(labels, num_classes=10):
        return np.eye(num_classes)[labels].T
    
    return (X_train.T, to_onehot(y_train)), (X_val.T, to_onehot(y_val)), (X_test.T, to_onehot(y_test))


# 训练函数
def train(model, X_train, y_train, X_val, y_val, epochs=50, batch_size=1,  # 将batch_size=1设为SGD标准配置
         lr=0.01, momentum=0.9, reg_lambda=0.001, lr_decay=0.95, save_path='best_model.npz'):
    best_val_acc = 0.0
    train_losses = []
    val_losses = []
    val_accs = []
    
    m = X_train.shape[1]
    for epoch in range(epochs):
        # 学习率衰减 (已经实现)
        current_lr = lr * (lr_decay ** epoch)
        
        # 随机打乱数据 (已经实现)
        permutation = np.random.permutation(m)
        X_shuffled = X_train[:, permutation]
        y_shuffled = y_train[:, permutation]
        
        # 小批量训练 (保持batch_size=1为真正的SGD)
        for i in range(0, m, batch_size):
            X_batch = X_shuffled[:, i:i+batch_size]
            y_batch = y_shuffled[:, i:i+batch_size]
            
            # 前向传播
            _ = model.forward(X_batch)
            
            # 反向传播 (交叉熵损失和L2正则化已经在compute_loss和backward中实现)
            grads = model.backward(y_batch, reg_lambda)
            
            # 使用SGD优化器更新参数
            model.update_params_sgd(grads, current_lr, momentum)
        
        # 以下代码保持不变（计算损失和准确率、保存最佳模型等）
        train_probs = model.forward(X_train)
        train_loss = model.compute_loss(y_train, reg_lambda)
        train_losses.append(train_loss)
        
        val_probs = model.forward(X_val)
        val_loss = model.compute_loss(y_val, reg_lambda)
        val_losses.append(val_loss)
        
        val_preds = np.argmax(val_probs, axis=0)
        val_acc = np.mean(val_preds == np.argmax(y_val, axis=0))
        val_accs.append(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save_weights(save_path)
        
        print(f'Epoch {epoch+1}/{epochs}, Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
    
    return train_losses, val_losses, val_accs