import matplotlib.pyplot as plt
import numpy as np
from model import ThreeLayerNN

# 测试函数
def test(model, X_test, y_test):
    probs = model.forward(X_test)
    preds = np.argmax(probs, axis=0)
    accuracy = np.mean(preds == np.argmax(y_test, axis=0))
    print(f'Test Accuracy: {accuracy:.4f}')
    return accuracy

# 可视化函数
def plot_training_curves(train_losses, val_losses, val_accs):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(val_accs, label='Validation Accuracy')
    plt.legend()
    plt.title('Validation Accuracy')
    plt.show()

def visualize_weights(weights):
    W1 = weights['W1']
    W1 = W1.reshape(-1, 32, 32, 3)
    
    plt.figure(figsize=(10, 10))
    for i in range(16):
        plt.subplot(4, 4, i+1)
        img = (W1[i] - W1[i].min()) / (W1[i].max() - W1[i].min())
        plt.imshow(img)
        plt.axis('off')
    plt.suptitle('First Layer Weight Visualizations')
    plt.show()
