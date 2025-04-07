import itertools
import os
from train import train, load_cifar10
from test import test
from model import ThreeLayerNN


# 参数查找模块
def grid_search(data_dir, epochs=30, output_dir='models'):
    """
    超参数网格搜索函数
    Args:
        data_dir: 数据目录路径
        epochs: 每个组合的训练轮次
        output_dir: 保存模型文件的输出目录
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 定义超参数网格
    hyperparams = {
        'hidden_size': [256, 512, 1024],
        'lr': [0.1, 0.01, 0.001],
        'reg_lambda': [0.1, 0.01, 0.001]
    }
    
    # 加载数据
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_cifar10(data_dir)
    
    # 遍历所有组合
    results = []
    for params in itertools.product(*hyperparams.values()):
        h_size, lr, reg = params
        model = ThreeLayerNN(3072, h_size, 10)
        
        print(f"\n=== 正在训练参数组合: hidden_size={h_size}, lr={lr}, reg_lambda={reg} ===")
        
        # 构建模型保存路径（放在指定文件夹中）
        model_filename = f'model_h{h_size}_lr{lr}_reg{reg}.npz'
        model_path = os.path.join(output_dir, model_filename)
        
        # 训练并验证（使用较小epochs加速搜索）
        _, _, val_accs = train(
            model, X_train, y_train, X_val, y_val,
            epochs=epochs, batch_size=128, lr=lr, reg_lambda=reg,
            save_path=model_path  # 使用完整路径保存模型
        )
        
        # 记录结果
        results.append({
            'hidden_size': h_size,
            'lr': lr,
            'reg_lambda': reg,
            'best_val_acc': max(val_accs),
            'model_path': model_path
        })
    
    # 输出最佳组合
    best = max(results, key=lambda x: x['best_val_acc'])
    print("\n=== 最佳参数组合 ===")
    print(f"隐藏层大小: {best['hidden_size']}")
    print(f"学习率: {best['lr']}")
    print(f"正则化强度: {best['reg_lambda']}")
    print(f"验证集准确率: {best['best_val_acc']:.4f}")
    print(f"模型保存路径: {best['model_path']}")
    
    return best