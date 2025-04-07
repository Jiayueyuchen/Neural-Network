import numpy as np
import matplotlib.pyplot as plt
from model import ThreeLayerNN
from train import train, load_cifar10
from test import test, plot_training_curves, visualize_weights
from hparam_search import grid_search

# 主程序
if __name__ == '__main__':
    import argparse
    
    # 配置命令行参数
    parser = argparse.ArgumentParser(description='神经网络分类器')
    parser.add_argument('--mode', choices=['train', 'test', 'search'], default='train',
                       help='运行模式: train-训练, test-测试, search-参数搜索')
    parser.add_argument('--data_dir', default=r'D:\a课程\第二学期\神经网络和深度学习\HW\HW1\cifar-10-batches-py',
                       help='CIFAR-10数据集目录')
    parser.add_argument('--model_path', default='best_model.npz',
                       help='模型权重文件路径（测试模式需要）')
    parser.add_argument('--output_dir', default='models',
                       help='模型保存目录（用于参数搜索和训练）')
    args = parser.parse_args()

    # 公共数据加载
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_cifar10(args.data_dir)

    if args.mode == 'train':
        # 标准训练流程
        model = ThreeLayerNN(3072, 1024, 10)
        train_losses, val_losses, val_accs = train(
            model, X_train, y_train, X_val, y_val,
            epochs=100, batch_size=128, lr=0.01, reg_lambda=0.001,
            save_path=f"{args.output_dir}/best_model.npz"  # 使用输出目录
        )
        plot_training_curves(train_losses, val_losses, val_accs)

    elif args.mode == 'test':
        # 测试流程
        model = ThreeLayerNN(3072, 1024, 10)
        model.load_weights(args.model_path)
        test_accuracy = test(model, X_test, y_test)
        visualize_weights(np.load(args.model_path))

    elif args.mode == 'search':
        # 参数搜索流程
        best_params = grid_search(
            data_dir=args.data_dir,
            epochs=30,  # 为加速搜索使用较少epoch
            output_dir=args.output_dir  # 传递输出目录参数
        )
        print("\n提示：可以使用以下参数进行完整训练：")
        print(f"--hidden_size {best_params['hidden_size']}")
        print(f"--lr {best_params['lr']} --reg_lambda {best_params['reg_lambda']}")
        print(f"最佳模型已保存至: {best_params['model_path']}")
        
# 示例用法：
# python main.py --mode train --data_dir cifar-10-batches-py --output_dir trained_models
# python main.py --mode test --model_path trained_models/best_model.npz
# python main.py --mode search --data_dir cifar-10-batches-py --output_dir search_models