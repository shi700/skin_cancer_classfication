import argparse
import torch
from data_loader import create_dataloaders
from model import MedFusionNet
from trainer import MedFusionNetTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="MedFusionNet Training（图像+元数据双分支）")
    parser.add_argument("--data_dir", type=str, default="./dataset",
                        help="数据集根目录（含images文件夹、labels.csv、metadata.csv）")
    parser.add_argument("--meta_csv", type=str, default="",
                        help="元数据CSV文件路径（默认从data_dir读取metadata.csv）")
    parser.add_argument("--batch_size", type=int, default=16, help="批次大小")
    parser.add_argument("--epochs", type=int, default=500, help="训练轮数")
    parser.add_argument("--lr", type=float, default=1e-4, help="初始学习率")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="训练设备")
    parser.add_argument("--pretrained", action="store_true", default=True, help="是否使用预训练权重")
    parser.add_argument("--resume", type=str, default="", help="恢复训练的checkpoint路径")
    parser.add_argument("--patience", type=int, default=20, help="早停机制的耐心值")
    parser.add_argument("--use_smote", action="store_true", default=False,
                        help="是否使用SMOTE处理类别不平衡")
    parser.add_argument("--use_class_balance", action="store_true", default=False,
                        help="是否使用类别平衡过采样")
    parser.add_argument("--min_samples", type=int, default=5,
                        help="类别平衡采样时的最小样本数阈值")
    parser.add_argument("--max_samples", type=int, default=50,
                        help="类别平衡采样时的最大样本数阈值")
    return parser.parse_args()


def main(args=None):
    if args is None:
        args = parse_args()

    print(f"Training Configuration: {args}")

    # 1. 创建数据加载器（含元数据）
    try:
        meta_csv_path = args.meta_csv if args.meta_csv else None
        train_loader, val_loader, class_names, meta_input_dim = create_dataloaders(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            use_smote=args.use_smote,
            use_class_balance=args.use_class_balance,
            min_samples=args.min_samples,
            max_samples=args.max_samples,
            meta_csv_path=meta_csv_path
        )
    except Exception as e:
        print(f"创建数据加载器失败: {str(e)}")
        print("请检查数据目录是否正确，是否包含images文件夹、labels.csv和metadata.csv")
        return

    num_classes = len(class_names)
    print(f"检测到{num_classes}个类别: {class_names}")
    print(f"元数据输入维度: {meta_input_dim}")

    # 2. 初始化模型（传入元数据维度）
    try:
        model = MedFusionNet(
            num_classes=num_classes,
            pretrained=args.pretrained,
            meta_input_dim=meta_input_dim
        )
        model.to(args.device)
    except Exception as e:
        print(f"初始化模型失败: {str(e)}")
        return

    # 3. 初始化训练器
    try:
        trainer = MedFusionNetTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_classes=num_classes,
            class_names=class_names,
            device=args.device,
            patience=args.patience
        )
    except Exception as e:
        print(f"初始化训练器失败: {str(e)}")
        return

    # 4. 恢复训练
    if args.resume:
        try:
            trainer.load_best_model(checkpoint_path=args.resume)
            print(f"从{args.resume}恢复训练")
        except Exception as e:
            print(f"恢复训练失败: {str(e)}")
            return

    # 5. 开始训练
    try:
        trainer.train(epochs=args.epochs)
    except Exception as e:
        print(f"训练过程出错: {str(e)}")
        return

    # 6. 输出最佳指标
    print(f"\nTraining Completed!")
    print(f"实际类别名称：{class_names}")
    print(f"Best Val Accuracy: {trainer.best_val_acc:.4f}")
    if hasattr(trainer, 'val_metrics'):
        print(f"Final Val Precision: {trainer.val_metrics.get('precision', [0])[-1]:.4f}")
        print(f"Final Val Recall: {trainer.val_metrics.get('recall', [0])[-1]:.4f}")
        print(f"Final Val AUC: {trainer.val_metrics.get('auc', [0])[-1]:.4f}")


if __name__ == "__main__":
    main()
