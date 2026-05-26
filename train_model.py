# -*- coding: utf-8 -*-
"""
训练脚本 - 生成PT格式的情感分析模型
"""

import sys
import os
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

from src.training.trainer_manager import TrainerManager

def main():
    print("=" * 60)
    print("🎯 训练情感分析模型")
    print("=" * 60)
    
    # 创建训练管理器
    trainer_manager = TrainerManager(model_type="textcnn", language="chinese")
    
    try:
        # 执行完整训练流程
        results = trainer_manager.full_training_pipeline(
            epochs=10,
            batch_size=64,
            max_samples=None  # 使用全部数据
        )
        
        print("\n" + "=" * 60)
        print("📊 训练结果汇总")
        print("=" * 60)
        print(f"模型类型: {results['model_type']}")
        print(f"语言: {results['language']}")
        print(f"训练轮数: {results['epochs']}")
        print(f"最佳验证准确率: {results['best_val_accuracy']:.4f}")
        print(f"最终训练准确率: {results['final_train_accuracy']:.4f}")
        print(f"最终验证准确率: {results['final_val_accuracy']:.4f}")
        print(f"测试准确率: {results['test_results']['accuracy']:.4f}")
        print(f"模型保存路径: {results['model_path']}")
        
        print("\n✅ 训练完成！")
        
    except Exception as e:
        print(f"\n❌ 训练失败: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
