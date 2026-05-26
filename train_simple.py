# -*- coding: utf-8 -*-
"""
简化版训练脚本 - 生成PT格式的情感分析模型
不依赖 Hugging Face datasets 库，使用本地CSV文件
"""

import sys
import os
import json
import random
from pathlib import Path
from collections import Counter

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score

from src.architectures.textcnn import TextCNN
from src.utils.config import Config
from src.utils.text_processor import TextProcessor

# 示例中文情感数据
SAMPLE_DATA = [
    {"text": "这家餐厅的食物非常美味，服务也很周到", "label": 1},
    {"text": "电影太棒了，我非常喜欢", "label": 1},
    {"text": "今天天气真好，心情很愉快", "label": 1},
    {"text": "产品质量很好，下次还会购买", "label": 1},
    {"text": "这本书写得非常精彩，值得一读", "label": 1},
    {"text": "旅行非常愉快，风景很美", "label": 1},
    {"text": "这家店的商品质量非常好", "label": 1},
    {"text": "服务态度很好，非常满意", "label": 1},
    {"text": "饭菜味道不错，价格实惠", "label": 1},
    {"text": "工作环境很好，同事们都很友好", "label": 1},
    {"text": "这家餐厅的食物很难吃，服务态度也很差", "label": 0},
    {"text": "电影太无聊了，浪费时间", "label": 0},
    {"text": "今天天气很差，心情郁闷", "label": 0},
    {"text": "产品质量很差，不会再买了", "label": 0},
    {"text": "这本书写得很烂，不值得一读", "label": 0},
    {"text": "旅行非常糟糕，很失望", "label": 0},
    {"text": "这家店的商品质量很差", "label": 0},
    {"text": "服务态度恶劣，非常不满意", "label": 0},
    {"text": "饭菜很难吃，价格很贵", "label": 0},
    {"text": "工作环境很差，同事们都不友好", "label": 0},
]

class SimpleDataset(Dataset):
    """简化的数据集类"""
    def __init__(self, data, vocab, max_length=50):
        self.data = data
        self.vocab = vocab
        self.max_length = max_length
        self.text_processor = TextProcessor("chinese")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        label = item['label']
        
        # 分词
        tokens = self.text_processor.tokenize(text)
        
        # 转换为ID序列
        token_ids = [self.vocab.get(token, self.vocab["<UNK>"]) for token in tokens]
        
        # 截断或填充
        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]
        else:
            token_ids = token_ids + [self.vocab["<PAD>"]] * (self.max_length - len(token_ids))
        
        return {
            'input_ids': torch.tensor(token_ids, dtype=torch.long),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def build_vocab(data, min_freq=1):
    """构建词汇表"""
    print("构建词汇表...")
    word_counts = Counter()
    text_processor = TextProcessor("chinese")
    
    for example in data:
        tokens = text_processor.tokenize(example["text"])
        word_counts.update(tokens)
    
    vocab = {"<PAD>": 0, "<UNK>": 1}
    vocab_size = 2
    
    for word, count in word_counts.most_common():
        if count >= min_freq:
            vocab[word] = vocab_size
            vocab_size += 1
    
    print(f"词汇表构建完成，大小: {len(vocab)}")
    return vocab

def train_model(model, train_loader, val_loader, epochs=10, device='cpu'):
    """训练模型"""
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    best_val_acc = 0.0
    best_model_state = None
    
    for epoch in range(epochs):
        # 训练
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_acc = train_correct / train_total
        
        # 验证
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = val_correct / val_total
        
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  训练损失: {train_loss/len(train_loader):.4f}, 准确率: {train_acc:.4f}")
        print(f"  验证损失: {val_loss/len(val_loader):.4f}, 准确率: {val_acc:.4f}")
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            print(f"  🏆 新的最佳模型！")
    
    # 恢复最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, best_val_acc

def test_model(model, test_loader, device='cpu'):
    """测试模型"""
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids)
            _, predicted = torch.max(outputs.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_predictions)
    print(f"测试准确率: {accuracy:.4f}")
    return accuracy

def main():
    print("=" * 60)
    print("🎯 训练中文情感分析模型")
    print("=" * 60)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 使用示例数据，复制多份以增加数据量
    data = SAMPLE_DATA * 50  # 20条 * 50 = 1000条数据
    
    # 打乱数据
    random.seed(42)
    random.shuffle(data)
    
    # 划分数据集
    total_size = len(data)
    train_size = int(total_size * 0.7)
    val_size = int(total_size * 0.15)
    
    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]
    
    print(f"\n数据集划分:")
    print(f"  训练集: {len(train_data)} 条")
    print(f"  验证集: {len(val_data)} 条")
    print(f"  测试集: {len(test_data)} 条")
    
    # 构建词汇表
    vocab = build_vocab(train_data)
    
    # 创建数据加载器
    batch_size = 16
    max_length = 50
    
    train_dataset = SimpleDataset(train_data, vocab, max_length)
    val_dataset = SimpleDataset(val_data, vocab, max_length)
    test_dataset = SimpleDataset(test_data, vocab, max_length)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 创建模型
    vocab_size = len(vocab)
    model = TextCNN.create_model(vocab_size)
    print(f"\n模型创建完成，词汇表大小: {vocab_size}")
    
    # 训练模型
    print("\n开始训练...")
    model, best_val_acc = train_model(model, train_loader, val_loader, epochs=15, device=device)
    
    # 测试模型
    print("\n开始测试...")
    test_acc = test_model(model, test_loader, device=device)
    
    # 保存模型
    model_dir = project_root / 'models'
    model_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = model_dir / 'textcnn_chinese.pth'
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'vocab': vocab,
        'model_config': Config.MODEL_CONFIGS['textcnn'],
        'language': 'chinese',
        'accuracy': {
            'best_val': best_val_acc,
            'test': test_acc
        }
    }
    
    torch.save(checkpoint, model_path)
    print(f"\n✅ 模型已保存到: {model_path}")
    
    # 保存词汇表单独文件（用于加载）
    vocab_path = model_dir / 'vocab_chinese.json'
    with open(vocab_path, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    print(f"✅ 词汇表已保存到: {vocab_path}")
    
    print("\n" + "=" * 60)
    print("训练完成！")
    print(f"最佳验证准确率: {best_val_acc:.4f}")
    print(f"测试准确率: {test_acc:.4f}")
    print("=" * 60)

if __name__ == "__main__":
    main()
