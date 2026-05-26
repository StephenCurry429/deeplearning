# 中文情感分析系统

> 📚 参考来源：[https://gitee.com/Snake-Konginchrist/deep-learning-text-sentiment-analysis.git](https://gitee.com/Snake-Konginchrist/deep-learning-text-sentiment-analysis.git)

基于深度学习的中文情感分析平台，支持多种模型架构。

## 📋 项目概览

- **模型类型**: TextCNN、BiLSTM、BERT
- **数据集**: 中文酒店评论数据集（7765条）
- **词汇表**: 3664个字符
- **准确率**: TextCNN约92%，BERT约95%

## 🚀 快速开始

### 1. 环境要求

- Python 3.10+
- PyTorch 2.0+
- Flask 2.0+
- Node.js 18+ (前端)

### 2. 安装依赖

```bash
# 安装Python依赖
pip install -r requirements.txt

# 安装前端依赖（可选）
cd webui
npm install
```

### 3. 启动服务

```bash
# 启动后端服务
python server.py

# 启动前端开发服务器（可选）
cd webui
npm run dev
```

### 4. 访问界面

- API服务: http://localhost:5000
- 前端界面: http://localhost:5173

## 📡 API 接口

### 单文本情感分析

```bash
POST /api/analyze
Content-Type: application/json

{"text": "要分析的文本", "model": "textcnn"}
```

**参数说明**:
- `text`: 要分析的文本内容
- `model`: 模型类型 (textcnn/bilstm/bert)，默认为textcnn

**响应示例**:
```json
{
    "text": "这个电影真好看！",
    "sentiment": "正面",
    "confidence": 99.63,
    "model_type": "TextCNN"
}
```

### 批量文本分析

```bash
POST /api/analyze/batch
Content-Type: application/json

{"texts": ["文本1", "文本2", "文本3"]}
```

### 健康检查

```bash
GET /api/health
```

### 模型列表

```bash
GET /api/models
```

## 📁 项目结构

```
├── src/                     # 源代码目录
│   ├── api/                 # API接口
│   │   ├── app.py           # Flask应用
│   │   └── training_api.py  # 训练相关API
│   ├── architectures/       # 模型架构
│   │   ├── textcnn.py       # TextCNN模型
│   │   ├── bilstm.py        # BiLSTM模型
│   │   └── bert_model.py    # BERT模型
│   ├── training/            # 训练模块
│   │   ├── trainer.py       # 训练器基类
│   │   ├── textcnn_trainer.py
│   │   ├── bilstm_trainer.py
│   │   ├── bert_trainer.py
│   │   └── trainer_manager.py
│   ├── services/            # 业务服务
│   │   └── sentiment_analyzer.py
│   ├── scripts/             # 脚本工具
│   │   └── dataset_loader.py
│   └── utils/               # 工具函数
│       ├── config.py        # 配置管理
│       └── text_processor.py
├── webui/                   # Vue前端
│   ├── src/                 # 前端源代码
│   └── public/              # 静态资源
├── docs/                    # 文档目录
├── models/                  # 模型文件
├── templates/               # 模板文件
├── server.py                # 服务器启动脚本
└── requirements.txt         # 依赖列表
```

## 🧠 模型说明

### TextCNN

基于卷积神经网络的文本分类模型：
- 使用字符级词嵌入
- 多尺寸卷积核提取特征
- 最大池化获取关键特征

### BiLSTM

基于双向长短时记忆网络：
- 双向LSTM捕捉上下文信息
- Dropout防止过拟合
- 全连接层分类

### BERT

预训练语言模型：
- 使用BERT-base-chinese
- 微调适应情感分析任务
- 最高准确率约95%

## 📊 性能指标

| 模型 | 准确率 | 模型大小 |
|------|--------|----------|
| TextCNN | ~92% | ~2.7MB |
| BiLSTM | ~91% | ~5.2MB |
| BERT | ~95% | ~418MB |

## 🔧 训练说明

如需重新训练模型：

```python
from src.training.trainer_manager import TrainerManager

# 创建训练管理器
manager = TrainerManager()

# 训练指定模型
manager.train_model('textcnn')  # 可选: textcnn, bilstm, bert
```

## 📚 文档

详细文档请查看 `docs/` 目录：

- [API.md](docs/API.md) - API接口文档
- [DATASETS.md](docs/DATASETS.md) - 数据集说明
- [DEPLOYMENT.md](docs/DEPLOYMENT.md) - 部署指南
- [DEVELOPMENT.md](docs/DEVELOPMENT.md) - 开发指南
- [TECHNICAL.md](docs/TECHNICAL.md) - 技术文档

## 📄 许可证

MIT License

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！