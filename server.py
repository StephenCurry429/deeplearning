import sys
import os
import json
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import torch

try:
    import onnxruntime as ort
    onnx_support = True
except ImportError:
    onnx_support = False

from architectures.textcnn import TextCNN

app = Flask(__name__)
CORS(app)

app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB

model = None
vocab = None
device = torch.device('cpu')
current_model_name = "textcnn_chinese"
current_model_format = "pth"
onnx_session = None

def load_vocab(vocab_path):
    """加载词汇表"""
    if vocab_path.exists():
        try:
            with open(vocab_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            pass
    return None

def load_pth_model(model_path):
    """加载 PyTorch .pth 或 .pt 模型"""
    global model, vocab, current_model_name, current_model_format, onnx_session
    
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        if 'vocab' in checkpoint:
            vocab = checkpoint['vocab']
        else:
            vocab_path = model_path.parent / f"{model_path.stem}_vocab.json"
            vocab = load_vocab(vocab_path)
            if vocab is None:
                vocab = create_default_vocab()
        
        # 优先从checkpoint中读取模型配置，如果不存在则使用默认配置
        if 'model_config' in checkpoint:
            model_config = checkpoint['model_config']
            print(f"从checkpoint加载模型配置: {model_config}")
        else:
            model_config = {
                'vocab_size': len(vocab),
                'embedding_dim': 300,
                'num_filters': 100,
                'filter_sizes': [3, 4, 5],
                'dropout': 0.5,
                'num_classes': 2
            }
        
        # 确保vocab_size正确
        model_config['vocab_size'] = len(vocab)
        
        model = TextCNN(model_config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        onnx_session = None
        current_model_format = "pth"
        current_model_name = Path(model_path).stem
        
        print(f"PTH模型加载成功! Name: {current_model_name}, Vocab size: {len(vocab)}")
        return True
    except Exception as e:
        print(f"加载PTH模型失败: {e}")
        return False

def load_onnx_model(model_path):
    """加载 ONNX 模型"""
    global model, vocab, current_model_name, current_model_format, onnx_session
    
    if not onnx_support:
        print("ONNX支持不可用，请安装 onnxruntime")
        return False
    
    try:
        onnx_session = ort.InferenceSession(str(model_path))
        
        vocab_path = model_path.parent / f"{model_path.stem}_vocab.json"
        vocab = load_vocab(vocab_path)
        if vocab is None:
            vocab = create_default_vocab()
        
        model = None
        current_model_format = "onnx"
        current_model_name = Path(model_path).stem
        
        print(f"ONNX模型加载成功! Name: {current_model_name}, Vocab size: {len(vocab)}")
        return True
    except Exception as e:
        print(f"加载ONNX模型失败: {e}")
        return False

def create_default_vocab():
    """创建默认词汇表"""
    vocab = {'<PAD>': 0, '<UNK>': 1}
    common_chars = "的了一是我不人在他有这个上们来到时大地为子中你说生国年着就那和要她出也得里后自以会家可下而过天去能对小多然于心学么之都好看起发当没成只如事把还用第样道想作种开美总从无情己面最女但现前些所同日手又行意动方它头经长儿回位分爱老因很给名法间斯知世什两次使身者被高已亲其进此话常与活正感"
    for char in common_chars:
        if char not in vocab:
            vocab[char] = len(vocab)
    return vocab

def load_model(model_path=None):
    """加载模型（自动检测格式）"""
    if model_path is None:
        model_path = project_root / 'models' / 'textcnn_chinese.pth'
    
    model_path = Path(model_path)
    
    if not model_path.exists():
        print(f"模型文件不存在: {model_path}")
        return False
    
    ext = model_path.suffix.lower()
    
    if ext in ['.pth', '.pt']:
        return load_pth_model(model_path)
    elif ext == '.onnx':
        return load_onnx_model(model_path)
    else:
        print(f"不支持的模型格式: {ext}")
        return False

def predict(text):
    """预测文本情感"""
    global model, vocab, onnx_session, current_model_format
    
    if vocab is None:
        return None, None
    
    indices = []
    for char in text[:128]:
        indices.append(vocab.get(char, vocab['<UNK>']))
    if len(indices) < 128:
        indices += [vocab['<PAD>']] * (128 - len(indices))
    
    input_ids = torch.tensor([indices], dtype=torch.long)
    
    if current_model_format == "onnx" and onnx_session:
        inputs = {onnx_session.get_inputs()[0].name: input_ids.numpy()}
        outputs = onnx_session.run(None, inputs)
        probabilities = torch.softmax(torch.tensor(outputs[0]), dim=1)
        predicted = torch.argmax(torch.tensor(outputs[0]), dim=1).item()
        confidence = probabilities[0][predicted].item()
    elif model:
        with torch.no_grad():
            outputs = model(input_ids)
            probabilities = torch.softmax(outputs, dim=1)
            predicted = torch.argmax(outputs, dim=1).item()
            confidence = probabilities[0][predicted].item()
    else:
        return None, None
    
    sentiment = "正面" if predicted == 1 else "负面"
    return sentiment, confidence

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': '请输入文本'}), 400
        
        sentiment, confidence = predict(text)
        
        if sentiment is None:
            return jsonify({'error': '模型未加载或预测失败'}), 500
        
        return jsonify({
            'text': text,
            'sentiment': sentiment,
            'confidence': round(confidence * 100, 2),
            'model_type': current_model_name,
            'model_format': current_model_format
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/upload-model', methods=['POST'])
def upload_model():
    try:
        if 'file' not in request.files:
            return jsonify({'error': '未选择文件'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': '未选择文件'}), 400
        
        filename = file.filename.lower()
        ext = Path(filename).suffix.lower()
        
        if ext not in ['.pt', '.pth', '.onnx']:
            return jsonify({'error': '仅支持 .pt, .pth 或 .onnx 格式的模型文件'}), 400
        
        models_dir = project_root / 'models'
        models_dir.mkdir(exist_ok=True)
        
        file_path = models_dir / file.filename
        
        if file_path.exists():
            version = 1
            while (models_dir / f"{file_path.stem}_{version}{ext}").exists():
                version += 1
            file_path = models_dir / f"{file_path.stem}_{version}{ext}"
        
        file.save(str(file_path))
        
        if load_model(file_path):
            return jsonify({
                'success': True,
                'message': f'模型上传并加载成功: {file_path.name}',
                'model_name': file_path.stem,
                'model_format': ext[1:]
            })
        else:
            os.remove(file_path)
            return jsonify({'error': '模型文件无效，无法加载'}), 400
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/models', methods=['GET'])
def get_models():
    models_dir = project_root / 'models'
    models_list = []
    
    if models_dir.exists():
        for ext in ['*.pt', '*.pth', '*.onnx']:
            for file in models_dir.glob(ext):
                models_list.append({
                    'name': file.stem,
                    'path': str(file),
                    'format': file.suffix[1:]
                })
    
    return jsonify({
        'models': models_list,
        'current_model': current_model_name,
        'current_format': current_model_format,
        'model_loaded': vocab is not None,
        'onnx_support': onnx_support
    })

@app.route('/load-model', methods=['POST'])
def load_model_by_name():
    try:
        data = request.get_json()
        model_name = data.get('model_name', '')
        
        if not model_name:
            return jsonify({'error': '请提供模型名称'}), 400
        
        models_dir = project_root / 'models'
        
        for ext in ['.pth', '.pt', '.onnx']:
            model_path = models_dir / f'{model_name}{ext}'
            if model_path.exists():
                if load_model(model_path):
                    return jsonify({
                        'success': True,
                        'message': f'模型加载成功: {model_name}',
                        'model_name': model_name,
                        'model_format': ext[1:]
                    })
                else:
                    return jsonify({'error': '模型加载失败'}), 500
        
        return jsonify({'error': f'模型文件不存在: {model_name}'}), 404
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy', 
        'model_loaded': vocab is not None,
        'current_model': current_model_name,
        'current_format': current_model_format,
        'onnx_support': onnx_support
    })

if __name__ == '__main__':
    print('=' * 60)
    print('启动情感分析服务器')
    print('=' * 60)
    
    load_model()
    print('服务器启动成功！')
    app.run(host='0.0.0.0', port=5000, debug=True)