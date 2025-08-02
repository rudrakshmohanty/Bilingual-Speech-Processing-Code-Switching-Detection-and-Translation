# Bilingual Speech Processing: Code-Switching Detection and Translation

A comprehensive speech processing system that handles bilingual audio data, performs automatic speech recognition (ASR), detects code-switching patterns, and provides intelligent translation between English and Tamil languages.

## 🎯 Project Overview

This project implements a sophisticated bilingual speech processing pipeline that:
- Transcribes bilingual (English-Tamil) audio using state-of-the-art ASR models
- Analyzes code-switching patterns in multilingual speech
- Converts English words to Tamil for linguistic consistency
- Provides comprehensive performance evaluation with advanced visualizations
- Supports multiple ASR models for comparative analysis

## 🏗️ System Architecture

```
Audio Input (.wav files)
    ↓
┌─────────────────────────┐
│   Speech Recognition    │
│  ├─ OpenAI Whisper     │
│  └─ Wav2Vec2 Tamil     │
└─────────────────────────┘
    ↓
┌─────────────────────────┐
│  Code-Switch Detection  │
│  └─ Language Analysis   │
└─────────────────────────┘
    ↓
┌─────────────────────────┐
│  English-Tamil          │
│  Translation            │
└─────────────────────────┘
    ↓
┌─────────────────────────┐
│  Performance Analysis   │
│  ├─ WER Calculation     │
│  ├─ Statistical Analysis│
│  └─ Interactive Viz     │
└─────────────────────────┘
```

## 🎙️ Dataset Information

### Custom Code-Switching Dataset
- **Size**: 36 bilingual audio samples
- **Languages**: English-Tamil code-switched speech
- **Format**: WAV files (16kHz sampling rate recommended)
- **Content**: Natural conversational speech with organic language switching
- **Ground Truth**: Manual transcriptions with accurate code-switching annotations
- **Availability**: Private dataset (not publicly available)

### Dataset Structure
```
CodeSwitchDataset/
├── CodeSwitch_Dataset_wav/     # Audio files
│   ├── Voice_1.wav
│   ├── Voice_2.wav
│   └── ...
└── CodeSwitch_36.csv          # Ground truth transcriptions
    ├── SNo
    ├── FileName
    └── Transcription
```

## 🚀 Features

### 1. Multi-Model ASR Support
- **OpenAI Whisper Large**: State-of-the-art multilingual ASR
- **Wav2Vec2 Tamil**: Specialized Tamil language model
- **GPU Acceleration**: CUDA support for faster processing

### 2. Advanced Code-Switching Analysis
- Automatic detection of language boundaries
- Multilingual content identification
- Code-switching pattern analysis
- Language composition statistics

### 3. Intelligent Translation
- **Model**: M2M100 English-to-Tamil translation
- **Technical Term Preservation**: Maintains domain-specific vocabulary
- **Context-Aware Translation**: Chunk-based processing for better accuracy
- **Fallback Mechanisms**: Error handling and graceful degradation

### 4. Comprehensive Evaluation
- **Word Error Rate (WER)** calculation
- Statistical performance metrics
- Comparative model analysis
- Interactive visualizations

### 5. Advanced Visualizations
- **Interactive Dashboards**: Plotly-based dynamic charts
- **Performance Heatmaps**: WER analysis by content categories
- **Trend Analysis**: Performance patterns across samples
- **Futuristic UI**: Modern, responsive design

## 🛠️ Technical Stack

### Core Libraries
```python
- whisper                  # OpenAI Whisper ASR
- transformers            # Hugging Face models
- torch                   # PyTorch deep learning
- librosa                 # Audio processing
- soundfile              # Audio I/O
```

### Analysis & Visualization
```python
- pandas                  # Data manipulation
- numpy                   # Numerical computing
- matplotlib              # Static plotting
- seaborn                # Statistical visualization
- plotly                 # Interactive dashboards
- jiwer                  # WER calculation
```

### Language Processing
```python
- langdetect             # Language detection
- transformers           # Translation models
- re                     # Regular expressions
```

## 📦 Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)
- 8GB+ RAM
- 5GB+ storage space

### Environment Setup
```bash
# Clone the repository
git clone https://github.com/your-username/Bilingual-Speech-Processing-Code-Switching-Detection-and-Translation.git
cd Bilingual-Speech-Processing-Code-Switching-Detection-and-Translation

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -U openai-whisper
pip install jiwer
pip install transformers
pip install torch torchvision torchaudio
pip install librosa soundfile
pip install pandas numpy matplotlib seaborn plotly
pip install langdetect scipy
```

### GPU Setup (Recommended)
```bash
# For CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 🎯 Usage

### 1. Basic Transcription Pipeline
```python
import whisper
import torch

# Load model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = whisper.load_model("large").to(device)

# Transcribe audio
result = model.transcribe("path/to/audio.wav", fp16=True)
transcription = result["text"]
```

### 2. Batch Processing
```python
import os
import pandas as pd

# Process multiple files
audio_folder = "path/to/audio/folder"
data = []

for file in sorted(os.listdir(audio_folder)):
    if file.endswith(".wav"):
        audio_path = os.path.join(audio_folder, file)
        result = model.transcribe(audio_path, fp16=True)
        data.append([file, result["text"]])

# Save results
df = pd.DataFrame(data, columns=["FileName", "Transcription"])
df.to_csv("transcriptions.csv", index=False)
```

### 3. WER Evaluation
```python
from jiwer import wer

# Calculate Word Error Rate
reference = "ground truth transcription"
hypothesis = "model transcription"
error_rate = wer(reference, hypothesis)
print(f"WER: {error_rate:.4f}")
```

### 4. English-Tamil Translation
```python
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

# Load translation model
model_name = "suriya7/English-to-Tamil"
tokenizer = M2M100Tokenizer.from_pretrained(model_name)
translator = M2M100ForConditionalGeneration.from_pretrained(model_name)

# Translate text
def translate_english_to_tamil(text):
    encoded = tokenizer(text, return_tensors="pt")
    translated = translator.generate(**encoded, forced_bos_token_id=tokenizer.get_lang_id("ta"))
    return tokenizer.batch_decode(translated, skip_special_tokens=True)[0]
```

## 📊 Performance Metrics

### Evaluation Criteria
1. **Word Error Rate (WER)**: Primary accuracy metric
2. **Character Error Rate (CER)**: Fine-grained accuracy
3. **Language Detection Accuracy**: Code-switching identification
4. **Translation Quality**: BLEU score for Tamil conversion

### Benchmark Results
- **Average WER**: Varies by model and content complexity
- **Multilingual Performance**: Enhanced for code-switched content
- **Processing Speed**: GPU acceleration provides 5-10x speedup
- **Memory Usage**: ~4GB VRAM for large models

## 📈 Visualization Features

### Interactive Dashboards
- **WER Distribution Analysis**: Histogram with statistical overlays
- **Performance vs Length**: Scatter plots with trend analysis
- **Top Error Samples**: Bar charts for worst-performing samples
- **Temporal Analysis**: Line charts showing performance trends
- **Correlation Heatmaps**: Cross-metric relationship analysis

### Export Formats
- **HTML**: Interactive web-based dashboards
- **PNG**: High-resolution static images
- **CSV**: Raw data for further analysis
- **PDF**: Report-ready visualizations

## 🔧 Configuration

### Model Configuration
```python
# Whisper model sizes
WHISPER_MODELS = {
    "tiny": "39M parameters, fastest",
    "base": "74M parameters, balanced",
    "small": "244M parameters, good quality",
    "medium": "769M parameters, better quality", 
    "large": "1550M parameters, best quality"
}

# Processing parameters
CONFIG = {
    "sample_rate": 16000,
    "chunk_size": 30,  # seconds
    "language_detection_threshold": 0.7,
    "translation_chunk_size": 5,  # words
    "gpu_memory_fraction": 0.8
}
```

### File Paths
```python
PATHS = {
    "audio_folder": "/path/to/audio/files",
    "ground_truth": "/path/to/ground_truth.csv",
    "output_dir": "/path/to/output",
    "model_cache": "/path/to/model/cache"
}
```

## 🎨 Advanced Features

### 1. Adaptive Processing
- **Dynamic chunk sizing** based on content complexity
- **Quality-speed trade-offs** with model selection
- **Memory optimization** for large-scale processing

### 2. Error Analysis
- **Confusion matrices** for common error patterns
- **Language-specific error rates**
- **Temporal error distribution**

### 3. Real-time Processing
- **Streaming transcription** for live audio
- **Incremental translation** with context preservation
- **Live performance monitoring**

## 🤝 Contributing

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Code formatting
black src/
isort src/

# Type checking
mypy src/
```

### Contribution Guidelines
1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **OpenAI** for the Whisper ASR model
- **Hugging Face** for transformer models and infrastructure
- **Facebook Research** for Wav2Vec2 architecture
- **Research Community** for code-switching detection methodologies

## 📞 Contact

**Project Maintainers:**
- **Niranjan**: ASR implementation and evaluation
- **Krithika**: Dataset creation and validation

**Repository**: [GitHub Link](https://github.com/your-username/Bilingual-Speech-Processing-Code-Switching-Detection-and-Translation)

## 🔮 Future Work

### Planned Enhancements
- **Real-time processing** with WebRTC integration
- **Multi-language support** beyond English-Tamil
- **Improved code-switching detection** with transformer models
- **Mobile deployment** with optimized models
- **API service** for cloud-based processing

### Research Directions
- **Few-shot learning** for low-resource languages
- **Contextual translation** with conversation history
- **Emotional speech analysis** in multilingual content
- **Cross-lingual speaker identification**

---

*This project represents cutting-edge research in multilingual speech processing and aims to bridge language barriers in diverse linguistic communities.*