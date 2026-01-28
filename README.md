# 🔍 DeepFake Video Detection System

A comprehensive AI-powered deepfake detection application built with Streamlit and Computer Vision techniques.

![DeepFake Detector](https://img.shields.io/badge/AI-Powered-blue)
![Python](https://img.shields.io/badge/Python-3.8+-green)
![OpenCV](https://img.shields.io/badge/OpenCV-4.9-red)

## 🎯 Features

- **Real-time Video Analysis**: Upload and analyze videos instantly
- **Multi-Frame Detection**: Analyzes 8 key frames for comprehensive detection
- **Advanced Algorithms**:
  - Frequency domain analysis (DCT artifacts)
  - Color consistency detection
  - Edge analysis for manipulation traces
  - Texture pattern recognition
  - Temporal consistency checking
- **Professional UI**: Modern, responsive interface with real-time metrics
- **Detailed Results**: Frame-by-frame analysis with confidence scores
- **Risk Assessment**: Automatic risk level classification

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd deepfake-detection
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the application**
```bash
streamlit run app.py
```

4. **Access the app**
Open your browser and navigate to `http://localhost:8501`

## 📦 Dependencies

- **Streamlit**: Web application framework
- **OpenCV**: Computer vision and image processing
- **NumPy**: Numerical computing
- **Pillow**: Image handling

## 🎓 How It Works

### Detection Pipeline

1. **Frame Extraction**: Extracts 8 representative frames from the uploaded video
2. **Face Detection**: Identifies facial regions using Haar Cascade
3. **Multi-Feature Analysis**:
   - **DCT Analysis**: Detects compression artifacts in frequency domain
   - **Color Analysis**: Examines color consistency in LAB color space
   - **Edge Detection**: Analyzes edge patterns using Canny edge detector
   - **Texture Analysis**: Evaluates texture using Sobel gradients
4. **Temporal Consistency**: Checks for frame-to-frame inconsistencies
5. **Scoring**: Combines all metrics into final confidence score

### Algorithm Details

#### Frequency Domain Analysis
```python
- Applies Discrete Cosine Transform (DCT)
- Analyzes high-frequency components
- Detects compression artifacts typical of deepfakes
```

#### Color Consistency Check
```python
- Converts to LAB color space
- Analyzes luminance channel variance
- Detects unnatural color patterns
```

#### Edge Analysis
```python
- Canny edge detection
- Calculates edge density
- Identifies unnatural boundaries
```

#### Texture Analysis
```python
- Sobel gradient computation
- Analyzes texture patterns
- Detects artificial skin textures
```

#### Temporal Analysis
```python
- Frame-to-frame difference calculation
- Variance in motion patterns
- Detects temporal inconsistencies
```

## 📊 Output Metrics

- **Confidence Score**: Overall probability of deepfake (0-100%)
- **Prediction**: AUTHENTIC / UNCERTAIN / DEEPFAKE DETECTED
- **Risk Level**: LOW / MEDIUM / HIGH
- **Frame Scores**: Individual analysis for each frame
- **Temporal Score**: Consistency across frames

## 🌐 Deployment

### Streamlit Community Cloud

1. Push your code to GitHub
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository
4. Deploy with one click!

### Hugging Face Spaces

1. Create a new Space on [Hugging Face](https://huggingface.co/spaces)
2. Upload your files
3. Set SDK to Streamlit
4. Your app will be live!

### Local Network Deployment

```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

## 🎨 UI Features

- **Modern Gradient Design**: Cyberpunk-inspired color scheme
- **Responsive Layout**: Works on desktop and mobile
- **Real-time Feedback**: Progress indicators during analysis
- **Interactive Metrics**: Hover effects and animations
- **Frame Gallery**: Visual display of analyzed frames

## 📝 Usage Guide

1. **Upload Video**: Click on upload area or drag & drop video file
2. **Preview**: Video preview displays automatically
3. **Analyze**: Click "Analyze Video" button
4. **Review Results**: 
   - Check overall prediction and confidence
   - Review frame-by-frame scores
   - Examine detailed metrics
5. **Interpret**: Use risk level and detailed explanations

## ⚠️ Important Notes

- **Not Legal Evidence**: Results are AI-assisted predictions, not definitive proof
- **Quality Dependent**: Detection accuracy depends on video quality
- **Evolving Technology**: Deepfake techniques constantly evolve
- **Multi-Verification**: Always verify through multiple methods
- **Educational Purpose**: Designed for research and educational use

## 🔬 Technical Specifications

### Supported Formats
- MP4 (H.264, H.265)
- AVI
- MOV
- MKV

### Performance
- Analysis time: 5-15 seconds (depending on video length)
- Frame extraction: 8 frames per video
- Face detection: Haar Cascade classifier
- Real-time progress updates

### System Requirements
- RAM: 2GB minimum (4GB recommended)
- Storage: 100MB for dependencies
- Processor: Any modern CPU
- No GPU required (CPU-based processing)

## 🛠️ Customization

### Adjust Detection Sensitivity

Modify thresholds in `app.py`:

```python
# Line ~150
if final_score > 0.6:  # Change threshold (0.6 = 60%)
    prediction = "DEEPFAKE DETECTED"
```

### Change Number of Frames

```python
# Line ~180
frames = extract_frames(video_path, num_frames=8)  # Adjust number
```

### Modify UI Colors

Edit CSS in `app.py`:

```css
/* Line ~30 */
background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 50%, #0f1419 100%);
```

## 📚 Academic Context

This project is developed as a **Final Year Project** demonstrating:
- Computer Vision techniques
- Machine Learning implementation
- Real-time video processing
- Modern web application development
- AI ethics and responsible use

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Deep learning model integration
- Additional detection algorithms
- Performance optimization
- UI/UX enhancements
- Multi-language support

## 📄 License

This project is for educational purposes. Please check with your institution regarding project submission guidelines.

## 🙏 Acknowledgments

- OpenCV community for computer vision tools
- Streamlit team for the amazing framework
- Research papers on deepfake detection
- Academic advisors and project guides

## 📧 Support

For questions or issues:
- Open an issue on GitHub
- Contact your project supervisor
- Review Streamlit documentation

## 🔮 Future Enhancements

- [ ] Deep learning model integration (CNN, ResNet)
- [ ] Audio analysis for voice deepfakes
- [ ] Batch processing for multiple videos
- [ ] Export detailed PDF reports
- [ ] API endpoint for external integration
- [ ] Database for analysis history
- [ ] User authentication system
- [ ] Advanced visualization dashboard

## 📊 Project Status

✅ **Ready for Deployment**
- Core detection algorithms implemented
- Professional UI completed
- Documentation comprehensive
- Deployment-ready code

---

**Made with ❤️ for academic excellence**

*Powered by Computer Vision & AI*
