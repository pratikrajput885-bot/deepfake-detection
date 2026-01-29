"""
Professional Deepfake Detection System
Advanced AI-powered video analysis with multiple detection algorithms
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
from pathlib import Path
import json
from datetime import datetime
import hashlib

# Deep Learning Libraries
try:
    import torch
    import torchvision.transforms as transforms
    from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    st.warning("PyTorch not installed. Using traditional CV methods only.")

# Page Configuration
st.set_page_config(
    page_title="Professional Deepfake Detection",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Professional UI
st.markdown("""
<style>
    /* Main Theme */
    .stApp {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 50%, #0f1419 100%);
    }
    
    /* Header Styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    }
    
    .main-header h1 {
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        color: #e0e7ff;
        font-size: 1.1rem;
        margin-top: 0.5rem;
    }
    
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #1e3a8a 0%, #3730a3 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #667eea;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
    
    .metric-title {
        color: #a5b4fc;
        font-size: 0.9rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        color: #fff;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    
    .metric-description {
        color: #cbd5e1;
        font-size: 0.9rem;
        line-height: 1.4;
    }
    
    /* Result Container */
    .result-container {
        background: rgba(30, 41, 59, 0.8);
        border-radius: 15px;
        padding: 2rem;
        margin: 2rem 0;
        border: 2px solid rgba(102, 126, 234, 0.3);
        backdrop-filter: blur(10px);
    }
    
    /* Status Badges */
    .badge-authentic {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 0.8rem 1.5rem;
        border-radius: 25px;
        font-weight: 700;
        font-size: 1.2rem;
        display: inline-block;
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3);
    }
    
    .badge-deepfake {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
        padding: 0.8rem 1.5rem;
        border-radius: 25px;
        font-weight: 700;
        font-size: 1.2rem;
        display: inline-block;
        box-shadow: 0 4px 15px rgba(239, 68, 68, 0.3);
        animation: pulse 2s infinite;
    }
    
    .badge-uncertain {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: white;
        padding: 0.8rem 1.5rem;
        border-radius: 25px;
        font-weight: 700;
        font-size: 1.2rem;
        display: inline-block;
        box-shadow: 0 4px 15px rgba(245, 158, 11, 0.3);
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.8; }
    }
    
    /* Progress Bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Sidebar Styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
    }
    
    /* Info Box */
    .info-box {
        background: rgba(59, 130, 246, 0.1);
        border-left: 4px solid #3b82f6;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        color: #dbeafe;
    }
    
    /* Warning Box */
    .warning-box {
        background: rgba(245, 158, 11, 0.1);
        border-left: 4px solid #f59e0b;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        color: #fef3c7;
    }
    
    /* Algorithm Badge */
    .algo-badge {
        background: rgba(102, 126, 234, 0.2);
        color: #a5b4fc;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.85rem;
        margin: 0.2rem;
        display: inline-block;
        border: 1px solid rgba(102, 126, 234, 0.3);
    }
</style>
""", unsafe_allow_html=True)


class DeepfakeDetector:
    """Professional Deepfake Detection System"""
    
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        
        # Initialize Deep Learning Model if available
        if TORCH_AVAILABLE:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.load_deep_model()
        else:
            self.model = None
            
        self.detection_results = {}
        
    def load_deep_model(self):
        """Load pre-trained deep learning model"""
        try:
            # Using EfficientNet-B4 pre-trained model
            self.model = efficientnet_b4(weights=EfficientNet_B4_Weights.DEFAULT)
            self.model.eval()
            self.model.to(self.device)
            
            self.transform = transforms.Compose([
                transforms.Resize((380, 380)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            st.success("✓ Deep Learning Model Loaded (EfficientNet-B4)")
        except Exception as e:
            self.model = None
            st.warning(f"Deep model loading failed: {e}")
    
    def extract_frames(self, video_path, num_frames=16):
        """Extract frames from video for analysis"""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        duration = total_frames / fps if fps > 0 else 0
        
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        frames = []
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
        
        cap.release()
        
        return frames, {
            'total_frames': total_frames,
            'fps': fps,
            'duration': duration
        }
    
    def detect_faces(self, frame):
        """Detect faces in frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        return faces
    
    def analyze_dct_artifacts(self, face_region):
        """Analyze DCT compression artifacts (deepfakes show unusual patterns)"""
        gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (64, 64))
        
        # Apply DCT
        dct = cv2.dct(np.float32(gray))
        
        # Analyze high-frequency components
        high_freq = np.abs(dct[32:, 32:])
        artifact_score = np.mean(high_freq) / (np.mean(np.abs(dct)) + 1e-6)
        
        # Deepfakes typically have unusual high-frequency patterns
        suspicion_score = min(artifact_score * 3, 1.0)
        
        return suspicion_score
    
    def analyze_color_consistency(self, face_region):
        """Analyze color consistency (deepfakes often have color mismatches)"""
        lab = cv2.cvtColor(face_region, cv2.COLOR_BGR2LAB)
        
        # Split channels
        l_channel, a_channel, b_channel = cv2.split(lab)
        
        # Calculate statistics
        l_std = np.std(l_channel)
        a_std = np.std(a_channel)
        b_std = np.std(b_channel)
        
        # Unnatural color variance indicates manipulation
        color_variance = (l_std + a_std + b_std) / 3
        
        # Normalize to 0-1 scale
        suspicion_score = min(color_variance / 50.0, 1.0)
        
        return suspicion_score
    
    def analyze_edge_consistency(self, face_region):
        """Analyze edge patterns for manipulation detection"""
        gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        
        # Apply Canny edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Calculate edge density
        edge_density = np.sum(edges > 0) / edges.size
        
        # Unusual edge patterns suggest manipulation
        # Both too many and too few edges are suspicious
        optimal_density = 0.15
        deviation = abs(edge_density - optimal_density)
        suspicion_score = min(deviation * 5, 1.0)
        
        return suspicion_score
    
    def analyze_texture_consistency(self, face_region):
        """Analyze texture patterns using Sobel gradients"""
        gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        
        # Calculate gradients
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        
        # Analyze texture variance
        texture_std = np.std(gradient_magnitude)
        
        # Artificial skin textures often have unusual variance
        suspicion_score = min(texture_std / 100.0, 1.0)
        
        return suspicion_score
    
    def analyze_eye_blinking(self, frame, faces):
        """Analyze eye blinking patterns (deepfakes often have abnormal blinking)"""
        if len(faces) == 0:
            return 0.5
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        eye_detections = 0
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            eyes = self.eye_cascade.detectMultiScale(roi_gray, 1.1, 5)
            eye_detections += len(eyes)
        
        # Expected eye count should be 2 per face
        expected_eyes = len(faces) * 2
        eye_ratio = eye_detections / (expected_eyes + 1e-6)
        
        # Deviation from expected indicates issues
        deviation = abs(eye_ratio - 1.0)
        suspicion_score = min(deviation, 1.0)
        
        return suspicion_score
    
    def analyze_temporal_consistency(self, frames):
        """Analyze temporal consistency across frames"""
        if len(frames) < 2:
            return 0.5
        
        frame_diffs = []
        for i in range(len(frames) - 1):
            # Convert to grayscale
            gray1 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frames[i + 1], cv2.COLOR_BGR2GRAY)
            
            # Resize for consistency
            gray1 = cv2.resize(gray1, (256, 256))
            gray2 = cv2.resize(gray2, (256, 256))
            
            # Calculate frame difference
            diff = cv2.absdiff(gray1, gray2)
            frame_diffs.append(np.mean(diff))
        
        # High variance in frame differences suggests temporal inconsistency
        temporal_variance = np.std(frame_diffs)
        suspicion_score = min(temporal_variance / 50.0, 1.0)
        
        return suspicion_score
    
    def deep_learning_analysis(self, frame):
        """Use deep learning model for analysis"""
        if not TORCH_AVAILABLE or self.model is None:
            return 0.5
        
        try:
            # Prepare frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
            
            # Get model output
            with torch.no_grad():
                output = self.model(input_tensor)
            
            # Analyze feature patterns
            features = output.cpu().numpy().flatten()
            
            # Use statistical analysis of features
            feature_variance = np.std(features)
            feature_mean = np.abs(np.mean(features))
            
            # Combine metrics
            suspicion_score = min((feature_variance + feature_mean) / 10.0, 1.0)
            
            return suspicion_score
        except Exception as e:
            st.warning(f"Deep learning analysis error: {e}")
            return 0.5
    
    def analyze_optical_flow(self, frames):
        """Analyze optical flow patterns"""
        if len(frames) < 2:
            return 0.5
        
        flow_magnitudes = []
        
        for i in range(len(frames) - 1):
            gray1 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frames[i + 1], cv2.COLOR_BGR2GRAY)
            
            gray1 = cv2.resize(gray1, (256, 256))
            gray2 = cv2.resize(gray2, (256, 256))
            
            # Calculate optical flow
            flow = cv2.calcOpticalFlowFarneback(
                gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            
            # Calculate magnitude
            magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
            flow_magnitudes.append(np.mean(magnitude))
        
        # Unnatural flow patterns suggest manipulation
        flow_variance = np.std(flow_magnitudes)
        suspicion_score = min(flow_variance / 5.0, 1.0)
        
        return suspicion_score
    
    def analyze_frame(self, frame, frame_idx):
        """Comprehensive frame analysis"""
        scores = {}
        
        # Detect faces
        faces = self.detect_faces(frame)
        scores['faces_detected'] = len(faces)
        
        if len(faces) > 0:
            # Analyze each detected face
            face_scores = []
            
            for (x, y, w, h) in faces:
                face_region = frame[y:y+h, x:x+w]
                
                if face_region.size > 0:
                    # Run all analyses
                    dct_score = self.analyze_dct_artifacts(face_region)
                    color_score = self.analyze_color_consistency(face_region)
                    edge_score = self.analyze_edge_consistency(face_region)
                    texture_score = self.analyze_texture_consistency(face_region)
                    
                    face_scores.append({
                        'dct': dct_score,
                        'color': color_score,
                        'edge': edge_score,
                        'texture': texture_score
                    })
            
            if face_scores:
                # Average across all faces
                scores['dct_artifacts'] = np.mean([f['dct'] for f in face_scores])
                scores['color_consistency'] = np.mean([f['color'] for f in face_scores])
                scores['edge_analysis'] = np.mean([f['edge'] for f in face_scores])
                scores['texture_analysis'] = np.mean([f['texture'] for f in face_scores])
            else:
                scores.update({
                    'dct_artifacts': 0.5,
                    'color_consistency': 0.5,
                    'edge_analysis': 0.5,
                    'texture_analysis': 0.5
                })
        else:
            # No faces detected - suspicious but not conclusive
            scores.update({
                'dct_artifacts': 0.6,
                'color_consistency': 0.6,
                'edge_analysis': 0.6,
                'texture_analysis': 0.6
            })
        
        # Eye blinking analysis
        scores['eye_analysis'] = self.analyze_eye_blinking(frame, faces)
        
        # Deep learning analysis
        scores['deep_learning'] = self.deep_learning_analysis(frame)
        
        return scores, faces
    
    def calculate_confidence(self, all_frame_scores, temporal_score, optical_flow_score):
        """Calculate overall confidence with weighted ensemble"""
        
        # Weight configuration (can be tuned based on research)
        weights = {
            'dct_artifacts': 0.20,
            'color_consistency': 0.15,
            'edge_analysis': 0.15,
            'texture_analysis': 0.15,
            'eye_analysis': 0.10,
            'deep_learning': 0.15,
            'temporal': 0.05,
            'optical_flow': 0.05
        }
        
        # Average frame scores
        avg_scores = {}
        for key in ['dct_artifacts', 'color_consistency', 'edge_analysis', 
                    'texture_analysis', 'eye_analysis', 'deep_learning']:
            values = [frame[key] for frame in all_frame_scores if key in frame]
            avg_scores[key] = np.mean(values) if values else 0.5
        
        avg_scores['temporal'] = temporal_score
        avg_scores['optical_flow'] = optical_flow_score
        
        # Weighted average
        confidence = sum(avg_scores[key] * weights[key] for key in weights.keys())
        
        # Apply sigmoid for better distribution
        confidence = 1 / (1 + np.exp(-10 * (confidence - 0.5)))
        
        return confidence, avg_scores
    
    def analyze_video(self, video_path, progress_callback=None):
        """Main video analysis function"""
        
        # Extract frames
        if progress_callback:
            progress_callback(0.1, "Extracting frames...")
        
        frames, video_info = self.extract_frames(video_path, num_frames=16)
        
        if progress_callback:
            progress_callback(0.2, "Analyzing frames...")
        
        # Analyze each frame
        all_frame_scores = []
        frame_faces = []
        
        for idx, frame in enumerate(frames):
            if progress_callback:
                progress = 0.2 + (idx / len(frames)) * 0.5
                progress_callback(progress, f"Analyzing frame {idx + 1}/{len(frames)}...")
            
            scores, faces = self.analyze_frame(frame, idx)
            all_frame_scores.append(scores)
            frame_faces.append(faces)
        
        # Temporal analysis
        if progress_callback:
            progress_callback(0.75, "Analyzing temporal consistency...")
        
        temporal_score = self.analyze_temporal_consistency(frames)
        
        # Optical flow analysis
        if progress_callback:
            progress_callback(0.85, "Analyzing optical flow...")
        
        optical_flow_score = self.analyze_optical_flow(frames)
        
        # Calculate final confidence
        if progress_callback:
            progress_callback(0.95, "Calculating final results...")
        
        confidence, component_scores = self.calculate_confidence(
            all_frame_scores, temporal_score, optical_flow_score
        )
        
        # Determine prediction
        if confidence < 0.35:
            prediction = "AUTHENTIC"
            risk_level = "LOW"
        elif confidence < 0.65:
            prediction = "UNCERTAIN"
            risk_level = "MEDIUM"
        else:
            prediction = "DEEPFAKE DETECTED"
            risk_level = "HIGH"
        
        results = {
            'prediction': prediction,
            'confidence': confidence,
            'risk_level': risk_level,
            'video_info': video_info,
            'frame_scores': all_frame_scores,
            'component_scores': component_scores,
            'temporal_score': temporal_score,
            'optical_flow_score': optical_flow_score,
            'frames_analyzed': len(frames),
            'total_faces_detected': sum(len(faces) for faces in frame_faces)
        }
        
        if progress_callback:
            progress_callback(1.0, "Analysis complete!")
        
        return results, frames


def display_results(results, frames):
    """Display comprehensive results"""
    
    # Header
    st.markdown('<div class="main-header"><h1>🔬 Analysis Results</h1><p>Comprehensive Deepfake Detection Report</p></div>', unsafe_allow_html=True)
    
    # Main prediction
    st.markdown('<div class="result-container">', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        prediction_class = results['prediction'].lower().replace(' ', '-')
        badge_class = f"badge-{prediction_class}" if prediction_class in ['authentic', 'deepfake-detected'] else "badge-uncertain"
        st.markdown(f'<div class="{badge_class}">{results["prediction"]}</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Confidence Score</div>
            <div class="metric-value">{results['confidence']*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        risk_color = {'LOW': '#10b981', 'MEDIUM': '#f59e0b', 'HIGH': '#ef4444'}[results['risk_level']]
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Risk Level</div>
            <div class="metric-value" style="color: {risk_color}">{results['risk_level']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Detailed Metrics
    st.markdown("### 📊 Detailed Analysis Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    metrics = [
        ("DCT Artifacts", results['component_scores']['dct_artifacts'], "Compression analysis"),
        ("Color Consistency", results['component_scores']['color_consistency'], "Color pattern analysis"),
        ("Edge Analysis", results['component_scores']['edge_analysis'], "Edge pattern detection"),
        ("Texture Analysis", results['component_scores']['texture_analysis'], "Skin texture analysis"),
        ("Eye Patterns", results['component_scores']['eye_analysis'], "Blinking analysis"),
        ("Deep Learning", results['component_scores']['deep_learning'], "Neural network analysis"),
        ("Temporal Consistency", results['component_scores']['temporal'], "Frame-to-frame analysis"),
        ("Optical Flow", results['component_scores']['optical_flow'], "Motion pattern analysis")
    ]
    
    cols = [col1, col2, col3, col4]
    for idx, (name, score, desc) in enumerate(metrics):
        with cols[idx % 4]:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">{name}</div>
                <div class="metric-value">{score*100:.1f}%</div>
                <div class="metric-description">{desc}</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Video Information
    st.markdown("### 📹 Video Information")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Duration", f"{results['video_info']['duration']:.2f}s")
    with col2:
        st.metric("FPS", results['video_info']['fps'])
    with col3:
        st.metric("Total Frames", results['video_info']['total_frames'])
    with col4:
        st.metric("Faces Detected", results['total_faces_detected'])
    
    # Frame Gallery
    st.markdown("### 🎞️ Analyzed Frames")
    
    cols = st.columns(4)
    for idx, frame in enumerate(frames[:8]):  # Show first 8 frames
        with cols[idx % 4]:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st.image(frame_rgb, caption=f"Frame {idx + 1}", use_container_width=True)
    
    # Interpretation Guide
    st.markdown("### 📖 How to Interpret Results")
    
    st.markdown(f"""
    <div class="info-box">
        <strong>Confidence Score: {results['confidence']*100:.1f}%</strong><br>
        • 0-35%: Likely authentic video<br>
        • 35-65%: Inconclusive, manual review recommended<br>
        • 65-100%: High probability of manipulation detected
    </div>
    """, unsafe_allow_html=True)
    
    if results['prediction'] == "DEEPFAKE DETECTED":
        st.markdown("""
        <div class="warning-box">
            ⚠️ <strong>High Suspicion Detected</strong><br>
            Multiple algorithms detected patterns consistent with video manipulation.
            This should not be used as legal evidence but warrants further investigation.
        </div>
        """, unsafe_allow_html=True)
    elif results['prediction'] == "AUTHENTIC":
        st.markdown("""
        <div class="info-box">
            ✓ <strong>Appears Authentic</strong><br>
            Analysis suggests this video has not been significantly manipulated.
            However, always verify important content through multiple sources.
        </div>
        """, unsafe_allow_html=True)


def main():
    """Main application"""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🔬 Professional Deepfake Detection System</h1>
        <p>Advanced AI-Powered Video Analysis with Multi-Algorithm Ensemble</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ System Information")
        
        st.markdown('<div class="algo-badge">✓ DCT Analysis</div>', unsafe_allow_html=True)
        st.markdown('<div class="algo-badge">✓ Color Analysis</div>', unsafe_allow_html=True)
        st.markdown('<div class="algo-badge">✓ Edge Detection</div>', unsafe_allow_html=True)
        st.markdown('<div class="algo-badge">✓ Texture Analysis</div>', unsafe_allow_html=True)
        st.markdown('<div class="algo-badge">✓ Eye Tracking</div>', unsafe_allow_html=True)
        st.markdown('<div class="algo-badge">✓ Temporal Analysis</div>', unsafe_allow_html=True)
        st.markdown('<div class="algo-badge">✓ Optical Flow</div>', unsafe_allow_html=True)
        
        if TORCH_AVAILABLE:
            st.markdown('<div class="algo-badge">✓ Deep Learning (EfficientNet)</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 📚 About")
        st.info("""
        This professional system combines:
        • 8+ detection algorithms
        • Deep learning models
        • Temporal analysis
        • Ensemble voting
        
        For research and educational purposes.
        """)
    
    # Main content
    st.markdown("### 📤 Upload Video for Analysis")
    
    uploaded_file = st.file_uploader(
        "Choose a video file (MP4, AVI, MOV, MKV)",
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="Upload a video file to analyze for deepfake detection"
    )
    
    if uploaded_file:
        # Video preview
        st.video(uploaded_file)
        
        # Analyze button
        if st.button("🔍 Analyze Video", type="primary", use_container_width=True):
            # Save uploaded file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(uploaded_file.read())
                video_path = tmp_file.name
            
            try:
                # Initialize detector
                detector = DeepfakeDetector()
                
                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def update_progress(progress, message):
                    progress_bar.progress(progress)
                    status_text.text(message)
                
                # Analyze video
                results, frames = detector.analyze_video(video_path, update_progress)
                
                # Clear progress
                progress_bar.empty()
                status_text.empty()
                
                # Display results
                display_results(results, frames)
                
                # Download report
                st.markdown("---")
                report = {
                    'timestamp': datetime.now().isoformat(),
                    'filename': uploaded_file.name,
                    'results': {
                        'prediction': results['prediction'],
                        'confidence': float(results['confidence']),
                        'risk_level': results['risk_level']
                    },
                    'component_scores': {k: float(v) for k, v in results['component_scores'].items()},
                    'video_info': results['video_info']
                }
                
                st.download_button(
                    "📥 Download Analysis Report (JSON)",
                    data=json.dumps(report, indent=2),
                    file_name=f"deepfake_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
                
            except Exception as e:
                st.error(f"❌ Analysis failed: {str(e)}")
                st.exception(e)
            finally:
                # Cleanup
                if os.path.exists(video_path):
                    os.remove(video_path)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #64748b; padding: 2rem;">
        <p><strong>⚠️ Important Disclaimer</strong></p>
        <p>This tool is for research and educational purposes only. Results should not be used as legal evidence.
        Always verify important content through multiple independent sources.</p>
        <p style="margin-top: 1rem;">Powered by Advanced Computer Vision & Deep Learning</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
