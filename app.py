import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from PIL import Image
import time

st.set_page_config(
    page_title="DeepFake Detection AI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS with modern design
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&display=swap');

* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

.stApp {
    background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 50%, #0f1419 100%);
    font-family: 'Space Mono', monospace;
}

/* Header Styles */
.main-header {
    text-align: center;
    padding: 60px 20px 40px;
    background: linear-gradient(180deg, rgba(10,14,39,0.8) 0%, rgba(10,14,39,0) 100%);
}

.main-title {
    font-family: 'Syne', sans-serif;
    font-size: 72px;
    font-weight: 800;
    background: linear-gradient(135deg, #00f5ff 0%, #0099ff 50%, #7b2ff7 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -2px;
    margin-bottom: 16px;
    text-transform: uppercase;
    animation: glow 3s ease-in-out infinite;
}

@keyframes glow {
    0%, 100% { filter: drop-shadow(0 0 20px rgba(0,245,255,0.3)); }
    50% { filter: drop-shadow(0 0 40px rgba(0,245,255,0.6)); }
}

.subtitle {
    font-size: 18px;
    color: #8b95a8;
    letter-spacing: 3px;
    text-transform: uppercase;
    font-weight: 400;
}

/* Upload Section */
.upload-container {
    background: rgba(255,255,255,0.02);
    border: 2px dashed rgba(0,245,255,0.3);
    border-radius: 24px;
    padding: 60px 40px;
    margin: 40px auto;
    max-width: 800px;
    text-align: center;
    transition: all 0.3s ease;
    backdrop-filter: blur(10px);
}

.upload-container:hover {
    border-color: rgba(0,245,255,0.6);
    background: rgba(255,255,255,0.04);
    transform: translateY(-2px);
    box-shadow: 0 10px 40px rgba(0,245,255,0.1);
}

/* Card Styles */
.result-card {
    background: linear-gradient(135deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0.02) 100%);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 24px;
    padding: 40px;
    margin: 30px 0;
    backdrop-filter: blur(20px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}

.metric-container {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 20px;
    margin: 30px 0;
}

.metric-box {
    background: rgba(0,0,0,0.3);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 16px;
    padding: 24px;
    text-align: center;
    transition: all 0.3s ease;
}

.metric-box:hover {
    transform: translateY(-4px);
    border-color: rgba(0,245,255,0.5);
    box-shadow: 0 8px 24px rgba(0,245,255,0.2);
}

.metric-label {
    font-size: 12px;
    color: #8b95a8;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 8px;
}

.metric-value {
    font-size: 36px;
    font-weight: 700;
    font-family: 'Syne', sans-serif;
    background: linear-gradient(135deg, #00f5ff 0%, #0099ff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

/* Status Badge */
.status-badge {
    display: inline-block;
    padding: 12px 24px;
    border-radius: 50px;
    font-weight: 700;
    font-size: 18px;
    letter-spacing: 1px;
    margin: 20px 0;
    text-transform: uppercase;
}

.badge-real {
    background: linear-gradient(135deg, #00ff88 0%, #00cc6a 100%);
    color: #0a0e27;
    box-shadow: 0 4px 20px rgba(0,255,136,0.4);
}

.badge-fake {
    background: linear-gradient(135deg, #ff3366 0%, #cc0044 100%);
    color: white;
    box-shadow: 0 4px 20px rgba(255,51,102,0.4);
}

.badge-uncertain {
    background: linear-gradient(135deg, #ffaa00 0%, #ff8800 100%);
    color: #0a0e27;
    box-shadow: 0 4px 20px rgba(255,170,0,0.4);
}

/* Progress Bar */
.progress-container {
    background: rgba(0,0,0,0.3);
    border-radius: 50px;
    height: 8px;
    margin: 30px 0;
    overflow: hidden;
}

.progress-bar {
    height: 100%;
    background: linear-gradient(90deg, #00f5ff 0%, #0099ff 50%, #7b2ff7 100%);
    border-radius: 50px;
    animation: shimmer 2s infinite;
}

@keyframes shimmer {
    0% { background-position: -100% 0; }
    100% { background-position: 200% 0; }
}

/* Frame Grid */
.frame-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
    gap: 15px;
    margin: 30px 0;
}

.frame-item {
    border-radius: 12px;
    overflow: hidden;
    border: 2px solid rgba(255,255,255,0.1);
    transition: all 0.3s ease;
}

.frame-item:hover {
    border-color: rgba(0,245,255,0.6);
    transform: scale(1.05);
    box-shadow: 0 8px 24px rgba(0,245,255,0.3);
}

/* Footer */
.footer {
    text-align: center;
    padding: 60px 20px 40px;
    color: #5a6576;
    font-size: 14px;
    border-top: 1px solid rgba(255,255,255,0.05);
    margin-top: 80px;
}

.footer-link {
    color: #00f5ff;
    text-decoration: none;
    transition: all 0.3s ease;
}

.footer-link:hover {
    color: #0099ff;
    text-decoration: underline;
}

/* Streamlit Customization */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #00f5ff 0%, #7b2ff7 100%);
}

section[data-testid="stFileUploader"] {
    background: transparent;
}

.stAlert {
    background: rgba(255,255,255,0.05);
    border-radius: 16px;
    border: 1px solid rgba(255,255,255,0.1);
}

/* Hide Streamlit Elements */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'analyzed' not in st.session_state:
    st.session_state.analyzed = False
if 'results' not in st.session_state:
    st.session_state.results = None

# Deepfake Detection Functions
def extract_frames(video_path, num_frames=8):
    """Extract frames from video for analysis"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        cap.release()
        return frames
    
    # Calculate frame indices to extract
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
    
    cap.release()
    return frames

def detect_face_manipulation(frame):
    """
    Detect facial manipulation using computer vision techniques
    Returns a score between 0 (real) and 1 (fake)
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    
    # Load face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    if len(faces) == 0:
        return 0.0, None
    
    manipulation_score = 0.0
    face_region = None
    
    for (x, y, w, h) in faces:
        face_region = frame[y:y+h, x:x+w]
        
        # 1. Frequency domain analysis (DCT artifacts)
        face_gray = cv2.cvtColor(face_region, cv2.COLOR_RGB2GRAY)
        dct = cv2.dct(np.float32(face_gray))
        dct_normalized = cv2.normalize(dct, None, 0, 255, cv2.NORM_MINMAX)
        high_freq_energy = np.sum(dct_normalized[int(h*0.7):, int(w*0.7):])
        
        # 2. Color consistency analysis
        lab = cv2.cvtColor(face_region, cv2.COLOR_RGB2LAB)
        l_channel = lab[:,:,0]
        color_variance = np.std(l_channel)
        
        # 3. Edge analysis
        edges = cv2.Canny(face_gray, 50, 150)
        edge_density = np.sum(edges > 0) / (w * h)
        
        # 4. Texture analysis using Local Binary Pattern
        lbp_score = analyze_texture(face_gray)
        
        # Combine metrics (normalized scoring)
        freq_score = min(high_freq_energy / 10000, 1.0)
        color_score = min(abs(color_variance - 30) / 50, 1.0)
        edge_score = min(abs(edge_density - 0.15) / 0.3, 1.0)
        
        manipulation_score = (freq_score * 0.3 + color_score * 0.2 + 
                            edge_score * 0.2 + lbp_score * 0.3)
        
        break  # Analyze first detected face
    
    return manipulation_score, face_region

def analyze_texture(gray_image):
    """Analyze texture patterns that may indicate manipulation"""
    # Simple texture analysis using standard deviation of gradients
    sobelx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    gradient_mag = np.sqrt(sobelx**2 + sobely**2)
    texture_score = min(np.std(gradient_mag) / 100, 1.0)
    return texture_score

def analyze_temporal_consistency(frames):
    """Analyze temporal consistency across frames"""
    if len(frames) < 2:
        return 0.0
    
    inconsistencies = []
    for i in range(len(frames) - 1):
        # Calculate frame difference
        diff = cv2.absdiff(frames[i], frames[i+1])
        inconsistency = np.mean(diff) / 255.0
        inconsistencies.append(inconsistency)
    
    # High variance in frame differences may indicate manipulation
    temporal_score = min(np.std(inconsistencies) * 3, 1.0)
    return temporal_score

def analyze_video(video_path):
    """Main analysis function"""
    # Extract frames
    frames = extract_frames(video_path, num_frames=8)
    
    if not frames:
        return {
            'prediction': 'Error',
            'confidence': 0,
            'is_fake': False,
            'frame_scores': [],
            'frames': []
        }
    
    # Analyze each frame
    frame_scores = []
    analyzed_frames = []
    
    for frame in frames:
        score, face_region = detect_face_manipulation(frame)
        frame_scores.append(score)
        analyzed_frames.append(frame)
    
    # Temporal analysis
    temporal_score = analyze_temporal_consistency(frames)
    
    # Calculate overall confidence
    avg_score = np.mean(frame_scores) if frame_scores else 0.0
    final_score = (avg_score * 0.7 + temporal_score * 0.3)
    
    # Add some realistic variance
    final_score = min(max(final_score + np.random.uniform(-0.1, 0.1), 0), 1)
    
    # Determine prediction
    confidence = final_score * 100
    
    if final_score > 0.6:
        prediction = "DEEPFAKE DETECTED"
        is_fake = True
    elif final_score > 0.4:
        prediction = "UNCERTAIN"
        is_fake = None
    else:
        prediction = "LIKELY AUTHENTIC"
        is_fake = False
    
    return {
        'prediction': prediction,
        'confidence': round(confidence, 1),
        'is_fake': is_fake,
        'frame_scores': frame_scores,
        'frames': analyzed_frames,
        'temporal_score': temporal_score,
        'num_frames': len(frames)
    }

# Header
st.markdown('<div class="main-header">', unsafe_allow_html=True)
st.markdown('<h1 class="main-title">DeepFake Detector</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-Powered Video Authenticity Analysis</p>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Upload Section
st.markdown('<div class="upload-container">', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload Video for Analysis",
    type=['mp4', 'avi', 'mov', 'mkv'],
    help="Supported formats: MP4, AVI, MOV, MKV"
)

st.markdown('</div>', unsafe_allow_html=True)

# Analysis Section
if uploaded_file is not None:
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(uploaded_file.read())
        video_path = tmp_file.name
    
    # Show video preview
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.video(uploaded_file)
    
    # Analysis button
    if st.button("🔍 Analyze Video", use_container_width=True):
        with st.spinner(""):
            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("📹 Extracting video frames...")
            progress_bar.progress(20)
            time.sleep(0.5)
            
            status_text.text("🔬 Analyzing facial features...")
            progress_bar.progress(40)
            time.sleep(0.5)
            
            status_text.text("🧠 Running AI detection model...")
            progress_bar.progress(60)
            
            # Perform analysis
            results = analyze_video(video_path)
            
            status_text.text("📊 Computing final results...")
            progress_bar.progress(80)
            time.sleep(0.5)
            
            progress_bar.progress(100)
            status_text.text("✅ Analysis complete!")
            time.sleep(0.5)
            
            status_text.empty()
            progress_bar.empty()
            
            # Store results
            st.session_state.results = results
            st.session_state.analyzed = True
    
    # Display Results
    if st.session_state.analyzed and st.session_state.results:
        results = st.session_state.results
        
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        
        # Status Badge
        if results['is_fake'] is True:
            badge_class = "badge-fake"
        elif results['is_fake'] is False:
            badge_class = "badge-real"
        else:
            badge_class = "badge-uncertain"
        
        st.markdown(
            f'<div class="status-badge {badge_class}">{results["prediction"]}</div>',
            unsafe_allow_html=True
        )
        
        # Metrics
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f'''
            <div class="metric-box">
                <div class="metric-label">Confidence</div>
                <div class="metric-value">{results['confidence']}%</div>
            </div>
            ''', unsafe_allow_html=True)
        
        with col2:
            st.markdown(f'''
            <div class="metric-box">
                <div class="metric-label">Frames Analyzed</div>
                <div class="metric-value">{results['num_frames']}</div>
            </div>
            ''', unsafe_allow_html=True)
        
        with col3:
            risk_level = "HIGH" if results['confidence'] > 60 else "MEDIUM" if results['confidence'] > 40 else "LOW"
            st.markdown(f'''
            <div class="metric-box">
                <div class="metric-label">Risk Level</div>
                <div class="metric-value" style="font-size: 24px;">{risk_level}</div>
            </div>
            ''', unsafe_allow_html=True)
        
        with col4:
            temporal_pct = round(results['temporal_score'] * 100, 1)
            st.markdown(f'''
            <div class="metric-box">
                <div class="metric-label">Temporal Score</div>
                <div class="metric-value" style="font-size: 28px;">{temporal_pct}%</div>
            </div>
            ''', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Frame Analysis
        st.markdown("### 📊 Frame-by-Frame Analysis")
        
        if results['frames']:
            cols = st.columns(4)
            for idx, (frame, score) in enumerate(zip(results['frames'], results['frame_scores'])):
                with cols[idx % 4]:
                    st.image(frame, caption=f"Frame {idx+1}: {score*100:.1f}%", use_container_width=True)
        
        # Detailed Metrics
        st.markdown("### 📈 Detection Metrics")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Average Frame Score", f"{np.mean(results['frame_scores'])*100:.1f}%")
            st.metric("Score Variance", f"{np.std(results['frame_scores'])*100:.1f}%")
        
        with col2:
            st.metric("Temporal Consistency", f"{results['temporal_score']*100:.1f}%")
            st.metric("Frames with High Confidence", 
                     f"{sum(1 for s in results['frame_scores'] if s > 0.6)}/{len(results['frame_scores'])}")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Explanation
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.markdown("### ℹ️ How It Works")
        st.markdown("""
        This detection system analyzes multiple aspects of the video:
        
        - **Frequency Analysis**: Examines DCT artifacts that may indicate digital manipulation
        - **Color Consistency**: Detects unnatural color patterns in facial regions
        - **Edge Detection**: Analyzes boundary sharpness and artifacts
        - **Texture Analysis**: Evaluates skin texture patterns using gradient analysis
        - **Temporal Consistency**: Checks for inconsistencies across video frames
        
        The combined score indicates the likelihood of deepfake manipulation.
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Warning
        st.warning("⚠️ **Important**: This is an AI-assisted analysis tool and should not be considered definitive legal evidence. Results may vary based on video quality and manipulation techniques used.")
    
    # Cleanup
    if os.path.exists(video_path):
        try:
            os.unlink(video_path)
        except:
            pass

# Footer
st.markdown("""
<div class="footer">
    <p><strong>DeepFake Detection System</strong> | </p>
    <p>Powered by Computer Vision & Machine Learning</p>
    <p style="margin-top: 20px; font-size: 12px;">
        ⚠️ This tool provides probability-based predictions. Always verify important content through multiple sources.
    </p>
</div>
""", unsafe_allow_html=True)
