# 🚀 Deployment Guide - DeepFake Detection System

## Table of Contents
1. [Streamlit Community Cloud](#streamlit-community-cloud)
2. [Hugging Face Spaces](#hugging-face-spaces)
3. [Local Deployment](#local-deployment)
4. [Docker Deployment](#docker-deployment)
5. [Troubleshooting](#troubleshooting)

---

## 1. Streamlit Community Cloud (Recommended - FREE)

### Step-by-Step Instructions

#### 1.1 Prepare Your Repository

1. **Create GitHub Repository**
   ```bash
   git init
   git add .
   git commit -m "Initial commit: DeepFake Detection System"
   git remote add origin https://github.com/YOUR_USERNAME/deepfake-detector.git
   git push -u origin main
   ```

2. **Ensure Files Are Present**
   - ✅ app.py
   - ✅ requirements.txt
   - ✅ README.md

#### 1.2 Deploy to Streamlit Cloud

1. Visit [share.streamlit.io](https://share.streamlit.io)
2. Click "New app"
3. Select your repository: `YOUR_USERNAME/deepfake-detector`
4. Set:
   - Main file path: `app.py`
   - Branch: `main`
5. Click "Deploy!"
6. Wait 2-5 minutes for deployment
7. Your app will be live at: `https://YOUR_USERNAME-deepfake-detector.streamlit.app`

#### 1.3 Configuration (Optional)

Create `.streamlit/config.toml`:

```toml
[theme]
primaryColor = "#00f5ff"
backgroundColor = "#0a0e27"
secondaryBackgroundColor = "#1a1f3a"
textColor = "#ffffff"

[server]
maxUploadSize = 200
```

---

## 2. Hugging Face Spaces (Alternative - FREE)

### Step-by-Step Instructions

#### 2.1 Create a Space

1. Go to [huggingface.co/spaces](https://huggingface.co/spaces)
2. Click "Create new Space"
3. Fill in:
   - Space name: `deepfake-detector`
   - License: `mit`
   - Select SDK: `Streamlit`
   - Space hardware: `CPU basic (free)`

#### 2.2 Upload Files

**Option A: Git Method**
```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/deepfake-detector
cd deepfake-detector
cp /path/to/your/app.py .
cp /path/to/your/requirements.txt .
git add .
git commit -m "Add DeepFake Detection app"
git push
```

**Option B: Web Interface**
- Upload `app.py` via web interface
- Upload `requirements.txt`
- Files will auto-deploy

#### 2.3 Additional Configuration

Create `README.md` in the Space:

```yaml
---
title: DeepFake Detector
emoji: 🔍
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: 1.31.0
app_file: app.py
pinned: false
---

# DeepFake Detection System
AI-powered video authenticity analysis
```

---

## 3. Local Deployment

### 3.1 Basic Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Run application
streamlit run app.py
```

### 3.2 Network Access (Share with Others)

```bash
# Allow access from other devices on your network
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

Access from other devices:
- Find your IP: `ipconfig` (Windows) or `ifconfig` (Mac/Linux)
- Access at: `http://YOUR_IP:8501`

### 3.3 Custom Port

```bash
streamlit run app.py --server.port 8080
```

### 3.4 Production Mode

```bash
streamlit run app.py --server.headless true --server.enableCORS false
```

---

## 4. Docker Deployment

### 4.1 Create Dockerfile

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy files
COPY requirements.txt .
COPY app.py .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run application
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### 4.2 Build and Run

```bash
# Build image
docker build -t deepfake-detector .

# Run container
docker run -p 8501:8501 deepfake-detector

# Run with volume (for development)
docker run -p 8501:8501 -v $(pwd):/app deepfake-detector
```

### 4.3 Docker Compose

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  deepfake-detector:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./app.py:/app/app.py
    environment:
      - STREAMLIT_SERVER_PORT=8501
    restart: unless-stopped
```

Run:
```bash
docker-compose up -d
```

---

## 5. Troubleshooting

### Common Issues

#### Issue: "ModuleNotFoundError: No module named 'cv2'"

**Solution:**
```bash
pip install opencv-python-headless
```

#### Issue: "libGL.so.1: cannot open shared object file"

**Solution (Linux):**
```bash
sudo apt-get update
sudo apt-get install -y libgl1-mesa-glx
```

#### Issue: Upload size limit exceeded

**Solution:** Create `.streamlit/config.toml`:
```toml
[server]
maxUploadSize = 200
```

#### Issue: App runs but shows blank page

**Solution:**
- Clear browser cache
- Check browser console for errors
- Verify Python version (3.8+)
- Reinstall streamlit: `pip install --upgrade streamlit`

#### Issue: Face detection not working

**Solution:**
- Ensure opencv-python-headless is installed
- Verify Haar Cascade files are accessible
- Check video format compatibility

#### Issue: Slow performance

**Solutions:**
- Reduce number of frames analyzed (change `num_frames=8` to `num_frames=4`)
- Use smaller video files for testing
- Deploy on platform with better resources

#### Issue: Memory error

**Solutions:**
- Reduce frame extraction count
- Process video in chunks
- Use cloud deployment with more RAM

### Platform-Specific Issues

#### Streamlit Cloud

**Issue:** Build fails
- Check requirements.txt format
- Ensure all dependencies are compatible
- Review build logs in Streamlit Cloud dashboard

**Issue:** App crashes after deployment
- Check memory usage (free tier: 1GB RAM)
- Reduce video processing parameters
- Monitor logs in Streamlit Cloud

#### Hugging Face

**Issue:** Space not building
- Verify `README.md` has correct YAML frontmatter
- Check SDK version compatibility
- Review Space logs

**Issue:** Slow loading
- Consider upgrading to paid tier for better performance
- Optimize code for faster processing

### Performance Optimization

1. **Reduce Frame Count**
   ```python
   frames = extract_frames(video_path, num_frames=4)  # Instead of 8
   ```

2. **Lower Video Resolution**
   ```python
   frame_rgb = cv2.resize(frame_rgb, (640, 480))
   ```

3. **Cache Results**
   ```python
   @st.cache_data
   def analyze_video(video_path):
       # Your code here
   ```

4. **Limit Upload Size**
   ```toml
   [server]
   maxUploadSize = 50  # MB
   ```

---

## Testing Deployment

### Quick Test Checklist

- [ ] App loads without errors
- [ ] Video upload works
- [ ] Analysis runs successfully
- [ ] Results display correctly
- [ ] Frame gallery renders
- [ ] Metrics are accurate
- [ ] UI is responsive
- [ ] No console errors

### Test Videos

Create test videos or use:
- Short clips (5-10 seconds)
- Various formats (MP4, AVI, MOV)
- Different resolutions
- With and without faces

---

## Monitoring & Maintenance

### Streamlit Cloud
- Monitor via dashboard: [share.streamlit.io](https://share.streamlit.io)
- Check logs for errors
- View analytics (views, errors, performance)

### Hugging Face
- Monitor Space status
- Check build logs
- View usage statistics

### Local Deployment
```bash
# Check running processes
ps aux | grep streamlit

# Monitor resource usage
htop  # or top

# Check application logs
streamlit run app.py 2>&1 | tee app.log
```

---

## Security Considerations

1. **Input Validation**
   - File size limits enforced
   - File type validation
   - Sanitize uploads

2. **Resource Limits**
   - Max upload size: 200MB
   - Processing timeout
   - Memory limits

3. **Privacy**
   - Videos are processed in-memory
   - Temporary files deleted after analysis
   - No permanent storage

---

## Support & Resources

- **Streamlit Docs**: [docs.streamlit.io](https://docs.streamlit.io)
- **OpenCV Docs**: [docs.opencv.org](https://docs.opencv.org)
- **Community Forum**: [discuss.streamlit.io](https://discuss.streamlit.io)
- **Hugging Face Docs**: [huggingface.co/docs](https://huggingface.co/docs)

---

## Next Steps After Deployment

1. **Share Your App**
   - Share URL with peers/professors
   - Add to portfolio
   - Document in project report

2. **Monitor Usage**
   - Track analytics
   - Gather feedback
   - Fix bugs

3. **Iterate**
   - Improve detection accuracy
   - Enhance UI/UX
   - Add features

4. **Document**
   - Update README
   - Create user guide
   - Write technical documentation

---

**Deployment Success! 🎉**

Your DeepFake Detection System is now live and ready to use!
