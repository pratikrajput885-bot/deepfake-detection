# 🚀 Quick Setup Guide - DeepFake Detection System

## Visual Step-by-Step Installation

---

## 📥 STEP 1: Download All Files

You should have these files downloaded:
```
✅ app.py
✅ requirements.txt
✅ README.md
✅ DEPLOYMENT.md
✅ TECHNICAL_DOCS.md
✅ test_algorithms.py
✅ PROJECT_STRUCTURE.md
✅ .gitignore
✅ config.toml (for .streamlit folder)
```

---

## 📁 STEP 2: Create Project Folder

### Windows:
```cmd
# Open Command Prompt
# Navigate to where you want the project (e.g., Desktop)
cd Desktop

# Create folder
mkdir deepfake-detection
cd deepfake-detection

# Create .streamlit folder
mkdir .streamlit
```

### Mac/Linux:
```bash
# Open Terminal
# Navigate to where you want the project
cd ~/Desktop

# Create folder
mkdir deepfake-detection
cd deepfake-detection

# Create .streamlit folder
mkdir .streamlit
```

---

## 📋 STEP 3: Organize Files

Move downloaded files to correct locations:

```
deepfake-detection/              👈 Main folder you created
│
├── app.py                       👈 Move here
├── requirements.txt             👈 Move here
├── README.md                    👈 Move here
├── DEPLOYMENT.md                👈 Move here
├── TECHNICAL_DOCS.md            👈 Move here
├── PROJECT_STRUCTURE.md         👈 Move here
├── test_algorithms.py           👈 Move here
├── .gitignore                   👈 Move here
│
└── .streamlit/                  👈 Folder you created
    └── config.toml              👈 Move config.toml here
```

### Visual Guide:

```
📂 Downloads/
   ├── app.py ─────────────────────┐
   ├── requirements.txt ───────────┤
   ├── README.md ──────────────────┤
   ├── config.toml ────────────┐   │
   └── ...                     │   │
                               │   │
                               ↓   ↓
📂 Desktop/
   └── 📂 deepfake-detection/
       ├── app.py              ✅
       ├── requirements.txt    ✅
       ├── README.md           ✅
       └── 📂 .streamlit/
           └── config.toml     ✅
```

---

## 🐍 STEP 4: Install Python (If Not Installed)

### Check if Python is installed:
```bash
python --version
# or
python3 --version
```

Should show: `Python 3.8.0` or higher

### If not installed:

**Windows:**
1. Go to [python.org/downloads](https://python.org/downloads)
2. Download Python 3.11
3. Run installer
4. ✅ Check "Add Python to PATH"
5. Click "Install Now"

**Mac:**
```bash
# Using Homebrew
brew install python3
```

**Linux:**
```bash
sudo apt update
sudo apt install python3 python3-pip
```

---

## 📦 STEP 5: Install Dependencies

Open terminal/command prompt in your project folder:

### Windows (Command Prompt):
```cmd
cd Desktop\deepfake-detection
python -m pip install -r requirements.txt
```

### Windows (PowerShell):
```powershell
cd Desktop\deepfake-detection
python -m pip install -r requirements.txt
```

### Mac/Linux:
```bash
cd ~/Desktop/deepfake-detection
pip3 install -r requirements.txt
```

**What gets installed:**
- streamlit (web framework)
- opencv-python-headless (computer vision)
- numpy (numerical computing)
- Pillow (image processing)

**Installation should take 2-5 minutes**

---

## ✅ STEP 6: Verify Installation

Run this test:

```bash
# Windows
python test_algorithms.py

# Mac/Linux
python3 test_algorithms.py
```

You should see:
```
============================================================
DeepFake Detection System - Algorithm Test Suite
============================================================
Testing Face Detection...
✅ Face detector loaded successfully

Testing DCT Analysis...
✅ DCT analysis working

Testing Edge Detection...
✅ Edge detection working

Testing Color Space Conversion...
✅ Color space conversion working

Testing Gradient Analysis...
✅ Gradient analysis working

============================================================
Test Results: 5/5 passed
============================================================

✅ All tests passed! System is ready to use.
```

---

## 🚀 STEP 7: Run the Application

### Start the app:

**Windows:**
```cmd
streamlit run app.py
```

**Mac/Linux:**
```bash
streamlit run app.py
```

### What happens:
```
You can now view your Streamlit app in your browser.

Local URL: http://localhost:8501
Network URL: http://192.168.1.x:8501
```

**Browser should open automatically!** 🎉

If not, manually open: `http://localhost:8501`

---

## 🎯 STEP 8: Test the Application

1. **Upload a video** (MP4, AVI, MOV)
2. **Click "🔍 Analyze Video"**
3. **Wait 5-15 seconds**
4. **View results!**

---

## 🌐 STEP 9: Deploy Online (Optional)

### Option A: Streamlit Community Cloud (FREE)

1. **Push to GitHub:**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/YOUR_USERNAME/deepfake-detection.git
   git push -u origin main
   ```

2. **Deploy:**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click "New app"
   - Select your repository
   - Click "Deploy"

### Option B: Hugging Face Spaces (FREE)

1. **Create account** at [huggingface.co](https://huggingface.co)
2. **Create new Space**
3. **Select Streamlit SDK**
4. **Upload files**
5. **Done!**

---

## 🔧 Troubleshooting

### Issue: "streamlit: command not found"

**Solution:**
```bash
# Windows
python -m pip install streamlit --upgrade

# Mac/Linux
pip3 install streamlit --upgrade
```

### Issue: "No module named cv2"

**Solution:**
```bash
pip uninstall opencv-python
pip install opencv-python-headless
```

### Issue: "Port 8501 is already in use"

**Solution:**
```bash
# Use different port
streamlit run app.py --server.port 8502
```

### Issue: App is slow

**Solutions:**
1. Use smaller video files (< 50MB)
2. Close other applications
3. Check your internet connection
4. Restart the app

### Issue: Upload fails

**Solutions:**
1. Check file format (MP4, AVI, MOV, MKV only)
2. Check file size (< 200MB)
3. Try a different video
4. Check video is not corrupted

---

## 📝 Quick Reference

### Start App:
```bash
streamlit run app.py
```

### Stop App:
```
Press Ctrl + C in terminal
```

### Clear Cache:
```bash
streamlit cache clear
```

### Update App:
```bash
# After editing app.py, save and refresh browser
# Streamlit auto-reloads!
```

### View on Other Devices (Same Network):
```bash
# Find your IP address
# Windows: ipconfig
# Mac/Linux: ifconfig

# Access at: http://YOUR_IP:8501
```

---

## 🎓 For Students - Project Submission

### Folder to Submit:

```
deepfake-detection.zip
│
Contains:
├── Source Code (app.py)
├── Documentation (README.md)
├── Technical Docs (TECHNICAL_DOCS.md)
├── Deployment Guide (DEPLOYMENT.md)
├── Requirements (requirements.txt)
└── Configuration (.streamlit/config.toml)
```

### Create ZIP:

**Windows:**
- Right-click folder → Send to → Compressed (zipped) folder

**Mac:**
- Right-click folder → Compress "deepfake-detection"

**Linux:**
```bash
zip -r deepfake-detection.zip deepfake-detection/
```

---

## ✅ Final Checklist

Before submitting/deploying:

- [ ] All files in correct folders
- [ ] Dependencies installed
- [ ] App runs locally without errors
- [ ] Tested with sample video
- [ ] Documentation complete
- [ ] Code is commented
- [ ] README is updated
- [ ] Screenshots taken
- [ ] Git repository created (if required)
- [ ] Deployed online (if required)

---

## 🎉 Success!

Your DeepFake Detection System is now:
✅ **Installed**
✅ **Running**
✅ **Ready to use**
✅ **Ready to deploy**

---

## 📞 Need Help?

Common resources:
- Streamlit Docs: [docs.streamlit.io](https://docs.streamlit.io)
- OpenCV Docs: [docs.opencv.org](https://docs.opencv.org)
- Python Docs: [docs.python.org](https://docs.python.org)

---

## 🚀 Next Steps

1. **Test thoroughly** with different videos
2. **Customize** the UI (edit app.py)
3. **Deploy online** for easy access
4. **Share** with your professor/peers
5. **Document** your findings
6. **Present** your project!

**Good luck with your project! 🎓**
