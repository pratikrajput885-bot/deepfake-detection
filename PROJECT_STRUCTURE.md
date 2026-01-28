# 📁 DeepFake Detection Project Structure

## Complete File & Folder Organization

```
deepfake-detection/
│
├── 📄 app.py                          # Main Streamlit application (REQUIRED)
├── 📄 requirements.txt                # Python dependencies (REQUIRED)
├── 📄 README.md                       # Project documentation (REQUIRED)
├── 📄 DEPLOYMENT.md                   # Deployment instructions
├── 📄 TECHNICAL_DOCS.md               # Technical documentation
├── 📄 test_algorithms.py              # Algorithm testing script
├── 📄 .gitignore                      # Git ignore file (create this)
│
├── 📁 .streamlit/                     # Streamlit configuration folder
│   └── 📄 config.toml                 # Streamlit settings
│
├── 📁 assets/                         # Optional: Images and media
│   ├── 📄 logo.png                    # Your project logo
│   ├── 📄 banner.png                  # Project banner
│   └── 📄 screenshots/                # App screenshots
│       ├── 📄 home.png
│       ├── 📄 upload.png
│       └── 📄 results.png
│
├── 📁 docs/                           # Optional: Additional documentation
│   ├── 📄 user_guide.pdf              # User manual
│   ├── 📄 presentation.pptx           # Project presentation
│   └── 📄 report.pdf                  # Final year project report
│
├── 📁 tests/                          # Optional: Test files
│   ├── 📄 test_detection.py           # Unit tests for detection
│   ├── 📄 test_video_processing.py    # Video processing tests
│   └── 📄 sample_videos/              # Sample test videos
│       ├── 📄 real_video.mp4
│       └── 📄 fake_video.mp4
│
├── 📁 notebooks/                      # Optional: Jupyter notebooks
│   ├── 📄 exploration.ipynb           # Data exploration
│   └── 📄 algorithm_testing.ipynb     # Algorithm development
│
└── 📁 utils/                          # Optional: Utility modules
    ├── 📄 __init__.py
    ├── 📄 video_processor.py          # Video processing utilities
    ├── 📄 face_detector.py            # Face detection module
    └── 📄 analyzer.py                 # Analysis functions

```

---

## 🎯 Minimal Required Structure (For Quick Deployment)

```
deepfake-detection/
│
├── 📄 app.py                    ✅ MUST HAVE
├── 📄 requirements.txt          ✅ MUST HAVE
└── 📄 README.md                 ✅ MUST HAVE
```

These 3 files are enough to deploy!

---

## 📋 Recommended Structure (For Academic Project)

```
deepfake-detection/
│
├── 📄 app.py                    ✅ Main app
├── 📄 requirements.txt          ✅ Dependencies
├── 📄 README.md                 ✅ Documentation
├── 📄 DEPLOYMENT.md             ⭐ How to deploy
├── 📄 TECHNICAL_DOCS.md         ⭐ Technical details
├── 📄 test_algorithms.py        ⭐ Testing script
├── 📄 .gitignore               ⭐ Git ignore
│
└── 📁 .streamlit/
    └── 📄 config.toml           ⭐ App configuration
```

---

## 🔧 Create .gitignore File

Create a file named `.gitignore` with this content:

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
build/
dist/
*.egg-info/

# Streamlit
.streamlit/secrets.toml

# IDE
.vscode/
.idea/
*.swp
*.swo
.DS_Store

# Testing
.pytest_cache/
.coverage
htmlcov/

# Videos (don't commit large files)
*.mp4
*.avi
*.mov
*.mkv
!sample_video.mp4

# Temporary files
*.tmp
*.log
temp/
tmp/
```

---

## 📂 How to Create This Structure

### Option 1: Manual Creation (Windows)

```cmd
mkdir deepfake-detection
cd deepfake-detection

# Create files (copy your files here)
copy path\to\app.py .
copy path\to\requirements.txt .
copy path\to\README.md .
copy path\to\DEPLOYMENT.md .
copy path\to\TECHNICAL_DOCS.md .
copy path\to\test_algorithms.py .

# Create folders
mkdir .streamlit
mkdir assets
mkdir docs
mkdir tests

# Create .gitignore
echo. > .gitignore
```

### Option 2: Manual Creation (Mac/Linux)

```bash
mkdir deepfake-detection
cd deepfake-detection

# Create files (copy your files here)
cp /path/to/app.py .
cp /path/to/requirements.txt .
cp /path/to/README.md .
cp /path/to/DEPLOYMENT.md .
cp /path/to/TECHNICAL_DOCS.md .
cp /path/to/test_algorithms.py .

# Create folders
mkdir -p .streamlit
mkdir -p assets
mkdir -p docs
mkdir -p tests

# Create .gitignore
touch .gitignore
```

### Option 3: Using Python Script

Create `setup_structure.py`:

```python
import os

# Define structure
structure = {
    'deepfake-detection': {
        'files': [
            'app.py',
            'requirements.txt',
            'README.md',
            'DEPLOYMENT.md',
            'TECHNICAL_DOCS.md',
            'test_algorithms.py',
            '.gitignore'
        ],
        'folders': {
            '.streamlit': ['config.toml'],
            'assets': [],
            'docs': [],
            'tests': []
        }
    }
}

# Create structure
def create_structure(base_path, struct):
    for folder, contents in struct.items():
        folder_path = os.path.join(base_path, folder)
        os.makedirs(folder_path, exist_ok=True)
        
        # Create files
        for file in contents.get('files', []):
            file_path = os.path.join(folder_path, file)
            if not os.path.exists(file_path):
                open(file_path, 'a').close()
                print(f"Created: {file_path}")
        
        # Create subfolders
        for subfolder, subfiles in contents.get('folders', {}).items():
            subfolder_path = os.path.join(folder_path, subfolder)
            os.makedirs(subfolder_path, exist_ok=True)
            for subfile in subfiles:
                subfile_path = os.path.join(subfolder_path, subfile)
                if not os.path.exists(subfile_path):
                    open(subfile_path, 'a').close()
                    print(f"Created: {subfile_path}")

if __name__ == '__main__':
    create_structure('.', structure)
    print("\n✅ Project structure created successfully!")
```

Run: `python setup_structure.py`

---

## 📤 Git Repository Setup

### 1. Initialize Git

```bash
cd deepfake-detection
git init
```

### 2. Add Files

```bash
git add .
```

### 3. Commit

```bash
git commit -m "Initial commit: DeepFake Detection System"
```

### 4. Create GitHub Repository

1. Go to [github.com](https://github.com)
2. Click "New Repository"
3. Name: `deepfake-detection`
4. Don't initialize with README (you already have one)
5. Click "Create repository"

### 5. Push to GitHub

```bash
git remote add origin https://github.com/YOUR_USERNAME/deepfake-detection.git
git branch -M main
git push -u origin main
```

---

## 📊 Visual Structure Diagram

```
🏠 deepfake-detection (ROOT)
│
├─── 🐍 PYTHON FILES
│    ├── app.py (700+ lines) - Main application
│    └── test_algorithms.py - Testing utilities
│
├─── 📋 CONFIGURATION FILES  
│    ├── requirements.txt - Dependencies
│    └── .gitignore - Git ignore rules
│
├─── 📚 DOCUMENTATION FILES
│    ├── README.md - Project overview
│    ├── DEPLOYMENT.md - Deployment guide
│    └── TECHNICAL_DOCS.md - Technical details
│
└─── ⚙️ SETTINGS FOLDER
     └── .streamlit/
         └── config.toml - Streamlit configuration
```

---

## 🎯 File Placement Guide

### Where to put each file:

| File | Location | Required? |
|------|----------|-----------|
| app.py | Root folder | ✅ YES |
| requirements.txt | Root folder | ✅ YES |
| README.md | Root folder | ✅ YES |
| DEPLOYMENT.md | Root folder | ⭐ Recommended |
| TECHNICAL_DOCS.md | Root folder | ⭐ Recommended |
| test_algorithms.py | Root folder | ⭐ Recommended |
| .gitignore | Root folder | ⭐ Recommended |
| config.toml | .streamlit/ folder | ⭐ Recommended |

---

## 🚀 Ready to Deploy Structure

```
deepfake-detection/
│
├── app.py                 👈 Copy here
├── requirements.txt       👈 Copy here
├── README.md             👈 Copy here
├── DEPLOYMENT.md         👈 Copy here
├── TECHNICAL_DOCS.md     👈 Copy here
├── test_algorithms.py    👈 Copy here
├── .gitignore           👈 Create new file
│
└── .streamlit/
    └── config.toml       👈 Copy here
```

### Quick Setup Commands:

```bash
# Create main folder
mkdir deepfake-detection
cd deepfake-detection

# Create subfolder
mkdir .streamlit

# Now copy all your downloaded files to these locations!
```

---

## ✅ Verification Checklist

After setting up, verify:

- [ ] app.py exists in root folder
- [ ] requirements.txt exists in root folder
- [ ] README.md exists in root folder
- [ ] .streamlit folder exists
- [ ] config.toml exists in .streamlit folder
- [ ] Can run: `streamlit run app.py`
- [ ] Git repository initialized (optional)
- [ ] Pushed to GitHub (optional)

---

## 🎓 For Academic Submission

Add these to your project folder:

```
deepfake-detection/
│
├── 📄 app.py
├── 📄 requirements.txt
├── 📄 README.md
├── 📄 DEPLOYMENT.md
├── 📄 TECHNICAL_DOCS.md
│
├── 📁 docs/
│   ├── 📄 project_report.pdf        👈 Your written report
│   ├── 📄 presentation.pptx         👈 Your presentation
│   ├── 📄 user_manual.pdf          👈 How to use guide
│   └── 📄 screenshots/
│       └── 📄 (app screenshots)
│
└── 📁 .streamlit/
    └── 📄 config.toml
```

---

**Need help? The structure is simple:**
1. Create a folder called `deepfake-detection`
2. Put all downloaded files in that folder
3. Create `.streamlit` subfolder
4. Put `config.toml` in `.streamlit` folder
5. Done! 🎉
