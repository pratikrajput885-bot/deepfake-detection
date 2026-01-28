"""
Test script for DeepFake Detection System
Run this to validate the detection algorithms
"""

import cv2
import numpy as np
from PIL import Image

def test_face_detection():
    """Test face detection capability"""
    print("Testing Face Detection...")
    
    # Create a simple test image with face-like features
    test_img = np.ones((480, 640, 3), dtype=np.uint8) * 200
    
    # Load face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    if face_cascade.empty():
        print("❌ Face detector failed to load")
        return False
    
    print("✅ Face detector loaded successfully")
    return True

def test_dct_analysis():
    """Test DCT (Discrete Cosine Transform) analysis"""
    print("\nTesting DCT Analysis...")
    
    # Create test image
    test_img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    
    try:
        dct = cv2.dct(np.float32(test_img))
        print("✅ DCT analysis working")
        return True
    except Exception as e:
        print(f"❌ DCT analysis failed: {e}")
        return False

def test_edge_detection():
    """Test Canny edge detection"""
    print("\nTesting Edge Detection...")
    
    # Create test image
    test_img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    
    try:
        edges = cv2.Canny(test_img, 50, 150)
        print("✅ Edge detection working")
        return True
    except Exception as e:
        print(f"❌ Edge detection failed: {e}")
        return False

def test_color_space_conversion():
    """Test LAB color space conversion"""
    print("\nTesting Color Space Conversion...")
    
    # Create test RGB image
    test_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    try:
        lab = cv2.cvtColor(test_img, cv2.COLOR_RGB2LAB)
        print("✅ Color space conversion working")
        return True
    except Exception as e:
        print(f"❌ Color space conversion failed: {e}")
        return False

def test_gradient_analysis():
    """Test Sobel gradient computation"""
    print("\nTesting Gradient Analysis...")
    
    # Create test image
    test_img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    
    try:
        sobelx = cv2.Sobel(test_img, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(test_img, cv2.CV_64F, 0, 1, ksize=3)
        print("✅ Gradient analysis working")
        return True
    except Exception as e:
        print(f"❌ Gradient analysis failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("DeepFake Detection System - Algorithm Test Suite")
    print("=" * 60)
    
    tests = [
        test_face_detection,
        test_dct_analysis,
        test_edge_detection,
        test_color_space_conversion,
        test_gradient_analysis
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "=" * 60)
    print(f"Test Results: {sum(results)}/{len(results)} passed")
    print("=" * 60)
    
    if all(results):
        print("\n✅ All tests passed! System is ready to use.")
    else:
        print("\n⚠️ Some tests failed. Please check OpenCV installation.")
    
    print("\nTo run the application:")
    print("  streamlit run app.py")

if __name__ == "__main__":
    main()
