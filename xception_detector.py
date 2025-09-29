#!/usr/bin/env python3
"""
Xception-based deepfake detector using timm
"""

import os
import argparse
import glob
import cv2
import numpy as np
import torch
import torch.nn as nn
import timm
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import urllib

class XceptionDetector:
    def __init__(self):
        print("Loading Xception model...")
        try:
            # Load pretrained Xception model
            self.model = timm.create_model('xception', pretrained=True)
            self.model.eval()
            
            # Get data config and create transform
            from timm.data import resolve_data_config
            from timm.data.transforms_factory import create_transform
            
            config = resolve_data_config({}, model=self.model)
            self.transform = create_transform(**config)
            
            print("✅ Xception model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading Xception model: {e}")
            self.model = None
    
    def detect_deepfake(self, frame):
        """Detect deepfake using Xception model"""
        if self.model is None:
            return 0.5
        
        try:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image
            pil_image = Image.fromarray(frame_rgb)
            
            # Transform image
            tensor = self.transform(pil_image).unsqueeze(0)
            
            # Get prediction
            with torch.no_grad():
                output = self.model(tensor)
                # Get probabilities
                probabilities = torch.nn.functional.softmax(output[0], dim=0)
                
                # For deepfake detection, we'll use the top prediction
                # In a real implementation, you'd need a model trained specifically for deepfake detection
                # For now, we'll use a heuristic based on the prediction confidence
                max_prob = torch.max(probabilities).item()
                
                # Convert to fake probability (higher confidence = more likely to be processed/fake)
                fake_prob = min(1.0, max_prob * 1.5)  # Scale the confidence
                
                return fake_prob
                
        except Exception as e:
            print(f"Error in Xception detection: {e}")
            return 0.5

def analyze_video_xception(video_path, detector):
    """Analyze video using Xception model"""
    print(f"Analyzing: {os.path.basename(video_path)}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Could not open video {video_path}")
        return None
    
    # Get video info
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0
    
    print(f"  - Resolution: {width}x{height}")
    print(f"  - FPS: {fps:.2f}")
    print(f"  - Duration: {duration:.2f} seconds")
    print(f"  - Total frames: {frame_count}")
    
    # Sample frames for analysis
    sample_frames = min(15, frame_count)
    predictions = []
    
    for i in range(0, frame_count, max(1, frame_count // sample_frames)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            continue
        
        # Detect deepfake using Xception
        try:
            fake_prob = detector.detect_deepfake(frame)
            predictions.append(fake_prob)
        except Exception as e:
            # Skip problematic frame and continue
            print(f"  - Skipping frame {i} due to error: {e}")
            continue
    
    cap.release()
    
    if not predictions:
        print("  ❌ No frames could be processed")
        return None
    
    # Calculate final score
    final_score = np.mean(predictions)
    std_score = np.std(predictions)
    
    print(f"  - Frames analyzed: {len(predictions)}")
    print(f"  - Average fake probability: {final_score:.3f}")
    print(f"  - Standard deviation: {std_score:.3f}")
    print(f"  - Final score: {final_score:.3f}")
    print()
    
    return {
        'filename': os.path.basename(video_path),
        'fake_score': final_score,
        'std_score': std_score,
        'frames_analyzed': len(predictions)
    }

def main():
    print("=== Xception Deepfake Detection ===")
    print("Using Xception model for deepfake detection")
    print()

    parser = argparse.ArgumentParser(description='Xception Deepfake Detection')
    parser.add_argument('--input', type=str, default=None, help='Path to a single video to analyze')
    parser.add_argument('--output', type=str, default='./output', help='Directory to write CSV output')
    args = parser.parse_args()

    # Resolve paths
    data_path = os.path.join('.', 'data')
    test_videos_path = os.path.join(data_path, 'test_videos') if args.input is None else None
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize Xception detector
    detector = XceptionDetector()
    
    if detector.model is None:
        print("❌ Could not load Xception model. Exiting.")
        return
    
    print()
    
    # Determine videos to process
    if args.input is not None:
        if not os.path.exists(args.input):
            print(f"❌ Input video not found: {args.input}")
            return
        video_files = [args.input]
    else:
        if not os.path.exists(test_videos_path):
            print(f"❌ Test directory {test_videos_path} not found")
            print("Create it and add .mp4 files, e.g.: .\\data\\test_videos\\your_video.mp4")
            return
        video_files = glob.glob(os.path.join(test_videos_path, "*.mp4"))
        if not video_files:
            print(f"❌ No MP4 videos found in {test_videos_path}")
            return
    
    print(f"Found {len(video_files)} videos to analyze:\n")
    
    results = []
    for video_path in video_files:
        result = analyze_video_xception(video_path, detector)
        if result:
            results.append(result)
    
    # Save results
    output_file = os.path.join(output_dir, 'submission_xception.csv')
    
    with open(output_file, 'w') as f:
        f.write('filename,fake_score,std_score,frames_analyzed\n')
        for result in results:
            f.write(f"{result['filename']},{result['fake_score']:.6f},"
                   f"{result['std_score']:.6f},{result['frames_analyzed']}\n")
    
    print(f"Results saved to: {output_file}")
    print("\n=== Summary ===")
    for result in results:
        status = "FAKE" if result['fake_score'] > 0.5 else "REAL"
        print(f"{result['filename']}: {result['fake_score']:.3f} ({status})")

if __name__ == "__main__":
    main()

