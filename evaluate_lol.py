"""
Low-Light Image Enhancement - Real Dataset Demo
Using LOL (Low-Light) Dataset for evaluation
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os


class LowLightEnhancer:
    """Enhanced Low-Light Image Enhancement"""
    
    def paper_baseline(self, img):
        """
        Baseline method - Simple Gamma Correction
        (Simplified version of paper's concept)
        """
        img_float = img.astype(np.float32) / 255.0
        gamma = 2.2  # Fixed gamma
        enhanced = np.power(img_float, 1.0 / gamma)
        return (enhanced * 255).astype(np.uint8)
    
    def improved_method(self, img):
        """
        Our Improved Method with Multiple Enhancements
        
        Improvements:
        1. Adaptive Gamma Correction - auto-adjusts based on image brightness
        2. CLAHE - better local contrast
        3. Denoising - reduces noise amplification
        4. Color Correction - preserves natural colors
        """
        # Convert to float
        img_float = img.astype(np.float32) / 255.0
        
        # Calculate brightness
        gray = cv2.cvtColor((img_float * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY) / 255.0
        mean_brightness = np.mean(gray)
        
        # IMPROVEMENT 1: Adaptive Gamma Correction
        if mean_brightness < 0.15:
            gamma = 2.8  # Very dark
        elif mean_brightness < 0.3:
            gamma = 2.2  # Dark
        else:
            gamma = 1.8  # Moderate
        
        gamma_corrected = np.power(img_float, 1.0 / gamma)
        
        # IMPROVEMENT 2: Contrast Enhancement with CLAHE
        gamma_uint = (gamma_corrected * 255).astype(np.uint8)
        lab = cv2.cvtColor(gamma_uint, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_clahe = clahe.apply(l)
        
        # Merge back
        lab_enhanced = cv2.merge([l_clahe, a, b])
        enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
        
        # IMPROVEMENT 3: Mild Denoising
        denoised = cv2.fastNlMeansDenoisingColored(enhanced, None, 5, 5, 7, 21)
        
        # IMPROVEMENT 4: Brightness adjustment if still too dark
        final_brightness = np.mean(cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)) / 255.0
        if final_brightness < 0.4:
            denoised = cv2.convertScaleAbs(denoised, alpha=1.2, beta=10)
        
        return denoised


def calculate_metrics(img):
    """Calculate image quality metrics"""
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    # Brightness (mean intensity)
    brightness = np.mean(gray) / 255.0
    
    # Entropy (information content)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = hist.flatten() / hist.sum()
    hist = hist[hist > 0]
    entropy = -np.sum(hist * np.log2(hist))
    
    # Contrast (standard deviation)
    contrast = np.std(gray) / 255.0
    
    return {
        'brightness': brightness,
        'entropy': entropy,
        'contrast': contrast
    }


def create_comparison_figure(low_img, high_img, baseline_result, improved_result, 
                            img_name, save_path):
    """Create comprehensive comparison figure"""
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, height_ratios=[2, 2, 1], hspace=0.3, wspace=0.25)
    
    # Calculate metrics
    metrics_low = calculate_metrics(low_img)
    metrics_high = calculate_metrics(high_img)
    metrics_base = calculate_metrics(baseline_result)
    metrics_impr = calculate_metrics(improved_result)
    
    # Convert BGR to RGB for display
    low_rgb = cv2.cvtColor(low_img, cv2.COLOR_BGR2RGB)
    high_rgb = cv2.cvtColor(high_img, cv2.COLOR_BGR2RGB)
    base_rgb = cv2.cvtColor(baseline_result, cv2.COLOR_BGR2RGB)
    impr_rgb = cv2.cvtColor(improved_result, cv2.COLOR_BGR2RGB)
    
    # Row 1: Original and Ground Truth
    ax1 = fig.add_subplot(gs[0, 0:2])
    ax1.imshow(low_rgb)
    ax1.set_title(f'Input: Low-Light Image\nBrightness: {metrics_low["brightness"]:.3f}', 
                  fontsize=14, fontweight='bold', color='darkred')
    ax1.axis('off')
    
    ax2 = fig.add_subplot(gs[0, 2:4])
    ax2.imshow(high_rgb)
    ax2.set_title(f'Ground Truth: Normal-Light Image\nBrightness: {metrics_high["brightness"]:.3f}', 
                  fontsize=14, fontweight='bold', color='darkgreen')
    ax2.axis('off')
    
    # Row 2: Enhanced Results
    ax3 = fig.add_subplot(gs[1, 0:2])
    ax3.imshow(base_rgb)
    ax3.set_title(f'Paper Baseline (Fixed Gamma)\nBrightness: {metrics_base["brightness"]:.3f} | ' +
                  f'Entropy: {metrics_base["entropy"]:.2f}', 
                  fontsize=14, fontweight='bold')
    ax3.axis('off')
    
    ax4 = fig.add_subplot(gs[1, 2:4])
    ax4.imshow(impr_rgb)
    ax4.set_title(f'Our Improved Method ✓\nBrightness: {metrics_impr["brightness"]:.3f} | ' +
                  f'Entropy: {metrics_impr["entropy"]:.2f}', 
                  fontsize=14, fontweight='bold', color='green')
    ax4.axis('off')
    
    # Row 3: Metrics Comparison
    ax5 = fig.add_subplot(gs[2, :])
    
    methods = ['Low-Light\n(Input)', 'Ground Truth', 'Paper Baseline', 'Our Improved']
    brightness_vals = [metrics_low['brightness'], metrics_high['brightness'], 
                      metrics_base['brightness'], metrics_impr['brightness']]
    entropy_vals = [metrics_low['entropy'], metrics_high['entropy'],
                   metrics_base['entropy'], metrics_impr['entropy']]
    contrast_vals = [metrics_low['contrast'], metrics_high['contrast'],
                    metrics_base['contrast'], metrics_impr['contrast']]
    
    x = np.arange(len(methods))
    width = 0.25
    
    bars1 = ax5.bar(x - width, brightness_vals, width, label='Brightness', 
                    color=['red', 'green', 'orange', 'darkgreen'])
    bars2 = ax5.bar(x, [e/10 for e in entropy_vals], width, label='Entropy/10',
                    color=['red', 'green', 'orange', 'darkgreen'], alpha=0.7)
    bars3 = ax5.bar(x + width, contrast_vals, width, label='Contrast',
                    color=['red', 'green', 'orange', 'darkgreen'], alpha=0.5)
    
    ax5.set_ylabel('Metric Value', fontsize=12, fontweight='bold')
    ax5.set_title('Quality Metrics Comparison', fontsize=14, fontweight='bold')
    ax5.set_xticks(x)
    ax5.set_xticklabels(methods, fontsize=11)
    ax5.legend(loc='upper left', fontsize=11)
    ax5.grid(True, alpha=0.3, axis='y')
    ax5.set_ylim([0, max(brightness_vals + [e/10 for e in entropy_vals] + contrast_vals) * 1.15])
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=9)
    
    plt.suptitle(f'Low-Light Enhancement Results - {img_name}', 
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return metrics_impr


def main():
    print("="*80)
    print("LOW-LIGHT IMAGE ENHANCEMENT - LOL DATASET EVALUATION")
    print("="*80)
    
    # Paths
    dataset_path = Path('/home/mohanganesh/rupesh_research/GWNet/data/lol_dataset/eval15')
    low_path = dataset_path / 'low'
    high_path = dataset_path / 'high'
    results_path = Path('/home/mohanganesh/rupesh_research/GWNet/results')
    results_path.mkdir(exist_ok=True, parents=True)
    
    # Clean old results
    for old_file in results_path.glob('*'):
        old_file.unlink()
    
    # Initialize enhancer
    enhancer = LowLightEnhancer()
    
    # Select test images (5 images for comprehensive evaluation)
    test_images = ['1.png', '111.png', '23.png', '493.png', '780.png']
    
    all_metrics = []
    
    print(f"\nProcessing {len(test_images)} test images from LOL dataset...")
    print("-"*80)
    
    for idx, img_name in enumerate(test_images, 1):
        print(f"\n[{idx}/{len(test_images)}] Processing: {img_name}")
        
        # Load images
        low_img = cv2.imread(str(low_path / img_name))
        high_img = cv2.imread(str(high_path / img_name))
        
        if low_img is None or high_img is None:
            print(f"  ✗ Error loading {img_name}")
            continue
        
        print(f"  Image size: {low_img.shape[1]}x{low_img.shape[0]}")
        
        # Apply methods
        baseline_result = enhancer.paper_baseline(low_img)
        improved_result = enhancer.improved_method(low_img)
        
        # Calculate metrics
        metrics = calculate_metrics(improved_result)
        all_metrics.append(metrics)
        
        # Save individual enhanced images
        cv2.imwrite(str(results_path / f'enhanced_{img_name}'), improved_result)
        
        # Create comparison figure
        save_path = results_path / f'comparison_{img_name}'
        create_comparison_figure(low_img, high_img, baseline_result, improved_result,
                                img_name, save_path)
        
        print(f"  ✓ Brightness: {metrics['brightness']:.3f} | Entropy: {metrics['entropy']:.2f}")
        print(f"  ✓ Saved: comparison_{img_name}")
    
    # Calculate average metrics
    avg_brightness = np.mean([m['brightness'] for m in all_metrics])
    avg_entropy = np.mean([m['entropy'] for m in all_metrics])
    avg_contrast = np.mean([m['contrast'] for m in all_metrics])
    
    print("\n" + "="*80)
    print("SUMMARY - Average Metrics Across All Test Images:")
    print("-"*80)
    print(f"  Average Brightness: {avg_brightness:.3f}")
    print(f"  Average Entropy:    {avg_entropy:.2f}")
    print(f"  Average Contrast:   {avg_contrast:.3f}")
    print("="*80)
    
    print(f"\n✓ All results saved in: {results_path}")
    print(f"✓ Total comparison images: {len(test_images)}")
    print(f"✓ Total enhanced images: {len(test_images)}")
    print("\n" + "="*80)
    print("PROJECT COMPLETE - Ready for presentation!")
    print("="*80)


if __name__ == "__main__":
    main()
