import numpy as np
import pywt
import cv2
import matplotlib.pyplot as plt
from PIL import Image

def wavelet_compress(image_path, compression_ratio):
    """
    Apply wavelet-based image compression
    
    Parameters:
    image_path (str): Path to input image
    compression_ratio (float): Compression ratio (0-100)
    
    Returns:
    numpy.ndarray: Compressed image
    """
    
    # Read the image
    image = cv2.imread(image_path)
    
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Convert to grayscale for simplicity (or process each channel separately)
    image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    
    # Normalize image to [0, 1]
    image_normalized = image_gray.astype(np.float32) / 255.0
    
    # Apply 2D Discrete Wavelet Transform (DWT) using 'bior1.3' wavelet
    coeffs = pywt.dwt2(image_normalized, 'bior1.3')
    
    # Extract the coefficients
    LL, (LH, HL, HH) = coeffs
    
    # Store original coefficients for comparison
    LL_original = LL.copy()
    
    # Sort and threshold the approximation coefficients (LL) based on compression ratio
    threshold = np.percentile(np.abs(LL), compression_ratio)
    LL_compressed = LL.copy()
    LL_compressed[np.abs(LL_compressed) < threshold] = 0
    
    # Reconstruct the image using inverse DWT
    compressed_coeffs = (LL_compressed, (LH, HL, HH))
    reconstructed_image = pywt.idwt2(compressed_coeffs, 'bior1.3')
    
    # Clip values to valid range [0, 1] and convert back to [0, 255]
    reconstructed_image = np.clip(reconstructed_image, 0, 1)
    compressed_image = (reconstructed_image * 255).astype(np.uint8)
    
    # Create visualization
    create_visualization(image_gray, LL_original, LL_compressed, LH, HL, HH, compressed_image)
    
    return compressed_image

def create_visualization(original, LL_orig, LL_comp, LH, HL, HH, compressed):
    """
    Create matplotlib figure to display the compressed image and its components
    """
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('Wavelet-Based Image Compression Analysis', fontsize=16, fontweight='bold')
    
    # Original image
    axes[0, 0].imshow(original, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Approximation coefficients (LL) - Original
    axes[0, 1].imshow(np.abs(LL_orig), cmap='hot')
    axes[0, 1].set_title('LL Coefficients (Original)')
    axes[0, 1].axis('off')
    
    # Approximation coefficients (LL) - Compressed
    axes[0, 2].imshow(np.abs(LL_comp), cmap='hot')
    axes[0, 2].set_title('LL Coefficients (Compressed)')
    axes[0, 2].axis('off')
    
    # Horizontal detail coefficients (LH)
    axes[0, 3].imshow(np.abs(LH), cmap='cool')
    axes[0, 3].set_title('LH Coefficients (Horizontal)')
    axes[0, 3].axis('off')
    
    # Vertical detail coefficients (HL)
    axes[1, 0].imshow(np.abs(HL), cmap='cool')
    axes[1, 0].set_title('HL Coefficients (Vertical)')
    axes[1, 0].axis('off')
    
    # Diagonal detail coefficients (HH)
    axes[1, 1].imshow(np.abs(HH), cmap='cool')
    axes[1, 1].set_title('HH Coefficients (Diagonal)')
    axes[1, 1].axis('off')
    
    # Compressed image
    axes[1, 2].imshow(compressed, cmap='gray')
    axes[1, 2].set_title('Compressed Image')
    axes[1, 2].axis('off')
    
    # Difference image
    difference = cv2.absdiff(original, compressed)
    axes[1, 3].imshow(difference, cmap='gray')
    axes[1, 3].set_title('Difference (Original - Compressed)')
    axes[1, 3].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Print compression statistics
    print_compression_stats(LL_orig, LL_comp)

def print_compression_stats(LL_orig, LL_comp):
    """
    Print compression statistics
    """
    total_coefficients = LL_orig.size
    non_zero_original = np.count_nonzero(LL_orig)
    non_zero_compressed = np.count_nonzero(LL_comp)
    
    compression_percentage = (1 - non_zero_compressed / non_zero_original) * 100
    
    print(f"\n=== Compression Statistics ===")
    print(f"Total coefficients: {total_coefficients}")
    print(f"Non-zero coefficients (original): {non_zero_original}")
    print(f"Non-zero coefficients (compressed): {non_zero_compressed}")
    print(f"Compression achieved: {compression_percentage:.2f}%")

def main():
    """
    Main function to execute the wavelet compression
    """
    # Specify input image path and compression ratio
    input_image_path = 'test1.jpeg'
    compression_ratio = 90  # Percentage of coefficients to keep
    
    try:
        # Perform wavelet compression
        compressed_image = wavelet_compress(input_image_path, compression_ratio)
        
        # Save the compressed image using OpenCV
        success = cv2.imwrite('compressed_test1.jpeg', compressed_image)
        
        if success:
            print(f"Compressed image saved successfully as 'compressed_test1.jpeg'")
        else:
            print("Error: Could not save compressed image")
            
    except FileNotFoundError:
        print(f"Error: Image file '{input_image_path}' not found.")
    except Exception as e:
        print(f"Error during compression: {str(e)}")

# Execute main function if script is run directly
if __name__ == '__main__':
    main()
