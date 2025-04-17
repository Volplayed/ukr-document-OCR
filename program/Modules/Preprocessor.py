import cv2
import numpy as np
from PIL import Image, ImageEnhance
from deskew import determine_skew
import matplotlib.pyplot as plt
import os

class Preprocessor:
    """
    Class for preprocessing images before OCR to improve recognition quality.
    Handles issues with image quality, skew/angle, lighting, and noise.
    """

    def __init__(self, debug_mode: bool = False):
        """
        Initialize the ImagePreprocessor.

        Args:
            debug_mode: If True, displays intermediate processing steps
        """
        self.debug_mode = debug_mode

    def load_image(self, image_path: str) -> np.ndarray:
        """
        Load an image from a file path.

        Args:
            image_path: Path to the image file

        Returns:
            Loaded image as numpy array
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        return image

    def _show_debug_image(self, image: np.ndarray, title: str):
        """Display image if debug mode is enabled."""
        if self.debug_mode:
            plt.figure(figsize=(10, 10))
            if len(image.shape) == 3:
                plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                plt.imshow(image, cmap='gray')
            plt.title(title)
            plt.axis('off')
            plt.show()

    def resize_image(self, image: np.ndarray, scale_percent: int = 100) -> np.ndarray:
        """
        Resize image by given percentage.

        Args:
            image: Input image
            scale_percent: Percentage to scale the image

        Returns:
            Resized image
        """
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        dim = (width, height)

        resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        self._show_debug_image(resized, "Resized Image")

        return resized

    def correct_skew(self, image: np.ndarray) -> np.ndarray:
        """
        Detect and correct skew in an image.

        Args:
            image: Input image

        Returns:
            Deskewed image
        """
        # Convert to grayscale if needed
        gray = image.copy()
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Determine the skew angle
        angle = determine_skew(gray)

        # If skew is minimal, return original image
        if abs(angle) < 0.5:
            return image

        # Rotate the image to correct the skew
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            image,
            M,
            (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )

        self._show_debug_image(rotated, f"Deskewed Image (angle: {angle:.2f}°)")
        return rotated

    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization).

        Args:
            image: Input image

        Returns:
            Contrast-enhanced image
        """
        # Convert to LAB color space
        if len(image.shape) == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)

            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)

            # Merge channels and convert back to BGR
            enhanced_lab = cv2.merge((cl, a, b))
            enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        else:
            # Apply CLAHE directly to grayscale image
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(image)

        self._show_debug_image(enhanced, "Contrast Enhanced")
        return enhanced

    def advanced_denoise(self, image: np.ndarray) -> np.ndarray:
        """
        Apply multiple denoising techniques for better noise reduction.

        Args:
            image: Input image

        Returns:
            Advanced denoised image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Step 3: Apply non-local means denoising for remaining noise
        final = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        self._show_debug_image(final, "Advanced Denoised")

        return final

    def morphological_operations(self, image: np.ndarray) -> np.ndarray:
        """
        Apply morphological operations to clean up the image.

        Args:
            image: Input image (should be binary/grayscale)

        Returns:
            Cleaned image
        """
        # Ensure image is grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Create kernels for morphological operations
        kernel = np.ones((2, 2), np.uint8)

        # Apply opening to remove small noise
        opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        self._show_debug_image(opening, "Opening")

        # Apply closing to fill small holes in text
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
        self._show_debug_image(closing, "Closing")

        return closing

    def adaptive_binarization(self, image: np.ndarray) -> np.ndarray:
        """
        Improved binarization with noise consideration.

        Args:
            image: Input image

        Returns:
            Better binarized image with less noise
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Try Otsu's thresholding first (works well for bimodal images)
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        self._show_debug_image(otsu, "Otsu Thresholding")

        # Also try adaptive thresholding
        adaptive = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 19, 11  # Increased block size and constant
        )
        self._show_debug_image(adaptive, "Adaptive Thresholding")

        # Combine both methods (this often preserves text better)
        combined = cv2.bitwise_and(otsu, adaptive)
        self._show_debug_image(combined, "Combined Binarization")

        return combined

    def process_image(self, image_path: str) -> np.ndarray:
        """
        Improved processing pipeline with better noise handling.

        Args:
            image_path: Path to the input image

        Returns:
            Fully preprocessed image ready for OCR
        """
        # Load image
        image = self.load_image(image_path)
        self._show_debug_image(image, "Original Image")

        image = self.resize_image(image, scale_percent=200)

        denoised = self.advanced_denoise(image)

        contrast = self.enhance_contrast(denoised)

        deskewed = self.correct_skew(contrast)

        binary = self.adaptive_binarization(deskewed)

        cleaned = self.morphological_operations(binary)

        return cleaned
    
if __name__ == "__main__":
    # Example usage
    preprocessor = Preprocessor(debug_mode=True)