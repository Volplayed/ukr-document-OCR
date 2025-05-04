import cv2
import numpy as np
from PIL import Image, ImageEnhance
from deskew import determine_skew
import matplotlib.pyplot as plt
import os

class Preprocessor:
    def __init__(self, debug_mode: bool = False):
        self.debug_mode = debug_mode

    def load_image(self, image_path: str) -> np.ndarray:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        return image

    def _show_debug_image(self, image: np.ndarray, title: str):
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
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        dim = (width, height)

        resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        self._show_debug_image(resized, "Resized Image")

        return resized

    def correct_skew(self, image: np.ndarray) -> np.ndarray:
        gray = image.copy()
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        angle = determine_skew(gray)

        if abs(angle) < 0.5:
            return image
        
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
        if len(image.shape) == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)

            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)

            enhanced_lab = cv2.merge((cl, a, b))
            enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        else:
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(image)

        self._show_debug_image(enhanced, "Contrast Enhanced")
        return enhanced

    def denoise(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        final = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        self._show_debug_image(final, "Denoised")

        return final

    def morphological_operations(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()


        kernel = np.ones((2, 2), np.uint8)

        opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        self._show_debug_image(opening, "Opening")

        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
        self._show_debug_image(closing, "Closing")

        return closing

    def adaptive_binarization(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        self._show_debug_image(otsu, "Otsu Thresholding")

        adaptive = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 19, 11
        )
        self._show_debug_image(adaptive, "Adaptive Thresholding")

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

        denoised = self.denoise(image)

        contrast = self.enhance_contrast(denoised)

        deskewed = self.correct_skew(contrast)

        binary = self.adaptive_binarization(deskewed)

        cleaned = self.morphological_operations(binary)

        return cleaned
    
if __name__ == "__main__":
    preprocessor = Preprocessor(debug_mode=True)