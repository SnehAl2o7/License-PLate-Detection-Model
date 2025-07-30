import os
import cv2 #type: ignore
import numpy as np #type: ignore
import pandas as pd #type: ignore
import matplotlib.pyplot as plt #type: ignore
from pathlib import Path
import torch #type: ignore
import torchvision.transforms as transforms #type: ignore
from PIL import Image
import easyocr #type: ignore
import pytesseract #type: ignore
import re
from ultralytics import YOLO #type: ignore
from sklearn.metrics import precision_score, recall_score, f1_score #type: ignore
import yaml
import shutil
from tqdm import tqdm #type: ignore
import argparse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LicensePlateDataset:
    """Class to handle dataset preparation and management"""
    
    def __init__(self, base_path="license_plate_dataset"):
        self.base_path = Path(base_path)
        self.images_path = self.base_path / "images"
        self.labels_path = self.base_path / "labels"
        self.train_path = self.base_path / "train"
        self.val_path = self.base_path / "val"
        self.test_path = self.base_path / "test"
        
    def create_directory_structure(self):
        """Create the required directory structure for YOLO training"""
        directories = [
            self.train_path / "images",
            self.train_path / "labels",
            self.val_path / "images", 
            self.val_path / "labels",
            self.test_path / "images",
            self.test_path / "labels"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        
        logger.info("Directory structure created successfully")
    
    def create_yaml_config(self):
        """Create YAML configuration file for YOLO training"""
        config = {
            'path': str(self.base_path.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': 1,  # number of classes
            'names': ['license_plate']
        }
        
        yaml_path = self.base_path / "dataset.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        logger.info(f"YAML config created at {yaml_path}")
        return yaml_path

class LicensePlateDetector:
    """Main class for license plate detection and OCR"""
    
    def __init__(self, model_path=None, confidence_threshold=0.5):
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.ocr_reader = None
        
        # Initialize OCR readers
        try:
            self.ocr_reader = easyocr.Reader(['en'])
            self.use_easyocr = True
            logger.info("EasyOCR initialized successfully")
        except Exception as e:
            logger.warning(f"EasyOCR initialization failed: {e}")
            self.use_easyocr = False
        
        # Load model if path provided
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            # Initialize with pre-trained YOLOv8 model
            self.model = YOLO('yolov8n.pt')
            logger.info("Initialized with YOLOv8n pre-trained model")
    
    def load_model(self, model_path):
        """Load trained YOLO model"""
        try:
            self.model = YOLO(model_path)
            logger.info(f"Model loaded successfully from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def train_model(self, dataset_yaml_path, epochs=100, img_size=640, batch_size=16):
        """Train YOLO model on license plate dataset"""
        try:
            # Initialize model for training
            model = YOLO('yolov8n.pt')
            
            # Train the model
            results = model.train(
                data=dataset_yaml_path,
                epochs=epochs,
                imgsz=img_size,
                batch=batch_size,
                name='license_plate_detection',
                patience=10,
                save=True,
                plots=True,
                device='auto'  # Automatically use GPU if available
            )
            
            # Save the best model
            best_model_path = "runs/detect/license_plate_detection/weights/best.pt"
            if os.path.exists(best_model_path):
                shutil.copy(best_model_path, "best_license_plate_model.pt")
                logger.info("Best model saved as 'best_license_plate_model.pt'")
            
            return results
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    def detect_license_plates(self, image_path, save_crops=True):
        """Detect license plates in an image"""
        if self.model is None:
            raise ValueError("Model not loaded. Please train or load a model first.")
        
        try:
            # Run inference
            results = self.model(image_path, conf=self.confidence_threshold)
            
            # Load original image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image from {image_path}")
            
            detections = []
            cropped_plates = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        
                        # Convert to integers
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        
                        # Crop license plate region
                        cropped_plate = image[y1:y2, x1:x2]
                        
                        detection_info = {
                            'bbox': (x1, y1, x2, y2),
                            'confidence': float(confidence),
                            'cropped_image': cropped_plate
                        }
                        
                        detections.append(detection_info)
                        cropped_plates.append(cropped_plate)
                        
                        # Draw bounding box on original image
                        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(image, f'License Plate: {confidence:.2f}', 
                                  (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            return detections, image, cropped_plates
            
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            raise
    
    def preprocess_for_ocr(self, image):
        """Preprocess cropped license plate for better OCR results"""
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Resize image for better OCR (maintain aspect ratio)
            height, width = gray.shape
            if width < 200:
                scale_factor = 200 / width
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            
            # Apply image enhancements
            # 1. Noise reduction
            denoised = cv2.medianBlur(gray, 3)
            
            # 2. Contrast enhancement
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(denoised)
            
            # 3. Threshold to get binary image
            _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # 4. Morphological operations to clean up
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            
            return cleaned
            
        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            return image
    
    def extract_text_easyocr(self, image):
        """Extract text using EasyOCR"""
        try:
            results = self.ocr_reader.readtext(image)
            texts = []
            confidences = []
            
            for (bbox, text, confidence) in results:
                # Clean the text
                cleaned_text = re.sub(r'[^A-Z0-9]', '', text.upper())
                if len(cleaned_text) >= 3:  # Minimum length for license plate
                    texts.append(cleaned_text)
                    confidences.append(confidence)
            
            return texts, confidences
            
        except Exception as e:
            logger.error(f"EasyOCR extraction failed: {e}")
            return [], []
    
    def extract_text_tesseract(self, image):
        """Extract text using Tesseract OCR"""
        try:
            # Configure Tesseract for license plates
            custom_config = r'--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
            
            text = pytesseract.image_to_string(image, config=custom_config)
            
            # Clean the text
            cleaned_text = re.sub(r'[^A-Z0-9]', '', text.upper().strip())
            
            # Get confidence score
            data = pytesseract.image_to_data(image, config=custom_config, output_type=pytesseract.Output.DICT)
            confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
            avg_confidence = np.mean(confidences) / 100.0 if confidences else 0.0
            
            return [cleaned_text] if cleaned_text else [], [avg_confidence] if cleaned_text else []
            
        except Exception as e:
            logger.error(f"Tesseract extraction failed: {e}")
            return [], []
    
    def read_license_plate(self, cropped_image):
        """Read text from cropped license plate image"""
        try:
            # Preprocess image
            processed_image = self.preprocess_for_ocr(cropped_image)
            
            all_texts = []
            all_confidences = []
            
            # Try EasyOCR if available
            if self.use_easyocr:
                texts_easy, conf_easy = self.extract_text_easyocr(processed_image)
                all_texts.extend([(text, conf, 'EasyOCR') for text, conf in zip(texts_easy, conf_easy)])
            
            # Try Tesseract OCR
            texts_tess, conf_tess = self.extract_text_tesseract(processed_image)
            all_texts.extend([(text, conf, 'Tesseract') for text, conf in zip(texts_tess, conf_tess)])
            
            if not all_texts:
                return None, 0.0, 'None'
            
            # Select best result based on confidence and text length
            best_result = max(all_texts, key=lambda x: (x[1], len(x[0])))
            
            return best_result[0], best_result[1], best_result[2]
            
        except Exception as e:
            logger.error(f"OCR failed: {e}")
            return None, 0.0, 'Error'
    
    def process_image(self, image_path, output_dir="output"):
        """Complete pipeline: detect license plates and read text"""
        try:
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Detect license plates
            detections, annotated_image, cropped_plates = self.detect_license_plates(image_path)
            
            results = []
            
            # Process each detected license plate
            for i, (detection, cropped_plate) in enumerate(zip(detections, cropped_plates)):
                # Read license plate text
                plate_text, confidence, ocr_method = self.read_license_plate(cropped_plate)
                
                result = {
                    'detection_id': i,
                    'bbox': detection['bbox'],
                    'detection_confidence': detection['confidence'],
                    'plate_text': plate_text,
                    'ocr_confidence': confidence,
                    'ocr_method': ocr_method
                }
                
                results.append(result)
                
                # Save cropped license plate
                crop_filename = f"license_plate_{i}.jpg"
                crop_path = os.path.join(output_dir, crop_filename)
                cv2.imwrite(crop_path, cropped_plate)
                
                logger.info(f"License Plate {i}: {plate_text} (Confidence: {confidence:.2f}, Method: {ocr_method})")
            
            # Save annotated image
            output_filename = f"detected_{os.path.basename(image_path)}"
            output_path = os.path.join(output_dir, output_filename)
            cv2.imwrite(output_path, annotated_image)
            
            return results, output_path
            
        except Exception as e:
            logger.error(f"Image processing failed: {e}")
            raise
    
    def evaluate_model(self, test_images_dir, ground_truth_file=None):
        """Evaluate model performance on test dataset"""
        try:
            test_images = list(Path(test_images_dir).glob("*.jpg")) + list(Path(test_images_dir).glob("*.png"))
            
            total_images = len(test_images)
            successful_detections = 0
            total_plates_detected = 0
            
            results = []
            
            for image_path in tqdm(test_images, desc="Evaluating"):
                try:
                    detections, _, _ = self.detect_license_plates(str(image_path))
                    
                    if detections:
                        successful_detections += 1
                        total_plates_detected += len(detections)
                        
                        for detection in detections:
                            plate_text, ocr_conf, ocr_method = self.read_license_plate(detection['cropped_image'])
                            # Example usage without command line arguments
                            results.append({
                                'image': image_path.name,
                                'detection_confidence': detection['confidence'],
                                'plate_text': plate_text,
                                'ocr_confidence': ocr_conf,
                                'ocr_method': ocr_method
                            })
                
                except Exception as e:
                    logger.warning(f"Failed to process {image_path}: {e}")
                    continue
            
            # Calculate metrics
            detection_rate = successful_detections / total_images if total_images > 0 else 0
            avg_plates_per_image = total_plates_detected / total_images if total_images > 0 else 0
            
            metrics = {
                'total_images': total_images,
                'successful_detections': successful_detections,
                'detection_rate': detection_rate,
                'total_plates_detected': total_plates_detected,
                'avg_plates_per_image': avg_plates_per_image,
                'results': results
            }
            
            logger.info(f"Evaluation Results:")
            logger.info(f"  Total Images: {total_images}")
            logger.info(f"  Successful Detections: {successful_detections}")
            logger.info(f"  Detection Rate: {detection_rate:.2%}")
            logger.info(f"  Total Plates Detected: {total_plates_detected}")
            logger.info(f"  Average Plates per Image: {avg_plates_per_image:.2f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            raise

def main():
    """Main function to demonstrate the license plate detection system"""
    parser = argparse.ArgumentParser(description='License Plate Detection and OCR System')
    parser.add_argument('--mode', choices=['train', 'detect', 'evaluate'], required=True,
                        help='Mode of operation')
    parser.add_argument('--input', type=str, help='/train/images/TEST.jpeg')
    parser.add_argument('--model', type=str, help='Path to trained model')
    parser.add_argument('--dataset', type=str, help='dataset.yaml')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--output', type=str, default='output', help='Output directory')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = LicensePlateDetector(model_path=args.model)
    
    if args.mode == 'train':
        if not args.dataset:
            logger.error("Dataset YAML file required for training")
            return
        
        logger.info("Starting training...")
        results = detector.train_model(args.dataset, epochs=args.epochs)
        logger.info("Training completed successfully")
    
    elif args.mode == 'detect':
        if not args.input:
            logger.error("Input image path required for detection")
            return
        
        logger.info(f"Processing image: {args.input}")
        results, output_path = detector.process_image(args.input, args.output)
        
        print("\nDetection Results:")
        for result in results:
            print(f"License Plate: {result['plate_text']} "
                  f"(Detection: {result['detection_confidence']:.2f}, "
                  f"OCR: {result['ocr_confidence']:.2f}, "
                  f"Method: {result['ocr_method']})")
        
        print(f"\nAnnotated image saved to: {output_path}")
    
    elif args.mode == 'evaluate':
        if not args.input:
            logger.error("Test images directory required for evaluation")
            return
        
        logger.info(f"Evaluating model on: {args.input}")
        metrics = detector.evaluate_model(args.input)
        
        # Save results to CSV
        if metrics['results']:
            df = pd.DataFrame(metrics['results'])
            csv_path = os.path.join(args.output, 'evaluation_results.csv')
            df.to_csv(csv_path, index=False)
            logger.info(f"Detailed results saved to: {csv_path}")

if __name__ == "__main__":
    try:
        # Initialize the detector
        detector = LicensePlateDetector()
        
        # Example: Process a single image
        # Make sure you have a test image
        test_image = "TEST.jpeg"  # Replace with your image path
        
        if os.path.exists(test_image):
            print("Processing test image...")
            results, output_path = detector.process_image(test_image)
            
            print("\nResults:")
            for i, result in enumerate(results):
                print(f"License Plate {i+1}: {result['plate_text']}")
                print(f"  Detection Confidence: {result['detection_confidence']:.2f}")
                print(f"  OCR Confidence: {result['ocr_confidence']:.2f}")
                print(f"  OCR Method: {result['ocr_method']}")
        else:
            print("Test image not found. Please provide a test image or use command line arguments.")
            print("\nUsage examples:")
            print("python license_plate_detector.py --mode detect --input test_image.jpg")
            print("python license_plate_detector.py --mode train --dataset dataset.yaml --epochs 100")
            print("python license_plate_detector.py --mode evaluate --input test_images_dir/")
    
    except Exception as e:
        logger.error(f"Error: {e}")
        print("\nTo use this system:")
        print("1. Install required packages: pip install ultralytics opencv-python easyocr pytesseract pandas matplotlib scikit-learn")
        print("2. Prepare your dataset in YOLO format")
        print("3. Run training or detection as needed")