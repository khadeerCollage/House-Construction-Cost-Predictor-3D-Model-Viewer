import torch
import cv2
import numpy as np
from PIL import Image
import logging
import os
import threading

logger = logging.getLogger(__name__)

class VitDetector:
    """
    Zero-Shot Vision Transformer for Floor Plan Symbol Detection.
    Uses Google's OWL-ViT to detect objects from text queries.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(VitDetector, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
            
        self._initialized = True
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.processor = None
        self.default_queries = [
            "a photo of a bed",
            "a photo of a door arc", 
            "a photo of a door swing",
            "a photo of a toilet",
            "a photo of a bathtub",
            "a photo of a kitchen stove",
            "a photo of a sofa",
            "a photo of a dining table"
        ]
        # Map long query to clean label
        self.query_labels = {
            "a photo of a bed": "bed",
            "a photo of a door arc": "door",
            "a photo of a door swing": "door",
            "a photo of a toilet": "toilet",
            "a photo of a bathtub": "bathtub",
            "a photo of a kitchen stove": "stove",
            "a photo of a sofa": "sofa",
            "a photo of a dining table": "dining table"
        }
        
    def load_model(self):
        """Lazy load the model to save memory until actually needed."""
        if self.model is not None:
            return
            
        logger.info(f"Loading OWL-ViT on {self.device}... (this may take a moment)")
        from transformers.models.owlvit import OwlViTProcessor, OwlViTForObjectDetection
        
        # Suppress some transformers warnings
        import warnings
        warnings.filterwarnings("ignore", category=UserWarning)
        
        self.processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
        self.model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
        self.model = self.model.to(self.device)
        self.model.eval()
        logger.info("OWL-ViT loaded successfully!")

    def detect_objects(self, image_input, threshold=0.08):
        """
        Detect objects in the image.
        Args:
            image_input: Path to image, or numpy array (BGR/RGB)
            threshold: Confidence score threshold
        Returns:
            List of dicts: [{'label': 'bed', 'score': 0.95, 'bbox': [x, y, w, h]}]
        """
        self.load_model()
        
        # Convert input to PIL Image
        if isinstance(image_input, str):
            image = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, np.ndarray):
            # If BGR from cv2, convert to RGB
            if len(image_input.shape) == 3 and image_input.shape[2] == 3:
                image = Image.fromarray(cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB))
            else:
                image = Image.fromarray(image_input)
        else:
            image = image_input # Assume already PIL
            
        # Run inference
        inputs = self.processor(text=[self.default_queries], images=image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # Get predictions
        target_sizes = torch.tensor([image.size[::-1]]).to(self.device)
        results = self.processor.image_processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=threshold)
        
        # Format results
        detected_objects = []
        
        i = 0  # We only passed one image
        boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]
        
        for box, score, label_idx in zip(boxes, scores, labels):
            score_val = score.item()
            label_text = self.default_queries[label_idx.item()]
            clean_label = self.query_labels[label_text]
            
            # Convert [xmin, ymin, xmax, ymax] to (x, y, w, h)
            box_np = box.cpu().numpy()
            x1, y1, x2, y2 = [int(v) for v in box_np]
            w = max(1, x2 - x1)
            h = max(1, y2 - y1)
            
            detected_objects.append({
                "label": clean_label,
                "score": score_val,
                "bbox": [x1, y1, w, h]
            })
            
        return detected_objects
