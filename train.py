import torch
from ultralytics import YOLO

class Train:
    def __init__(self, model_path, data_yaml, epochs=100, img_size=640):
        self.model_path = model_path
        self.data_yaml = data_yaml
        self.epochs = epochs
        self.img_size = img_size

    def train_model(self):
        """Train YOLO model on the dataset."""
        # Load model
        model = YOLO(self.model_path)

        # Check for GPU availability
        device = 0 if torch.cuda.is_available() else 'cpu'
        if device == 0:
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("Warning: No GPU detected, training on CPU may be slow.")

        # Train model
        train_results = model.train(
            data=self.data_yaml,
            epochs=self.epochs,
            imgsz=self.img_size,
            device=device
        )

        return train_results
