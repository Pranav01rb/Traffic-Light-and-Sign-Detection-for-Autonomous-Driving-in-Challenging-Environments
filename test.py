import os
from ultralytics import YOLO

class Test:
    def __init__(self, model_path, data_yaml):
        self.model_path = model_path
        self.data_yaml = data_yaml

    def evaluate_model(self):
        """Evaluate YOLO model specifically on the test dataset."""
        # Load trained model
        model = YOLO(self.model_path)

        # Ensure we are passing the correct dataset config for test evaluation
        eval_results = model.val(data=self.data_yaml) 
        print("Test Dataset Evaluation Results:", eval_results)

    def test_on_sample_image(self, test_image, output_folder="test_img_res"):
        """Run inference on a test image and save results."""
        os.makedirs(output_folder, exist_ok=True)

        # Load trained model
        model = YOLO(self.model_path)

        # Run inference
        results = model(test_image, save=True)

        # Save and display results
        output_path = os.path.join(output_folder, os.path.basename(test_image))
        results[0].save(output_path)
        results[0].show()
