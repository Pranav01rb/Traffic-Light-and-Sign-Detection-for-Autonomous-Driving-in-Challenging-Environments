import os
from train import Train # type: ignore
from test import Test

def main():
    # Paths for training and testing
    train_data_yaml = "/path/to/data.yaml"
    initial_model_path = "yolo11n.pt"
    trained_model_path = "/path/to/best_trained_model/best.pt"

    # Train the model
    print("Starting model training...")
    trainer = Train(model_path=initial_model_path, data_yaml=train_data_yaml, epochs=100, img_size=640)
    train_results = trainer.train_model()

    # Log training results
    print("Training completed!")
    print(f"Training Results: {train_results}")
    
    # Test the trained model on the test dataset
    print("Evaluating model on the test dataset...")
    tester = Test(model_path=trained_model_path, data_yaml=train_data_yaml)
    tester.evaluate_model()

    # Test the model on a sample image
    test_image = "/path/to/test_image.jpg"
    print(f"Testing model on sample image: {test_image}")
    tester.test_on_sample_image(test_image)

if __name__ == "__main__":
    main()
