import os
import cv2
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging


# Set up logging for debugging
logging.basicConfig(level=logging.INFO)

# Base path for data
BASE_PATH = 'data'
IMG_DIR = f"{BASE_PATH}/train/images"
LABEL_DIR = f"{BASE_PATH}/train/labels"

def read_label(img_name):
    """
    Reads label data from a text file and returns bounding box information.

    Args:
        img_name (str): Name of the image (without extension).

    Returns:
        list: A list of tuples containing class_id, x_center, y_center, width, and height.
    """
    label_path = f"{LABEL_DIR}/{img_name}.txt"
    boxes = []

    try:
        with open(label_path, 'r') as f:
            for line in f.readlines():
                # Parse each line in the label file
                label_data = line.strip().split(' ')
                class_id = label_data[0]
                x_center, y_center, width, height = map(float, label_data[1:])
                boxes.append((class_id, x_center, y_center, width, height))
        logging.info(f"Successfully read labels for {img_name}")
    except FileNotFoundError:
        logging.error(f"Label file not found: {label_path}")
    except Exception as e:
        logging.error(f"Error reading label file {label_path}: {e}")

    return boxes

def draw_bounding_boxes(img, boxes):
    """
    Draws bounding boxes on an image.

    Args:
        img (numpy.ndarray): The image on which to draw bounding boxes.
        boxes (list): A list of bounding box data (class_id, x_center, y_center, width, height).

    Returns:
        numpy.ndarray: The image with bounding boxes drawn.
    """
    height, width, _ = img.shape

    for box in boxes:
        class_id, x_center, y_center, box_width, box_height = box

        # Convert normalized coordinates to pixel values
        x_center = int(x_center * width)
        y_center = int(y_center * height)
        box_width = int(box_width * width)
        box_height = int(box_height * height)

        # Calculate top-left and bottom-right corners of the bounding box
        x1 = int(x_center - box_width / 2)
        y1 = int(y_center - box_height / 2)
        x2 = int(x_center + box_width / 2)
        y2 = int(y_center + box_height / 2)

        # Draw rectangle and class label
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img,
            f"Class {class_id}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    return img

def display_img(img_path):
    """
    Displays an image with bounding boxes.

    Args:
        img_path (str): Path to the image file.
    """
    try:
        # Read the image
        img_name = os.path.basename(img_path).split('.jpg')[0]
        img = cv2.imread(img_path)

        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")

        # Convert BGR to RGB for displaying with matplotlib
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Read labels and draw bounding boxes
        boxes = read_label(img_name)
        img_with_boxes = draw_bounding_boxes(img_rgb.copy(), boxes)

        # Display the image
        plt.figure(figsize=(8, 8))
        plt.imshow(img_with_boxes)
        plt.title(f"Image: {img_name}")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    except FileNotFoundError as e:
        logging.error(e)
    except Exception as e:
        logging.error(f"Error displaying image {img_path}: {e}")


def add_motion_blur(img, kernel_size=15, angle=0):
    """
    Apply motion blur to the image.
    
    Parameters:
        img (numpy.ndarray): Input image.
        kernel_size (int): Size of the motion blur kernel.
        angle (int): Angle of motion blur.

    Returns:
        numpy.ndarray: Blurred image.
    """
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[(kernel_size - 1) // 2, :] = np.ones(kernel_size) / kernel_size

    rot_matrix = cv2.getRotationMatrix2D(((kernel_size - 1) / 2, (kernel_size - 1) / 2), angle, 1.0)
    kernel = cv2.warpAffine(kernel, rot_matrix, (kernel_size, kernel_size))

    return cv2.filter2D(img, -1, kernel)

def add_fog(image, intensity=0.5):
    """
    Simulate fog by overlaying a blurred white layer.

    Parameters:
        image (numpy.ndarray): Input image.
        intensity (float): Fog intensity (0 to 1).

    Returns:
        numpy.ndarray: Foggy image.
    """
    fog_layer = np.ones_like(image, dtype=np.uint8) * 255
    fog_layer = cv2.GaussianBlur(fog_layer, (21, 21), 0)
    return cv2.addWeighted(image, 1 - intensity, fog_layer, intensity, 0)

def add_rain(image, drop_length=20, drop_width=1, rain_density=0.05):
    """
    Simulate rain by drawing streaks on an overlay.

    Parameters:
        image (numpy.ndarray): Input image.
        drop_length (int): Length of raindrops.
        drop_width (int): Width of raindrops.
        rain_density (float): Density of rain (0 to 1).

    Returns:
        numpy.ndarray: Rainy image.
    """
    rain_layer = np.zeros_like(image, dtype=np.uint8)
    num_drops = int(rain_density * image.shape[0] * image.shape[1])

    for _ in range(num_drops):
        x = np.random.randint(0, image.shape[1])
        y = np.random.randint(0, image.shape[0])
        cv2.line(rain_layer, (x, y), (x, y + drop_length), (200, 200, 200), drop_width)

    rain_layer = cv2.GaussianBlur(rain_layer, (5, 5), 0)
    return cv2.addWeighted(image, 0.8, rain_layer, 0.2, 0)

def add_haze(image, intensity=0.3, brightness_factor=0.7):
    """
    Simulate haze by adding a brightness-reducing overlay.

    Parameters:
        image (numpy.ndarray): Input image.
        intensity (float): Haze intensity (0 to 1).
        brightness_factor (float): Brightness reduction factor.

    Returns:
        numpy.ndarray: Hazy image.
    """
    hazy_image = image.astype(np.float32) / 255.0
    fog_layer = np.ones_like(hazy_image, dtype=np.float32) * intensity
    hazy_image = cv2.addWeighted(hazy_image, brightness_factor, fog_layer, 1 - brightness_factor, 0)
    return (hazy_image * 255).astype(np.uint8)

def add_color_shift(image, hue_shift, sat_shift, val_shift):
    """
    Shift the color balance of an image.

    Parameters:
        image (numpy.ndarray): Input image.
        hue_shift (float): Shift in hue (degrees).
        sat_shift (float): Shift in saturation.
        val_shift (float): Shift in brightness.

    Returns:
        numpy.ndarray: Color-shifted image.
    """
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv_image[..., 0] = (hsv_image[..., 0] + hue_shift) % 180
    hsv_image[..., 1] = np.clip(hsv_image[..., 1] + sat_shift, 0, 255)
    hsv_image[..., 2] = np.clip(hsv_image[..., 2] + val_shift, 0, 255)
    return cv2.cvtColor(hsv_image.astype(np.uint8), cv2.COLOR_HSV2BGR)

def augment_image(img, transforms):
    """
    Apply multiple augmentations to an image.

    Parameters:
        img (numpy.ndarray): Input image.
        transforms (list): List of transformations to apply.

    Returns:
        tuple: (Applied transformations as a string, Augmented image)
    """
    augmented_img = img.copy()
    applied_transforms = []

    for t in transforms:
        if t == 'motion blur':
            kernel_size = np.random.randint(10, 19)
            angle = np.random.randint(0, 12)
            augmented_img = add_motion_blur(augmented_img, kernel_size, angle)
        elif t == 'fog':
            intensity = np.random.uniform(0.3, 0.7)
            augmented_img = add_fog(augmented_img, intensity)
        elif t == 'rain':
            drop_length = np.random.randint(10, 21)
            drop_width = np.random.choice([1, 2])
            density = np.random.uniform(0.01, 0.05)
            augmented_img = add_rain(augmented_img, drop_length, drop_width, density)
        elif t == 'haze':
            intensity = np.random.uniform(0.2, 0.6)
            brightness = np.random.uniform(0.5, 0.8)
            augmented_img = add_haze(augmented_img, intensity, brightness)
        elif t == 'color shift':
            hue_shift = np.random.uniform(-10, 10)
            sat_shift = np.random.uniform(-30, 30)
            val_shift = np.random.uniform(-30, 30)
            augmented_img = add_color_shift(augmented_img, hue_shift, sat_shift, val_shift)
        applied_transforms.append(t)

    return ', '.join(applied_transforms), augmented_img

def create_augmented_dataset(input_folder, output_folder, augmentation_plan):
    """
    Apply specified augmentations to a dataset and save the results.

    Args:
        input_folder (str): Path to input images.
        output_folder (str): Path to save augmented images.
        augmentation_plan (list of dict): List where each dict contains:
            - "transformations": List of transformations to apply
            - "num_images": Number of images to generate per input image
    """
    os.makedirs(output_folder, exist_ok=True)
    image_paths = glob(os.path.join(input_folder, "*.*"))

    for img_path in tqdm(image_paths, desc="Augmenting Images"):
        img = cv2.imread(img_path)
        if img is None:
            logging.error(f"Skipping invalid image: {img_path}")
            continue

        filename = os.path.basename(img_path).split('.')[0]

        for plan in augmentation_plan:
            transformations = plan["transformations"]
            num_images = plan["num_images"]

            for i in range(num_images):
                augmented_img, transform_name = augment_image(img, transformations)
                output_filename = f"{filename}_{transform_name}_{i+1}.jpg"
                cv2.imwrite(os.path.join(output_folder, output_filename), augmented_img)


if __name__ == "__main__":

    # Define dataset paths
    input_folder = "path/to/dataset"
    output_folder = "path/to/output_folder"

    # Define the augmentation plan 
    augmentation_plan = [
        {"transformations": ["motion blur"], "num_images": 2},  # Apply motion blur, generate 2 images
        {"transformations": ["fog"], "num_images": 3},  # Apply fog, generate 3 images
        {"transformations": ["rain"], "num_images": 1},  # Apply rain, generate 1 image
        {"transformations": ["motion blur", "fog"], "num_images": 2},  # Apply motion blur + fog, generate 2 images
        {"transformations": ["rain", "fog"], "num_images": 1},  # Apply rain + fog, generate 1 image
    ]

    # Preview sample images before augmentation
    image_paths = glob(os.path.join(input_folder, "*.*"))
    if not image_paths:
        logging.error("No images found in the input folder. Please check the path.")
    else:
        display_img(image_paths, num_samples=5)

    # Apply augmentations
    create_augmented_dataset(input_folder, output_folder, augmentation_plan)
