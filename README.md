# Traffic-Light-and-Sign-Detection-for-Autonomous-Driving-in-Challenging-Environments

## Overview
This project focuses on detecting and classifying traffic lights and signs in real-world environments, particularly under challenging conditions such as motion blur, fog, rain, haze, and color shifts. The project leverages advanced data augmentation techniques to improve the robustness of detection models and includes tools for creating augmented datasets.

## Key Features
1. Data Augmentation: Apply transformations like motion blur, fog, rain, haze, and color shifts to simulate adverse conditions.
2. Augmented Dataset Creation: Automatically generate augmented datasets from raw images using a customizable augmentation pipeline.
3. Modular Codebase: Includes reusable functions for image processing and dataset generation.
4. Support for Multiple Transformations: Combine multiple augmentations (e.g., motion blur + fog) to create diverse datasets.

## Augmentation Techniques
The following transformations are supported:
1. Motion Blur: Simulates motion blur with customizable kernel size and angle.
2. Fog: Adds fog-like effects with adjustable intensity.
3. Rain: Simulates raindrops with adjustable drop length, width, and density.
4. Haze: Adds haze effects with adjustable intensity and brightness.
5. Color Shift: Shifts hue, saturation, and brightness values randomly.

## Dataset
The project uses the LISA Traffic Sign Dataset, sourced from Roboflow. This dataset provides high-quality annotations of traffic signs under various conditions. It is further augmented with synthetic data to simulate rare and challenging scenarios such as motion blur caused by vehicle speed, adverse weather conditions like fog, rain, and haze, lighting variations using color shifts.
