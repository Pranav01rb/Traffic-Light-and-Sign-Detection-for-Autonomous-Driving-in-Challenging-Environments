# Traffic-Light-and-Sign-Detection-for-Autonomous-Driving-in-Challenging-Environments

## Overview
This project focuses on detecting and classifying traffic lights and signs in real-world environments, particularly under challenging conditions such as motion blur, fog, rain, haze, and color shifts. The project leverages advanced data augmentation techniques to improve the robustness of detection models and includes tools for creating augmented datasets.

## Key Features
#### Data Augmentation:
Apply transformations like motion blur, fog, rain, haze, and color shifts to simulate adverse conditions.

#### Augmented Dataset Creation: 
Automatically generate augmented datasets from raw images using a customizable augmentation pipeline.

#### Modular Codebase: 
Includes reusable functions for image processing and dataset generation.

#### Support for Multiple Transformations: 
Combine multiple augmentations (e.g., motion blur + fog) to create diverse datasets.

## Augmentation Techniques
The following transformations are supported:
#### Motion Blur: 
Simulates motion blur with customizable kernel size and angle.
#### Fog: 
Adds fog-like effects with adjustable intensity.
#### Rain: 
Simulates raindrops with adjustable drop length, width, and density.
#### Haze: 
Adds haze effects with adjustable intensity and brightness.
#### Color Shift: 
Shifts hue, saturation, and brightness values randomly.
