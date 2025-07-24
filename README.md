# 3D Reconstruction

## Introduction

This project aims to facilitate 3D object modeling, pose estimation, movable part segmentation, and URDF generation using Intel RealSense cameras and various algorithms and tools such as RODIN, SAM3D, and REART. It provides a comprehensive pipeline for robot manipulation and simulation.

## Features

- **Data Acquisition**: Synchronized depth and color image capture using Intel RealSense cameras.
- **3D Modeling**: 3D mesh generation with RODIN.
- **Pose Estimation and Adjustment**: Pose estimation and optimization using OOPS.
- **Movable Part Segmentation**: Segmentation of movable parts and joint estimation using SAM3D and REART.
- **URDF Generation**: URDF file generation for robotics simulation.

## Environment Dependencies

- Python 3.8+
- pyrealsense2
- numpy
- opencv-python, opencv-contrib-python
- open3d
- trimesh
- sapien
- networkx
- matplotlib
- pygltf
- jsonpickle
- python-dotenv

To install the dependencies, run:

```
pip install -r requirements.txt
```

## Project Structure

- `capture_images.py`: Data acquisition from Intel RealSense cameras.
- `constants.py`: Project configuration, including paths and parameters.
- `main.py`: Main entry point coordinating the project workflow.
- `record_affordance.py`: Recording 3D coordinates of interaction points.
- `run_oops.py`: Running OOPS for pose estimation and optimization.
- `run_REART.py`: Running REART for movable part segmentation and joint estimation.
- `run_rodin.py`: Generating 3D mesh models using RODIN API.
- `run_SAM3D.py`: Running SAM3D for part segmentation.
- `util.py`: Utility functions for pose estimation and point cloud generation.

## Quick Start

1. **Install Dependencies**:

    ```
    pip install -r requirements.txt
    ```

2. **Data Acquisition**:

    ```
    python capture_images.py
    ```

    Press `r` to start/stop recording and `s` to save frames.

3. **Run the Main Pipeline**:

    ```
    python main.py
    ```

## Usage Examples

- **Record Interaction Points**:

    ```
    python record_affordance.py --dataset_path /path/to/dataset
    ```

    Follow the instructions to click interaction points on the image, and the program will compute and save their 3D coordinates.

- **Run OOPS**:

    ```
    python run_oops.py --dataset_path /path/to/dataset
    ```

    This script invokes OOPS for depth rendering, pose estimation, and optimization.

## Notes

- Ensure the Blender executable path is correctly configured in `constants.py` for SAM3D rendering.
- For RODIN usage, apply for an API key and configure it in `secret_config.py`.
- Some algorithms like REART may require significant computational resources. It is recommended to run them on a machine with a GPU.

## Contribution Guidelines

Contributions and suggestions for improvement are welcome! Please fork the repository, create a new branch, and submit a Pull Request.

## License

This project is licensed under the Apache 2.0 License. See the LICENSE file for details.
