# Find Contour using NVIDIA NPP

## Overview

This sample demonstrates how to detect and extract contours from an image using NVIDIA’s [NPP library](https://docs.nvidia.com/cuda/npp/index.html). It performs unified labeling, label compression, and geometric contour reconstruction on grayscale input images.

## Key Concepts

- Unified Framework (UF) Labeling
- Connected Component Analysis
- Image Contour Extraction using CUDA/NPP

## Requirements

- **Operating System:** Linux or Windows
- **CPU Architecture:** x86_64
- **GPU Support:** [CUDA-enabled GPUs (SM 7.0, 7.2, 7.5, 8.0, and above)](https://developer.nvidia.com/cuda-gpus)
- **Toolkit:** [CUDA Toolkit 11.4 or later](https://developer.nvidia.com/cuda-downloads)
- **Libraries:** NPP, CMake, and CUDA Runtime

---

## Build Instructions

### Linux

```bash
mkdir build
cd build
cmake ..
make
```

### Windows

```cmd
mkdir build
cd build
cmake -DCMAKE_GENERATOR_PLATFORM=x64 ..
```
Then open `findContour.sln` in Visual Studio (2017 or later) and build the solution.

---

## Usage

```bash
./findContour
```

**Example Output:**

```bash

./build/findContour 
Done. Compressed Labels: 274

```

---

## Input & Output

### Input Image

- `CircuitBoard_2048x1024_8u.raw`
- Format: 8-bit unsigned grayscale
- Dimensions: 2048 x 1024

![Input Image](/NPP/findContour/CircuitBoard_2048x1024_8u.jpg)

---

### Output Visualizations

#### 1. Label Markers (UF-based)

- **File:** `CircuitBoard_LabelMarkersUF_8Way_2048x1024_32u.raw`
- **Generated by:** `nppiLabelMarkersUF_8u32u_C1R_Ctx`

![Label Markers](/NPP/findContour/CircuitBoard_LabelMarkersUF_8Way_2048x1024_32u.jpg)

---

#### 2. Compressed Labels

- **File:** `CircuitBoard_CompressedMarkerLabelsUF_8Way_2048x1024_32u.raw`
- **Generated by:** `nppiCompressMarkerLabelsUF_32u_C1IR_Ctx`

![Compressed Labels](/NPP/findContour/CircuitBoard_CompressedMarkerLabelsUF_8Way_2048x1024_32u.jpg)

---

#### 3. Contour Pixels

- **File:** `CircuitBoard_Contours_8Way_2048x1024_8u.raw`
- **Generated by:** `nppiCompressedMarkerLabelsUFInfo_32u_C1R_Ctx`

![Contours](/NPP/findContour/CircuitBoard_Contours_8Way_2048x1024_8u.jpg)

---

#### 4. Reconstructed Contours (Ordered Geometry)

- **File:** `CircuitBoard_ContoursReconstructed_8Way_2048x1024_8u.raw`
- **Generated by:** `nppiCompressedMarkerLabelsUFContoursOutputGeometryLists_C1R`

![Reconstructed Contours](/NPP/findContour/CircuitBoard_ContoursReconstructed_8Way_2048x1024_8u.jpg)

