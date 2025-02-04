# GoudaCell

GoudaCell is a comprehensive toolkit for cell image analysis, focusing on accurate segmentation and labeling of different cell types. This project aims to streamline the workflow from initial image processing to final cell type annotation.

## Features

- Image segmentation using Napari and Cellpose
- Interactive cell type labeling
- Standardized file organization and naming conventions

## Installation

GoudaCell uses conda for environment management. **We strongly recommend that this is done locally to avoid GUI issues.** To get started building a conda environment for running Napari:

```bash
# Clone the repository
git clone https://github.com/cheeseman-lab/goudacell.git
cd goudacell

# Create and activate the conda environment
conda env create -f environment.yml
conda activate napari
```

## Image Labeling Workflow

### 1. Launch Napari
```bash
conda activate napari
napari
```

### 2. Load and Prepare Images
- Drag and drop your microscopy files (.dv, .nd2, .tiff format supported)
- Select napari-aicsimageio when prompted for reader choice
- Use max projection for better visualization
- Convert to 2D view when needed

### 3. Merge Channels to RGB
For multichannel images, you'll need to merge them to RGB before segmentation:
1. Load your multichannel image - this creates separate layers in Napari
2. Set each channel to the appropriate color (red, green, or blue) using the colormap settings
3. Select all layers (typically 3)
4. Right-click and select "Merge to RGB" to create a single RGB image
   - This step is crucial as cellpose-napari cannot directly segment stacked/multichannel images
5. Now you can specify:
   - Cytoplasmic background channel as main channel for segmentation
   - Nuclear channel as a secondary nuclear channel for segmentation
   
> **Note**: Direct segmentation of stacked/multichannel images is not supported in cellpose-napari. Always merge to RGB first.

### 4. Generate Cell Segmentation
1. Access Cellpose plugin: Plugins -> cellpose
2. Configure settings:
   - Model: cyto3
   - Channel: Your cytoplasmic background
   - Optional nuclear channel: Your nuclear channel
   - Diameter: ~150 pixels (adjust based on your cells, you can also use estimate function)
   - You can manipulate cellprob and flow threshold as needed
3. Run segmentation
4. Review results

### 5. Refine Segmentation (if needed)
1. Use the paint tool (press 'P') to add missing parts of cells (select correct cell color for each refine you do)
2. Use the erase tool (press 'E') to remove incorrect segments
3. The fill tool can be used to merge cells that were incorrectly split
4. Take your time here - good segmentation is crucial for accurate labeling!

### 6. Label Cell Types
1. Duplicate the segmentation layer
2. Access Annotator plugin: Plugins -> Annotator
3. Split RGB images if useful
4. Label different cell types using the Fill tool:
   - Press '4' or click fill (bucket) icon
   - Change the label colors to make them more distinguishable for labeling
   - Set Label numbers for different cell types (overwriting the original cell-by-cell labels)
   - Click to fill cells with appropriate labels

### 7. Save Your Work
- Save files using these conventions:
    - `[filename]_mask.tif`: Original segmentation
    - `[filename]_labels.tif`: Cell type labels
    - `[filename]_max_proj_[channel].tif`: Raw projected images
- Make a description.txt file that documents your work (see template)

## File Organization

```
goudacell/
├── data/                   # Create this for saving data
│   ├── description.txt     # Description of dataset (especially which label signifies which cell type)
│   ├── raw/                # Original microscopy files
│   ├── processed/          # Processed images and masks
│   └── labels/             # Manual annotations
└── environment.yml         # Conda environment specification
```

## Coming Soon

- Improved use of Napari (without funky merge to RGB)
- Use of other segmentation models