# GoudaCell

GoudaCell is a comprehensive toolkit for cell image analysis, focusing on accurate segmentation and labeling of different cell types. This project aims to streamline the workflow from initial image processing to final cell type annotation.

## Features

- Image segmentation using Napari with multiple engines:
  - Cellpose for traditional cell segmentation
  - StarDist for star-convex object detection
  - MicroSAM (Segment Anything for Microscopy) for interactive and automatic segmentation
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
- Select napari-aicsimageio when prompted for reader choice *if you are not using .tiff format*
- Use max projection for better visualization
- Convert to 2D view when needed

### 3. Choose Segmentation Method

#### Option A: Cellpose Segmentation
For multichannel images using Cellpose, you'll need to merge them to RGB before segmentation:
1. Load your multichannel image - this creates separate layers in Napari
2. Set each channel to the appropriate color (red, green, or blue) using the colormap settings
3. Select all layers (typically 3)
4. Right-click and select "Merge to RGB" to create a single RGB image
   - This step is crucial as cellpose-napari cannot directly segment stacked/multichannel images
5. Now you can specify:
   - Cytoplasmic background channel as main channel for segmentation
   - Nuclear channel as a secondary nuclear channel for segmentation
   
> **Note**: Direct segmentation of stacked/multichannel images is not supported in cellpose-napari. Always merge to RGB first.

#### Option B: StarDist Segmentation
StarDist is particularly effective for nucleus segmentation and densely packed objects:
1. Access the StarDist plugin: Plugins -> StarDist
2. Configure settings:
   - Select appropriate model (e.g., "2D_versatile_fluo" for fluorescent nuclei)
   - Adjust probability threshold and NMS threshold as needed
3. Run segmentation
4. Review results

#### Option C: MicroSAM Segmentation
MicroSAM provides interactive and automatic segmentation options:
1. Access MicroSAM: Plugins -> Segment Anything for Microscopy -> Annotator 2d
2. Select image and model from the Embedding menu
   - For cell segmentation, try the "Light Microscopy" model
3. Use point prompts (positive/negative) or box prompts for interactive segmentation
4. Click "Segment Object" (or press S) to generate the segmentation
5. For automatic segmentation, click "Automatic Segmentation"
6. Review results

### 4. Generate Cell Segmentation
Follow the steps for your chosen segmentation method (Cellpose, StarDist, or MicroSAM)

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

## Working with Pre-made Masks

If you already have segmentation masks from another source or want to refine/label existing masks:

### 1. Loading Pre-made Masks
```bash
conda activate napari
napari
```
- Drag and drop your mask file (typically a .tif file with integer labels)
- If also analyzing original images, load them in the same Napari session
- Ensure your mask is properly aligned with the original image if both are loaded

### 2. Refining Pre-made Masks
1. Select the mask layer in the layer list
2. Use Napari's label editing tools:
   - Paint tool (press 'P') to extend or add labels
   - Erase tool (press 'E') to remove incorrect segments
   - Fill tool (press 'F') to fill holes

3. **Merging incorrectly split cells**:
   - Select the color picker/dropper tool (press 'D')
   - Click on one of the cells you wish to merge to select its label/color
   - Switch to the fill tool (press 'F')
   - Click on the adjacent cell that should be merged with the first cell
   - This will replace the second cell's label with the first cell's label, effectively merging them

4. Adjust label colors as needed for better visualization:
   - Right-click on the mask layer → "Color Cycler"
   - Choose a color scheme that makes individual cells clearly visible

### 3. Converting Segmentation to Cell Type Labels
1. Duplicate the refined mask layer (right-click → "Duplicate")
2. Rename the new layer to indicate it contains cell type labels
3. Use the Annotator plugin (Plugins → Annotator)
4. Create a new label scheme:
   - Set specific label numbers for different cell types (e.g., 1 for type A, 2 for type B)
   - Document your labeling scheme in description.txt
5. Use the Fill tool to assign cell type labels:
   - Select the label number for the cell type
   - Click on cells to fill them with the appropriate label

### 4. Save Refined Masks and Labels
- Save your refined mask: Right-click on mask layer → "Save Layer(s)"
  - Use naming convention: `[filename]_refined_mask.tif`
- Save your cell type labels: Right-click on labels layer → "Save Layer(s)"
  - Use naming convention: `[filename]_labels.tif`
- Update description.txt with information about your refinements and labeling scheme

## File Organization

```
goudacell/
├── data/                   # Create this for saving data
│   ├── description.txt     # Description of dataset (especially which label signifies which cell type)
│   ├── raw/                # Original microscopy files
│   ├── processed/          # Processed images and masks
│   ├── external_masks/     # Pre-made masks from other sources
│   └── labels/             # Manual annotations
└── environment.yml         # Conda environment specification
```
