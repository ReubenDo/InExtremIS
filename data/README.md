# Downloading the data used for the experiments

In this work, we used a large (N=242) dataset for Vestibular Schwannoma segmentation. This dataset is publicly available on 
[TCIA](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70229053). 

This readme explains how to download and pre-process the raw data from TCIA. We also provide an open access to the extreme points and pre-computed geodesics used in this work.

## Download the fully annotated TCIA-VS dataset

### Option 1 - Downloading the T2 scans only and their segmentation maps (Recommended):

Please follow the following steps:

**Step 1**: Download the NBIA Data Retriever: 
* Please follow the instructions [here](https://wiki.cancerimagingarchive.net/display/NBIA/Downloading+TCIA+Images).

**Step 2**: Download the T2 scans only:
* Open `manifest-T2.tcia` with NBIA Data Retriever and download the T2 images (DICOM, 6GB) with the "Descriptive Directory Name" format.

**Step 3**: DICOM to Nifti conversion:
* Install dependencies: `pip install -r preprocess_requirements.txt`
* Execute the conversion script: 
`python3 convert.py --input <input_folder> --output <output_folder>`
  * `<input_folder>` is the directory containing the raw T2 images (e.g. `/home/admin/manifest-T2/Vestibular-Schwannoma-SEG/`).
  * `<output_folder>` is the directory in which the pre-processed data will be saved.

**Step 4**:  Download the fully annotated segmentation masks [here](https://zenodo.org/record/5081986/files/full_annotations.zip?download=1).

### Option 2 - Downloading the full dataset and manually convert contours into segmentation masks:
Please follow the instructions from the [VS_Seg repository](https://github.com/KCL-BMEIS/VS_Seg/tree/master/preprocessing).

## Download the extreme points and pre-computed geodesics
The manual and simulated extreme points can be found [here](https://zenodo.org/record/5081986/files/extreme_points.zip?download=1). 
The pre-computed geodesics using the image gradient information (`grad` folder) and with the additional Euclidean distance (`grad_eucl` folder) can be found [here](https://zenodo.org/record/5081986/files/precomputed_geodesics.zip?download=1).

## Citations
If you use this VS data, please cite:

Shapey, J., Wang, G., Dorent, R., Dimitriadis, A., Li, W., Paddick, I., Kitchen, N., Bisdas, S., Saeed, S. R., Ourselin, S., Bradford, R., & Vercauteren, T. (2021). An artificial intelligence framework for automatic segmentation and volumetry of vestibular schwannomas from contrast-enhanced T1-weighted and high-resolution T2-weighted MRI. Journal of Neurosurgery, 134(1), 171–179. https://doi.org/10.3171/2019.9.jns191949

If you use the extreme points, please additionally cite:

```
@article{InExtremIS2021Dorent,
         author={Dorent, Reuben and Joutard, Samuel and Shapey, Jonathan and
         Kujawa, Aaron and Modat, Marc and Ourselin, S\'ebastien and Vercauteren, Tom},
         title={Inter Extreme Points Geodesics for End-to-End Weakly Supervised Image Segmentation},
         journal={MICCAI},
         year={2021},
}
```
## Credits
The conversion script is based on https://github.com/KCL-BMEIS/VS_Seg/tree/master/preprocessing.
