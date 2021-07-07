# Downloading the data used for the experiments

In this work, we used a large (N=242) dataset for Vestibular Schwannoma segmentation. This dataset is publicly available on 
[TCIA](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70229053). 

This readme explains how to download and pre-process the raw data from TCIA. We also provide an open access to the extreme points and pre-computed geodesics used in this work.

Our code is based on the [VS_Seg](https://github.com/KCL-BMEIS/VS_Seg) repository.

## Download the fully annotated TCIA-VS dataset

To download the dataset, please follow the following steps:

**Step 1**: Download the NBIA Data Retriever [here](https://wiki.cancerimagingarchive.net/display/NBIA/Downloading+TCIA+Images).

**Step 2**: Download the imaging data:
* *Option 1*: Download the full TCIA dataset [here](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70229053):
  * Images and Radiation Therapy Structures (DICOM, 26 GB) in "Descriptive Directory Name" format.
* *Option 2*: Download the T2 scans only:
  * Open `manifest-T2.tcia` with NBIA Data Retriever and download the T2 images (DICOM, 6GB) "Descriptive Directory Name" format.

**Step 3**:  Download the contours (JSON, zip, 16 MB) [here](https://wiki.cancerimagingarchive.net/download/attachments/70229053/Vestibular-Schwannoma-SEG%20contours%20Mar%202021.zip?api=v2).

**Step 4**: Convert the images and contours in the Nifti format:
  * Install dependencies: `pip install -r requirements.txt`
  * Execute the conversion script: 
  `python3 convert.py --input <input_folder> --output <output_folder>`
    * `<input_folder>` is the path to the `Vestibular-Schwannoma-SEG` directory
    * `<output_folder>` is the directory in which the pre-processed data will be saved

## Download the extreme points and pre-computed geodesics
The manual and simulated extreme points can be found here (not available yet). 
The pre-computed geodesics using the gradient information (`grad` folder) and with the additional Euclidean distance (`grad_eucl`) can be found here (not available yet)

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
