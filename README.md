# BreastDCEDL

A comprehensive deep learning-ready dataset of pretreatment 3D dynamic contrast-enhanced MRI (DCE-MRI) scans from **2,070 breast cancer patients**, combining data from three major clinical trials: **I-SPY2** (n=982), **I-SPY1** (n=172), and **Duke** (n=916).

## 📄 Publication

**BreastDCEDL: Curating a Comprehensive DCE-MRI Dataset and Developing a Transformer Implementation**  
[Read on arXiv](https://doi.org/10.48550/arXiv.2506.12190)

## 🔍 Dataset Versions

### MinCrop Version
- **3 tumor-centered scans per patient**: pre-contrast, early post-contrast, late post-contrast
- **Standardized size**: All scans cropped to 256×256 pixels around the main tumor
- **Fully available on Zenodo**: [Download MinCrop Dataset](https://zenodo.org/records/15627233)
- **Used for**: Training deep learning models with RGB fusion from 3 main time points for pCR and HER2 prediction in published research

### Full Version
- **Complete DCE-MRI sequences**: 3-12 time points per patient
- **Original resolution**: Preserves full field of view and spatial information
- **Availability**:
  - **I-SPY1 (Full)**: [Download from Zenodo](https://zenodo.org/records/15627233)
  - **I-SPY2 (Full)**: [Download from Zenodo](https://zenodo.org/records/15627233)
  - **Duke**: Download from [TCIA](https://www.cancerimagingarchive.net/) and convert using provided code

## 📁 Repository Contents

| File | Description |
|------|-------------|
| `BreastDCEDL_metadata.csv` | Metadata for the full size BreastDCEDL dataset |
| `BreastDCEDL_metadata_min_crop.csv` | Metadata for the MinCrop version |
| `BreastDCEDL_demo.ipynb` | Visualize data examples included in this repository |
| `BreastDCEDL_demo_on_local_data_min_crop.ipynb` | Explore and work with MinCrop data after downloading from Zenodo |
| `BreastDCEDL_vit_predict.ipynb` | Predict pCR using trained ViT model |

Directories: ISPY1, ISPY2 and DUKE contains dataset specific code for preprocessing metadata, dicom exploring and converting


## 🎯 Benchmark Tasks

Three standardized classification tasks with preserved train/validation/test splits:

| Task | Description | Distribution | Best Performance |
|------|-------------|--------------|-----------------|
| **pCR Prediction** | Pathological complete response to neoadjuvant therapy | 29.5% positive (n=428/1452) | AUC 0.94 (ViT, HR+/HER2−)¹ |
| **HER2 Status** | HER2 expression | 22.1% positive (n=458/2070) | AUC 0.74 (Dual-Attention ResNet)² |
| **HR Status** | Hormone receptor positivity | 64.2% positive (n=1327/2070) | - |

¹Results from [Fridman et al., 2025 - BreastDCEDL](https://doi.org/10.48550/arXiv.2506.12190)  
²Results from [Fridman & Goldstein, 2025 - Dual-Attention ResNet](https://arxiv.org/abs/2510.13897)

### Data Splits

| Split | pCR N | pCR+ | pCR− | HR N | HR+ | HR− | HER2 N | HER2+ | HER2− |
|-------|-------|------|------|------|-----|-----|--------|-------|-------|
| **Training** | 1099 | 322 | 777 | 1529 | 987 | 542 | 1528 | 345 | 1183 |
| **Validation** | 176 | 53 | 123 | 268 | 167 | 101 | 268 | 58 | 210 |
| **Test** | 177 | 53 | 124 | 271 | 173 | 98 | 269 | 56 | 213 |
| **Total** | 1452 | 428 | 1024 | 2068 | 1327 | 741 | 2065 | 459 | 1606 |

*N = number of patients with available labels for each biomarker

## 🏥 Dataset Details

### I-SPY2 (n=982)
- **Sequences**: 3-12 time points (typically 7)
- **Annotations**: Full 3D tumor segmentations at 3 selected time points
![Example from I-SPY1](https://github.com/naomifridman/BreastDCEDL/blob/main/images/spy2_example.png?raw=true)
### I-SPY1 (n=172)
- **Sequences**: 3-5 usable DCE scans
- **Annotations**: Full 3D tumor segmentations

![Example from I-SPY1](https://github.com/naomifridman/BreastDCEDL/blob/main/images/spy1_example.png?raw=true)

### Duke (n=916)
- **NAC subset**: 298 patients with pCR labels (only 31% received neoadjuvant chemotherapy)
- **Sequences**: 1 pre-contrast + 2-4 post-contrast scans
- **Annotations**: Bounding box of largest tumor (no full segmentation)

![Example from Duke](https://github.com/naomifridman/BreastDCEDL/blob/main/images/duke_example.png?raw=true)

## 🚀 Quick Start

### Option 1: Explore Sample Data (No Download Required)
Open `BreastDCEDL_demo.ipynb` to visualize example data included in this repository.

### Option 2: Work with Full MinCrop Dataset
1. Download the MinCrop dataset from [Zenodo](https://zenodo.org/records/15627233)
2. Open `BreastDCEDL_demo_on_local_data_min_crop.ipynb`
3. Follow the notebook to explore and analyze the data

## 🔬 Clinical Background

Dynamic Contrast-Enhanced MRI (DCE-MRI) is a key imaging technique for breast cancer evaluation. It captures tissue perfusion dynamics through sequential 3D scans before and after contrast agent administration. In breast cancer:
- **Enhancement patterns**: Malignant tumors typically show rapid initial enhancement followed by washout or plateau
- **Clinical protocol**: Radiologists analyze pre-contrast and multiple post-contrast phases (typically 3-12 time points)
- **Predictive value**: Enhancement dynamics correlate with treatment response and can predict pathological complete response (pCR) to neoadjuvant therapy

For detailed methodology, see [Fridman et al., 2025](https://doi.org/10.48550/arXiv.2506.12190)

## 📚 Citations

If you use the BreastDCEDL dataset or code in your research, please cite both the article and dataset:

### Article Citation (Required)
```bibtex
@article{fridman2025breastdcedl,
  title={BreastDCEDL: A Comprehensive Breast Cancer DCE-MRI Dataset and Transformer Implementation for Treatment Response Prediction},
  author={Fridman, Naomi and Solway, Bubby and Fridman, Tomer and Barnea, Itamar and Goldstein, Anat},
  journal={arXiv preprint arXiv:2506.12190},
  year={2025},
  doi={10.48550/arXiv.2506.12190}
}
```

### Dataset Citation (Required when using the data)
```bibtex
@dataset{fridman2025breastdcedl_dataset,
  author       = {Fridman, Naomi and others},
  title        = {BreastDCEDL: Curated DCE-MRI Dataset},
  year         = {2025},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.15627233}
}
```

**Note**: Please include both citations in any publication that uses the BreastDCEDL dataset or associated code.

## 🔗 Resources

- **Original Data Source**: [The Cancer Imaging Archive (TCIA)](https://www.cancerimagingarchive.net/)
- **I-SPY2 Trial**: ClinicalTrials.gov NCT01042379
- **Zenodo Repository**: [https://zenodo.org/records/15627233](https://zenodo.org/records/15627233)

## 📝 License

Please refer to the original data sources for licensing information.
