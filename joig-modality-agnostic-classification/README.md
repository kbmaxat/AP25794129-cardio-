# JOIG modality-agnostic cardiac classification

Reproducibility implementation for the manuscript's binary cardiac image classification experiments across echocardiography and cardiac MRI. The default experiment follows the reported VGG16 transfer-learning protocol; ResNet50 and MobileNetV2 are included as comparison backbones.

## Implemented protocol

- ImageNet-pretrained VGG16 with global average pooling, a 256-unit ReLU layer, dropout 0.5, and one sigmoid logit.
- Modality-aware preprocessing: 3×3 median filtering for echocardiography and mild Gaussian smoothing for MRI.
- Resize to 224×224, grayscale-to-RGB conversion, ImageNet normalization, and training-only augmentation.
- Patient-grouped, stratified five-fold cross-validation to prevent subject leakage.
- Frozen-backbone training followed by fine-tuning of the last two VGG blocks.
- Accuracy, precision, recall, specificity, F1, ROC-AUC, confusion-matrix counts, fold predictions, and checkpoints.
- Frozen-checkpoint evaluation on external ACDC/CAMUS manifests and Grad-CAM utilities.

The repository does not contain clinical images. Public datasets must be obtained under their original terms. The local/internal cohort can be added later as manifest rows.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

## Data manifest

See `data/README.md` and `data/example_manifest.csv`. Paths are resolved relative to the manifest file. A unique `patient_id` is mandatory because the split is patient-level.

```bash
joig-cardio validate-manifest data/example_manifest.csv --skip-file-check
```

## Training and external evaluation

```bash
joig-cardio cross-validate data/internal_manifest.csv --config configs/default.json --output outputs/vgg16
joig-cardio evaluate data/acdc_manifest.csv outputs/vgg16/fold_0.pt --output outputs/acdc_fold0
joig-cardio evaluate data/camus_manifest.csv outputs/vgg16/fold_0.pt --output outputs/camus_fold0
```

Run each fold checkpoint on each external set and report the mean and standard deviation across folds. Do not fine-tune on ACDC or CAMUS when reproducing the manuscript's external-validation protocol.

## Reproducibility note

The code implements the method described in the article, but the article's numerical results cannot be regenerated until the exact internal training cohort and the downloaded public-dataset manifests are supplied. Dataset files, checkpoints, and outputs are excluded from Git.
