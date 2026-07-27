# Materials and Methods Template for the Cardiac Segmentation Benchmark

This template is aligned with the pipeline currently implemented in the repository as of July 1, 2026. It can be adapted for submission to *Radiology: Artificial Intelligence* after the final experimental runs are complete and the reported metrics are replaced with the actual results.

## 1. Study Design

We conducted a retrospective benchmark study to evaluate whether a structured image-preprocessing pipeline improves downstream cardiac image segmentation across public cardiovascular imaging datasets. The experimental framework was designed to isolate the effect of preprocessing from model architecture by training the same 2D U-Net baseline under multiple preprocessing conditions while preserving patient-level separation between development and evaluation cohorts. All experiments were executed from a manifest-driven pipeline to reduce manual intervention, improve traceability, and prevent data leakage across train, validation, and test partitions.

## 2. Datasets

Two publicly available datasets were used in the present implementation:

1. **Automated Cardiac Diagnosis Challenge (ACDC)**: short-axis cine cardiac magnetic resonance (CMR) data with expert annotations for end-diastolic and end-systolic frames. In the implemented local manifest, positive-mask slices were extracted from volumetric NIfTI files and converted into slice-level training instances while preserving patient identifiers and frame provenance.
2. **CAMUS**: transthoracic echocardiography data with 2-chamber and 4-chamber annotations at end-diastole and end-systole. The implemented local manifest uses the NIfTI-exported public release and retains the official dataset split metadata when available.

Images and masks were indexed through CSV manifests containing, at minimum, `patient_id`, `phase`, `image_path`, and `mask_path`. Additional metadata fields included `dataset`, `subset`, `view`, `frame_id`, `slice_index`, and `source_patient_id`, enabling controlled subgroup analyses and patient-level split verification. Public data were stored outside the repository root, whereas only the manifest files and experiment metadata were maintained under version control.

## 3. Manifest Construction and Data Governance

The preprocessing and training framework was built around a manifest-first design. For ACDC, slice-level rows were generated from 3D NIfTI volumes by loading the original cine frame and its corresponding ground-truth mask, enumerating the slice index, and retaining only slices with nonzero foreground content. For CAMUS, each annotated end-diastolic and end-systolic frame was represented as a single manifest row. Each patient identifier was made globally unique at the combined-corpus level (e.g., `ACDC_patient001`, `CAMUS_patient0001`) to eliminate cross-dataset identifier collisions.

Patient-level leakage was prevented programmatically. When a dataset provided explicit train/validation/test metadata, these splits were respected. When a full validation topology was not available, patient-level random splitting was used with stratification over available metadata fields such as dataset, clinical group, and view. A validation routine explicitly checked for overlap between train, validation, and test patient identifiers before model training was allowed to proceed.

## 4. Image Loading and Representation

The current implementation supports PNG, NumPy, and NIfTI inputs. NIfTI volumes were loaded using `nibabel`, and slice selection was performed through the stored `slice_index` field. All images were converted into single-channel floating-point arrays and normalized to the \([0,1]\) range before any optional preprocessing operation. Binary training targets were derived from the provided masks by collapsing all nonzero label values into a foreground class, thereby converting the problem into foreground-versus-background segmentation.

In the current ACDC extension, the same loader also supports multiclass label preservation. Rather than collapsing all nonzero voxels into one foreground mask, the loader can retain the original ACDC label map \(\{0,1,2,3\}\), corresponding to background, right ventricular cavity, myocardium, and left ventricular cavity. In that branch, label maps are resized with nearest-neighbor interpolation, invalid labels are reassigned to background, and one-hot training targets are generated on the fly for multiclass optimization.

## 5. Preprocessing Pipeline

All preprocessing modes were implemented through a common function-level interface so that only one factor varied between experiments: the selected image-filtering strategy. The current repository supports six modes:

1. `none`
2. `gaussian`
3. `wavelet`
4. `nlm`
5. `clahe`
6. `hybrid`

The `gaussian` mode applies Gaussian smoothing. The `wavelet` mode performs wavelet-domain denoising with soft thresholding. The `nlm` mode applies nonlocal means denoising using a noise-adaptive filtering strength. The `clahe` mode performs contrast-limited adaptive histogram equalization. The `hybrid` mode applies wavelet denoising, followed by nonlocal means filtering, followed by CLAHE.

For model training, the preprocessed image and its corresponding binary mask were resized to \(256 \times 256\) pixels. Images were resized with bilinear interpolation, whereas masks were resized with nearest-neighbor interpolation to preserve discrete label structure.

## 6. Data Augmentation

To reduce overfitting and improve model robustness, paired augmentations were applied to the training set only. The current implementation includes horizontal flipping, vertical flipping, random rotations in multiples of 90 degrees, low-amplitude intensity gain/bias perturbation, and additive Gaussian noise. Augmentations were applied jointly to the image-mask pair to preserve spatial consistency. Validation and test data were processed deterministically without random augmentation.

## 7. Segmentation Model

The implemented baseline model is a compact 2D U-Net with a four-level encoder-decoder topology and skip connections between corresponding resolution stages. Each block consists of two convolutional layers followed by batch normalization and ReLU activation. Downsampling is performed through max pooling, whereas upsampling is performed through transposed convolution. The default configuration uses a single input channel, a single output channel, and 32 base feature channels in the first layer. The final layer produces a logits map, which is transformed by a sigmoid function during evaluation.

This baseline was intentionally selected as a controlled and interpretable architectural reference rather than as a claim of state-of-the-art performance. The scientific objective of the benchmark is to quantify the effect of preprocessing under a stable model backbone.

For the ACDC multiclass extension, the same architectural backbone is reused with four output channels instead of one. This design preserves the controlled-benchmark logic: only the supervision target changes, whereas the encoder-decoder topology, optimization stack, augmentation policy, and split governance remain fixed.

## 8. Training Configuration

Training was performed in Python 3.12 with PyTorch. The current local environment was configured with CUDA-enabled PyTorch and executed on an NVIDIA GeForce RTX 5070 GPU when available; otherwise, the same code path falls back to CPU execution. Optimization was performed with AdamW using a default learning rate of \(1 \times 10^{-3}\), weight decay of \(1 \times 10^{-4}\), and a cosine annealing learning-rate schedule. The default batch size was 8, and the baseline configuration used 10 training epochs for each preprocessing mode.

The loss function combined binary cross-entropy with a soft Dice loss in equal proportion:

\[
\mathcal{L}_{\text{seg}} = 0.5\,\mathcal{L}_{\text{BCE}} + 0.5\,\mathcal{L}_{\text{Dice}}.
\]

Soft Dice loss was computed from sigmoid probabilities as:

\[
\mathcal{L}_{\text{Dice}} = 1 - \frac{2 \sum_i p_i y_i + \varepsilon}{\sum_i p_i + \sum_i y_i + \varepsilon},
\]

where \(p_i\) denotes the predicted foreground probability at pixel \(i\), \(y_i\) denotes the corresponding binary ground-truth label, and \(\varepsilon\) is a numerical stabilization constant.

Model checkpoints were selected according to the lowest validation loss. Training history, resolved split files, best checkpoints, patient-level summaries, and test-set predictions were stored under the experiment output directory for full auditability.

For the multiclass ACDC extension, the loss function was modified to combine categorical cross-entropy with a multiclass soft Dice term computed from softmax probabilities:

\[
\mathcal{L}_{\text{multi}} = 0.5\,\mathcal{L}_{\text{CE}} + 0.5\,\mathcal{L}_{\text{Dice-multi}}.
\]

The multiclass Dice component was computed after excluding the background channel from the overlap term:

\[
\mathcal{L}_{\text{Dice-multi}} =
1 -
\frac{1}{C'}
\sum_{c=1}^{C'}
\frac{2 \sum_i p_{ic} y_{ic} + \varepsilon}
{\sum_i p_{ic} + \sum_i y_{ic} + \varepsilon},
\]

where \(p_{ic}\) denotes the softmax probability for class \(c\) at pixel \(i\), \(y_{ic}\) denotes the corresponding one-hot target, and \(C'\) is the number of foreground classes included in the Dice term. This formulation was introduced to support the reviewer-driven extension from binary foreground segmentation to class-resolved anatomical segmentation while preserving comparability with the original training protocol.

## 9. Evaluation Protocol

Performance was evaluated at both the slice level and the patient level. Slice-level metrics were first computed for each test instance and then aggregated at the patient level by averaging all numeric metrics for a given patient and preprocessing mode. The currently implemented evaluation metrics include Dice similarity coefficient, intersection over union (IoU), and the 95th-percentile Hausdorff distance (HD95). In addition, inference latency per image was measured during forward passes on the evaluation set, and the total number of trainable parameters was computed directly from the network.

The principal segmentation metrics were defined as follows:

\[
\mathrm{Dice} = \frac{2|A \cap B|}{|A| + |B|},
\]

\[
\mathrm{IoU} = \frac{|A \cap B|}{|A \cup B|},
\]

\[
\mathrm{HD95}(A,B) = \mathrm{percentile}_{95}\big(d(A,B)\big),
\]

where \(A\) and \(B\) denote the foreground pixel sets of the reference and predicted masks, respectively, and \(d(A,B)\) denotes the bidirectional set of surface-to-surface distances.

Inference efficiency was summarized using the mean per-image inference time (milliseconds per image) and the number of trainable model parameters. These quantities were included because the intended target journal emphasizes not only segmentation accuracy but also implementation realism and deployability.

For the multiclass ACDC extension, the evaluation module exports per-class metrics (`class_1_*`, `class_2_*`, `class_3_*`) together with macro-averaged Dice, IoU, and HD95 across foreground classes. Slice-level outputs are written to `test_slice_level.csv`, patient-level aggregates are written to `test_patient_level.csv`, and full run metadata are stored in `summary.json`. This output contract is intentionally manuscript-friendly and can be mapped directly into class-wise result tables for the revised study.

## 10. Reproducibility

All experiments were executed from command-line entry points linked to version-controlled code. Random seeds were fixed across Python, NumPy, and PyTorch. The repository stores the following artifacts for each run: resolved split manifests, training history, checkpoint files, runtime logs, slice-level outputs, patient-level outputs, and a JSON summary of the full experiment configuration. This design was chosen to support manuscript-level reproducibility and later extension toward additional architectures, external validation experiments, and statistical comparison across preprocessing modes.

The currently implemented command-line entry points are:

```bash
python -m cardiac_image_system.experiments.train_unet_baseline --manifest data/manifests_local/segmentation_public_combined.csv --output-dir outputs/unet_baseline_combined_none --mode none
```

for the binary benchmark, and

```bash
python -m cardiac_image_system.experiments.train_unet_multiclass --manifest data/manifests_local/segmentation_public_combined.csv --output-dir outputs/unet_multiclass_acdc_none --mode none --dataset-filter ACDC
```

for the class-resolved ACDC extension. A convenience PowerShell wrapper, `scripts/run_unet_multiclass_acdc_top3.ps1`, is provided for the reviewer-prioritized `none`, `wavelet`, and `nlm` comparison set.

## 11. Suggested Manuscript Wording for the Final Study

The final manuscript should explicitly state that the current benchmark evaluates the effect of preprocessing on downstream segmentation rather than claiming universal clinical superiority of any single preprocessing strategy. If the study is later expanded to additional architectures, this section can be revised from a single-backbone benchmark to a multi-architecture validation study while retaining the same manifest-driven experimental framework.
