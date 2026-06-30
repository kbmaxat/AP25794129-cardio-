# Architecture

## Goal

Implement a reproducible research prototype for preprocessing cardiovascular biomedical images.

## Components

1. `cardiac_image_system.core.io`
   - image loading;
   - image saving;
   - float normalization.

2. `cardiac_image_system.core.preprocessing`
   - none;
   - Gaussian;
   - Wavelet;
   - NLM;
   - CLAHE;
   - Hybrid.

3. `cardiac_image_system.core.segmentation`
   - Otsu proxy segmentation.

4. `cardiac_image_system.core.metrics`
   - PSNR;
   - SSIM;
   - Dice;
   - IoU;
   - HD95;
   - relative area error.

5. `cardiac_image_system.experiments`
   - comparison runner;
   - ablation runner;
   - CSV output.

6. `backend.app`
   - minimal FastAPI demo.

## Research boundaries

This implementation supports experimental validation of preprocessing. It does not perform autonomous diagnosis.
