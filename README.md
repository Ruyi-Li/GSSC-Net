# GSSC-Net

This repository is the **pre-acceptance public release** associated with the GSSC-Net manuscript.

At this stage, the repository provides the dataset-loading pipeline, dataset-specific experimental configuration, a generic multimodal feature-extraction backbone, and training/testing scripts. **The paper-specific core implementation is intentionally not included in this pre-acceptance release.** The complete implementation will be released after paper acceptance.

## Files

- `dataset.py`: data loading, preprocessing, patch extraction, and data augmentation.
- `parameter.py`: dataset selection and public experimental configuration.
- `net.py`: generic public multimodal backbone.
- `train.py`: cross-entropy training pipeline.
- `test.py`: OA/AA/Kappa and per-class evaluation.

## Dataset index

- `0`: Houston2013
- `1`: Augsburg City
- `2`: Beijing (legacy compatibility)
- `3`: Berlin

Change `data_type` in `parameter.py` to select the dataset.

## Note

Dataset paths in `dataset.py` are local placeholders and should be changed to the user's own dataset locations before running.
