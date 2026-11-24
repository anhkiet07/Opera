# OPERA datasets

Public benchmark data downloaded for a later distribution analysis. No distribution analysis or model training has been performed.

## Layout and provenance

- `forecasting/activity`: Human Activity forecasting, raw and processed data from t-PatchGNN.
- `forecasting/physionet`: PhysioNet 2012 forecasting, raw and processed archives from t-PatchGNN.
- `forecasting/ushcn`: USHCN forecasting, raw benchmark CSV and processed tensors from t-PatchGNN.
- `classification/P12`: Raindrop P12 processed dataset, including the preprocessing resources and splits shipped upstream.
- `classification/P19`: Raindrop P19 processed dataset. The custom P19_hospB split used by OPERA is not supplied by this results-only workspace and has not been recreated.
- `classification/PAM`: Raindrop PAMAP2 classification dataset. This differs from the Human Activity forecasting dataset.
- `native_shift/eicu_demo`: public eICU demo source data. The OPERA eicu_demo feature selection, region split, and task preprocessing are not included in this workspace and have not been recreated.

## Important matching limits

QuITE documents t-PatchGNN forecasting preprocessing and Raindrop classification preprocessing as its benchmark sources. The original OPERA training code, exact cached inputs, split indices, and normalization artifacts are absent here. These downloads therefore provide the public upstream benchmark data, not verified byte-identical OPERA training inputs.

Forecast horizons are generated from the same base forecasting data; they do not require separate dataset downloads. The public eICU demo is a separate resource from the credentialed full eICU database.

## Sources

- https://github.com/Meaningfull9502/QuITE
- https://github.com/usail-hkust/t-PatchGNN
- https://github.com/mims-harvard/Raindrop
- P12: https://doi.org/10.6084/m9.figshare.19514341.v1
- P19: https://doi.org/10.6084/m9.figshare.19514338.v1
- PAM: https://doi.org/10.6084/m9.figshare.19514347.v1
- eICU demo: https://physionet.org/content/eicu-crd-demo/2.0.1/

`download_manifest.json` records the downloaded files, source URLs, byte sizes, and SHA-256 checksums. `validation.json` records archive integrity checks and extracted file counts after successful extraction. Original archives are retained.

## Download status

The six main benchmark datasets are complete and their downloaded file sizes and SHA-256 hashes have been verified. eICU demo is incomplete because PhysioNet repeatedly timed out during TLS handshake or transfer. Partial files are not usable datasets. No background downloads remain. See `download_status.json`.
