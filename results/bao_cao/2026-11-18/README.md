# tong_ket_2_tuan_2026-11-18

- runs: 564 (282 vanilla/opera pairs), sweep files: 564, excluded: 659
- sweep conditions (20): identity, random_dropout:0.7, temporal_stride:3, variable_dropout:0.5, temporal_stride:3+variable_group_dropout:0.3, random_dropout:0.1, random_dropout:0.2, random_dropout:0.3, random_dropout:0.4, random_dropout:0.5, random_dropout:0.6, temporal_stride:2, temporal_stride:4, variable_dropout:0.2, variable_dropout:0.3, variable_dropout:0.4, temporal_stride:3+variable_group_dropout:0.1, temporal_stride:3+variable_group_dropout:0.2, temporal_stride:3+variable_group_dropout:0.4, temporal_stride:3+variable_group_dropout:0.5
- protocol conditions checked against results.json (rel tol 0.0001, as in scripts/reevaluate.py): identity, random_dropout:0.7, temporal_stride:3, variable_dropout:0.5, temporal_stride:3+variable_group_dropout:0.3
- runs/<run>/: results.json, config.yaml, train.log (model.pt omitted)
- sweep/<run>.json: output of scripts/severity_sweep.py (protocol replay verified)
- slurm_logs/: copy of logs/ (train jobs, severity-sweep jobs, local launcher logs)
- excluded_runs.csv: every bench run not exported and why

| dataset | backbone | seeds |
|---|---|---|
| activity | itransformer | 0 |
| activity | patchtst | 0 |
| activity | segrnn | 0,1,2 |
| activity | tmix | 0 |
| activity_h1000 | cru | 0,1,2,3,4 |
| activity_h1000 | dlinear | 0,1,2,3,4 |
| activity_h1000 | grafiti | 0,1,2,3,4 |
| activity_h1000 | grud | 0,1,2,3,4 |
| activity_h1000 | mtand | 0,1,2,3,4 |
| activity_h1000 | segrnn | 0,1,2,3,4 |
| activity_h1000 | tpatchgnn | 0,1,2,3,4 |
| activity_h1000 | warpformer | 0,1,2,3,4 |
| activity_h2000 | grafiti | 0,1,2,3,4 |
| activity_h2000 | grud | 0,1,2,3,4 |
| activity_h2000 | mtand | 0,1,2,3,4 |
| activity_h2000 | segrnn | 0,1,2 |
| activity_h2000 | tpatchgnn | 0,1,2,3,4 |
| activity_h2000 | warpformer | 0,1,2,3,4 |
| cgmacros | grafiti | 0,1,2,3,4 |
| cgmacros | grud | 0,1,2,3,4 |
| cgmacros | itransformer | 0,1,2,3,4 |
| cgmacros | mtand | 0,1,2,3,4 |
| cgmacros | patchtst | 0,1,2,3,4 |
| cgmacros | segrnn | 0,1,2 |
| cgmacros | tmix | 0,1,2,3,4 |
| cgmacros | tpatchgnn | 0,1,2,3,4 |
| cgmacros | warpformer | 0,1,2,3,4 |
| noaa_isd | grud | 0,1,2,3,4 |
| noaa_isd | mtand | 0,1,2,3,4 |
| noaa_isd | segrnn | 0,1,2 |
| noaa_isd | warpformer | 0,1,2,3,4 |
| physionet | segrnn | 0,1,2 |
| physionet_h12 | dlinear | 0,1,2,3,4 |
| physionet_h12 | grafiti | 0,1,2,3,4 |
| physionet_h12 | grud | 0,1,2,3,4 |
| physionet_h12 | mtand | 0,1,2,3,4 |
| physionet_h12 | segrnn | 0,1,2,3,4 |
| physionet_h12 | tpatchgnn | 0,1,2,3,4 |
| physionet_h12 | warpformer | 0,1,2,3,4 |
| physionet_h36 | grafiti | 0,1,2,3,4 |
| physionet_h36 | grud | 0,1,2,3,4 |
| physionet_h36 | mtand | 0,1,2,3,4 |
| physionet_h36 | segrnn | 0,1,2 |
| physionet_h36 | tpatchgnn | 0,1,2,3,4 |
| physionet_h36 | warpformer | 0,1,2,3,4 |
| ushcn_12mo | segrnn | 0,1,2 |
| ushcn_1mo | segrnn | 0,1,2 |
| ushcn_6mo | cru | 0,1,2,3,4 |
| ushcn_6mo | dlinear | 0,1,2,3,4 |
| ushcn_6mo | grafiti | 0,1,2,3,4 |
| ushcn_6mo | grud | 0,1,2,3,4 |
| ushcn_6mo | mtand | 0,1,2,3,4 |
| ushcn_6mo | segrnn | 0,1,2,3,4 |
| ushcn_6mo | tpatchgnn | 0,1,2,3,4 |
| ushcn_6mo | warpformer | 0,1,2,3,4 |
| vitaldb | dlinear | 0,1,2,3,4 |
| vitaldb | grafiti | 0,1,2,3,4 |
| vitaldb | grud | 0,1,2,3,4 |
| vitaldb | mtand | 0,1,2,3,4 |
| vitaldb | segrnn | 0,1,2,3,4 |
| vitaldb | tpatchgnn | 0,1,2,3,4 |
| vitaldb | warpformer | 0,1,2,3,4 |
