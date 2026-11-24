### Findings from the dataset-only diagnostics

- **Activity 3s -> 1s:** 200 windows; median history/target entries 276/93; 0.00% have no observed target; 0.00% of history channels are empty on average.
- **PhysioNet 24h -> 24h:** 200 windows; median history/target entries 222/171; 0.00% have no observed target; 31.60% of history channels are empty on average.
- **USHCN 24 -> 1:** 200 windows; median history/target entries 178/7; 2.50% have no observed target; 1.40% of history channels are empty on average.
- **USHCN 24 -> 12:** 200 windows; median history/target entries 178/90; 0.00% have no observed target; 1.60% of history channels are empty on average.

- On 100 common USHCN record/start pairs, the two horizons have identical historical observation counts. Target horizon and window sampling must therefore be distinguished from source-mask differences.
- Re-observation effects are reported alongside realized retention: removing points, a time block, and channels does not remove the same information even when nominal severity is 50%.
- These measures identify weak observation support; they do not establish unpredictability, prove a real acquisition-policy shift, or define a universal acceptable missingness threshold.
- Source channels/windows are correlated, and overlapping windows are not independent samples. No confidence intervals or significance claims are made.