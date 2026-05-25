# BA2 Evaluation Summary

## NIST
- Variants total: 32
- Variants with any failed NIST test: 32 (1.0)
- Mean passed tests per variant: 11.03125
- Mean failed tests per variant: 3.53125
- Core tests (excluding template tests): 9 with fails out of 13
- Worst tests by fail count:
- non_overlapping_template_matching: n_fail=32, pass_rate=0.0
- overlapping_template_matching: n_fail=32, pass_rate=0.0
- serial: n_fail=12, pass_rate=0.625
- discrete_fourier_transform: n_fail=8, pass_rate=0.75
- approximate_entropy: n_fail=8, pass_rate=0.75
- Fail hotspots (mean failed tests per variant):
- memory_type=opensimplex: 3.75
- memory_type=perlin: 3.3125
- seed_strategy=neighborhood3: 3.625
- seed_strategy=window_mean_3x3: 3.4375
- quant_k=100000.0: 4.875
- quant_k=10000000.0: 2.1875

## Avalanche
- Rows evaluated: 352
- Rows skipped: 32
- Mean hamming ratio: 0.4999777390198274
- Min/Max hamming ratio: 0.4991121292114258 / 0.5008206367492676

## Periodicity
- Variants total: 16
- Mean lag match ratio: 0.0039002063227634803
- Max lag match ratio: 0.0039857153799019605
- Detected prefix period variants: 0

## Reconstruction
- Variants total: 16
- Mean reconstruction R²: 5.8118038459217813e-05
- Mean reconstruction RMSE: 0.5795946252466264
- Mean R² by chaos engine:
- lorenz: -2.75445346968467e-05
- rossler: 0.00014378061161528233

## BA2 Notes
- Use NIST fail clusters to discuss parameter sensitivity (`dt`, `warmup`, `quant_k`, seed strategy, memory model).
- Use avalanche deviation from 0.5 as diffusion indicator.
- Use periodicity indicators as finite-precision risk signal.
