# ChaosCrypto WP2 – Report
_Generated: 2026-05-17T10:43:22.245985+00:00 UTC_

## Inputs
- Benchmark CSV: `benchmark.csv` (16 variants aggregated)
- Analyze CSV: `analysis.csv` (16 variants aggregated)
- Token: not stored; only fingerprints in source CSV

## Scope
- Profile: alice
- Coord: (12,34)
- nbytes: 262144
- Seed strategies: ['neighborhood3', 'window_mean_3x3']
- Memory types: ['opensimplex', 'perlin']
- Varying parameters:
  - dt: [0.01]
  - warmup: [100, 1000]
  - quant_k: [100000.0, 10000000.0]
  - size: [128]
  - scale: [0.1]

## Benchmark Summary
Phase timing overview (mean across aggregated variants):
- Field generation mean (s): 0.199139
- Seed derivation mean (s): 0.000098
- Keystream generation mean (s): 0.112891
- XOR mean (s): 0.010798

Top throughput overall (mean over repeats):

| dt | warmup | quant_k | size | scale | seed_strategy | memory_type | mean_t_field_s | mean_t_seed_s | mean_t_keystream_s | mean_t_xor_s | mean_tp_bps | keystream_sha256 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.01 | 1000 | 100000.0 | 128 | 0.1 | neighborhood3 | opensimplex | 0.248147 | 1.75338e-05 | 0.100687 | 0.0105975 | 2.6e+06 | e23a539e5f39ac497009325f1e579c547bd43c5e77c65630dfa89ce38195ec42 |
| 0.01 | 1000 | 10000000.0 | 128 | 0.1 | window_mean_3x3 | opensimplex | 0.249303 | 0.000172007 | 0.104003 | 0.00949103 | 2.53e+06 | 8dcd61627ade0fb7ea5c32baee722cd01fbe386b3f861a82c43a532052bbe60c |
| 0.01 | 100 | 10000000.0 | 128 | 0.1 | window_mean_3x3 | opensimplex | 0.262879 | 0.000137495 | 0.104309 | 0.0103022 | 2.52e+06 | af51d8f8b974d54ead65417d7543aeb07381ab873896625e41809f0c6cd93e08 |
| 0.01 | 100 | 10000000.0 | 128 | 0.1 | window_mean_3x3 | perlin | 0.108252 | 0.000166034 | 0.105184 | 0.00982059 | 2.5e+06 | b83b247d757d2c99822563501928a1f100a1c28e990525a6c2289c2e30c6d9a7 |
| 0.01 | 1000 | 100000.0 | 128 | 0.1 | neighborhood3 | perlin | 0.117529 | 1.39343e-05 | 0.105863 | 0.00946409 | 2.49e+06 | 95b23e0a407d1323f7bd7da9e289cebed647fa01969a49e3f6864136902e29de |

Top throughput per seed_strategy (best across memory types):

| seed_strategy | dt | warmup | quant_k | memory_type | mean_tp_bps | keystream_sha256 |
|---|---|---|---|---|---|---|
| neighborhood3 | 0.01 | 1000 | 100000.0 | opensimplex | 2.6e+06 | e23a539e5f39ac497009325f1e579c547bd43c5e77c65630dfa89ce38195ec42 |
| window_mean_3x3 | 0.01 | 1000 | 10000000.0 | opensimplex | 2.53e+06 | 8dcd61627ade0fb7ea5c32baee722cd01fbe386b3f861a82c43a532052bbe60c |

Top throughput per seed_strategy and memory_type:

| seed_strategy | memory_type | dt | warmup | quant_k | size | scale | mean_tp_bps | keystream_sha256 |
|---|---|---|---|---|---|---|---|---|
| neighborhood3 | opensimplex | 0.01 | 1000 | 100000.0 | 128 | 0.1 | 2.6e+06 | e23a539e5f39ac497009325f1e579c547bd43c5e77c65630dfa89ce38195ec42 |
| neighborhood3 | perlin | 0.01 | 1000 | 100000.0 | 128 | 0.1 | 2.49e+06 | 95b23e0a407d1323f7bd7da9e289cebed647fa01969a49e3f6864136902e29de |
| window_mean_3x3 | opensimplex | 0.01 | 1000 | 10000000.0 | 128 | 0.1 | 2.53e+06 | 8dcd61627ade0fb7ea5c32baee722cd01fbe386b3f861a82c43a532052bbe60c |
| window_mean_3x3 | perlin | 0.01 | 100 | 10000000.0 | 128 | 0.1 | 2.5e+06 | b83b247d757d2c99822563501928a1f100a1c28e990525a6c2289c2e30c6d9a7 |

## Analyze Summary
- Bit ones ratio min/mean/max: (0.499701, 0.499905, 0.500082)
- Runs norm diff min/mean/max: (-0.000153, -3.7e-05, 0.00017)
- Byte chi2 norm min/mean/max: (0.981969, 1.061766, 1.133206)
- Autocorr lag1 min/mean/max: (-0.000341, 7.5e-05, 0.000307)

Per seed_strategy / memory_type (mean values):
| seed_strategy | memory_type | bit_ones_ratio | byte_chi2_norm | runs_norm_diff | autocorr_lag_1 |
|---|---|---|---|---|---|
| neighborhood3 | opensimplex | 0.500018 | 1.030799 | 0.000008 | -0.000017 |
| neighborhood3 | perlin | 0.499920 | 1.063791 | -0.000043 | 0.000086 |
| window_mean_3x3 | opensimplex | 0.499749 | 1.046652 | -0.000023 | 0.000045 |
| window_mean_3x3 | perlin | 0.499931 | 1.105821 | -0.000092 | 0.000183 |

| dt | warmup | quant_k | size | scale | seed_strategy | memory_type | bit_ones_ratio | byte_chi2_norm | runs_norm_diff | autocorr_lag_1 | keystream_sha256 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.01 | 100 | 100000.0 | 128 | 0.1 | neighborhood3 | opensimplex | 0.500073 | 0.981969 | -0.000152 | 0.000304 | b8ce10e014ed53eb9dfab2ce1e17997224a1d5923f62c790ec74ea5c0e611b26 |
| 0.01 | 100 | 100000.0 | 128 | 0.1 | neighborhood3 | perlin | 0.499829 | 1.012517 | -0.000056 | 0.000112 | 265c51e4280be77c31982f3c55a2b1b339de41d0027ccc77c4a1a9e931ea12ae |
| 0.01 | 100 | 100000.0 | 128 | 0.1 | window_mean_3x3 | opensimplex | 0.499796 | 1.000971 | 0.000059 | -0.000118 | 4c63b5572566e8129ff2fa6013d20b5bb7351efc6737ce642fee443db8dcf837 |
| 0.01 | 100 | 100000.0 | 128 | 0.1 | window_mean_3x3 | perlin | 0.500058 | 1.079505 | -0.000058 | 0.000115 | 8961d882e761a738a1d6523ee39a6a7a145531a4aa5c66a0d0dd5f162ec333a1 |
| 0.01 | 100 | 10000000.0 | 128 | 0.1 | neighborhood3 | opensimplex | 0.499956 | 1.077045 | 0.000170 | -0.000341 | 227a776499831b3a19400f1f0f8cbbb909411312a9ec371b40a08699e81e050d |
| 0.01 | 100 | 10000000.0 | 128 | 0.1 | neighborhood3 | perlin | 0.500011 | 1.115142 | -0.000033 | 0.000065 | 38dd69fdca01279ca3fc369fda15eee742eb40cf176fed4791c76a8a87ff283e |
| 0.01 | 100 | 10000000.0 | 128 | 0.1 | window_mean_3x3 | opensimplex | 0.499701 | 1.091900 | -0.000106 | 0.000211 | ba43d8241d8995c8698218892b767df22ed4339b47a7dcbe975c29f6297d624f |
| 0.01 | 100 | 10000000.0 | 128 | 0.1 | window_mean_3x3 | perlin | 0.499812 | 1.133038 | -0.000126 | 0.000253 | 62be714da77289db7bb3f0350c9e789d5b2a05631f32ce23239b3c617e46e768 |
| 0.01 | 1000 | 100000.0 | 128 | 0.1 | neighborhood3 | opensimplex | 0.500082 | 0.987691 | -0.000153 | 0.000307 | 65adf6de4f974e02aa7c79a65b3ee3a86d536089958cbf816c7500b2c1ed385e |
| 0.01 | 1000 | 100000.0 | 128 | 0.1 | neighborhood3 | perlin | 0.499826 | 1.010950 | -0.000055 | 0.000110 | 3dfa2a9120d6e39ab222e25a4bab83bc0acb478bacdfb21d499b997213ff23e6 |
| 0.01 | 1000 | 100000.0 | 128 | 0.1 | window_mean_3x3 | opensimplex | 0.499797 | 1.003139 | 0.000060 | -0.000121 | aedec9c276e06b6870bf0e5d01d54429b81c38655ae02d9b6fd6814fda4e166c |
| 0.01 | 1000 | 100000.0 | 128 | 0.1 | window_mean_3x3 | perlin | 0.500045 | 1.077536 | -0.000057 | 0.000114 | 906251a32a63629653e2577725454cff63e9916f31eb45bbadc73edbcd26674d |
| 0.01 | 1000 | 10000000.0 | 128 | 0.1 | neighborhood3 | opensimplex | 0.499961 | 1.076492 | 0.000168 | -0.000336 | 6f5d0740c3ea0fb01fe774cc3a1a863207bb92a569520837374723626f860fb4 |
| 0.01 | 1000 | 10000000.0 | 128 | 0.1 | neighborhood3 | perlin | 0.500015 | 1.116554 | -0.000029 | 0.000058 | 7705c6adc5b804bc28da2f3bd31b685d6137c4bf5cd16c47cf875037e427cc59 |
| 0.01 | 1000 | 10000000.0 | 128 | 0.1 | window_mean_3x3 | opensimplex | 0.499704 | 1.090596 | -0.000104 | 0.000207 | 8b8da750fe2c91131cd3d7d2c58b066b9edf6d09e6088647295920a2dee26b5a |
| 0.01 | 1000 | 10000000.0 | 128 | 0.1 | window_mean_3x3 | perlin | 0.499809 | 1.133206 | -0.000125 | 0.000251 | 8944f4d6d7c5ee71b68f554de3e240a90b8db5fb7620d68786e01568d894e844 |

## NIST Summary
- Variants with NIST results: 16
- Total passed tests: 175
- Total failed tests: 55
- Total skipped tests: 7
- NIST runtime mean/min/max (s): (12.115, 11.128, 13.519)

| test | pass | fail | skip | mean_p_value |
|---|---|---|---|---|
| frequency_monobit | 28 | 4 | 0 | 0.344090 |
| block_frequency | 26 | 6 | 0 | 0.323662 |
| runs | 30 | 2 | 0 | 0.393651 |
| longest_run_of_ones | 32 | 0 | 0 | 0.525597 |
| binary_matrix_rank | 32 | 0 | 0 | 0.524980 |
| discrete_fourier_transform | 24 | 8 | 0 | 0.366514 |
| non_overlapping_template_matching | 0 | 32 | 0 | 0.000000 |
| overlapping_template_matching | 0 | 32 | 0 | 0.000000 |
| maurers_universal | 32 | 0 | 0 | 0.696362 |
| linear_complexity | 32 | 0 | 0 | 0.445408 |
| serial | 20 | 12 | 0 | 0.334280 |
| approximate_entropy | 24 | 8 | 0 | 0.443690 |
| cumulative_sums | 28 | 4 | 0 | 0.337709 |
| random_excursions | 22 | 3 | 7 | 0.127180 |
| random_excursions_variant | 23 | 2 | 7 | 0.114840 |

## Best Candidates (heuristic score)
Top 5 overall:
| dt | warmup | quant_k | size | scale | seed_strategy | memory_type | score | perf_score | rand_score | bit_ones_ratio | autocorr_lag_1 | runs_norm_diff | byte_chi2_norm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.01 | 1000 | 100000.0 | 128 | 0.1 | neighborhood3 | opensimplex | 0.992387 | 1.000000 | 0.987312 | 0.500082 | 0.000307 | -0.000153 | 0.987691 |
| 0.01 | 1000 | 100000.0 | 128 | 0.1 | neighborhood3 | perlin | 0.975197 | 0.954738 | 0.988837 | 0.499826 | 0.000110 | -0.000055 | 1.010950 |
| 0.01 | 1000 | 100000.0 | 128 | 0.1 | window_mean_3x3 | opensimplex | 0.956278 | 0.895961 | 0.996489 | 0.499797 | -0.000121 | 0.000060 | 1.003139 |
| 0.01 | 1000 | 10000000.0 | 128 | 0.1 | window_mean_3x3 | opensimplex | 0.937696 | 0.969611 | 0.916419 | 0.499704 | 0.000207 | -0.000104 | 1.090596 |
| 0.01 | 100 | 10000000.0 | 128 | 0.1 | window_mean_3x3 | opensimplex | 0.935626 | 0.966087 | 0.915318 | 0.499701 | 0.000211 | -0.000106 | 1.091900 |

Top 3 per seed_strategy:
| seed_strategy | dt | warmup | quant_k | size | scale | memory_type | score | bit_ones_ratio | autocorr_lag_1 | runs_norm_diff | byte_chi2_norm |
|---|---|---|---|---|---|---|---|---|---|---|---|
| neighborhood3 | 0.01 | 1000 | 100000.0 | 128 | 0.1 | opensimplex | 0.992387 | 0.500082 | 0.000307 | -0.000153 | 0.987691 |
| neighborhood3 | 0.01 | 1000 | 100000.0 | 128 | 0.1 | perlin | 0.975197 | 0.499826 | 0.000110 | -0.000055 | 1.010950 |
| neighborhood3 | 0.01 | 1000 | 10000000.0 | 128 | 0.1 | opensimplex | 0.928854 | 0.499961 | -0.000336 | 0.000168 | 1.076492 |
| window_mean_3x3 | 0.01 | 1000 | 100000.0 | 128 | 0.1 | opensimplex | 0.956278 | 0.499797 | -0.000121 | 0.000060 | 1.003139 |
| window_mean_3x3 | 0.01 | 1000 | 10000000.0 | 128 | 0.1 | opensimplex | 0.937696 | 0.499704 | 0.000207 | -0.000104 | 1.090596 |
| window_mean_3x3 | 0.01 | 100 | 10000000.0 | 128 | 0.1 | opensimplex | 0.935626 | 0.499701 | 0.000211 | -0.000106 | 1.091900 |

## Plots
![](plots/bench/bench_throughput_dt0p01_q100000p0_neighborhood3_opensimplex.png)
![](plots/bench/bench_throughput_dt0p01_q10000000p0_neighborhood3_opensimplex.png)
![](plots/bench/bench_throughput_dt0p01_q100000p0_window_mean_3x3_opensimplex.png)
![](plots/bench/bench_throughput_dt0p01_q10000000p0_window_mean_3x3_opensimplex.png)
![](plots/bench/bench_throughput_dt0p01_q100000p0_neighborhood3_perlin.png)
![](plots/bench/bench_throughput_dt0p01_q10000000p0_neighborhood3_perlin.png)
![](plots/bench/bench_throughput_dt0p01_q100000p0_window_mean_3x3_perlin.png)
![](plots/bench/bench_throughput_dt0p01_q10000000p0_window_mean_3x3_perlin.png)
![](plots/analyze_bit/analyze_bit_balance_dt0p01_q100000p0_neighborhood3_opensimplex.png)
![](plots/analyze_bit/analyze_bit_balance_dt0p01_q10000000p0_neighborhood3_opensimplex.png)
![](plots/analyze_bit/analyze_bit_balance_dt0p01_q100000p0_window_mean_3x3_opensimplex.png)
![](plots/analyze_bit/analyze_bit_balance_dt0p01_q10000000p0_window_mean_3x3_opensimplex.png)
![](plots/analyze_bit/analyze_bit_balance_dt0p01_q100000p0_neighborhood3_perlin.png)
![](plots/analyze_bit/analyze_bit_balance_dt0p01_q10000000p0_neighborhood3_perlin.png)
![](plots/analyze_bit/analyze_bit_balance_dt0p01_q100000p0_window_mean_3x3_perlin.png)
![](plots/analyze_bit/analyze_bit_balance_dt0p01_q10000000p0_window_mean_3x3_perlin.png)
![](plots/analyze_autocorr/analyze_autocorr_lag1_dt0p01_q100000p0_neighborhood3_opensimplex.png)
![](plots/analyze_autocorr/analyze_autocorr_lag1_dt0p01_q10000000p0_neighborhood3_opensimplex.png)
![](plots/analyze_autocorr/analyze_autocorr_lag1_dt0p01_q100000p0_window_mean_3x3_opensimplex.png)
![](plots/analyze_autocorr/analyze_autocorr_lag1_dt0p01_q10000000p0_window_mean_3x3_opensimplex.png)
![](plots/analyze_autocorr/analyze_autocorr_lag1_dt0p01_q100000p0_neighborhood3_perlin.png)
![](plots/analyze_autocorr/analyze_autocorr_lag1_dt0p01_q10000000p0_neighborhood3_perlin.png)
![](plots/analyze_autocorr/analyze_autocorr_lag1_dt0p01_q100000p0_window_mean_3x3_perlin.png)
![](plots/analyze_autocorr/analyze_autocorr_lag1_dt0p01_q10000000p0_window_mean_3x3_perlin.png)
![](plots/analyze_chi2/analyze_byte_chi2_norm_dt0p01_q100000p0_neighborhood3_opensimplex.png)
![](plots/analyze_chi2/analyze_byte_chi2_norm_dt0p01_q10000000p0_neighborhood3_opensimplex.png)
![](plots/analyze_chi2/analyze_byte_chi2_norm_dt0p01_q100000p0_window_mean_3x3_opensimplex.png)
![](plots/analyze_chi2/analyze_byte_chi2_norm_dt0p01_q10000000p0_window_mean_3x3_opensimplex.png)
![](plots/analyze_chi2/analyze_byte_chi2_norm_dt0p01_q100000p0_neighborhood3_perlin.png)
![](plots/analyze_chi2/analyze_byte_chi2_norm_dt0p01_q10000000p0_neighborhood3_perlin.png)
![](plots/analyze_chi2/analyze_byte_chi2_norm_dt0p01_q100000p0_window_mean_3x3_perlin.png)
![](plots/analyze_chi2/analyze_byte_chi2_norm_dt0p01_q10000000p0_window_mean_3x3_perlin.png)

## Appendix
- CSV columns: benchmark includes timing/throughput; analyze includes keystream statistics.
- Reproducibility: same config → identical hashes/metrics.

## Methodology Notes
- Benchmark results are averaged over `repeats` runs (as configured; BA1 kit uses repeats=3).
- Analyze metrics are deterministic per variant and computed once per variant (no repeats by default).
- Environment details for the run are stored in `out/ba1/run_meta.txt` (UTC date, python version, uname, pip freeze).
- Statistical metrics describe properties of the generated keystream for the tested length; they do not prove cryptographic security.