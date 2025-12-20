# ChaosCrypto WP2 – Report

## Inputs
- Benchmark CSV: `/mnt/c/Users/tobia/Projects/FH/WP2/out/ba1/bench/results.csv` (24 variants aggregated)
- Analyze CSV: `/mnt/c/Users/tobia/Projects/FH/WP2/out/ba1/analyze/analysis.csv` (24 variants aggregated)
- Token: not stored; only fingerprints in source CSV

## Scope
- Profile: alice
- Coord: (12,34)
- nbytes: 1048576
- Seed strategies: ['neighborhood3', 'window_mean_3x3']
- Memory types: ['opensimplex', 'perlin']
- Varying parameters:
  - dt: [0.01]
  - warmup: [100, 1000, 5000]
  - quant_k: [100000.0, 10000000.0]
  - size: [128]
  - scale: [0.1]

## Benchmark Summary
Top throughput overall (mean over repeats):

| dt | warmup | quant_k | size | scale | seed_strategy | memory_type | mean_t_keystream_s | mean_tp_bps | keystream_sha256 |
|---|---|---|---|---|---|---|---|---|---|
| 0.01 | 100 | 100000.0 | 128 | 0.1 | neighborhood3 | perlin | 0.30106 | 3.48e+06 | 265c51e4280be77c31982f3c55a2b1b339de41d0027ccc77c4a1a9e931ea12ae |
| 0.01 | 100 | 100000.0 | 128 | 0.1 | neighborhood3 | opensimplex | 0.302151 | 3.47e+06 | b8ce10e014ed53eb9dfab2ce1e17997224a1d5923f62c790ec74ea5c0e611b26 |
| 0.01 | 100 | 100000.0 | 128 | 0.1 | window_mean_3x3 | perlin | 0.303892 | 3.45e+06 | 8961d882e761a738a1d6523ee39a6a7a145531a4aa5c66a0d0dd5f162ec333a1 |
| 0.01 | 100 | 10000000.0 | 128 | 0.1 | neighborhood3 | opensimplex | 0.30717 | 3.41e+06 | 227a776499831b3a19400f1f0f8cbbb909411312a9ec371b40a08699e81e050d |
| 0.01 | 100 | 100000.0 | 128 | 0.1 | window_mean_3x3 | opensimplex | 0.307625 | 3.41e+06 | 4c63b5572566e8129ff2fa6013d20b5bb7351efc6737ce642fee443db8dcf837 |

Top throughput per seed_strategy:

| seed_strategy | dt | warmup | quant_k | size | scale | memory_type | mean_tp_bps | keystream_sha256 |
|---|---|---|---|---|---|---|---|---|
| neighborhood3 | 0.01 | 100 | 100000.0 | 128 | 0.1 | opensimplex | 3.47e+06 | b8ce10e014ed53eb9dfab2ce1e17997224a1d5923f62c790ec74ea5c0e611b26 |
| neighborhood3 | 0.01 | 100 | 100000.0 | 128 | 0.1 | perlin | 3.48e+06 | 265c51e4280be77c31982f3c55a2b1b339de41d0027ccc77c4a1a9e931ea12ae |
| window_mean_3x3 | 0.01 | 100 | 100000.0 | 128 | 0.1 | opensimplex | 3.41e+06 | 4c63b5572566e8129ff2fa6013d20b5bb7351efc6737ce642fee443db8dcf837 |
| window_mean_3x3 | 0.01 | 100 | 100000.0 | 128 | 0.1 | perlin | 3.45e+06 | 8961d882e761a738a1d6523ee39a6a7a145531a4aa5c66a0d0dd5f162ec333a1 |

## Analyze Summary
- Bit ones ratio min/mean/max: (0.499577, 0.500057, 0.500528)
- Runs norm diff min/mean/max: (-0.000371, -1.7e-05, 0.000411)
- Byte chi2 norm min/mean/max: (0.896563, 1.051495, 1.135279)
- Autocorr lag1 min/mean/max: (-0.000823, 3.4e-05, 0.000742)

Per seed_strategy / memory_type (mean values):
| seed_strategy | memory_type | bit_ones_ratio | byte_chi2_norm | runs_norm_diff | autocorr_lag_1 |
|---|---|---|---|---|---|
| neighborhood3 | opensimplex | 0.500275 | 1.065996 | 0.000172 | -0.000345 |
| neighborhood3 | perlin | 0.499981 | 1.031055 | -0.000095 | 0.000190 |
| window_mean_3x3 | opensimplex | 0.499950 | 1.008068 | -0.000053 | 0.000106 |
| window_mean_3x3 | perlin | 0.500021 | 1.100860 | -0.000092 | 0.000184 |

| dt | warmup | quant_k | size | scale | seed_strategy | memory_type | bit_ones_ratio | byte_chi2_norm | runs_norm_diff | autocorr_lag_1 | keystream_sha256 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.01 | 100 | 100000.0 | 128 | 0.1 | neighborhood3 | opensimplex | 0.500501 | 1.012630 | -0.000054 | 0.000108 | b8ce10e014ed53eb9dfab2ce1e17997224a1d5923f62c790ec74ea5c0e611b26 |
| 0.01 | 100 | 100000.0 | 128 | 0.1 | neighborhood3 | perlin | 0.499951 | 0.953018 | 0.000173 | -0.000346 | 265c51e4280be77c31982f3c55a2b1b339de41d0027ccc77c4a1a9e931ea12ae |
| 0.01 | 100 | 100000.0 | 128 | 0.1 | window_mean_3x3 | opensimplex | 0.500310 | 0.899140 | -0.000094 | 0.000188 | 4c63b5572566e8129ff2fa6013d20b5bb7351efc6737ce642fee443db8dcf837 |
| 0.01 | 100 | 100000.0 | 128 | 0.1 | window_mean_3x3 | perlin | 0.500280 | 1.079013 | 0.000026 | -0.000052 | 8961d882e761a738a1d6523ee39a6a7a145531a4aa5c66a0d0dd5f162ec333a1 |
| 0.01 | 100 | 10000000.0 | 128 | 0.1 | neighborhood3 | opensimplex | 0.500032 | 1.126147 | 0.000411 | -0.000822 | 227a776499831b3a19400f1f0f8cbbb909411312a9ec371b40a08699e81e050d |
| 0.01 | 100 | 10000000.0 | 128 | 0.1 | neighborhood3 | perlin | 0.500010 | 1.116042 | -0.000352 | 0.000704 | 38dd69fdca01279ca3fc369fda15eee742eb40cf176fed4791c76a8a87ff283e |
| 0.01 | 100 | 10000000.0 | 128 | 0.1 | window_mean_3x3 | opensimplex | 0.499577 | 1.115259 | -0.000011 | 0.000022 | ba43d8241d8995c8698218892b767df22ed4339b47a7dcbe975c29f6297d624f |
| 0.01 | 100 | 10000000.0 | 128 | 0.1 | window_mean_3x3 | perlin | 0.499784 | 1.129220 | -0.000216 | 0.000432 | 62be714da77289db7bb3f0350c9e789d5b2a05631f32ce23239b3c617e46e768 |
| 0.01 | 1000 | 100000.0 | 128 | 0.1 | neighborhood3 | opensimplex | 0.500505 | 1.013369 | -0.000060 | 0.000120 | 65adf6de4f974e02aa7c79a65b3ee3a86d536089958cbf816c7500b2c1ed385e |
| 0.01 | 1000 | 100000.0 | 128 | 0.1 | neighborhood3 | perlin | 0.499941 | 0.950057 | 0.000171 | -0.000341 | 3dfa2a9120d6e39ab222e25a4bab83bc0acb478bacdfb21d499b997213ff23e6 |
| 0.01 | 1000 | 100000.0 | 128 | 0.1 | window_mean_3x3 | opensimplex | 0.500311 | 0.896563 | -0.000092 | 0.000184 | aedec9c276e06b6870bf0e5d01d54429b81c38655ae02d9b6fd6814fda4e166c |
| 0.01 | 1000 | 100000.0 | 128 | 0.1 | window_mean_3x3 | perlin | 0.500270 | 1.076170 | 0.000016 | -0.000032 | 906251a32a63629653e2577725454cff63e9916f31eb45bbadc73edbcd26674d |
| 0.01 | 1000 | 10000000.0 | 128 | 0.1 | neighborhood3 | opensimplex | 0.500036 | 1.121121 | 0.000411 | -0.000823 | 6f5d0740c3ea0fb01fe774cc3a1a863207bb92a569520837374723626f860fb4 |
| 0.01 | 1000 | 10000000.0 | 128 | 0.1 | neighborhood3 | perlin | 0.500016 | 1.112573 | -0.000358 | 0.000715 | 7705c6adc5b804bc28da2f3bd31b685d6137c4bf5cd16c47cf875037e427cc59 |
| 0.01 | 1000 | 10000000.0 | 128 | 0.1 | window_mean_3x3 | opensimplex | 0.499581 | 1.115135 | -0.000008 | 0.000015 | 8b8da750fe2c91131cd3d7d2c58b066b9edf6d09e6088647295920a2dee26b5a |
| 0.01 | 1000 | 10000000.0 | 128 | 0.1 | window_mean_3x3 | perlin | 0.499777 | 1.135279 | -0.000204 | 0.000409 | 8944f4d6d7c5ee71b68f554de3e240a90b8db5fb7620d68786e01568d894e844 |
| 0.01 | 5000 | 100000.0 | 128 | 0.1 | neighborhood3 | opensimplex | 0.500528 | 1.009965 | -0.000076 | 0.000152 | 6e30cc395858c5787e7b02a7d7230147cf9e37aba125d6a276ef6de1b9c3c859 |
| 0.01 | 5000 | 100000.0 | 128 | 0.1 | neighborhood3 | perlin | 0.499937 | 0.939493 | 0.000168 | -0.000336 | 5e559163977b073f75fd2c73e24688c652f365163e9096ca11c4d6d69c6b919c |
| 0.01 | 5000 | 100000.0 | 128 | 0.1 | window_mean_3x3 | opensimplex | 0.500325 | 0.905113 | -0.000100 | 0.000200 | a77608ec1121bd678cc072bf3694a7611a8e68ee862e5f37e102853e73888a5e |
| 0.01 | 5000 | 100000.0 | 128 | 0.1 | window_mean_3x3 | perlin | 0.500237 | 1.056376 | 0.000017 | -0.000033 | aacea544bd8f862cd02ab9c02551b45321dc2e1bef15427d657bc3953f98009d |
| 0.01 | 5000 | 10000000.0 | 128 | 0.1 | neighborhood3 | opensimplex | 0.500049 | 1.112743 | 0.000403 | -0.000806 | 0f107415bf46cf117a574364307f940ef3f691f9ae662bcdfefe0ca8c4740c79 |
| 0.01 | 5000 | 10000000.0 | 128 | 0.1 | neighborhood3 | perlin | 0.500032 | 1.115146 | -0.000371 | 0.000742 | 5a8ae8094938263242f89dfa257213977b5ceee803fb0714e27f781480b7c91c |
| 0.01 | 5000 | 10000000.0 | 128 | 0.1 | window_mean_3x3 | opensimplex | 0.499594 | 1.117199 | -0.000012 | 0.000024 | 1ed6d7988da345441a7468162f3ddf2680d289641fa4417473bace0e6aeced57 |
| 0.01 | 5000 | 10000000.0 | 128 | 0.1 | window_mean_3x3 | perlin | 0.499781 | 1.129103 | -0.000190 | 0.000381 | 42acd5799babb39196739560a5004cd0ed65fc2475041abee31d456df13bcff8 |

## Best Candidates (heuristic score)
Top 5 overall:
| dt | warmup | quant_k | size | scale | seed_strategy | memory_type | score | perf_score | rand_score | bit_ones_ratio | autocorr_lag_1 | runs_norm_diff | byte_chi2_norm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.01 | 100 | 100000.0 | 128 | 0.1 | neighborhood3 | opensimplex | 0.990865 | 0.996841 | 0.986881 | 0.500501 | 0.000108 | -0.000054 | 1.012630 |
| 0.01 | 100 | 100000.0 | 128 | 0.1 | neighborhood3 | perlin | 0.972765 | 1.000000 | 0.954609 | 0.499951 | -0.000346 | 0.000173 | 0.953018 |
| 0.01 | 5000 | 100000.0 | 128 | 0.1 | neighborhood3 | opensimplex | 0.959112 | 0.913690 | 0.989393 | 0.500528 | 0.000152 | -0.000076 | 1.009965 |
| 0.01 | 1000 | 100000.0 | 128 | 0.1 | neighborhood3 | opensimplex | 0.958568 | 0.917210 | 0.986140 | 0.500505 | 0.000120 | -0.000060 | 1.013369 |
| 0.01 | 100 | 100000.0 | 128 | 0.1 | window_mean_3x3 | perlin | 0.952099 | 0.990549 | 0.926465 | 0.500280 | -0.000052 | 0.000026 | 1.079013 |

Top 3 per seed_strategy:
| seed_strategy | dt | warmup | quant_k | size | scale | memory_type | score | bit_ones_ratio | autocorr_lag_1 | runs_norm_diff | byte_chi2_norm |
|---|---|---|---|---|---|---|---|---|---|---|---|
| neighborhood3 | 0.01 | 100 | 100000.0 | 128 | 0.1 | opensimplex | 0.990865 | 0.500501 | 0.000108 | -0.000054 | 1.012630 |
| neighborhood3 | 0.01 | 100 | 100000.0 | 128 | 0.1 | perlin | 0.972765 | 0.499951 | -0.000346 | 0.000173 | 0.953018 |
| neighborhood3 | 0.01 | 5000 | 100000.0 | 128 | 0.1 | opensimplex | 0.959112 | 0.500528 | 0.000152 | -0.000076 | 1.009965 |
| window_mean_3x3 | 0.01 | 100 | 100000.0 | 128 | 0.1 | perlin | 0.952099 | 0.500280 | -0.000052 | 0.000026 | 1.079013 |
| window_mean_3x3 | 0.01 | 5000 | 100000.0 | 128 | 0.1 | perlin | 0.940700 | 0.500237 | -0.000033 | 0.000017 | 1.056376 |
| window_mean_3x3 | 0.01 | 100 | 100000.0 | 128 | 0.1 | opensimplex | 0.936311 | 0.500310 | 0.000188 | -0.000094 | 0.899140 |

## Plots
![](/mnt/c/Users/tobia/Projects/FH/WP2/out/ba1/report/plots/bench/bench_throughput_dt0p01_q100000p0.png)
![](/mnt/c/Users/tobia/Projects/FH/WP2/out/ba1/report/plots/bench/bench_throughput_dt0p01_q10000000p0.png)
![](/mnt/c/Users/tobia/Projects/FH/WP2/out/ba1/report/plots/analyze_bit/analyze_bit_balance_dt0p01_q100000p0.png)
![](/mnt/c/Users/tobia/Projects/FH/WP2/out/ba1/report/plots/analyze_bit/analyze_bit_balance_dt0p01_q10000000p0.png)
![](/mnt/c/Users/tobia/Projects/FH/WP2/out/ba1/report/plots/analyze_autocorr/analyze_autocorr_lag1_dt0p01_q100000p0.png)
![](/mnt/c/Users/tobia/Projects/FH/WP2/out/ba1/report/plots/analyze_autocorr/analyze_autocorr_lag1_dt0p01_q10000000p0.png)
![](/mnt/c/Users/tobia/Projects/FH/WP2/out/ba1/report/plots/analyze_chi2/analyze_byte_chi2_norm_dt0p01_q100000p0.png)
![](/mnt/c/Users/tobia/Projects/FH/WP2/out/ba1/report/plots/analyze_chi2/analyze_byte_chi2_norm_dt0p01_q10000000p0.png)

## Appendix
- CSV columns: benchmark includes timing/throughput; analyze includes keystream statistics.
- Reproducibility: same config → identical hashes/metrics.