"""Project-wide constants for the WP2 MVP."""

DEFAULT_MEMORY_SIZE = 128
DEFAULT_MEMORY_SCALE = 0.1

LCL_SIGMA = 10.0  # Lorenz sigma
LCL_RHO = 28.0    # Lorenz rho
LCL_BETA = 8.0 / 3.0  # Lorenz beta
RSL_A = 0.2           # Roessler a
RSL_B = 0.2           # Roessler b
RSL_C = 5.7           # Roessler c

DEFAULT_DT = 0.01
DEFAULT_WARMUP = 1000
DEFAULT_QUANT_K = 1e5

VERSION = "1"
MEMORY_TYPE = "opensimplex"
SEED_STRATEGY = "neighborhood3"
SAMPLING_TYPE = "quantize_byte"
CHAOS_ENGINE = "lorenz"
ENCODING = "utf-8"
