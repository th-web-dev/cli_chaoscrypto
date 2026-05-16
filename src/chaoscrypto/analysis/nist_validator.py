from __future__ import annotations

import json
import math
from dataclasses import dataclass
from statistics import NormalDist
from time import perf_counter
from typing import Any, Dict, Iterable, List

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.special import erfc, gammaincc

NIST_DEFAULT_ALPHA = 0.01
NIST_TEST_NAMES = [
    "frequency_monobit",
    "block_frequency",
    "runs",
    "longest_run_of_ones",
    "binary_matrix_rank",
    "discrete_fourier_transform",
    "non_overlapping_template_matching",
    "overlapping_template_matching",
    "maurers_universal",
    "linear_complexity",
    "serial",
    "approximate_entropy",
    "cumulative_sums",
    "random_excursions",
    "random_excursions_variant",
]

NIST_CSV_FIELDS: List[str] = [
    "nist_alpha",
    "nist_passed_count",
    "nist_failed_count",
    "nist_skipped_count",
    "nist_total_runtime_s",
    "nist_failed_tests",
]
for _name in NIST_TEST_NAMES:
    prefix = f"nist_{_name}"
    NIST_CSV_FIELDS.extend(
        [
            f"{prefix}_status",
            f"{prefix}_pass",
            f"{prefix}_p_value",
            f"{prefix}_details_json",
        ]
    )

_NORMAL = NormalDist()
_NON_OVERLAPPING_TEMPLATE_STRINGS = (
    "000000001",
    "000001111",
    "000111111",
    "001010101",
    "010101010",
    "111000111",
    "111110000",
    "111111111",
)
_NON_OVERLAPPING_TEMPLATE_BITS = {
    template: np.fromiter((1 if ch == "1" else 0 for ch in template), dtype=np.uint8)
    for template in _NON_OVERLAPPING_TEMPLATE_STRINGS
}
_OVERLAPPING_TEMPLATE_BITS = np.ones(9, dtype=np.uint8)
_MIN_BITS_BY_TEST: Dict[str, int] = {
    "frequency_monobit": 1,
    "block_frequency": 128,
    "runs": 100,
    "longest_run_of_ones": 128,
    "binary_matrix_rank": 1024,
    "discrete_fourier_transform": 1000,
    "non_overlapping_template_matching": 1024,
    "overlapping_template_matching": 1032,
    "maurers_universal": 387840,
    "linear_complexity": 500,
    "serial": 128,
    "approximate_entropy": 128,
    "cumulative_sums": 1,
    "random_excursions": 1,
    "random_excursions_variant": 1,
}


@dataclass(frozen=True)
class _Result:
    name: str
    status: str
    p_value: float | None
    passed: bool | None
    details: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "p_value": self.p_value,
            "pass": self.passed,
            "details": self.details,
        }


def _skip(name: str, reason: str, **details: Any) -> Dict[str, Any]:
    payload = {"reason": reason, **details}
    return _Result(name=name, status="skip", p_value=None, passed=None, details=payload).to_dict()


def _single(name: str, p_value: float, alpha: float, **details: Any) -> Dict[str, Any]:
    p = float(max(0.0, min(1.0, p_value)))
    passed = p >= alpha
    return _Result(name=name, status="pass" if passed else "fail", p_value=p, passed=passed, details=details).to_dict()


def _multi(name: str, p_values: Iterable[float | None], alpha: float, **details: Any) -> Dict[str, Any]:
    finite = [float(p) for p in p_values if p is not None and not math.isnan(p)]
    if not finite:
        return _skip(name, "no_finite_p_values", **details)
    overall = min(finite)
    passed = all(p >= alpha for p in finite)
    payload = dict(details)
    payload["subtest_p_values"] = finite
    return _Result(
        name=name,
        status="pass" if passed else "fail",
        p_value=overall,
        passed=passed,
        details=payload,
    ).to_dict()


def _bits_from_bytes(data: bytes) -> np.ndarray:
    if not data:
        return np.empty(0, dtype=np.uint8)
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8)).astype(np.uint8, copy=False)


def _safe_gammaincc(a: float, x: float) -> float:
    if a <= 0:
        return 0.0
    return float(gammaincc(a, x))


def _pattern_counts(bits: np.ndarray, m: int, *, circular: bool) -> np.ndarray:
    if m <= 0:
        return np.array([bits.size], dtype=np.int64)
    n = int(bits.size)
    if n == 0:
        return np.zeros(1 << m, dtype=np.int64)
    if circular:
        extended = np.concatenate([bits, bits[: m - 1]])
        total = n
    else:
        if n < m:
            return np.zeros(1 << m, dtype=np.int64)
        extended = bits
        total = n - m + 1
    weights = (1 << np.arange(m - 1, -1, -1, dtype=np.uint32)).astype(np.uint32)
    counts = np.zeros(1 << m, dtype=np.int64)
    chunk = 250_000
    for start in range(0, total, chunk):
        length = min(chunk, total - start)
        view = sliding_window_view(extended[start : start + length + m - 1], m)
        values = view.astype(np.uint32, copy=False) @ weights
        counts += np.bincount(values, minlength=1 << m)
    return counts


def _window_matches(block: np.ndarray, template_bits: np.ndarray) -> np.ndarray:
    width = template_bits.size
    if block.size < width:
        return np.empty(0, dtype=bool)
    windows = sliding_window_view(block, width)
    return np.all(windows == template_bits, axis=1)


def _count_non_overlapping_matches(block: np.ndarray, template_bits: np.ndarray) -> int:
    matches = np.flatnonzero(_window_matches(block, template_bits))
    if matches.size == 0:
        return 0
    count = 0
    next_allowed = -1
    step = template_bits.size
    for idx in matches.tolist():
        if idx >= next_allowed:
            count += 1
            next_allowed = idx + step
    return count


def _longest_run_in_block(block: np.ndarray) -> int:
    max_run = 0
    cur = 0
    for bit in block:
        if bit:
            cur += 1
            if cur > max_run:
                max_run = cur
        else:
            cur = 0
    return max_run


def _gf2_rank_u32(rows: np.ndarray) -> int:
    rank = 0
    rows_u = rows.astype(np.uint64, copy=True)
    n_rows = rows_u.size
    for bit_idx in range(31, -1, -1):
        mask = np.uint64(1 << bit_idx)
        pivot = None
        for row_idx in range(rank, n_rows):
            if rows_u[row_idx] & mask:
                pivot = row_idx
                break
        if pivot is None:
            continue
        if pivot != rank:
            rows_u[[rank, pivot]] = rows_u[[pivot, rank]]
        pivot_row = rows_u[rank]
        for row_idx in range(n_rows):
            if row_idx != rank and (rows_u[row_idx] & mask):
                rows_u[row_idx] ^= pivot_row
        rank += 1
        if rank == n_rows:
            break
    return rank


def _berlekamp_massey_int(bits: np.ndarray) -> int:
    c = 1
    b = 1
    l = 0
    m = -1
    history = 0
    for n_idx, bit in enumerate(bits):
        history = (history << 1) | int(bit)
        discrepancy = (c & history).bit_count() & 1
        if discrepancy:
            t = c
            c ^= b << (n_idx - m)
            if l <= n_idx // 2:
                l = n_idx + 1 - l
                m = n_idx
                b = t
    return l


def _frequency_monobit(bits: np.ndarray, alpha: float) -> Dict[str, Any]:
    n = bits.size
    if n == 0:
        return _skip("frequency_monobit", "empty_bitstream")
    transformed = 2 * bits.astype(np.int64) - 1
    s_obs = abs(transformed.sum()) / math.sqrt(n)
    p_value = erfc(s_obs / math.sqrt(2.0))
    return _single("frequency_monobit", p_value, alpha, n_bits=n, s_obs=float(s_obs))


def _block_frequency(bits: np.ndarray, alpha: float) -> Dict[str, Any]:
    n = bits.size
    if n < 128:
        return _skip("block_frequency", "requires_at_least_128_bits", n_bits=n)
    block_size = min(128, max(8, n // 100))
    n_blocks = n // block_size
    if n_blocks < 2:
        return _skip("block_frequency", "insufficient_full_blocks", n_bits=n, block_size=block_size)
    trimmed = bits[: n_blocks * block_size].reshape(n_blocks, block_size)
    pis = trimmed.mean(axis=1)
    chi2 = 4.0 * block_size * float(np.square(pis - 0.5).sum())
    p_value = _safe_gammaincc(n_blocks / 2.0, chi2 / 2.0)
    return _single("block_frequency", p_value, alpha, n_bits=n, block_size=block_size, n_blocks=n_blocks, chi_square=chi2)


def _runs(bits: np.ndarray, alpha: float) -> Dict[str, Any]:
    n = bits.size
    if n < 100:
        return _skip("runs", "requires_at_least_100_bits", n_bits=n)
    pi = float(bits.mean())
    tau = 2.0 / math.sqrt(n)
    if abs(pi - 0.5) >= tau:
        return _single("runs", 0.0, alpha, n_bits=n, pi=pi, tau=tau, precondition_met=False)
    v_obs = 1 + int(np.count_nonzero(bits[1:] != bits[:-1]))
    denom = 2.0 * math.sqrt(2.0 * n) * pi * (1.0 - pi)
    p_value = erfc(abs(v_obs - 2.0 * n * pi * (1.0 - pi)) / denom)
    return _single("runs", p_value, alpha, n_bits=n, pi=pi, v_obs=v_obs, precondition_met=True)


def _longest_run(bits: np.ndarray, alpha: float) -> Dict[str, Any]:
    n = bits.size
    if n < 128:
        return _skip("longest_run_of_ones", "requires_at_least_128_bits", n_bits=n)
    if n < 6272:
        block_size = 8
        categories = [1, 2, 3]
        probs = [0.2148, 0.3672, 0.2305, 0.1875]
    elif n < 750000:
        block_size = 128
        categories = [4, 5, 6, 7, 8]
        probs = [0.1174, 0.2430, 0.2493, 0.1752, 0.1027, 0.1124]
    else:
        block_size = 10000
        categories = [10, 11, 12, 13, 14, 15]
        probs = [0.0882, 0.2092, 0.2483, 0.1933, 0.1208, 0.0675, 0.0727]
    n_blocks = n // block_size
    if n_blocks == 0:
        return _skip("longest_run_of_ones", "insufficient_full_blocks", n_bits=n, block_size=block_size)
    trimmed = bits[: n_blocks * block_size].reshape(n_blocks, block_size)
    counts = np.zeros(len(probs), dtype=np.int64)
    for block in trimmed:
        value = _longest_run_in_block(block)
        if value <= categories[0]:
            counts[0] += 1
        elif value >= categories[-1] + 1:
            counts[-1] += 1
        else:
            matched = False
            for idx, cat in enumerate(categories[1:], start=1):
                if value == cat:
                    counts[idx] += 1
                    matched = True
                    break
            if not matched:
                counts[-2] += 1
    expected = n_blocks * np.array(probs)
    chi2 = float(((counts - expected) ** 2 / expected).sum())
    p_value = _safe_gammaincc((len(probs) - 1) / 2.0, chi2 / 2.0)
    return _single(
        "longest_run_of_ones",
        p_value,
        alpha,
        n_bits=n,
        block_size=block_size,
        n_blocks=n_blocks,
        category_counts=counts.tolist(),
    )


def _binary_matrix_rank(bits: np.ndarray, alpha: float) -> Dict[str, Any]:
    n = bits.size
    matrix_bits = 32 * 32
    n_matrices = n // matrix_bits
    if n_matrices == 0:
        return _skip("binary_matrix_rank", "requires_at_least_1024_bits", n_bits=n)
    trimmed = bits[: n_matrices * matrix_bits].reshape(n_matrices, 32, 32)
    row_weights = (1 << np.arange(31, -1, -1, dtype=np.uint64)).astype(np.uint64)
    full_rank = 0
    rank31 = 0
    for matrix in trimmed:
        row_masks = matrix.astype(np.uint64, copy=False) @ row_weights
        rank = _gf2_rank_u32(row_masks)
        if rank == 32:
            full_rank += 1
        elif rank == 31:
            rank31 += 1
    remainder = n_matrices - full_rank - rank31
    probs = np.array([0.2888, 0.5776, 0.1336], dtype=np.float64)
    counts = np.array([full_rank, rank31, remainder], dtype=np.float64)
    expected = probs * n_matrices
    chi2 = float(((counts - expected) ** 2 / expected).sum())
    p_value = math.exp(-chi2 / 2.0)
    return _single(
        "binary_matrix_rank",
        p_value,
        alpha,
        n_bits=n,
        n_matrices=n_matrices,
        rank_32=full_rank,
        rank_31=rank31,
        rank_le_30=remainder,
    )


def _discrete_fourier_transform(bits: np.ndarray, alpha: float) -> Dict[str, Any]:
    n = bits.size
    if n < 1000:
        return _skip("discrete_fourier_transform", "requires_at_least_1000_bits", n_bits=n)
    sequence = 2 * bits.astype(np.int16) - 1
    spectrum = np.fft.fft(sequence)
    magnitudes = np.abs(spectrum[: n // 2])
    threshold = math.sqrt(math.log(1.0 / 0.05) * n)
    expected = 0.95 * n / 2.0
    observed = float(np.count_nonzero(magnitudes < threshold))
    deviation = (observed - expected) / math.sqrt(n * 0.95 * 0.05 / 4.0)
    p_value = erfc(abs(deviation) / math.sqrt(2.0))
    return _single(
        "discrete_fourier_transform",
        p_value,
        alpha,
        n_bits=n,
        threshold=threshold,
        expected_below_threshold=expected,
        observed_below_threshold=observed,
    )


def _non_overlapping_template_matching(bits: np.ndarray, alpha: float) -> Dict[str, Any]:
    n = bits.size
    template_length = 9
    if n < 8 * 128:
        return _skip("non_overlapping_template_matching", "requires_at_least_1024_bits", n_bits=n, template_length=template_length)
    n_blocks = 8
    block_len = n // n_blocks
    blocks = bits[: block_len * n_blocks].reshape(n_blocks, block_len)
    mu = (block_len - template_length + 1) / (2.0**template_length)
    var = block_len * ((1.0 / (2.0**template_length)) - ((2.0 * template_length - 1.0) / (2.0 ** (2 * template_length))))
    if var <= 0:
        return _skip("non_overlapping_template_matching", "non_positive_variance", n_bits=n, template_length=template_length)
    p_values: List[float] = []
    details: Dict[str, Any] = {
        "template_length": template_length,
        "n_blocks": n_blocks,
        "block_length": block_len,
        "templates": {},
    }
    for template, template_bits in _NON_OVERLAPPING_TEMPLATE_BITS.items():
        counts = [_count_non_overlapping_matches(block, template_bits) for block in blocks]
        chi2 = float(sum(((count - mu) ** 2) / var for count in counts))
        p_value = _safe_gammaincc(n_blocks / 2.0, chi2 / 2.0)
        p_values.append(p_value)
        details["templates"][template] = {"p_value": p_value, "counts": counts}
    return _multi("non_overlapping_template_matching", p_values, alpha, **details)


def _overlapping_template_matching(bits: np.ndarray, alpha: float) -> Dict[str, Any]:
    n = bits.size
    template_length = 9
    block_len = 1032
    n_blocks = n // block_len
    if n_blocks == 0:
        return _skip("overlapping_template_matching", "requires_at_least_1032_bits", n_bits=n, template_length=template_length)
    blocks = bits[: n_blocks * block_len].reshape(n_blocks, block_len)
    counts = [int(np.count_nonzero(_window_matches(block, _OVERLAPPING_TEMPLATE_BITS))) for block in blocks]
    lam = (block_len - template_length + 1) / (2.0**template_length)
    eta = lam / 2.0
    probs = [math.exp(-eta)]
    for bucket in range(1, 5):
        probs.append(math.exp(-eta) * eta / (2.0**bucket))
    probs.append(1.0 - sum(probs))
    categories = np.zeros(6, dtype=np.int64)
    for count in counts:
        categories[min(count, 5)] += 1
    expected = n_blocks * np.array(probs, dtype=np.float64)
    chi2 = float(((categories - expected) ** 2 / expected).sum())
    p_value = _safe_gammaincc(5.0 / 2.0, chi2 / 2.0)
    return _single(
        "overlapping_template_matching",
        p_value,
        alpha,
        n_bits=n,
        template="111111111",
        block_length=block_len,
        n_blocks=n_blocks,
        category_counts=categories.tolist(),
    )


_UNIVERSAL_PARAMS = [
    (1059061760, 16, 15.167379, 3.421),
    (496435200, 15, 14.167488, 3.419),
    (231669760, 14, 13.167693, 3.416),
    (107560960, 13, 12.168070, 3.410),
    (49643520, 12, 11.168765, 3.401),
    (22753280, 11, 10.170032, 3.384),
    (10342400, 10, 9.1723243, 3.356),
    (4654080, 9, 8.1764248, 3.311),
    (2068480, 8, 7.1836656, 3.238),
    (904960, 7, 6.1962507, 3.125),
    (387840, 6, 5.2177052, 2.954),
]


def _maurers_universal(bits: np.ndarray, alpha: float) -> Dict[str, Any]:
    n = bits.size
    params = next((item for item in _UNIVERSAL_PARAMS if n >= item[0]), None)
    if params is None:
        return _skip("maurers_universal", "requires_at_least_387840_bits", n_bits=n)
    _, block_length, expected_value, variance = params
    q = 10 * (2**block_length)
    k = n // block_length - q
    if k <= 0:
        return _skip("maurers_universal", "insufficient_blocks_after_training", n_bits=n, block_length=block_length)
    trimmed = bits[: (q + k) * block_length]
    blocks = trimmed.reshape(q + k, block_length)
    weights = (1 << np.arange(block_length - 1, -1, -1, dtype=np.uint32)).astype(np.uint32)
    values = (blocks.astype(np.uint32) @ weights).tolist()
    table = [0] * (2**block_length)
    for idx in range(q):
        table[values[idx]] = idx + 1
    total = 0.0
    for idx in range(q, q + k):
        value = values[idx]
        last = table[value]
        distance = idx + 1 - last
        table[value] = idx + 1
        total += math.log2(distance)
    fn = total / k
    c = 0.7 - 0.8 / block_length + (4.0 + 32.0 / block_length) * (k ** (-3.0 / block_length)) / 15.0
    sigma = c * math.sqrt(variance / k)
    p_value = erfc(abs(fn - expected_value) / (math.sqrt(2.0) * sigma))
    return _single("maurers_universal", p_value, alpha, n_bits=n, block_length=block_length, q=q, k=k, fn=fn)


def _linear_complexity(bits: np.ndarray, alpha: float) -> Dict[str, Any]:
    n = bits.size
    block_size = 500 if n < 1_000_000 else 1000
    n_blocks = n // block_size
    if n_blocks == 0:
        return _skip("linear_complexity", "insufficient_full_blocks", n_bits=n, block_size=block_size)
    trimmed = bits[: n_blocks * block_size].reshape(n_blocks, block_size)
    mean = (
        block_size / 2.0
        + (9.0 + (-1.0) ** (block_size + 1)) / 36.0
        - ((block_size / 3.0) + (2.0 / 9.0)) / (2.0**block_size)
    )
    buckets = np.zeros(7, dtype=np.int64)
    for block in trimmed:
        complexity = _berlekamp_massey_int(block)
        t = ((-1) ** block_size) * (complexity - mean) + (2.0 / 9.0)
        if t <= -2.5:
            buckets[0] += 1
        elif t <= -1.5:
            buckets[1] += 1
        elif t <= -0.5:
            buckets[2] += 1
        elif t <= 0.5:
            buckets[3] += 1
        elif t <= 1.5:
            buckets[4] += 1
        elif t <= 2.5:
            buckets[5] += 1
        else:
            buckets[6] += 1
    probs = np.array([0.01047, 0.03125, 0.12500, 0.50000, 0.25000, 0.06250, 0.020833], dtype=np.float64)
    expected = n_blocks * probs
    chi2 = float(((buckets - expected) ** 2 / expected).sum())
    p_value = _safe_gammaincc(3.0, chi2 / 2.0)
    return _single(
        "linear_complexity",
        p_value,
        alpha,
        n_bits=n,
        block_size=block_size,
        n_blocks=n_blocks,
        category_counts=buckets.tolist(),
    )


def _serial(bits: np.ndarray, alpha: float) -> Dict[str, Any]:
    n = bits.size
    if n < 128:
        return _skip("serial", "requires_at_least_128_bits", n_bits=n)
    m = min(16, max(2, int(math.log2(n)) - 2))

    def psi(m_bits: int) -> float:
        if m_bits <= 0:
            return 0.0
        counts = _pattern_counts(bits, m_bits, circular=True)
        return (float((counts.astype(np.float64) ** 2).sum()) * (2**m_bits) / n) - n

    psi_m = psi(m)
    psi_m1 = psi(m - 1)
    psi_m2 = psi(m - 2)
    delta1 = psi_m - psi_m1
    delta2 = psi_m - 2.0 * psi_m1 + psi_m2
    p1 = _safe_gammaincc((2 ** (m - 1)) / 2.0, delta1 / 2.0)
    p2 = _safe_gammaincc((2 ** (m - 2)) / 2.0, delta2 / 2.0)
    return _multi("serial", [p1, p2], alpha, n_bits=n, m=m, delta_1=delta1, delta_2=delta2)


def _approximate_entropy(bits: np.ndarray, alpha: float) -> Dict[str, Any]:
    n = bits.size
    if n < 128:
        return _skip("approximate_entropy", "requires_at_least_128_bits", n_bits=n)
    m = min(10, max(2, int(math.log2(n)) - 6))

    def phi(m_bits: int) -> float:
        counts = _pattern_counts(bits, m_bits, circular=True).astype(np.float64)
        probs = counts / n
        probs = probs[probs > 0]
        return float(np.sum(probs * np.log(probs)))

    phi_m = phi(m)
    phi_m1 = phi(m + 1)
    ap_en = phi_m - phi_m1
    chi2 = 2.0 * n * (math.log(2.0) - ap_en)
    p_value = _safe_gammaincc(2 ** (m - 1), chi2 / 2.0)
    return _single("approximate_entropy", p_value, alpha, n_bits=n, m=m, approximate_entropy=ap_en)


def _cum_sums_p_value(sequence: np.ndarray) -> float:
    n = sequence.size
    cumulative = np.cumsum(sequence)
    z = int(np.max(np.abs(cumulative)))
    if z == 0:
        return 1.0
    sum_a = 0.0
    start = int(math.floor((-n / z + 1.0) / 4.0))
    end = int(math.floor((n / z - 1.0) / 4.0))
    for k in range(start, end + 1):
        sum_a += _NORMAL.cdf((4 * k + 1) * z / math.sqrt(n)) - _NORMAL.cdf((4 * k - 1) * z / math.sqrt(n))
    sum_b = 0.0
    start = int(math.floor((-n / z - 3.0) / 4.0))
    end = int(math.floor((n / z - 1.0) / 4.0))
    for k in range(start, end + 1):
        sum_b += _NORMAL.cdf((4 * k + 3) * z / math.sqrt(n)) - _NORMAL.cdf((4 * k + 1) * z / math.sqrt(n))
    return float(1.0 - sum_a + sum_b)


def _cumulative_sums(bits: np.ndarray, alpha: float) -> Dict[str, Any]:
    n = bits.size
    if n == 0:
        return _skip("cumulative_sums", "empty_bitstream")
    sequence = 2 * bits.astype(np.int16) - 1
    forward = _cum_sums_p_value(sequence)
    reverse = _cum_sums_p_value(sequence[::-1])
    return _multi("cumulative_sums", [forward, reverse], alpha, n_bits=n, forward_p_value=forward, reverse_p_value=reverse)


def _random_excursions(bits: np.ndarray, alpha: float) -> Dict[str, Any]:
    n = bits.size
    if n == 0:
        return _skip("random_excursions", "empty_bitstream")
    sequence = 2 * bits.astype(np.int16) - 1
    cumulative = np.concatenate([[0], np.cumsum(sequence), [0]])
    zero_positions = np.where(cumulative == 0)[0]
    n_cycles = zero_positions.size - 1
    if n_cycles < 500:
        return _skip("random_excursions", "requires_at_least_500_cycles", n_bits=n, cycles=n_cycles)
    states = [-4, -3, -2, -1, 1, 2, 3, 4]
    details: Dict[str, Any] = {"cycles": int(n_cycles), "states": {}}
    p_values: List[float] = []
    cycles = [cumulative[zero_positions[idx] : zero_positions[idx + 1] + 1] for idx in range(n_cycles)]
    for state in states:
        visits = np.zeros(6, dtype=np.int64)
        for cycle in cycles:
            count = int(np.count_nonzero(cycle == state))
            visits[min(count, 5)] += 1
        abs_state = abs(state)
        probs = np.array(
            [
                1.0 - 1.0 / (2.0 * abs_state),
                1.0 / (4.0 * abs_state * abs_state),
                (1.0 - 1.0 / (2.0 * abs_state)) / (4.0 * abs_state * abs_state),
                ((1.0 - 1.0 / (2.0 * abs_state)) ** 2) / (4.0 * abs_state * abs_state),
                ((1.0 - 1.0 / (2.0 * abs_state)) ** 3) / (4.0 * abs_state * abs_state),
                (1.0 / (2.0 * abs_state)) * ((1.0 - 1.0 / (2.0 * abs_state)) ** 4),
            ],
            dtype=np.float64,
        )
        expected = n_cycles * probs
        chi2 = float(((visits - expected) ** 2 / expected).sum())
        p_value = _safe_gammaincc(2.5, chi2 / 2.0)
        p_values.append(p_value)
        details["states"][str(state)] = {"p_value": p_value, "visit_buckets": visits.tolist()}
    return _multi("random_excursions", p_values, alpha, **details)


def _random_excursions_variant(bits: np.ndarray, alpha: float) -> Dict[str, Any]:
    n = bits.size
    if n == 0:
        return _skip("random_excursions_variant", "empty_bitstream")
    sequence = 2 * bits.astype(np.int16) - 1
    cumulative = np.concatenate([[0], np.cumsum(sequence), [0]])
    zero_positions = np.where(cumulative == 0)[0]
    n_cycles = zero_positions.size - 1
    if n_cycles < 500:
        return _skip("random_excursions_variant", "requires_at_least_500_cycles", n_bits=n, cycles=n_cycles)
    details: Dict[str, Any] = {"cycles": int(n_cycles), "states": {}}
    p_values: List[float] = []
    for state in list(range(-9, 0)) + list(range(1, 10)):
        count = int(np.count_nonzero(cumulative == state))
        denom = math.sqrt(2.0 * n_cycles * (4.0 * abs(state) - 2.0))
        p_value = erfc(abs(count - n_cycles) / denom)
        p_values.append(p_value)
        details["states"][str(state)] = {"p_value": p_value, "count": count}
    return _multi("random_excursions_variant", p_values, alpha, **details)


def run_full_nist_suite(keystream_bytes: bytes, alpha: float = NIST_DEFAULT_ALPHA) -> Dict[str, Any]:
    bits = _bits_from_bytes(keystream_bytes)
    runners = [
        ("frequency_monobit", _frequency_monobit),
        ("block_frequency", _block_frequency),
        ("runs", _runs),
        ("longest_run_of_ones", _longest_run),
        ("binary_matrix_rank", _binary_matrix_rank),
        ("discrete_fourier_transform", _discrete_fourier_transform),
        ("non_overlapping_template_matching", _non_overlapping_template_matching),
        ("overlapping_template_matching", _overlapping_template_matching),
        ("maurers_universal", _maurers_universal),
        ("linear_complexity", _linear_complexity),
        ("serial", _serial),
        ("approximate_entropy", _approximate_entropy),
        ("cumulative_sums", _cumulative_sums),
        ("random_excursions", _random_excursions),
        ("random_excursions_variant", _random_excursions_variant),
    ]
    started = perf_counter()
    tests: Dict[str, Dict[str, Any]] = {}
    summary = {"passed": 0, "failed": 0, "skipped": 0}
    failed_tests: List[str] = []

    for name, fn in runners:
        t0 = perf_counter()
        min_bits = _MIN_BITS_BY_TEST.get(name, 1)
        if bits.size < min_bits:
            result = _skip(name, "insufficient_bits", n_bits=int(bits.size), required_min_bits=int(min_bits))
        else:
            result = fn(bits, alpha)
        elapsed = perf_counter() - t0
        result["details"]["elapsed_s"] = elapsed
        tests[name] = result
        if result["status"] == "pass":
            summary["passed"] += 1
        elif result["status"] == "fail":
            summary["failed"] += 1
            failed_tests.append(name)
        else:
            summary["skipped"] += 1

    summary["failed_tests"] = failed_tests
    summary["total_runtime_s"] = perf_counter() - started
    return {"alpha": float(alpha), "n_bits": int(bits.size), "tests": tests, "summary": summary}


def flatten_nist_results(results: Dict[str, Any]) -> Dict[str, Any]:
    summary = results.get("summary", {})
    flat: Dict[str, Any] = {
        "nist_alpha": results.get("alpha"),
        "nist_passed_count": summary.get("passed"),
        "nist_failed_count": summary.get("failed"),
        "nist_skipped_count": summary.get("skipped"),
        "nist_total_runtime_s": summary.get("total_runtime_s"),
        "nist_failed_tests": ",".join(summary.get("failed_tests", [])),
    }
    tests = results.get("tests", {})
    for name in NIST_TEST_NAMES:
        payload = tests.get(name) or {}
        prefix = f"nist_{name}"
        flat[f"{prefix}_status"] = payload.get("status")
        flat[f"{prefix}_pass"] = payload.get("pass")
        flat[f"{prefix}_p_value"] = payload.get("p_value")
        flat[f"{prefix}_details_json"] = json.dumps(payload.get("details", {}), sort_keys=True)
    return flat
