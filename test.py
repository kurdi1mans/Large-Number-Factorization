#!/usr/bin/env python3
"""
Generate a CSV with 20 random 200-digit integers and their prime factorizations.
For each factor we include (Factor i, Exponent i, PrimeIndex i, PrimeIndexNote i).

- Exact prime indices are used for small primes (<= EXACT_LIMIT).
- For larger primes, we use a fast approximation (logarithmic integral) to avoid MemoryError.
"""

import random
import math
import pandas as pd
import sympy as sp
import mpmath as mp

# ------------------ config ------------------
SAMPLES = 20
DIGITS = 200
EXACT_LIMIT = 1_000_000  # use exact primepi for primes <= this value
OUTPUT_FILE = "200_digit_numbers_with_prime_factors_sorted.csv"
RNG_SEED = 42
# --------------------------------------------

mp.mp.dps = 50  # precision for mpmath (affects li, logs) — 50 digits is plenty

def rand_prime_with_digits(d: int) -> int:
    """Return a random prime with exactly d decimal digits."""
    low = 10**(d - 1)
    high = 10**d - 1
    return int(sp.randprime(low, high + 1))

def make_factored_n_digits(n_digits: int) -> tuple[int, list[tuple[int, int]]]:
    """
    Build a composite number with known prime factorization and exactly n_digits digits.
    Strategy:
      - Multiply 3–5 random large primes whose total digits sum to ~ (n_digits - 5..20).
      - Pad with 2^k * 5^k to reach exactly n_digits.
    Returns:
      n (int), factors (sorted list of (prime, exponent)).
    """
    t = random.choice([3, 4, 5])
    target = random.randint(n_digits - 20, n_digits - 5)  # digit budget for large primes

    # split target into t parts (min 20 digits each to keep generation quick)
    parts = []
    remaining = target
    for i in range(t - 1):
        max_for_this = remaining - 20 * (t - 1 - i)
        d = random.randint(20, max_for_this)
        parts.append(d)
        remaining -= d
    parts.append(remaining)
    random.shuffle(parts)

    # clamp digits for speed (20..80)
    parts = [max(20, min(80, d)) for d in parts]

    primes = [rand_prime_with_digits(d) for d in parts]
    n = 1
    factor_map = {}
    for p in primes:
        n *= p
        factor_map[p] = factor_map.get(p, 0) + 1

    D = len(str(n))
    if D > n_digits:
        # extremely unlikely with the chosen targets; retry
        return make_factored_n_digits(n_digits)

    # pad to exact length with 10^k = 2^k * 5^k
    k = n_digits - D
    if k > 0:
        n *= (2 ** k) * (5 ** k)
        factor_map[2] = factor_map.get(2, 0) + k
        factor_map[5] = factor_map.get(5, 0) + k

    assert len(str(n)) == n_digits

    factors = sorted(factor_map.items())  # (prime, exponent)
    return n, factors

def prime_index_safe(p: int, exact_limit: int = EXACT_LIMIT) -> tuple[int, str]:
    """
    Return (index, note) where index is the position of prime p in the sequence of primes.
    - Exact for p <= exact_limit (uses sympy.primepi).
    - Approximate for larger p using logarithmic integral Li(p).
    """
    p = int(p)
    if p <= exact_limit:
        return int(sp.primepi(p)), "exact"

    # Approximation: π(p) ~ Li(p). When Li is unavailable/slow, a refined PNT can be used.
    # We’ll use mpmath.li for stability on large inputs.
    # For very small p we guard to avoid domain issues.
    if p < 17:
        return int(sp.primepi(p)), "exact"

    # Use li(p) with a mild correction (common: x/log x * (1 + 1/log x))
    # We'll take the integer part of li(p).
    approx = int(mp.li(p))
    return approx, "approx"

def generate_dataset(
    count: int = SAMPLES,
    n_digits: int = DIGITS,
    output_file: str = OUTPUT_FILE,
    exact_limit: int = EXACT_LIMIT,
    seed: int = RNG_SEED,
) -> None:
    random.seed(seed)
    rows = []
    seen = set()

    # First, build the base table with numbers and symbolic factorization strings
    for i in range(count):
        n, factors = make_factored_n_digits(n_digits)
        while n in seen:
            n, factors = make_factored_n_digits(n_digits)
        seen.add(n)

        # Sort factors by prime index (safe ordering)
        factors_with_idx = []
        for base, exp in factors:
            idx, note = prime_index_safe(base, exact_limit=exact_limit)
            factors_with_idx.append((base, exp, idx, note))
        factors_with_idx.sort(key=lambda t: t[2])  # by prime index

        # Flatten row
        row = [i + 1, str(n)]
        for base, exp, idx, note in factors_with_idx:
            row.extend([str(base), exp, idx, note])
        rows.append(row)

    # Determine the maximum number of factor slots needed
    max_slots = max((len(r) - 2) // 4 for r in rows)  # each slot: Factor, Exponent, PrimeIndex, PrimeIndexNote

    # Build headers
    columns = ["#", f"{n_digits}-digit Number"]
    for j in range(max_slots):
        k = j + 1
        columns.extend([f"Factor {k}", f"Exponent {k}", f"PrimeIndex {k}", f"PrimeIndexNote {k}"])

    # Pad rows to equal length
    target_len = 2 + 4 * max_slots
    for r in rows:
        r += [""] * (target_len - len(r))

    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(output_file, index=False)
    print(f"CSV saved to {output_file}")

if __name__ == "__main__":
    generate_dataset()
