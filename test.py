#!/usr/bin/env python3
import random
import pandas as pd
import sympy as sp
import mpmath as mp

# ------------------ config ------------------
SAMPLES = 20
DIGITS = 1000
EXACT_LIMIT = 1_000_000  # exact prime index for primes <= this value
OUTPUT_FILE = "200_digit_numbers_with_prime_factors_sorted.csv"
RNG_SEED = 42
# --------------------------------------------

mp.mp.dps = 50  # precision for li/logs

def rand_prime_with_digits(d: int) -> int:
    low = 10**(d - 1)
    high = 10**d - 1
    return int(sp.randprime(low, high + 1))

def make_factored_n_digits(n_digits: int):
    """Compose a number with exactly n_digits and known factorization."""
    t = random.choice([3, 4, 5])
    target = random.randint(n_digits - 20, n_digits - 5)

    parts, remaining = [], target
    for i in range(t - 1):
        max_for_this = remaining - 20 * (t - 1 - i)
        d = random.randint(20, max_for_this)
        parts.append(d)
        remaining -= d
    parts.append(remaining)
    random.shuffle(parts)

    # clamp digits to keep prime generation fast
    parts = [max(20, min(80, d)) for d in parts]

    primes = [rand_prime_with_digits(d) for d in parts]
    n = 1
    factor_map = {}
    for p in primes:
        n *= p
        factor_map[p] = factor_map.get(p, 0) + 1

    # pad with 10^k = 2^k * 5^k to reach exactly n_digits
    D = len(str(n))
    if D > n_digits:
        return make_factored_n_digits(n_digits)
    k = n_digits - D
    if k > 0:
        n *= (2**k) * (5**k)
        factor_map[2] = factor_map.get(2, 0) + k
        factor_map[5] = factor_map.get(5, 0) + k

    assert len(str(n)) == n_digits
    return n, sorted(factor_map.items())  # list[(prime, exponent)]

def prime_index_safe(p: int, exact_limit: int = EXACT_LIMIT):
    """
    Return (global_index, note), where note ∈ {"exact","approx"}.
    Uses exact primepi for small primes; approximates large ones to avoid MemoryError.
    """
    p = int(p)
    if p <= exact_limit:
        return int(sp.primepi(p)), "exact"
    # approximate with logarithmic integral
    # (π(x) ~ Li(x) is very accurate for large x)
    return int(mp.li(p)), "approx"

def generate_dataset(count=SAMPLES, n_digits=DIGITS, output_file=OUTPUT_FILE, seed=RNG_SEED):
    random.seed(seed)
    rows, seen = [], set()

    for i in range(count):
        n, factors = make_factored_n_digits(n_digits)
        while n in seen:
            n, factors = make_factored_n_digits(n_digits)
        seen.add(n)

        # Compute global prime indices (NOT positions) and sort by that index
        factors_with_idx = []
        for base, exp in factors:
            idx, note = prime_index_safe(base)
            factors_with_idx.append((base, exp, idx, note))
        factors_with_idx.sort(key=lambda t: t[2])  # sort by global index

        flat = [i + 1, str(n)]
        for base, exp, idx, note in factors_with_idx:
            flat.extend([str(base), exp, idx, note])
        rows.append(flat)

    # Determine maximum number of factor slots and build headers
    max_slots = max((len(r) - 2) // 4 for r in rows)
    columns = ["#", f"{n_digits}-digit Number"]
    for j in range(max_slots):
        k = j + 1
        columns += [f"Factor {k}", f"Exponent {k}", f"PrimeIndex {k}", f"PrimeIndexNote {k}"]

    # Pad rows and save
    target_len = 2 + 4 * max_slots
    for r in rows:
        r += [""] * (target_len - len(r))

    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(output_file, index=False)
    print(f"CSV saved to {output_file}")

if __name__ == "__main__":
    # quick sanity check: 197 is the 45th prime
    assert sp.primepi(197) == 45, "Sanity check failed: 197 should be the 45th prime"
    generate_dataset()
