import sympy as sp
import pandas as pd
import random

def rand_prime_with_digits(d):
    """Return a random prime with exactly d decimal digits."""
    low = 10**(d-1)
    high = 10**d - 1
    return sp.randprime(low, high+1)

def make_factored_200_digit_number():
    """
    Build a composite number with known prime factorization that has exactly 200 digits.
    Strategy:
      - Multiply 3–5 random large primes whose digits total ~180–195.
      - Pad with 2^k * 5^k to reach exactly 200 digits.
    """
    t = random.choice([3,4,5])  # number of large primes
    target = random.randint(180, 195)  # digit budget for large primes

    # split into t parts (at least 20 digits each)
    parts = []
    remaining = target
    for i in range(t-1):
        max_for_this = remaining - 20*(t-1-i)
        d = random.randint(20, max_for_this)
        parts.append(d)
        remaining -= d
    parts.append(remaining)
    random.shuffle(parts)

    # generate primes
    primes = [rand_prime_with_digits(max(20, min(80, d))) for d in parts]
    n = 1
    factor_map = {}
    for p in primes:
        n *= p
        factor_map[p] = factor_map.get(p, 0) + 1

    # pad with powers of 10
    D = len(str(n))
    if D > 200:
        return make_factored_200_digit_number()
    k = 200 - D
    if k > 0:
        n *= (2**k) * (5**k)
        factor_map[2] = factor_map.get(2, 0) + k
        factor_map[5] = factor_map.get(5, 0) + k

    assert len(str(n)) == 200
    return n, sorted(factor_map.items())

def generate_dataset(count=20, output_file="200_digit_numbers_with_prime_factors_sorted.csv"):
    rows = []
    seen = set()

    for i in range(count):
        n, factors = make_factored_200_digit_number()
        while n in seen:  # avoid duplicates
            n, factors = make_factored_200_digit_number()
        seen.add(n)

        # sort by prime index
        factors_sorted = sorted(factors, key=lambda x: sp.primepi(x[0]))

        row = [i+1, str(n)]
        for base, exp in factors_sorted:
            prime_index = sp.primepi(base)
            row.extend([str(base), exp, prime_index])
        rows.append(row)

    # build column names
    max_factors = max(len(factors) for _, factors in [make_factored_200_digit_number() for _ in range(5)])
    columns = ["#", "200-digit Number"]
    for i in range(15):  # allow up to 15 factors; adjust if needed
        columns.extend([f"Factor {i+1}", f"Exponent {i+1}", f"PrimeIndex {i+1}"])

    df = pd.DataFrame(rows, columns=columns[:len(rows[0])])
    df.to_csv(output_file, index=False)
    print(f"CSV file saved as {output_file}")

if __name__ == "__main__":
    random.seed(42)
    generate_dataset()
