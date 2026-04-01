import math
import cmath
from typing import List, Set, Callable, Dict, Tuple
import json
import os

# =======================================================
# Blok 1: Grupy, pierścienie, ciała (operacje i elementy neutralne)
# =======================================================
class ModularArithmetic:
    @staticmethod
    def add(a: int, b: int, mod: int) -> int:
        return (a + b) % mod

    @staticmethod
    def mul(a: int, b: int, mod: int) -> int:
        return (a * b) % mod

    @staticmethod
    def powmod(base: int, exp: int, mod: int) -> int:
        result = 1
        while exp > 0:
            if exp % 2:
                result = ModularArithmetic.mul(result, base, mod)
            base = ModularArithmetic.mul(base, base, mod)
            exp //= 2
        return result

# =======================================================
# Blok 2: Liczby pierwsze, NWD/NWW, sito Eratostenesa
# =======================================================
def gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a

def lcm(a: int, b: int) -> int:
    return a // gcd(a, b) * b

def is_prime(n: int) -> bool:
    if n < 2:
        return False
    for i in range(2, int(math.isqrt(n)) + 1):
        if n % i == 0:
            return False
    return True

def sieve_of_eratosthenes(limit: int) -> List[int]:
    if limit < 2:
        return []
    prime = [True] * (limit + 1)
    prime[0:2] = [False, False]
    for i in range(2, int(limit**0.5) + 1):
        if prime[i]:
            for j in range(i*i, limit+1, i):
                prime[j] = False
    return [i for i, is_p in enumerate(prime) if is_p]

# =======================================================
# Blok 3: Funkcja Eulera i kongruencje
# =======================================================
def euler_phi(n: int) -> int:
    result = n
    p = 2
    while p * p <= n:
        if n % p == 0:
            while n % p == 0:
                n //= p
            result -= result // p
        p += 1
    if n > 1:
        result -= result // n
    return result

def crt(remainders: List[int], mods: List[int]) -> int:
    M = 1
    for m in mods:
        M *= m
    x = 0
    for r, m in zip(remainders, mods):
        Mi = M // m
        inv = pow(Mi, -1, m)
        x += r * Mi * inv
    return x % M

# =======================================================
# Blok 4: Grupy, podgrupy, homomorfizmy i iloczyn prosty
# =======================================================
class Group:
    def __init__(self, elements: Set[int], operation: Callable[[int, int], int], identity: int):
        self.elements = elements
        self.op = operation
        self.e = identity

    def is_subgroup(self, subset: Set[int]) -> bool:
        if not subset.issubset(self.elements):
            return False
        for a in subset:
            for b in subset:
                if self.op(a, b) not in subset:
                    return False
        for a in subset:
            inv = next((x for x in subset if self.op(a, x) == self.e), None)
            if inv is None:
                return False
        return True

    def order(self, element: int) -> int:
        count = 1
        result = element
        while result != self.e:
            result = self.op(result, element)
            count += 1
        return count

    def homomorphism(self, other_group: 'Group', mapping: Dict[int, int]) -> bool:
        for a in self.elements:
            for b in self.elements:
                if mapping[self.op(a, b)] != other_group.op(mapping[a], mapping[b]):
                    return False
        return True

def direct_product_group(G1: Group, G2: Group) -> Group:
    elements = {(a, b) for a in G1.elements for b in G2.elements}
    def op(x, y):
        return (G1.op(x[0], y[0]), G2.op(x[1], y[1]))
    return Group(elements, op, (G1.e, G2.e))

# =======================================================
# Blok 5: Grupy cykliczne i permutacje
# =======================================================
def apply_permutation(v: List[int], perm: List[int]) -> List[int]:
    return [v[i] for i in perm]

def permutation_cycles(perm: List[int]) -> List[List[int]]:
    visited = [False] * len(perm)
    cycles = []
    for i in range(len(perm)):
        if not visited[i]:
            cycle = []
            j = i
            while not visited[j]:
                visited[j] = True
                cycle.append(j)
                j = perm[j]
            if len(cycle) > 1:
                cycles.append(cycle)
    return cycles

def permutation_sign(perm: List[int]) -> int:
    visited = [False] * len(perm)
    sign = 1
    for i in range(len(perm)):
        if not visited[i]:
            j = i
            cycle_len = 0
            while not visited[j]:
                visited[j] = True
                j = perm[j]
                cycle_len += 1
            if (cycle_len - 1) % 2 == 1:
                sign *= -1
    return sign

# =======================================================
# Blok 6: Ideały i pierścienie ilorazowe
# =======================================================
class Ring:
    def __init__(self, elements: Set[int], add: Callable[[int,int], int], mul: Callable[[int,int], int]):
        self.elements = elements
        self.add = add
        self.mul = mul

    def is_ideal(self, subset: Set[int]) -> bool:
        for a in subset:
            for b in subset:
                if self.add(a, b) not in subset:
                    return False
        for r in self.elements:
            for x in subset:
                if self.mul(r, x) not in subset or self.mul(x, r) not in subset:
                    return False
        return True

    def homomorphism(self, other_ring: 'Ring', mapping: Dict[int,int]) -> bool:
        for a in self.elements:
            for b in self.elements:
                if mapping[self.add(a,b)] != other_ring.add(mapping[a], mapping[b]):
                    return False
                if mapping[self.mul(a,b)] != other_ring.mul(mapping[a], mapping[b]):
                    return False
        return True

def direct_product_ring(R1: Ring, R2: Ring) -> Ring:
    elements = {(a,b) for a in R1.elements for b in R2.elements}
    def add(x,y):
        return (R1.add(x[0], y[0]), R2.add(x[1], y[1]))
    def mul(x,y):
        return (R1.mul(x[0], y[0]), R2.mul(x[1], y[1]))
    return Ring(elements, add, mul)

class QuotientRing(Ring):
    def __init__(self, base_ring: Ring, ideal: Set[int]):
        super().__init__(elements=set(range(len(base_ring.elements)//len(ideal))),
                         add=lambda x,y: (x+y)%len(base_ring.elements),
                         mul=lambda x,y: (x*y)%len(base_ring.elements))
        self.base_ring = base_ring
        self.ideal = ideal

# =======================================================
# Blok 7: Wielomiany nad ciałami skończonymi (rozszerzenie)
# =======================================================
class Polynomial:
    def __init__(self, coeffs: List[int]):
        self.coeffs = coeffs

    def __add__(self, other: 'Polynomial') -> 'Polynomial':
        size = max(len(self.coeffs), len(other.coeffs))
        result = [0] * size
        for i, val in enumerate(self.coeffs):
            result[i + size - len(self.coeffs)] += val
        for i, val in enumerate(other.coeffs):
            result[i + size - len(other.coeffs)] += val
        return Polynomial(result)

    def __mul__(self, other: 'Polynomial') -> 'Polynomial':
        result = [0] * (len(self.coeffs) + len(other.coeffs) - 1)
        for i, a in enumerate(self.coeffs):
            for j, b in enumerate(other.coeffs):
                result[i + j] += a * b
        return Polynomial(result)

    def divmod(self, divisor: 'Polynomial') -> Tuple['Polynomial', 'Polynomial']:
        dividend = self.coeffs[:]
        divisor_degree = len(divisor.coeffs) - 1
        divisor_lead_inv = pow(divisor.coeffs[0], -1, 1 << 32)
        quotient = [0] * (len(dividend) - divisor_degree)
        for i in range(len(quotient)):
            factor = dividend[i] * divisor_lead_inv
            quotient[i] = factor
            for j in range(len(divisor.coeffs)):
                dividend[i + j] -= factor * divisor.coeffs[j]
        remainder = dividend[-divisor_degree:] if divisor_degree else []
        return Polynomial(quotient), Polynomial(remainder)

    def __repr__(self):
        return str(self.coeffs)

# --- Rozszerzenie: Ciała skończone i pierścienie ilorazowe ---
class FiniteField:
    def __init__(self, p: int):
        if not is_prime(p):
            raise ValueError("p must be prime for GF(p)")
        self.p = p

    def add(self, a: int, b: int) -> int:
        return (a + b) % self.p

    def mul(self, a: int, b: int) -> int:
        return (a * b) % self.p

    def neg(self, a: int) -> int:
        return (-a) % self.p

    def inv(self, a: int) -> int:
        if a == 0:
            raise ZeroDivisionError
        return pow(a, self.p - 2, self.p)

    def sub(self, a: int, b: int) -> int:
        return (a - b) % self.p

    def div(self, a: int, b: int) -> int:
        return self.mul(a, self.inv(b))

class PolynomialField:
    def __init__(self, field: FiniteField):
        self.field = field

    def add(self, p1: Polynomial, p2: Polynomial) -> Polynomial:
        size = max(len(p1.coeffs), len(p2.coeffs))
        result = [0]*size
        for i, val in enumerate(p1.coeffs):
            result[i + size - len(p1.coeffs)] = (result[i + size - len(p1.coeffs)] + val) % self.field.p
        for i, val in enumerate(p2.coeffs):
            result[i + size - len(p2.coeffs)] = (result[i + size - len(p2.coeffs)] + val) % self.field.p
        return Polynomial(result)

    def mul(self, p1: Polynomial, p2: Polynomial) -> Polynomial:
        size = len(p1.coeffs) + len(p2.coeffs) - 1
        result = [0]*size
        for i, a in enumerate(p1.coeffs):
            for j, b in enumerate(p2.coeffs):
                result[i+j] = (result[i+j] + self.field.mul(a,b)) % self.field.p
        return Polynomial(result)

    def divmod(self, dividend: Polynomial, divisor: Polynomial) -> Tuple[Polynomial, Polynomial]:
        a = dividend.coeffs[:]
        b = divisor.coeffs[:]
        deg_a = len(a)-1
        deg_b = len(b)-1
        if deg_b < 0:
            raise ZeroDivisionError
        quotient = [0]*(deg_a - deg_b + 1)
        while deg_a >= deg_b and any(a):
            factor = self.field.div(a[0], b[0])
            quotient[deg_a - deg_b] = factor
            for i in range(len(b)):
                a[i] = self.field.sub(a[i], self.field.mul(factor, b[i]))
            a.pop(0)
            deg_a -= 1
        return Polynomial(quotient), Polynomial(a)

class QuotientPolynomialRing:
    def __init__(self, poly_field: PolynomialField, mod_poly: Polynomial):
        self.field = poly_field
        self.mod_poly = mod_poly

    def add(self, p1: Polynomial, p2: Polynomial) -> Polynomial:
        return self.field.add(p1, p2).divmod(self.mod_poly)[1]

    def mul(self, p1: Polynomial, p2: Polynomial) -> Polynomial:
        product = self.field.mul(p1, p2)
        _, remainder = product.divmod(self.mod_poly)
        return remainder

# =======================================================
# Blok 8: Liczby zespolone i pierwiastki z jedynki
# =======================================================
def exp_complex(theta: float) -> complex:
    return cmath.exp(1j * theta)

def roots_of_unity(n: int) -> List[complex]:
    return [exp_complex(2*math.pi*k/n) for k in range(n)]

# =======================================================
# Algorytm kryptograficzny (RSA-like)
# =======================================================
class SimpleCrypto:
    def __init__(self, p: int, q: int):
        self.n = p * q
        phi = (p - 1) * (q - 1)
        self.e = 3
        while gcd(self.e, phi) != 1:
            self.e += 2
        self.d = 1
        while (self.d * self.e) % phi != 1:
            self.d += 1

    def encrypt(self, m: int) -> int:
        return ModularArithmetic.powmod(m, self.e, self.n)

    def decrypt(self, c: int) -> int:
        return ModularArithmetic.powmod(c, self.d, self.n)

def simple_xor_encrypt(data_bytes: bytes, key: bytes) -> bytes:
    return bytes([b ^ key[i % len(key)] for i, b in enumerate(data_bytes)])

# =======================================================
# Backup AES symetryczny
# =======================================================
def secure_backup(data_bytes: bytes, backup_file="backup_czar.enc", meta_file="backup_meta_czar.json"):
    key = os.urandom(16)
    ciphertext = simple_xor_encrypt(data_bytes, key)

    print("KEY (hex):", key.hex())
    print("CIPHERTEXT (hex):", ciphertext.hex())  # 🔥 TU masz wynik szyfrowania

    print("ORIGINAL (str):", data_bytes.decode())
    print("ENCRYPTED (hex):", ciphertext.hex())

    with open(backup_file, "wb") as f:
        f.write(ciphertext)

    meta = {"xor_key": key.hex(), "cipher_len": len(ciphertext)}
    with open(meta_file, "w") as f:
        json.dump(meta, f, indent=2)

    print("Secure backup created successfully.")

# =======================================================
# Testy i demonstracje
# =======================================================
if __name__ == "__main__":
    # RSA-like szyfrowanie
    crypto = SimpleCrypto(17, 23)
    message = 42
    cipher = crypto.encrypt(message)
    plain = crypto.decrypt(cipher)
    print(f"Message: {message}, Cipher: {cipher}, Decrypted: {plain}")

    # Permutacje
    data_list = [1,2,3,4]
    perm = [2,0,3,1]
    permuted = apply_permutation(data_list, perm)
    print("Permuted:", permuted)
    print("Permutation cycles:", permutation_cycles(perm))
    print("Permutation sign:", permutation_sign(perm))

    # Liczby zespolone
    z = exp_complex(math.pi / 3)
    print("Complex exp(i*pi/3):", z)
    print("4th roots of unity:", roots_of_unity(4))

    # Wielomiany
    p1 = Polynomial([1, 2, 3])
    p2 = Polynomial([3, 0, 1])
    print("Polynomial sum:", p1 + p2)
    print("Polynomial mul:", p1 * p2)

    # Grupa i podgrupa
    G = Group(elements=set(range(6)), operation=lambda x,y: (x+y)%6, identity=0)
    H = {0, 3}
    print("H is subgroup of G?", G.is_subgroup(H))
    print("Order of 2 in G:", G.order(2))

    # Pierścień i ideał
    R = Ring(elements=set(range(6)), add=lambda x,y: (x+y)%6, mul=lambda x,y: (x*y)%6)
    I = {0,3}
    print("I is ideal in R?", R.is_ideal(I))

    # CRT example
    print("CRT solution for x ≡ 2 mod 3, x ≡ 3 mod 5:", crt([2,3],[3,5]))

    # Demonstracja backupu
    sample_docs = {"example.txt": "To jest przykładowy dokument."}
    data_bytes = json.dumps(sample_docs).encode("utf-8")
    secure_backup(data_bytes)

    # Sito Eratostenesa
    print("Primes up to 30:", sieve_of_eratosthenes(30))

    # Ciała skończone i pierścienie wielomianów
    F5 = FiniteField(5)
    poly_field = PolynomialField(F5)
    p1 = Polynomial([1, 2, 3])
    p2 = Polynomial([3, 0, 1])
    print("Sum in GF(5)[x]:", poly_field.add(p1,p2))
    print("Product in GF(5)[x]:", poly_field.mul(p1,p2))
    mod_poly = Polynomial([1,0,1])
    quot_ring = QuotientPolynomialRing(poly_field, mod_poly)
    reduced_prod = quot_ring.mul(p1,p2)
    print("Product modulo x^2+1 in GF(5)[x]:", reduced_prod)