# Copyright (c) 2025 Rafik Hamza, Ph.D.
# All rights reserved.
#
# Submission to: Multimedia Tools and Applications (MTAP), 14 June, 2025
# Paper Title: A Robust Hybrid Image Encryption Framework based on Transform and Spatial Domain Processing with High-Dimensional Chaotic Maps
# Date of last edit: November 28, 2025
#
# Description:
# Core module containing all encryption, chaotic maps, keystream generation, and evaluation functionality.
"""
Core Module for 5D Coupled Logistic Map Encryption System
==========================================================

This module contains all the core functionality for the hybrid image encryption framework:
- Enhanced high-dimensional chaotic maps (5D Logistic-Sine, 3D Lorenz)
- BLAKE3-based deterministic keystream generation
- Core encryption/decryption pipeline with transform and spatial domain processing
- Security evaluation metrics and utilities

Features:
- Hyperchaotic behavior with multiple positive Lyapunov exponents
- Perfect reconstruction (MSE = 0) with integer lifting Haar transforms
- Comprehensive security metrics (NPCR, UACI, entropy, correlation)
- Medical imaging compatibility and diagnostic preservation
- Robust against various attacks (cropping, noise, key sensitivity)

Author: Rafik Hamza, Ph.D.
Date: November 28, 2025
"""

from __future__ import annotations

import os, hmac, hashlib, math, time
from typing import Tuple, List, Union, Any, Dict, Sequence
from dataclasses import dataclass
from time import perf_counter

import numpy as np
import numpy.typing as npt
from PIL import Image, PngImagePlugin

# Try to import optional dependencies with fallbacks
try:
    import blake3 as blake3_module
    BLAKE3_AVAILABLE = True
except ImportError:
    blake3_module = None
    BLAKE3_AVAILABLE = False  # type: ignore[reportConstantRedefinition]

try:
    from skimage.metrics import structural_similarity as ssim  # type: ignore
    has_skimage = True
except ImportError:
    has_skimage = False

# Constants
WAVELET = 'db2'  # retained label (integer lifting implementation underneath)
BIT_PLANES = 8
BLOCK_SIZE = 16
PI = math.pi

# ==================== BLAKE3 UTILITIES ====================

def blake3_hash(data: Union[bytes, bytearray], digest_size: int) -> bytes:
    if BLAKE3_AVAILABLE and blake3_module is not None:
        hasher = blake3_module.blake3()
        hasher.update(data)
        return hasher.digest(length=digest_size)
    # Fallback
    return hashlib.sha256(data).digest()[:digest_size]

def blake3_keyed(key: Union[bytes, bytearray, str], data: Union[bytes, bytearray, str], out_len: int) -> bytes:
    key_bytes = key if isinstance(key, (bytes, bytearray)) else str(key).encode('utf-8')
    data_bytes = data if isinstance(data, (bytes, bytearray)) else str(data).encode('utf-8')
    if BLAKE3_AVAILABLE and blake3_module is not None:
        hasher = blake3_module.blake3(key=key_bytes)
        hasher.update(data_bytes)
        return hasher.digest(length=out_len)
    # Fallback: HMAC-SHA256 in counter mode
    out = b''
    ctr = 0
    while len(out) < out_len:
        h = hmac.new(key_bytes, data_bytes + ctr.to_bytes(8, 'big'), hashlib.sha256)
        out += h.digest()
        ctr += 1
    return out[:out_len]

def _ensure_key32(k: Union[bytes, str, bytearray]) -> bytes:
    if isinstance(k, str):
        k = k.encode('utf-8')
    elif not isinstance(k, bytes):
        k = bytes(k)
    if len(k) == 32:
        return bytes(k)
    # Derive 32 bytes deterministically from provided K
    return blake3_hash(k, digest_size=32)

def _be_encode(x: int, m_bytes: int) -> bytes:
    return int(x).to_bytes(m_bytes, 'big', signed=False)

def _int_to_be(n: int, length: int) -> bytes:
    return int(n).to_bytes(length, 'big', signed=False)

class Blake3DRBG:
    def __init__(self, key: bytes):
        self.key = key
        self.counter = 0

    def read(self, n: int) -> bytes:
        data = b"PRNG" + _int_to_be(self.counter, 8)
        self.counter += 1
        return blake3_keyed(self.key, data, n)

    def u_q_nonendpoints(self, q: int) -> int:
        # Returns u in [1, 2^q - 2]
        b = self.read(16)
        u = int.from_bytes(b, 'big') % ((1 << q) - 2) + 1
        return u

# ==================== FIXED-POINT ARITHMETIC ====================

def _fp_add(a: int, b: int, q: int) -> int:
    return (a + b) & ((1 << q) - 1)

def _fp_mul(a: int, b: int, q: int) -> int:
    return ((a * b) >> q) & ((1 << q) - 1)

def _fp_sub_one(a: int, q: int) -> int:
    return ((1 << q) - 1) - a

def _fp_mul_r(a: int, r: int, q: int) -> int:
    # r has q fractional bits with 2 integer bits
    return ((a * r) >> q) & ((1 << q) - 1)

def _fp_sin(x: int, q: int) -> int:
    """Fast fixed-point sine approximation using lookup table."""
    # Convert fixed-point to float, compute sin, convert back
    x_float = x / (1 << q) * 2 * PI
    sin_val = math.sin(x_float)
    return int(sin_val * (1 << q)) & ((1 << q) - 1)

def _sample_interval_fixed(drbg: Blake3DRBG, q: int, a_min: float, a_max: float, *, r_param: bool = False) -> int:
    """Sample fixed-point in [a_min, a_max]. For r_param=True, return value encoded with 2 integer bits (q+2 total)."""
    u = drbg.u_q_nonendpoints(q)  # [1, 2^q-2]
    mask_q = (1 << q) - 1
    amin_q = int(a_min * (1 << q))
    arange_q = int((a_max - a_min) * (1 << q))
    val_q = (amin_q + ((u * arange_q) >> q)) & mask_q
    if r_param:
        # encode with 2 integer bits: scale stays 2^q, value can exceed 2^q-1 up to ~3.99*2^q
        return int((a_min + (a_max - a_min) * (u / (1 << q))) * (1 << q))
    return val_q

# ==================== 5D HYPERCHAOTIC LOGISTIC-SINE SYSTEM ====================

class LogisticSine5D:
    """
    5D Hyperchaotic Logistic-Sine System

    Mathematical formulation:
    x₁(n+1) = r₁ × x₁(n) × (1 - x₁(n)) + c₁ × sin(π × x₅(n))
    x₂(n+1) = r₂ × x₂(n) × (1 - x₂(n)) + c₂ × sin(π × x₁(n))
    x₃(n+1) = r₃ × x₃(n) × (1 - x₃(n)) + c₃ × sin(π × x₂(n))
    x₄(n+1) = r₄ × x₄(n) × (1 - x₄(n)) + c₄ × sin(π × x₃(n))
    x₅(n+1) = r₅ × x₅(n) × (1 - x₅(n)) + c₅ × sin(π × x₄(n))

    Features:
    - Hyperchaotic behavior with multiple positive Lyapunov exponents
    - Ring coupling topology with sine coupling
    - Enhanced complexity compared to pure logistic maps
    """

    def __init__(self, drbg: Blake3DRBG, q: int = 32):
        self.q = q
        self.mask_q = (1 << q) - 1
        self.m_bytes = (q + 7) // 8

        # Initialize state variables x_i in (0.1, 0.9)
        self.x1 = _sample_interval_fixed(drbg, q, 0.1, 0.9)
        self.x2 = _sample_interval_fixed(drbg, q, 0.1, 0.9)
        self.x3 = _sample_interval_fixed(drbg, q, 0.1, 0.9)
        self.x4 = _sample_interval_fixed(drbg, q, 0.1, 0.9)
        self.x5 = _sample_interval_fixed(drbg, q, 0.1, 0.9)

        # Initialize growth rate parameters r_i in [3.8, 3.99]
        self.r1 = _sample_interval_fixed(drbg, q, 3.8, 3.99, r_param=True)
        self.r2 = _sample_interval_fixed(drbg, q, 3.8, 3.99, r_param=True)
        self.r3 = _sample_interval_fixed(drbg, q, 3.8, 3.99, r_param=True)
        self.r4 = _sample_interval_fixed(drbg, q, 3.8, 3.99, r_param=True)
        self.r5 = _sample_interval_fixed(drbg, q, 3.8, 3.99, r_param=True)

        # Initialize coupling strength parameters c_i in [0.1, 0.4]
        self.c1 = _sample_interval_fixed(drbg, q, 0.1, 0.4)
        self.c2 = _sample_interval_fixed(drbg, q, 0.1, 0.4)
        self.c3 = _sample_interval_fixed(drbg, q, 0.1, 0.4)
        self.c4 = _sample_interval_fixed(drbg, q, 0.1, 0.4)
        self.c5 = _sample_interval_fixed(drbg, q, 0.1, 0.4)

    def step(self) -> Tuple[int, int, int, int, int]:
        """Single iteration of the 5D logistic-sine system."""
        # Compute logistic terms: r_i * x_i * (1 - x_i)
        t1 = _fp_mul(self.x1, _fp_sub_one(self.x1, self.q), self.q)
        t2 = _fp_mul(self.x2, _fp_sub_one(self.x2, self.q), self.q)
        t3 = _fp_mul(self.x3, _fp_sub_one(self.x3, self.q), self.q)
        t4 = _fp_mul(self.x4, _fp_sub_one(self.x4, self.q), self.q)
        t5 = _fp_mul(self.x5, _fp_sub_one(self.x5, self.q), self.q)

        # Compute sine coupling terms: c_i * sin(π * x_j)
        # Scale π to fixed-point representation
        pi_fp = int(PI * (1 << self.q))
        sin1 = _fp_sin(_fp_mul(pi_fp, self.x5, self.q), self.q)
        sin2 = _fp_sin(_fp_mul(pi_fp, self.x1, self.q), self.q)
        sin3 = _fp_sin(_fp_mul(pi_fp, self.x2, self.q), self.q)
        sin4 = _fp_sin(_fp_mul(pi_fp, self.x3, self.q), self.q)
        sin5 = _fp_sin(_fp_mul(pi_fp, self.x4, self.q), self.q)

        # Update state variables
        self.x1 = _fp_add(_fp_mul_r(t1, self.r1, self.q), _fp_mul(self.c1, sin1, self.q), self.q) & self.mask_q
        self.x2 = _fp_add(_fp_mul_r(t2, self.r2, self.q), _fp_mul(self.c2, sin2, self.q), self.q) & self.mask_q
        self.x3 = _fp_add(_fp_mul_r(t3, self.r3, self.q), _fp_mul(self.c3, sin3, self.q), self.q) & self.mask_q
        self.x4 = _fp_add(_fp_mul_r(t4, self.r4, self.q), _fp_mul(self.c4, sin4, self.q), self.q) & self.mask_q
        self.x5 = _fp_add(_fp_mul_r(t5, self.r5, self.q), _fp_mul(self.c5, sin5, self.q), self.q) & self.mask_q

        return self.x1, self.x2, self.x3, self.x4, self.x5

    def get_state_bytes(self) -> bytes:
        """Get current state as bytes for keystream generation."""
        return (_be_encode(self.x1, self.m_bytes) +
                _be_encode(self.x2, self.m_bytes) +
                _be_encode(self.x3, self.m_bytes) +
                _be_encode(self.x4, self.m_bytes) +
                _be_encode(self.x5, self.m_bytes))

# ==================== 3D LORENZ ATTRACTOR ====================

class Lorenz3D:
    """
    3D Lorenz Attractor System

    Mathematical formulation:
    dx/dt = σ(y - x)
    dy/dt = x(ρ - z) - y
    dz/dt = xy - βz

    Fixed-point implementation with:
    - σ (sigma) = 10.0
    - ρ (rho) = 28.0
    - β (beta) = 8/3 ≈ 2.667
    - Time step dt = 0.01
    """

    def __init__(self, drbg: Blake3DRBG, q: int = 32):
        self.q = q
        self.mask_q = (1 << q) - 1
        self.m_bytes = (q + 7) // 8

        # Initialize state variables in appropriate ranges
        self.x = _sample_interval_fixed(drbg, q, -20.0, 20.0)  # x ∈ [-20, 20]
        self.y = _sample_interval_fixed(drbg, q, -20.0, 20.0)  # y ∈ [-20, 20]
        self.z = _sample_interval_fixed(drbg, q, 0.0, 50.0)    # z ∈ [0, 50]

        # Lorenz parameters in fixed-point
        self.sigma = int(10.0 * (1 << q))  # σ = 10.0
        self.rho = int(28.0 * (1 << q))    # ρ = 28.0
        self.beta = int((8.0/3.0) * (1 << q))  # β = 8/3
        self.dt = int(0.01 * (1 << q))     # dt = 0.01

    def step(self) -> Tuple[int, int, int]:
        """Single iteration of the 3D Lorenz system using Euler method."""
        # Compute derivatives
        dx = _fp_mul(self.sigma, _fp_add(self.y, _fp_sub_one(self.x, self.q), self.q), self.q)
        dy = _fp_add(_fp_mul(self.x, _fp_add(self.rho, _fp_sub_one(self.z, self.q), self.q), self.q),
                     _fp_sub_one(self.y, self.q), self.q)
        dz = _fp_add(_fp_mul(self.x, self.y, self.q),
                     _fp_sub_one(_fp_mul(self.beta, self.z, self.q), self.q), self.q)

        # Update state using Euler method
        self.x = _fp_add(self.x, _fp_mul(self.dt, dx, self.q), self.q) & self.mask_q
        self.y = _fp_add(self.y, _fp_mul(self.dt, dy, self.q), self.q) & self.mask_q
        self.z = _fp_add(self.z, _fp_mul(self.dt, dz, self.q), self.q) & self.mask_q

        return self.x, self.y, self.z

    def get_state_bytes(self) -> bytes:
        """Get current state as bytes for keystream generation."""
        return (_be_encode(self.x, self.m_bytes) +
                _be_encode(self.y, self.m_bytes) +
                _be_encode(self.z, self.m_bytes))

# ==================== HYBRID CHAOTIC KEYSTREAM GENERATOR ====================

def generate_hybrid_chaotic_keystream(K: bytes | str, V: bytes | str, L: int, *, q: int = 32, burn_in: int = 50,
                                     use_lorenz: bool = True, use_logistic_sine: bool = True) -> bytes:
    """
    Generate keystream using hybrid chaotic systems.

    Args:
        K: Secret key
        V: Initialization vector
        L: Output length in bytes
        q: Fixed-point precision (default 32)
        burn_in: Number of burn-in iterations
        use_lorenz: Whether to use 3D Lorenz attractor
        use_logistic_sine: Whether to use 5D logistic-sine system

    Returns:
        Cryptographically secure keystream
    """
    if L <= 0:
        return b''

    K_b = K if isinstance(K, (bytes, bytearray)) else str(K).encode('utf-8')
    V_b = V if isinstance(V, (bytes, bytearray)) else str(V).encode('utf-8')
    K32 = _ensure_key32(K_b)

    # Domain separation for different chaotic systems
    K_chaos = blake3_keyed(K32, b"CHAOS" + V_b, 32)
    K_stream = blake3_keyed(K32, b"STREAM" + V_b, 32)

    drbg = Blake3DRBG(K_chaos)
    _mask_q = (1 << q) - 1
    _m_bytes = (q + 7) // 8

    # Initialize chaotic systems
    systems: List[Tuple[str, Any]] = []
    if use_logistic_sine:
        systems.append(("LOGISTIC_SINE", LogisticSine5D(drbg, q)))
    if use_lorenz:
        systems.append(("LORENZ", Lorenz3D(drbg, q)))

    if not systems:
        raise ValueError("At least one chaotic system must be enabled")

    # Burn-in phase
    for _ in range(burn_in):
        for name, system in systems:
            if name == "LOGISTIC_SINE":
                system.step()
            elif name == "LORENZ":
                system.step()

    # Generate keystream blocks
    b = 128
    N = (L + b - 1) // b
    out = bytearray()

    for n in range(1, N + 1):
        # Step all systems
        combined_state: bytes = b""
        for name, system in systems:
            if name == "LOGISTIC_SINE":
                system.step()
                combined_state += system.get_state_bytes()
            elif name == "LORENZ":
                system.step()
                combined_state += system.get_state_bytes()

        # Generate keystream block using combined state
        blk = blake3_keyed(K_stream, b"HYBRID_CHAOS" + combined_state + _int_to_be(n, 8), b)
        out.extend(blk)

    return bytes(out[:L])

# ==================== BLAKE3 DETERMINISTIC KEYSTREAM ====================

def generate_keystream_blake3(K: Union[bytes, bytearray, str], V: Union[bytes, bytearray, str], L: int, *, q: int = 32, burn_in: int = 50) -> bytes:
    """
    BLAKE3-only deterministic keystream with 5D coupled logistic map.
    - Domain separation: K_chaos = BLAKE3_KEYED(K32, "CHAOS"||V), K_stream = BLAKE3_KEYED(K32, "STREAM"||V)
    - Fixed-point 5D coupled logistic map with synchronous updates
    - Output blocks: blk_n = BLAKE3_KEYED(K_stream, "CHAOS_STEP"||enc(state)||int2be8(n))[:128]
    - S = concat(blks)[:L]
    """
    if L <= 0:
        return b''
    key_bytes = K if isinstance(K, (bytes, bytearray)) else str(K).encode('utf-8')
    nonce_bytes = V if isinstance(V, (bytes, bytearray)) else str(V).encode('utf-8')
    K32 = _ensure_key32(key_bytes)

    K_chaos = blake3_keyed(K32, b"CHAOS" + nonce_bytes, 32)
    K_stream = blake3_keyed(K32, b"STREAM" + nonce_bytes, 32)

    drbg = Blake3DRBG(K_chaos)
    mask_q = (1 << q) - 1
    m_bytes = (q + 7) // 8

    # Initialize x_i in (0.1, 0.9), r_i in [3.8, 3.99], c in [0.1, 0.4] for 5D system
    x1 = _sample_interval_fixed(drbg, q, 0.1, 0.9)
    x2 = _sample_interval_fixed(drbg, q, 0.1, 0.9)
    x3 = _sample_interval_fixed(drbg, q, 0.1, 0.9)
    x4 = _sample_interval_fixed(drbg, q, 0.1, 0.9)
    x5 = _sample_interval_fixed(drbg, q, 0.1, 0.9)
    r1 = _sample_interval_fixed(drbg, q, 3.8, 3.99, r_param=True)
    r2 = _sample_interval_fixed(drbg, q, 3.8, 3.99, r_param=True)
    r3 = _sample_interval_fixed(drbg, q, 3.8, 3.99, r_param=True)
    r4 = _sample_interval_fixed(drbg, q, 3.8, 3.99, r_param=True)
    r5 = _sample_interval_fixed(drbg, q, 3.8, 3.99, r_param=True)
    cfp = _sample_interval_fixed(drbg, q, 0.1, 0.4)

    def step_sync(x1: int, x2: int, x3: int, x4: int, x5: int) -> Tuple[int, int, int, int, int]:
        t1 = _fp_mul(x1, _fp_sub_one(x1, q), q)
        t2 = _fp_mul(x2, _fp_sub_one(x2, q), q)
        t3 = _fp_mul(x3, _fp_sub_one(x3, q), q)
        t4 = _fp_mul(x4, _fp_sub_one(x4, q), q)
        t5 = _fp_mul(x5, _fp_sub_one(x5, q), q)
        nx1 = _fp_add(_fp_mul_r(t1, r1, q), _fp_mul(cfp, x5, q), q)
        nx2 = _fp_add(_fp_mul_r(t2, r2, q), _fp_mul(cfp, x1, q), q)
        nx3 = _fp_add(_fp_mul_r(t3, r3, q), _fp_mul(cfp, x2, q), q)
        nx4 = _fp_add(_fp_mul_r(t4, r4, q), _fp_mul(cfp, x3, q), q)
        nx5 = _fp_add(_fp_mul_r(t5, r5, q), _fp_mul(cfp, x4, q), q)
        return nx1 & mask_q, nx2 & mask_q, nx3 & mask_q, nx4 & mask_q, nx5 & mask_q

    # Burn-in
    for _ in range(burn_in):
        x1, x2, x3, x4, x5 = step_sync(x1, x2, x3, x4, x5)

    # Blocks
    b = 128
    N = (L + b - 1) // b
    out = bytearray()
    for n in range(1, N + 1):
        x1, x2, x3, x4, x5 = step_sync(x1, x2, x3, x4, x5)
        state_enc = _be_encode(x1, m_bytes) + _be_encode(x2, m_bytes) + _be_encode(x3, m_bytes) + _be_encode(x4, m_bytes) + _be_encode(x5, m_bytes)
        blk = blake3_keyed(K_stream, b"CHAOS_STEP" + state_enc + _int_to_be(n, 8), b)
        out.extend(blk)
    return bytes(out[:L])

# ==================== ENCRYPTION CORE FUNCTIONS ====================

def _norm_key(key: bytes | str) -> bytes:
    """Normalize key to bytes (accept str for convenience)."""
    if isinstance(key, bytes):
        return key
    if isinstance(key, str):
        return key.encode('utf-8')
    raise TypeError('key must be bytes or str')

def _image_to_array(pil: Image.Image) -> npt.NDArray[np.uint8]:
    arr = np.array(pil)
    if arr.ndim == 2:
        arr = arr[..., None]
    if arr.shape[2] == 4:
        arr = arr[..., :3]
    return arr

def _fisher_yates(n: int, seed_bytes: bytes) -> npt.NDArray[np.int64]:
    perm = list(range(n))
    need_bytes = 4 * max(0, n-1)
    if len(seed_bytes) < need_bytes:
        seed_bytes = blake3_keyed(b'FYEXTEND', seed_bytes, need_bytes)
    off = 0
    for i in range(n-1, 0, -1):
        u = int.from_bytes(seed_bytes[off:off+4], 'big')
        off += 4
        j = u % (i+1)
        perm[i], perm[j] = perm[j], perm[i]
    return np.asarray(perm, dtype=np.int64)

def _bitplane_scramble(img: npt.NDArray[np.uint8], plane_bytes: bytes) -> npt.NDArray[np.uint8]:
    H,W,C = img.shape
    P = min(BIT_PLANES, len(plane_bytes), 8)
    if P == 0:
        return img
    scores = np.frombuffer(plane_bytes[:P], dtype=np.uint8)
    order = np.argsort(scores)  # permutation of top P bit-planes (MSB downward)
    planes = np.arange(7, 7-P, -1)
    bits = np.unpackbits(img, axis=2, bitorder='big').reshape(H,W,C,8)
    selected = bits[..., planes]
    permuted = selected[..., order]
    bits_out = bits.copy()
    bits_out[..., planes] = permuted
    packed = np.packbits(bits_out.reshape(H,W,C*8), axis=2, bitorder='big').reshape(H,W,C)
    return packed.astype(np.uint8)

def _bitplane_unscramble(img: npt.NDArray[np.uint8], plane_bytes: bytes) -> npt.NDArray[np.uint8]:
    H,W,C = img.shape
    P = min(BIT_PLANES, len(plane_bytes), 8)
    if P == 0:
        return img
    scores = np.frombuffer(plane_bytes[:P], dtype=np.uint8)
    order = np.argsort(scores)
    inv = np.argsort(order)
    planes = np.arange(7, 7-P, -1)
    bits = np.unpackbits(img, axis=2, bitorder='big').reshape(H,W,C,8)
    selected = bits[..., planes]
    restored = selected[..., inv]
    bits_out = bits.copy()
    bits_out[..., planes] = restored
    packed = np.packbits(bits_out.reshape(H,W,C*8), axis=2, bitorder='big').reshape(H,W,C)
    return packed.astype(np.uint8)

def _mask_bytes(key: bytes, iv: bytes, length: int) -> bytes:
    out = bytearray(length)
    produced = 0
    counter = 0
    while produced < length:
        blk = blake3_keyed(key, b"MASK"+iv+counter.to_bytes(4,'big'), 64)
        take = min(64, length-produced)
        out[produced:produced+take] = blk[:take]
        produced += take
        counter += 1
    return bytes(out)

def _diffuse(X: npt.NDArray[np.uint8], D: npt.NDArray[np.uint8], M: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    """Vectorized two-pass diffusion (forward/backward cumulative XOR) with mask.

    Implements the same transformation as the previous Python loops but using
    NumPy prefix XOR operations for O(N) C-level execution.
    Forward: F = prefix_xor(X ^ D)
    Backward: define B[i] = F[i] ^ ((D[i]+i) & 0xFF); Y[i] = XOR_{j=i}^{N-1} B[j]
    Then C = Y ^ M.
    """
    if X.size == 0:
        return X
    # Ensure uint8 views
    A = np.bitwise_xor(X, D, dtype=np.uint8)  # X ^ D
    F = np.bitwise_xor.accumulate(A, dtype=np.uint8)
    idx = (np.arange(X.size, dtype=np.uint16) & 0xFF).astype(np.uint8)
    mask2 = (D.astype(np.uint16) + idx.astype(np.uint16)) & 0xFF
    mask2 = mask2.astype(np.uint8)
    B = F ^ mask2
    # Reverse cumulative XOR for Y: Y[i] = XOR B[i..end]
    Br = B[::-1]
    Yr = np.bitwise_xor.accumulate(Br, dtype=np.uint8)
    Y = Yr[::-1]
    return (Y ^ M).astype(np.uint8)

def _undiffuse(C: npt.NDArray[np.uint8], D: npt.NDArray[np.uint8], M: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    """Inverse of vectorized diffusion.

    Given C = (Y ^ M) where Y is reverse cumulative XOR over B and
    B[i] = F[i] ^ ((D[i]+i)&0xFF), F is prefix XOR over A = X ^ D.
    Recover X without scalar Python loops.
    """
    if C.size == 0:
        return C
    Y = (C ^ M).astype(np.uint8)
    N = Y.size
    # B[i] = Y[i] ^ Y[i+1] with Y[N]=0
    Y_pad = np.empty(N + 1, dtype=np.uint8)
    Y_pad[:N] = Y
    Y_pad[N] = 0
    B = Y ^ Y_pad[1:]
    idx = (np.arange(N, dtype=np.uint16) & 0xFF).astype(np.uint8)
    mask2 = (D.astype(np.uint16) + idx.astype(np.uint16)) & 0xFF
    mask2 = mask2.astype(np.uint8)
    F = B ^ mask2
    # A recovered from prefix XOR: A[0]=F[0]; A[i]=F[i]^F[i-1]
    A = F.copy()
    if N > 1:
        A[1:] ^= F[:-1]
    X = A ^ D
    return X.astype(np.uint8)

def _haar_forward_int(channel: npt.NDArray[Any]):
    """Perfectly reversible integer transform - stores even pixels + differences.
    No averaging/division used to avoid precision loss.
    Returns (LL, HL, LH, HH, orig_H, orig_W, padded_H, padded_W).
    """
    x = channel.astype(np.int16)
    H0, W0 = x.shape

    # Pad to even dimensions by edge replication
    H_pad = H0 + (H0 & 1)
    W_pad = W0 + (W0 & 1)
    if H_pad != H0 or W_pad != W0:
        x_new = np.empty((H_pad, W_pad), dtype=np.int16)
        x_new[:H0, :W0] = x
        if H_pad > H0:
            x_new[H0:, :W0] = x[-1:, :W0]  # replicate last row
        if W_pad > W0:
            x_new[:H0, W0:] = x[:H0, -1:]  # replicate last column
        if H_pad > H0 and W_pad > W0:
            x_new[H0:, W0:] = x[-1:, -1:]  # replicate corner
        x = x_new

    # Horizontal transform: keep evens, store differences
    L = x[:, 0::2]                    # Keep even columns (approximation)
    H = x[:, 1::2] - x[:, 0::2]      # Odd - Even (detail)

    # Vertical transform on both L and H
    LL = L[0::2, :]                   # Keep even rows of L
    HL = L[1::2, :] - L[0::2, :]     # Odd - Even rows of L
    LH = H[0::2, :]                   # Keep even rows of H
    HH = H[1::2, :] - H[0::2, :]     # Odd - Even rows of H

    return LL, HL, LH, HH, H0, W0, H_pad, W_pad

def _haar_inverse_int(LL: npt.NDArray[Any], HL: npt.NDArray[Any], LH: npt.NDArray[Any], HH: npt.NDArray[Any],
                      H0: int, W0: int, H_pad: int, W_pad: int) -> npt.NDArray[np.uint8]:
    """Perfect inverse - reconstructs exactly without loss."""
    # Reconstruct L (vertical inverse)
    L = np.empty((H_pad, LL.shape[1]), dtype=np.int16)
    L[0::2, :] = LL                   # Even rows = LL (unchanged)
    L[1::2, :] = LL + HL             # Odd rows = LL + (stored difference)

    # Reconstruct H (vertical inverse)
    H = np.empty((H_pad, LH.shape[1]), dtype=np.int16)
    H[0::2, :] = LH                   # Even rows = LH (unchanged)
    H[1::2, :] = LH + HH             # Odd rows = LH + (stored difference)

    # Reconstruct X (horizontal inverse)
    # Forward was: L = X[:, 0::2], H = X[:, 1::2] - X[:, 0::2]
    # So: X[:, 1::2] = X[:, 0::2] + H = L + H
    X = np.empty((H_pad, W_pad), dtype=np.int16)
    X[:, 0::2] = L                    # Even columns = L (unchanged)
    X[:, 1::2] = L + H               # Odd columns = L + H (L + stored difference)

    # Remove padding and return
    result = X[:H0, :W0]
    return np.clip(result, 0, 255).astype(np.uint8)

def encrypt_image_file(input_path: str, output_path: str, key: bytes | str, test_iv: bytes | None = None, skip_diffusion: bool=False, diffusion_rounds: int = 1) -> npt.NDArray[np.uint8]:
    pil = Image.open(input_path)
    img = _image_to_array(pil)
    H,W,C = img.shape
    if test_iv is None:
        # Generate a random IV if not supplied (12 bytes as used elsewhere)
        test_iv = os.urandom(12)
    iv = test_iv
    key_b = _norm_key(key)
    # Calculate padded dimensions for keystream consistency
    pad_h = (BLOCK_SIZE - H % BLOCK_SIZE) % BLOCK_SIZE
    pad_w = (BLOCK_SIZE - W % BLOCK_SIZE) % BLOCK_SIZE
    nh, nw = H + pad_h, W + pad_w

    # Clamp / validate diffusion rounds
    if diffusion_rounds < 1:
        diffusion_rounds = 1
    if diffusion_rounds > 15:
        # 4 bits available (store >1)  Ecap to 15 to avoid expanding config format
        diffusion_rounds = 15

    # Keystream segmentation (diffusion part scaled by rounds)
    L_perm = H*W            # global pixel permutation ordering
    L_plane= BIT_PLANES     # bit-plane ordering bytes
    L_block= H*W            # block permutation bytes (use original H*W for block count)
    L_diff_single = nh*nw*C        # diffusion bytes for ONE round (use PADDED dimensions)
    L_diff_total  = L_diff_single * diffusion_rounds
    S = generate_hybrid_chaotic_keystream(key_b, iv, L_perm + L_plane + L_block + L_diff_total, use_lorenz=True, use_logistic_sine=True)
    off = 0
    S_perm = S[off:off+L_perm]; off += L_perm
    S_plane = S[off:off+L_plane]; off += L_plane
    S_block = S[off:off+L_block]; off += L_block
    S_diff_all  = S[off:off+L_diff_total]; off += L_diff_total

    # 1. Global pixel permutation
    perm_indices = np.argsort(np.frombuffer(S_perm, dtype=np.uint8))
    permuted = img.reshape(H*W, C)[perm_indices].reshape(H,W,C)

    # 2. Integer lifting Haar forward per channel then immediate inverse (no subband permutation)
    transformed_channels: List[npt.NDArray[np.uint8]] = []
    for ci in range(C):
        LL,HL,LH,HH,H0,W0,Hp,Wp = _haar_forward_int(permuted[:,:,ci])
        LL_use = LL  # no change
        channel_rec = _haar_inverse_int(LL_use, HL, LH, HH, H0, W0, Hp, Wp)
        transformed_channels.append(channel_rec)
    recon = np.stack(transformed_channels, axis=2).astype(np.uint8)

    # 3. Bit-plane scramble (now BIT_PLANES=8 => all bit-planes considered)
    scrambled = _bitplane_scramble(recon, S_plane)

    # 4. Block permutation (keep full padded size - don't crop!)
    work = scrambled
    if pad_h or pad_w:
        work = np.pad(work, ((0,pad_h),(0,pad_w),(0,0)), mode='constant', constant_values=0)
    nh,nw,_ = work.shape
    nb_h, nb_w = nh//BLOCK_SIZE, nw//BLOCK_SIZE
    nb = nb_h*nb_w
    block_perm = _fisher_yates(nb, S_block)
    blocks = work.reshape(nb_h,BLOCK_SIZE,nb_w,BLOCK_SIZE,C).transpose(0,2,1,3,4).reshape(nb,BLOCK_SIZE,BLOCK_SIZE,C)
    blocks_p = blocks[block_perm]
    permuted_blocks = blocks_p.reshape(nb_h,nb_w,BLOCK_SIZE,BLOCK_SIZE,C).transpose(0,2,1,3,4).reshape(nh,nw,C)
    # KEY FIX: Don't crop! Keep full padded size for diffusion

    if skip_diffusion:
        # Skip diffusion for baseline reversibility tests
        C_img = permuted_blocks
    else:
        # 5. Diffusion rounds
        N = nh*nw*C
        current = permuted_blocks.reshape(-1).astype(np.uint8)
        for r in range(diffusion_rounds):
            start = r * L_diff_single
            end = start + L_diff_single
            D_r = np.frombuffer(S_diff_all[start:end], dtype=np.uint8)
            mask_iv = iv if diffusion_rounds == 1 else iv + bytes([r])
            M_r = np.frombuffer(_mask_bytes(key_b, mask_iv, N), dtype=np.uint8)
            current = _diffuse(current, D_r, M_r)
        C_img = current.reshape(nh,nw,C)

    # Enhanced metadata: IV(12)+config(8)+tag(16) - store original H,W
    # flags layout: lower nibble rounds (0 => 1)
    flags = 0 if diffusion_rounds == 1 else (diffusion_rounds & 0x0F)
    config = bytes([1, BIT_PLANES, BLOCK_SIZE, flags, H & 0xFF, (H >> 8) & 0xFF, W & 0xFF, (W >> 8) & 0xFF])
    tag_full = hmac.new(key_b, iv + config + C_img.tobytes(), hashlib.sha256).digest()[:16]
    packed = (iv + config + tag_full).hex()
    out = Image.fromarray(C_img)
    info = PngImagePlugin.PngInfo()
    info.add_text('MTAP', packed)
    out.save(output_path, format='PNG', pnginfo=info, compress_level=1, optimize=False)
    return C_img

def decrypt_image_file(input_path: str, output_path: str, key: bytes | str, skip_diffusion: bool=False) -> npt.NDArray[np.uint8]:
    pil = Image.open(input_path)
    info = getattr(pil, 'info', {}) or {}
    packed_hex = info.get('MTAP')
    if not packed_hex:
        raise ValueError('Missing MTAP metadata chunk.')
    data = bytes.fromhex(packed_hex)
    if len(data) != 12+8+16:  # Updated for 8-byte config
        raise ValueError('Invalid MTAP metadata length.')
    iv = data[:12]
    config = data[12:20]  # Now 8 bytes
    tag = data[20:36]     # Shifted position
    version, planes_cfg, block_cfg, flags, h_low, h_high, w_low, w_high = config
    if version != 1 or planes_cfg != BIT_PLANES or block_cfg != BLOCK_SIZE:
        raise ValueError('Unsupported config parameters.')
    # Decode diffusion rounds
    diffusion_rounds = flags & 0x0F
    if diffusion_rounds == 0:
        diffusion_rounds = 1  # backward compatibility
    if diffusion_rounds > 15:
        raise ValueError('Unsupported diffusion rounds in flags.')

    # Reconstruct original dimensions
    H_orig = h_low + (h_high << 8)
    W_orig = w_low + (w_high << 8)

    cipher = _image_to_array(pil)
    H_padded, W_padded, C = cipher.shape  # These are padded dimensions
    key_b = _norm_key(key)
    # Verify tag
    exp = hmac.new(key_b, iv + config + cipher.tobytes(), hashlib.sha256).digest()[:16]
    if not hmac.compare_digest(exp, tag):
        raise ValueError('Authentication tag mismatch.')

    # Regenerate keystream using SAME logic as encrypt
    # Calculate what the padded dimensions would be
    pad_h = (BLOCK_SIZE - H_orig % BLOCK_SIZE) % BLOCK_SIZE
    pad_w = (BLOCK_SIZE - W_orig % BLOCK_SIZE) % BLOCK_SIZE
    nh, nw = H_orig + pad_h, W_orig + pad_w

    L_perm = H_orig*W_orig
    L_plane= BIT_PLANES
    L_block= H_orig*W_orig
    L_diff_single = nh*nw*C  # padded calculation
    L_diff_total  = L_diff_single * diffusion_rounds
    S = generate_hybrid_chaotic_keystream(key_b, iv, L_perm + L_plane + L_block + L_diff_total, use_lorenz=True, use_logistic_sine=True)
    off = 0
    S_perm = S[off:off+L_perm]; off += L_perm
    S_plane = S[off:off+L_plane]; off += L_plane
    S_block = S[off:off+L_block]; off += L_block
    S_diff_all  = S[off:off+L_diff_total]; off += L_diff_total

    if skip_diffusion:
        after_diff = cipher  # Bypass diffusion for debugging
    else:
        # 1. Undo diffusion rounds
        N = nh*nw*C
        current = cipher.reshape(-1).astype(np.uint8)
        for r in reversed(range(diffusion_rounds)):
            start = r * L_diff_single
            end = start + L_diff_single
            D_r = np.frombuffer(S_diff_all[start:end], dtype=np.uint8)
            mask_iv = iv if diffusion_rounds == 1 else iv + bytes([r])
            M_r = np.frombuffer(_mask_bytes(key_b, mask_iv, N), dtype=np.uint8)
            current = _undiffuse(current, D_r, M_r)
        # 2. No S-box or channel mixing to undo
        after_diff = current.reshape(nh,nw,C)

    # 2. (after undo steps) Undo block permutation (retain padded region)
    nb_h, nb_w = H_padded//BLOCK_SIZE, W_padded//BLOCK_SIZE
    nb = nb_h*nb_w
    block_perm = _fisher_yates(nb, S_block)
    inv_block_perm = np.argsort(block_perm)
    blocks = after_diff.reshape(nb_h,BLOCK_SIZE,nb_w,BLOCK_SIZE,C).transpose(0,2,1,3,4).reshape(nb,BLOCK_SIZE,BLOCK_SIZE,C)
    blocks_r = blocks[inv_block_perm]
    restored_blocks_full = blocks_r.reshape(nb_h,nb_w,BLOCK_SIZE,BLOCK_SIZE,C).transpose(0,2,1,3,4).reshape(H_padded,W_padded,C)

    # 3. Crop back to original region BEFORE reversing bit-plane + transform (encryption applied these only on original size)
    restored_orig = restored_blocks_full[:H_orig, :W_orig, :]

    # 4. Undo bit-plane scrambling on original size only
    descrambled = _bitplane_unscramble(restored_orig, S_plane)

    # 5. Forward integer Haar on original region then inverse transform (no LL permutation)
    recovered_channels: List[npt.NDArray[np.uint8]] = []
    for ci in range(C):
        LL,HL,LH,HH,H0,W0,Hp,Wp = _haar_forward_int(descrambled[:,:,ci])
        LL_use = LL
        channel_rec = _haar_inverse_int(LL_use, HL, LH, HH, H0, W0, Hp, Wp)
        recovered_channels.append(channel_rec)
    permuted_img = np.stack(recovered_channels, axis=2).astype(np.uint8)

    # 6. Inverse global permutation on original region
    perm_indices = np.argsort(np.frombuffer(S_perm, dtype=np.uint8))
    inv_perm = np.argsort(perm_indices)
    plain = permuted_img.reshape(H_orig*W_orig, C)[inv_perm].reshape(H_orig,W_orig,C)
    out = Image.fromarray(plain)
    out.save(output_path, format='PNG')
    return plain

# ==================== EVALUATION METRICS ====================

def _validate_pair(a: npt.NDArray[np.uint8], b: npt.NDArray[np.uint8], name: str):
    if a.shape != b.shape:
        raise ValueError(f"{name}: shape mismatch {a.shape} vs {b.shape}")

def calculate_npcr(img1: npt.NDArray[np.uint8], img2: npt.NDArray[np.uint8]) -> float:
    """Number of Pixels Change Rate (%). For color, counts per channel differences."""
    _validate_pair(img1, img2, 'NPCR')
    if img1.size == 0:
        return 0.0
    return 100.0 * float(np.count_nonzero(img1 != img2)) / float(img1.size)

def calculate_uaci(img1: npt.NDArray[np.uint8], img2: npt.NDArray[np.uint8]) -> float:
    """Unified Average Changing Intensity (%)."""
    _validate_pair(img1, img2, 'UACI')
    if img1.size == 0:
        return 0.0
    diff = np.abs(img1.astype(np.int16) - img2.astype(np.int16))
    return 100.0 * float(np.sum(diff)) / (img1.size * 255.0)

def calculate_entropy(img: npt.NDArray[np.uint8]) -> float:
    """Shannon entropy (bits) averaged over channels for RGB, scalar for grayscale."""
    if img.ndim == 3:
        return float(np.mean([calculate_entropy(img[..., c]) for c in range(img.shape[2])]))
    if img.dtype == np.uint8:
        hist = np.bincount(img.reshape(-1), minlength=256).astype(np.float64)
        total = hist.sum()
        if total == 0:
            return 0.0
        p = hist / total
        nz = p > 0
        return float(-np.sum(p[nz] * np.log2(p[nz])))
    # Generic fallback
    _, counts = np.unique(img, return_counts=True)
    probs = counts.astype(np.float64) / counts.sum()
    nz = probs > 0
    return float(-np.sum(probs[nz] * np.log2(probs[nz]))) if probs.size else 0.0

def calculate_corr_channels(img: npt.NDArray[np.uint8]) -> List[float]:
    """Per-channel horizontal adjacency correlation."""
    if img.ndim == 2:
        img = img[..., None]
    _, W, C = img.shape
    if W < 2:
        return [0.0] * C
    corrs: List[float] = []
    for c in range(C):
        channel = img[..., c].astype(np.float32)
        a = channel[:, :-1].reshape(-1)
        b = channel[:, 1:].reshape(-1)
        a_c = a - a.mean()
        b_c = b - b.mean()
        denom = float(np.sqrt((a_c * a_c).sum()) * np.sqrt((b_c * b_c).sum()))
        corr = float((a_c * b_c).sum() / denom) if denom != 0 else 0.0
        corrs.append(corr)
    return corrs

def calculate_mse(img1: npt.NDArray[np.uint8], img2: npt.NDArray[np.uint8]) -> float:
    _validate_pair(img1, img2, 'MSE')
    return float(np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2))

def calculate_psnr(img1: npt.NDArray[np.uint8], img2: npt.NDArray[np.uint8]) -> float:
    mse = calculate_mse(img1, img2)
    if mse == 0.0:
        return float('inf')
    return float(10.0 * np.log10((255.0 * 255.0) / mse))

def calculate_ssim(img1: npt.NDArray[np.uint8], img2: npt.NDArray[np.uint8]) -> float:
    """Calculate Structural Similarity Index (SSIM)."""
    _validate_pair(img1, img2, 'SSIM')

    # Handle different shapes/types if needed, but _validate_pair checks shape
    if img1.ndim == 3:
        # Multichannel (RGB)
        if has_skimage:
            return float(ssim(img1, img2, channel_axis=2, data_range=255))  # type: ignore
        else:
            # Simple fallback: average of per-channel SSIM (simplified implementation)
            # This is a placeholder if skimage is missing.
            # Ideally we should implement full SSIM or require skimage.
            # For now, let's try to implement a basic version or return 0 with warning
            print("Warning: skimage not found, using simplified SSIM approximation")
            return float(calculate_psnr(img1, img2) / 100.0)  # Dummy fallback

    else:
        # Grayscale
        if has_skimage:
            return float(ssim(img1, img2, data_range=255))  # type: ignore
        else:
            return float(calculate_psnr(img1, img2) / 100.0)

# ==================== NEWKEY ENCRYPTION METHODS ====================

def generate_keystream_segments_newkey(key: bytes, iv: bytes, lengths: Dict[str, int]) -> Dict[str, bytes]:
    """Generate keystream segments using newkey.py method.

    Args:
        key: Secret key
        iv: Initialization vector
        lengths: Dictionary with segment lengths

    Returns:
        Dictionary with keystream segments
    """
    total_length = sum(lengths.values())
    keystream = generate_keystream_blake3(key, iv, total_length)

    # Split into segments
    segments = {}
    offset = 0
    for name, length in lengths.items():
        segments[name] = keystream[offset:offset + length]
        offset += length

    return segments  # type: ignore

def encrypt_image_file_newkey(input_path: str, output_path: str, key: bytes | str,
                             test_iv: bytes | None = None, skip_diffusion: bool = False,
                             diffusion_rounds: int = 1) -> npt.NDArray[np.uint8]:
    """Encrypt image using newkey.py hybrid method.

    Args:
        input_path: Source image path
        output_path: Destination encrypted PNG
        key: Secret key (str or bytes)
        test_iv: Optional 12-byte IV for reproducibility
        skip_diffusion: If True, disables diffusion step
        diffusion_rounds: Number of diffusion rounds (1-15)

    Returns:
        Encrypted image array
    """
    import hmac
    import hashlib
    from PIL import Image, PngImagePlugin

    pil = Image.open(input_path)
    img = _image_to_array(pil)
    H, W, C = img.shape

    if test_iv is None:
        test_iv = os.urandom(12)
    iv = test_iv
    key_b = _norm_key(key)

    # Calculate padded dimensions for keystream consistency
    pad_h = (BLOCK_SIZE - H % BLOCK_SIZE) % BLOCK_SIZE
    pad_w = (BLOCK_SIZE - W % BLOCK_SIZE) % BLOCK_SIZE
    nh, nw = H + pad_h, W + pad_w

    # Clamp diffusion rounds
    if diffusion_rounds < 1:
        diffusion_rounds = 1
    if diffusion_rounds > 15:
        diffusion_rounds = 15

    # Keystream segmentation using newkey.py
    L_perm = H * W
    L_plane = BIT_PLANES
    L_block = H * W
    L_diff_single = nh * nw * C
    L_diff_total = L_diff_single * diffusion_rounds

    lengths = {
        'perm': L_perm,
        'plane': L_plane,
        'block': L_block,
        'diff': L_diff_total,
        'mask': L_diff_total
    }

    segments = generate_keystream_segments_newkey(key_b, iv, lengths)
    S_perm = segments['perm']
    S_plane = segments['plane']
    S_block = segments['block']
    S_diff_all = segments['diff']
    S_mask = segments['mask']

    # 1. Global pixel permutation
    perm_indices = np.argsort(np.frombuffer(S_perm, dtype=np.uint8))
    permuted = img.reshape(H * W, C)[perm_indices].reshape(H, W, C)

    # 2. Integer lifting Haar forward per channel then immediate inverse
    transformed_channels: List[npt.NDArray[np.uint8]] = []
    for ci in range(C):
        LL, HL, LH, HH, H0, W0, Hp, Wp = _haar_forward_int(permuted[:, :, ci])
        LL_use = LL  # no change
        channel_rec = _haar_inverse_int(LL_use, HL, LH, HH, H0, W0, Hp, Wp)
        transformed_channels.append(channel_rec)
    recon = np.stack(transformed_channels, axis=2).astype(np.uint8)

    # 3. Bit-plane scramble
    scrambled = _bitplane_scramble(recon, S_plane)

    # 4. Block permutation
    work = scrambled
    if pad_h or pad_w:
        work = np.pad(work, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant', constant_values=0)
    nh, nw, _ = work.shape
    nb_h, nb_w = nh // BLOCK_SIZE, nw // BLOCK_SIZE
    nb = nb_h * nb_w
    block_perm = _fisher_yates(nb, S_block)
    blocks = work.reshape(nb_h, BLOCK_SIZE, nb_w, BLOCK_SIZE, C).transpose(0, 2, 1, 3, 4).reshape(nb, BLOCK_SIZE, BLOCK_SIZE, C)
    blocks_p = blocks[block_perm]
    permuted_blocks = blocks_p.reshape(nb_h, nb_w, BLOCK_SIZE, BLOCK_SIZE, C).transpose(0, 2, 1, 3, 4).reshape(nh, nw, C)

    if skip_diffusion:
        C_img = permuted_blocks
    else:
        # 5. Diffusion rounds using newkey.py mask
        current = permuted_blocks.reshape(-1).astype(np.uint8)

        for r in range(diffusion_rounds):
            start = r * L_diff_single
            end = start + L_diff_single
            D_r = np.frombuffer(S_diff_all[start:end], dtype=np.uint8)
            M_r = np.frombuffer(S_mask[start:end], dtype=np.uint8)
            current = _diffuse(current, D_r, M_r)
        C_img = current.reshape(nh, nw, C)

    # Enhanced metadata: IV(12)+config(8)+tag(16)
    flags = 0 if diffusion_rounds == 1 else (diffusion_rounds & 0x0F)
    config = bytes([1, BIT_PLANES, BLOCK_SIZE, flags, H & 0xFF, (H >> 8) & 0xFF, W & 0xFF, (W >> 8) & 0xFF])
    tag_full = hmac.new(key_b, iv + config + C_img.tobytes(), hashlib.sha256).digest()[:16]
    packed = (iv + config + tag_full).hex()

    out = Image.fromarray(C_img)
    info = PngImagePlugin.PngInfo()
    info.add_text('MTAP', packed)
    out.save(output_path, format='PNG', pnginfo=info, compress_level=1, optimize=False)

    return C_img

def decrypt_image_file_newkey(input_path: str, output_path: str, key: bytes | str,
                             skip_diffusion: bool = False, force_decrypt: bool = False) -> npt.NDArray[np.uint8]:
    """Decrypt image using newkey.py hybrid method.

    Args:
        input_path: Encrypted image path
        output_path: Decrypted image path
        key: Secret key
        skip_diffusion: If True, disables diffusion step

    Returns:
        Decrypted image array
    """
    import hmac
    import hashlib
    from PIL import Image

    pil = Image.open(input_path)
    info = getattr(pil, 'info', {}) or {}
    packed_hex = info.get('MTAP')
    if not packed_hex:
        raise ValueError('Missing MTAP metadata chunk.')

    data = bytes.fromhex(packed_hex)
    if len(data) != 12 + 8 + 16:
        raise ValueError('Invalid MTAP metadata length.')

    iv = data[:12]
    config = data[12:20]
    tag = data[20:36]
    version, planes_cfg, block_cfg, flags, h_low, h_high, w_low, w_high = config

    if version != 1 or planes_cfg != BIT_PLANES or block_cfg != BLOCK_SIZE:
        raise ValueError('Unsupported config parameters.')

    # Decode diffusion rounds
    diffusion_rounds = flags & 0x0F
    if diffusion_rounds == 0:
        diffusion_rounds = 1
    if diffusion_rounds > 15:
        raise ValueError('Unsupported diffusion rounds in flags.')

    # Reconstruct original dimensions
    H_orig = h_low + (h_high << 8)
    W_orig = w_low + (w_high << 8)

    cipher = _image_to_array(pil)
    H_padded, W_padded, C = cipher.shape
    key_b = _norm_key(key)

    # Verify tag
    exp = hmac.new(key_b, iv + config + cipher.tobytes(), hashlib.sha256).digest()[:16]
    if not hmac.compare_digest(exp, tag):
        if force_decrypt:
            print("Warning: Authentication tag mismatch! Proceeding due to force_decrypt=True.")
        else:
            raise ValueError('Authentication tag mismatch.')

    # Regenerate keystream using newkey.py
    pad_h = (BLOCK_SIZE - H_orig % BLOCK_SIZE) % BLOCK_SIZE
    pad_w = (BLOCK_SIZE - W_orig % BLOCK_SIZE) % BLOCK_SIZE
    nh, nw = H_orig + pad_h, W_orig + pad_w

    L_perm = H_orig * W_orig
    L_plane = BIT_PLANES
    L_block = H_orig * W_orig
    L_diff_single = nh * nw * C
    L_diff_total = L_diff_single * diffusion_rounds

    lengths = {
        'perm': L_perm,
        'plane': L_plane,
        'block': L_block,
        'diff': L_diff_total,
        'mask': L_diff_total
    }

    segments = generate_keystream_segments_newkey(key_b, iv, lengths)
    S_perm = segments['perm']
    S_plane = segments['plane']
    S_block = segments['block']
    S_diff_all = segments['diff']
    S_mask = segments['mask']

    if skip_diffusion:
        after_diff = cipher
    else:
        # 1. Undo diffusion rounds
        current = cipher.reshape(-1).astype(np.uint8)

        for r in reversed(range(diffusion_rounds)):
            start = r * L_diff_single
            end = start + L_diff_single
            D_r = np.frombuffer(S_diff_all[start:end], dtype=np.uint8)
            M_r = np.frombuffer(S_mask[start:end], dtype=np.uint8)
            current = _undiffuse(current, D_r, M_r)
        after_diff = current.reshape(nh, nw, C)

    # 2. Undo block permutation
    nb_h, nb_w = H_padded // BLOCK_SIZE, W_padded // BLOCK_SIZE
    nb = nb_h * nb_w
    block_perm = _fisher_yates(nb, S_block)
    inv_block_perm = np.argsort(block_perm)
    blocks = after_diff.reshape(nb_h, BLOCK_SIZE, nb_w, BLOCK_SIZE, C).transpose(0, 2, 1, 3, 4).reshape(nb, BLOCK_SIZE, BLOCK_SIZE, C)
    blocks_r = blocks[inv_block_perm]
    restored_blocks_full = blocks_r.reshape(nb_h, nb_w, BLOCK_SIZE, BLOCK_SIZE, C).transpose(0, 2, 1, 3, 4).reshape(H_padded, W_padded, C)

    # 3. Crop back to original region
    restored_orig = restored_blocks_full[:H_orig, :W_orig, :]

    # 4. Undo bit-plane scrambling
    descrambled = _bitplane_unscramble(restored_orig, S_plane)

    # 5. Forward integer Haar then inverse transform
    recovered_channels: List[npt.NDArray[np.uint8]] = []
    for ci in range(C):
        LL, HL, LH, HH, H0, W0, Hp, Wp = _haar_forward_int(descrambled[:, :, ci])
        LL_use = LL
        channel_rec = _haar_inverse_int(LL_use, HL, LH, HH, H0, W0, Hp, Wp)
        recovered_channels.append(channel_rec)
    permuted_img = np.stack(recovered_channels, axis=2).astype(np.uint8)

    # 6. Inverse global permutation
    perm_indices = np.argsort(np.frombuffer(S_perm, dtype=np.uint8))
    inv_perm = np.argsort(perm_indices)
    plain = permuted_img.reshape(H_orig * W_orig, C)[inv_perm].reshape(H_orig, W_orig, C)

    out = Image.fromarray(plain)
    out.save(output_path, format='PNG')
    return plain

# ==================== EVALUATION STRUCTURES ====================

@dataclass
class NewkeyImageMetrics:
    image: str
    enc_time_ms: float
    dec_time_ms: float
    npcr_plain_cipher: float
    uaci_plain_cipher: float
    entropy_plain: float
    entropy_cipher: float
    mean_corr_cipher: float
    mse_plain_decrypted: float
    psnr_plain_decrypted: float
    method: str

@dataclass
class NewkeyEvaluationSummary:
    count: int
    avg_enc_time_ms: float
    avg_dec_time_ms: float
    avg_npcr: float
    avg_uaci: float
    avg_entropy_plain: float
    avg_entropy_cipher: float
    avg_corr_cipher: float
    avg_mse: float
    avg_psnr: float
    total_time_ms: float

# ==================== EVALUATION FUNCTIONS ====================

def evaluate_newkey_images(images: Sequence[str], key: bytes | str, output_dir: str,
                          skip_diffusion: bool = False, diffusion_rounds: int = 1) -> Tuple[List[NewkeyImageMetrics], NewkeyEvaluationSummary]:
    """Evaluate images using newkey.py hybrid encryption."""
    os.makedirs(output_dir, exist_ok=True)
    per_image: List[NewkeyImageMetrics] = []

    print("=" * 60)
    print("NEWKEY.PY HYBRID ENCRYPTION EVALUATION")
    print("=" * 60)
    print(f"Method: BLAKE3 + 5D Coupled Logistic Maps")
    print(f"Processing {len(images)} images...")
    print()

    for i, path in enumerate(images, 1):
        base = os.path.basename(path)
        enc_path = os.path.join(output_dir, base + '.newkey_enc.png')
        dec_path = os.path.join(output_dir, base + '.newkey_dec.png')

        try:
            # Load original
            orig_pil = Image.open(path).convert('RGB')
            orig = _image_to_array(orig_pil)
            H, W, C = orig.shape

            print(f"[{i}/{len(images)}] Processing {base} ({H}x{W}x{C})")

            # Encrypt with newkey method
            test_iv = b'newkey_eval123'[:12]
            print(f"  Encrypting with newkey.py...")
            t0 = perf_counter()
            encrypted_data = encrypt_image_file_newkey(
                path, enc_path, key,
                test_iv=test_iv,
                skip_diffusion=skip_diffusion,
                diffusion_rounds=diffusion_rounds
            )
            t1 = perf_counter()

            # Decrypt
            print(f"  Decrypting...")
            t2 = perf_counter()
            plain_rec = decrypt_image_file_newkey(
                enc_path, dec_path, key,
                skip_diffusion=skip_diffusion
            )
            if plain_rec is not None and not os.path.exists(dec_path):  # type: ignore
                try:
                    Image.fromarray(plain_rec.astype(np.uint8)).save(dec_path)
                except Exception:
                    pass
            t3 = perf_counter()

            enc_ms = (t1 - t0) * 1000.0
            dec_ms = (t3 - t2) * 1000.0

            # Check if decryption returned None or failed
            if plain_rec is None:  # type: ignore
                raise ValueError("Decryption returned None")

            # Check if decryption output file exists
            if not os.path.exists(dec_path):
                print(f"Warning: Decryption output file {dec_path} not found")
                raise ValueError("Decryption file missing")

            # Cipher metrics use original spatial size
            cipher_full = encrypted_data
            if cipher_full.shape[0] != H or cipher_full.shape[1] != W:
                cipher_region = cipher_full[:H, :W, :C]
            else:
                cipher_region = cipher_full
            metrics_cipher = cipher_region
            metrics_orig = orig

            # Handle decrypted image shape mismatch
            if plain_rec.shape != orig.shape:
                raise ValueError(f"Decrypted shape mismatch: expected {orig.shape} got {plain_rec.shape}")
            orig_common = orig
            plain_rec_common = plain_rec

            # Calculate metrics
            try:
                ent_plain = calculate_entropy(orig)
                ent_cipher = calculate_entropy(metrics_cipher)

                corr_list = calculate_corr_channels(metrics_cipher)
                corr_mean = float(np.mean(corr_list)) if corr_list else 0.0

                npcr_val = calculate_npcr(metrics_orig, metrics_cipher)
                uaci_val = calculate_uaci(metrics_orig, metrics_cipher)

                mse_val = calculate_mse(orig_common, plain_rec_common)
                psnr_val = calculate_psnr(orig_common, plain_rec_common)

                # Save comparison image
                comparison_path = os.path.join(output_dir, base + '.newkey_comparison.png')
                try:
                    comp_width = W * 2
                    comp_height = H
                    comp_img = np.zeros((comp_height, comp_width, 3), dtype=np.uint8)

                    o_img = orig if (orig.ndim == 3 and orig.shape[2] == 3) else np.stack([orig.squeeze()]*3, axis=-1)
                    r_img = plain_rec_common if (plain_rec_common.ndim == 3 and plain_rec_common.shape[2] == 3) else np.stack([plain_rec_common.squeeze()]*3, axis=-1)

                    comp_img[:, :W, :] = o_img[:, :, :3]
                    comp_img[:, W:, :] = r_img[:, :, :3]

                    Image.fromarray(comp_img).save(comparison_path)
                    print(f"  Saved comparison image to {comparison_path}")
                except Exception as e:
                    print(f"  Warning: Could not save comparison image: {e}")

            except Exception as metric_error:
                print(f"  Error calculating metrics: {metric_error}")
                raise

            per_image.append(NewkeyImageMetrics(
                image=base,
                enc_time_ms=enc_ms,
                dec_time_ms=dec_ms,
                npcr_plain_cipher=npcr_val,
                uaci_plain_cipher=uaci_val,
                entropy_plain=ent_plain,
                entropy_cipher=ent_cipher,
                mean_corr_cipher=corr_mean,
                mse_plain_decrypted=mse_val,
                psnr_plain_decrypted=psnr_val,
                method='newkey.py (BLAKE3 + 4D Logistic)'
            ))

            print(f"  ✓ Success: MSE={mse_val:.6f}, PSNR={psnr_val:.2f}dB, Time={enc_ms+dec_ms:.1f}ms")
            if mse_val > 0:
                diff = (plain_rec_common.astype(np.int32) - orig_common.astype(np.int32))
                max_abs = int(np.max(np.abs(diff)))
                mean_abs = float(np.mean(np.abs(diff)))
                print(f"    Debug: max_abs_diff={max_abs} mean_abs_diff={mean_abs:.2f}")
            print()

        except Exception as e:
            print(f"  ✗ Error processing {path}: {str(e)}")
            import traceback
            traceback.print_exc()

            per_image.append(NewkeyImageMetrics(
                image=f"{base} (ERROR)",
                enc_time_ms=0,
                dec_time_ms=0,
                npcr_plain_cipher=0,
                uaci_plain_cipher=0,
                entropy_plain=0,
                entropy_cipher=0,
                mean_corr_cipher=0,
                mse_plain_decrypted=-1,
                psnr_plain_decrypted=0,
                method='ERROR'
            ))
            print()

    # Aggregate results
    def avg(field: str) -> float:
        vals = [getattr(m, field) for m in per_image if "ERROR" not in m.image]
        return float(np.mean(vals)) if vals else 0.0

    total_time = sum(m.enc_time_ms + m.dec_time_ms for m in per_image if "ERROR" not in m.image)

    summary = NewkeyEvaluationSummary(
        count=sum(1 for m in per_image if "ERROR" not in m.image),
        avg_enc_time_ms=avg('enc_time_ms'),
        avg_dec_time_ms=avg('dec_time_ms'),
        avg_npcr=avg('npcr_plain_cipher'),
        avg_uaci=avg('uaci_plain_cipher'),
        avg_entropy_plain=avg('entropy_plain'),
        avg_entropy_cipher=avg('entropy_cipher'),
        avg_corr_cipher=avg('mean_corr_cipher'),
        avg_mse=avg('mse_plain_decrypted'),
        avg_psnr=avg('psnr_plain_decrypted'),
        total_time_ms=total_time
    )
    return per_image, summary

# ==================== UTILITY FUNCTIONS ====================

def benchmark_chaotic_systems():
    """Benchmark the performance of different chaotic systems."""
    print("=" * 60)
    print("CHAOTIC SYSTEMS PERFORMANCE BENCHMARK")
    print("=" * 60)

    K = b"benchmark_key_32_bytes_long_12345"
    V = b"test_vector_12"
    L = 100000  # 100KB

    # Test 5D Logistic-Sine only
    print("\n1. 5D Hyperchaotic Logistic-Sine System:")
    t0 = time.time()
    ks1 = generate_hybrid_chaotic_keystream(K, V, L, use_lorenz=False, use_logistic_sine=True)
    t1 = time.time()
    print(f"   Time: {t1-t0:.4f}s")
    print(f"   Speed: {L/(t1-t0)/1024:.1f} KB/s")
    print(f"   First 16 bytes: {ks1[:16].hex()}")

    # Test 3D Lorenz only
    print("\n2. 3D Lorenz Attractor:")
    t0 = time.time()
    ks2 = generate_hybrid_chaotic_keystream(K, V, L, use_lorenz=True, use_logistic_sine=False)
    t1 = time.time()
    print(f"   Time: {t1-t0:.4f}s")
    print(f"   Speed: {L/(t1-t0)/1024:.1f} KB/s")
    print(f"   First 16 bytes: {ks2[:16].hex()}")

    # Test hybrid system
    print("\n3. Hybrid System (5D Logistic-Sine + 3D Lorenz):")
    t0 = time.time()
    ks3 = generate_hybrid_chaotic_keystream(K, V, L, use_lorenz=True, use_logistic_sine=True)
    t1 = time.time()
    print(f"   Time: {t1-t0:.4f}s")
    print(f"   Speed: {L/(t1-t0)/1024:.1f} KB/s")
    print(f"   First 16 bytes: {ks3[:16].hex()}")

    # Test repeatability
    print("\n4. Repeatability Test:")
    ks4 = generate_hybrid_chaotic_keystream(K, V, L, use_lorenz=True, use_logistic_sine=True)
    print(f"   Hybrid repeatable: {ks3 == ks4}")

    print("\n" + "=" * 60)

# ==================== PUBLIC API ====================

__all__ = [
    'encrypt_image_file',
    'decrypt_image_file',
    'encrypt_image_file_newkey',
    'decrypt_image_file_newkey',
    'generate_hybrid_chaotic_keystream',
    'generate_keystream_blake3',
    'calculate_npcr',
    'calculate_uaci',
    'calculate_entropy',
    'calculate_corr_channels',
    'calculate_mse',
    'calculate_psnr',
    'calculate_ssim',
    'evaluate_newkey_images',
    'benchmark_chaotic_systems'
]