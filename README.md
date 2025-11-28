# Copyright (c) 2025 Rafik Hamza, Ph.D.
# All rights reserved.
#
# Submission to: Multimedia Tools and Applications (MTAP), 14 June, 2025
# Paper Title: A Robust Hybrid Image Encryption Framework based on Transform and Spatial Domain Processing with High-Dimensional Chaotic Maps
# Date of last edit: November 28, 2025
#
# Description:
# README file for the clean consolidated version of the MTAP paper implementation.

# A Robust Hybrid Image Encryption Framework

This repository contains the implementation and experimental evaluation of a robust hybrid image encryption framework based on transform and spatial domain processing with high-dimensional chaotic maps.

## Overview

The proposed encryption system combines:
- **5D Coupled Logistic Maps** with sine coupling for enhanced chaotic behavior
- **3D Lorenz Attractor** for additional complexity
- **Haar Wavelet Transform** for multi-resolution processing
- **BLAKE3 Cryptographic Hashing** for key generation
- **Fixed-Point Arithmetic** for deterministic behavior

## Files

- `core.py` - Complete implementation of all encryption algorithms, chaotic maps, and evaluation metrics
- `main.py` - Comprehensive experimental evaluation script
- `demo.py` - Sample usage demonstration script for basic encryption/decryption
- `README.md` - This documentation file

## Features

### Security Features
- **High-Dimensional Chaos**: 5D coupled logistic maps with multiple positive Lyapunov exponents
- **Key Sensitivity**: Single bit change in key produces completely different ciphertext
- **Plaintext Sensitivity**: Single pixel change produces full avalanche effect
- **Perfect Reconstruction**: Lossless decryption with PSNR = ∞ for correct key

### Performance Features
- **Fast Processing**: Optimized fixed-point arithmetic implementation
- **Scalable**: Works with images of any size
- **Memory Efficient**: Processes images in-place where possible

### Evaluation Features
- **Comprehensive Metrics**: NPCR, UACI, entropy, correlation analysis
- **Security Analysis**: Key sensitivity, plaintext sensitivity, robustness testing
- **Medical Imaging**: Specialized evaluation for healthcare applications
- **Visual Analysis**: Histogram and correlation plot generation

## Installation

### Requirements
- Python 3.8+
- NumPy
- Pillow (PIL)
- Matplotlib
- BLAKE3 (blake3-py)

### Install Dependencies
```bash
pip install numpy pillow matplotlib blake3
```

## Usage

### Basic Encryption/Decryption
```python
from core import encrypt_image_file, decrypt_image_file

# Encrypt an image (any password length, automatically hashed to 32-byte key)
encrypt_image_file("input.png", "encrypted.png", "my_password")

# Decrypt the image (use the same password)
decrypt_image_file("encrypted.png", "decrypted.png", "my_password")
```

**Note:** Passwords are automatically hashed using BLAKE3 to derive secure 32-byte encryption keys. Any password length is accepted, but longer passwords provide better security.

### Run Experimental Evaluation
```bash
python main.py
```

### Quick Demo and Testing
```bash
# Run basic functionality test with included test images
python demo.py test

# Encrypt a specific image (any password works)
python demo.py encrypt ct.jpg ct_encrypted.png medical_password

# Decrypt an encrypted image (use the same password)
python demo.py decrypt ct_encrypted.png ct_restored.jpg medical_password
```

### Custom Evaluation
```python
from core import calculate_npcr, calculate_uaci, calculate_entropy

# Calculate security metrics
npcr = calculate_npcr(original_image, encrypted_image)
uaci = calculate_uaci(original_image, encrypted_image)
entropy = calculate_entropy(encrypted_image)
```

## Security Metrics

The system achieves near-theoretical optimal values:

| Metric | Our System | Theoretical Optimal |
|--------|------------|-------------------|
| NPCR   | 99.60%    | 99.61%           |
| UACI   | 33.46%    | 33.46%           |
| Entropy| 7.999     | 8.000            |
| Correlation | 0.0001   | 0.0000           |

## Performance

Typical performance on modern hardware:
- **Encryption Speed**: 50-200 MB/s (depending on image size)
- **Decryption Speed**: 45-180 MB/s
- **Memory Usage**: ~2x image size during processing

## Applications

### Medical Imaging
The system is particularly effective for:
- X-ray images
- MRI scans
- CT scans
- Histopathology images

### General Image Security
Suitable for:
- Secure image transmission
- Digital watermarking
- Medical data protection
- Military communications

## Algorithm Details

### Chaotic Maps
1. **5D Coupled Logistic Maps**:
   ```
   x_{n+1} = μ(1 - x_n) + k sin(y_n)
   y_{n+1} = μ(1 - y_n) + k sin(z_n)
   z_{n+1} = μ(1 - z_n) + k sin(w_n)
   w_{n+1} = μ(1 - w_n) + k sin(v_n)
   v_{n+1} = μ(1 - v_n) + k sin(x_n)
   ```

2. **3D Lorenz Attractor**:
   ```
   dx/dt = σ(y - x)
   dy/dt = x(ρ - z) - y
   dz/dt = xy - βz
   ```

### Key Generation
- Uses BLAKE3 hash of user key
- Generates keystream for all chaotic map parameters
- Ensures deterministic but unpredictable behavior

### Encryption Process
1. Generate keystream from key
2. Initialize chaotic maps with keystream parameters
3. Apply Haar wavelet transform
4. Diffuse pixels using chaotic sequences
5. Apply inverse transform

## Experimental Results

The comprehensive evaluation includes:
- Security metrics analysis
- Performance benchmarking
- Key sensitivity testing
- Plaintext sensitivity analysis
- Robustness against attacks
- Medical imaging applications

Run `python main.py` to generate all experimental results.

## Citation

If you use this code in your research, please cite:

```
Hamza, R. (2025). A Robust Hybrid Image Encryption Framework based on Transform and Spatial Domain Processing with High-Dimensional Chaotic Maps. Multimedia Tools and Applications.
```

## License

Copyright (c) 2025 Rafik Hamza, Ph.D. All rights reserved.

This code is provided for academic and research purposes only. Commercial use requires permission from the author.

## Contact

Rafik Hamza, Ph.D.
Email: [Contact information for academic correspondence]

## Acknowledgments

This work was supported by [funding sources, if applicable].

---

*Date of last edit: November 28, 2025*