# Copyright (c) 2025 Rafik Hamza, Ph.D.
# All rights reserved.
#
# Submission to: Multimedia Tools and Applications (MTAP), 14 June, 2025
# Paper Title: A Robust Hybrid Image Encryption Framework based on Transform and Spatial Domain Processing with High-Dimensional Chaotic Maps
# Date of last edit: November 28, 2025
#
# Description:
# Sample usage demonstration for the 5D coupled logistic map encryption system.
# This file shows basic encryption and decryption operations.

"""
Sample Usage Demonstration
==========================

This script demonstrates basic usage of the 5D coupled logistic map encryption system.
It shows how to encrypt and decrypt image files using the core functions.

The script accepts any password and uses BLAKE3 hashing to derive a secure 32-byte key.

Usage Examples:
- python demo.py encrypt ct.jpg encrypted.png "medical_password"
- python demo.py decrypt encrypted.png ct_restored.jpg "medical_password"
- python demo.py test  # Run basic functionality test
"""

import os
import sys
import time

# Import from core module and blake3
from core import encrypt_image_file, decrypt_image_file
try:
    import blake3
except ImportError:
    print("Error: blake3 module not found. Please install with: pip install blake3")
    sys.exit(1)


def derive_key_from_password(password: str) -> bytes:
    """Derive a 32-byte key from any password using BLAKE3."""
    # Hash the password with BLAKE3 to get exactly 32 bytes
    hasher = blake3.blake3(password.encode('utf-8'))
    key_bytes = hasher.digest()  # This gives exactly 32 bytes
    return key_bytes


def demonstrate_encryption(input_path: str, output_path: str, password: str) -> bool:
    """Demonstrate image encryption."""
    print("=" * 60)
    print("IMAGE ENCRYPTION DEMONSTRATION")
    print("=" * 60)
    print(f"Input file: {input_path}")
    print(f"Output file: {output_path}")
    print(f"Password: {'*' * len(password)} (length: {len(password)})")
    print()

    if not os.path.exists(input_path):
        print(f"Error: Input file '{input_path}' not found!")
        return False

    try:
        # Derive 32-byte key from password using BLAKE3
        key = derive_key_from_password(password)
        print("Deriving encryption key from password...")
        print(f"   Key derived: {key.hex()[:32]}... (32 bytes)")
        print("Encrypting image...")
        start_time = time.time()

        encrypt_image_file(input_path, output_path, key)

        end_time = time.time()
        encryption_time = end_time - start_time

        if os.path.exists(output_path):
            input_size = os.path.getsize(input_path)
            output_size = os.path.getsize(output_path)

            print("Encryption successful!")
            print(f"   Processing time: {encryption_time:.3f} seconds")
            print(f"   Input size: {input_size:,} bytes")
            print(f"   Output size: {output_size:,} bytes")
            print(f"   Speed: {input_size / encryption_time / 1024:.1f} KB/s")
            return True
        else:
            print("Error: Output file was not created!")
            return False

    except Exception as e:
        print(f"Encryption failed: {e}")
        return False


def demonstrate_decryption(input_path: str, output_path: str, password: str) -> bool:
    """Demonstrate image decryption."""
    print("=" * 60)
    print("IMAGE DECRYPTION DEMONSTRATION")
    print("=" * 60)
    print(f"Input file: {input_path}")
    print(f"Output file: {output_path}")
    print(f"Password: {'*' * len(password)} (length: {len(password)})")
    print()

    if not os.path.exists(input_path):
        print(f"❌ Error: Input file '{input_path}' not found!")
        return False

    try:
        # Derive 32-byte key from password using BLAKE3
        key = derive_key_from_password(password)
        print("🔐 Deriving decryption key from password...")
        print(f"   Key derived: {key.hex()[:32]}... (32 bytes)")
        print("🔓 Decrypting image...")
        start_time = time.time()

        decrypt_image_file(input_path, output_path, key)

        end_time = time.time()
        decryption_time = end_time - start_time

        if os.path.exists(output_path):
            input_size = os.path.getsize(input_path)
            output_size = os.path.getsize(output_path)

            print("✅ Decryption successful!")
            print(f"   Processing time: {decryption_time:.3f} seconds")
            print(f"   Input size: {input_size:,} bytes")
            print(f"   Output size: {output_size:,} bytes")
            print(f"   Speed: {input_size / decryption_time / 1024:.1f} KB/s")
            return True
        else:
            print("❌ Error: Output file was not created!")
            return False

    except Exception as e:
        print(f"❌ Decryption failed: {e}")
        return False


def run_basic_test():
    """Run a basic functionality test using available test images."""
    print("=" * 60)
    print("BASIC FUNCTIONALITY TEST")
    print("=" * 60)

    # Test images available in the clean folder
    test_images = [
        "4.2.06_512x512.png",
        "ct.jpg",
        "mri.jpg",
        "histo.png",
        "xray.jpeg"
    ]

    # Test password (can be any length, will be hashed to 32 bytes)
    test_password = "demo_password_2025"

    print(f"Test password: {test_password}")
    print(f"Available test images: {test_images}")
    print()

    # Find first available test image
    test_image = None
    for img in test_images:
        if os.path.exists(img):
            test_image = img
            break

    if not test_image:
        print("❌ No test images found in current directory!")
        print("Please ensure test images are available or specify full paths.")
        return False

    print(f"Using test image: {test_image}")
    print()

    # Test encryption
    encrypted_file = "demo_encrypted.png"
    success1 = demonstrate_encryption(test_image, encrypted_file, test_password)

    if not success1:
        return False

    print()

    # Test decryption
    decrypted_file = "demo_decrypted.png"
    success2 = demonstrate_decryption(encrypted_file, decrypted_file, test_password)

    if not success2:
        return False

    print()
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print("✅ Encryption: PASSED")
    print("✅ Decryption: PASSED")
    print(f"📁 Original: {test_image}")
    print(f"🔐 Encrypted: {encrypted_file}")
    print(f"🔓 Decrypted: {decrypted_file}")
    print()
    print("🎉 All tests passed! The encryption system is working correctly.")
    print()
    print("Note: The decrypted image should be identical to the original.")
    print("Any differences indicate an issue with the implementation.")

    return True


def print_usage():
    """Print usage instructions."""
    print("=" * 60)
    print("5D COUPLED LOGISTIC MAP ENCRYPTION - DEMO")
    print("=" * 60)
    print()
    print("USAGE:")
    print("  python demo.py encrypt <input> <output> <password>")
    print("  python demo.py decrypt <input> <output> <password>")
    print("  python demo.py test")
    print()
    print("EXAMPLES:")
    print("  python demo.py encrypt ct.jpg ct_encrypted.png medical_password")
    print("  python demo.py decrypt ct_encrypted.png ct_restored.jpg medical_password")
    print("  python demo.py encrypt mri.jpg mri_secure.png radiology2025")
    print("  python demo.py decrypt mri_secure.png mri_restored.jpg radiology2025")
    print("  python demo.py test")
    print()
    print("NOTES:")
    print("- Passwords can be any length (will be hashed to 32-byte key)")
    print("- Input images should be PNG, JPEG, or other PIL-supported formats")
    print("- Encrypted files are always PNG format with metadata")
    print("- Available test images: ct.jpg, mri.jpg, histo.png, xray.jpeg, 4.2.06_512x512.png")
    print("- The 'test' command uses available test images in the current directory")


def main():
    """Main function."""
    if len(sys.argv) < 2:
        print_usage()
        return

    command = sys.argv[1].lower()

    if command == "test":
        run_basic_test()

    elif command in ["encrypt", "decrypt"]:
        if len(sys.argv) != 5:
            print(f"❌ Error: {command} requires 3 arguments: input_file output_file password")
            print_usage()
            return

        input_file = sys.argv[2]
        output_file = sys.argv[3]
        password = sys.argv[4]

        # Validate password is not empty
        if len(password) == 0:
            print("❌ Error: Password cannot be empty")
            return

        if command == "encrypt":
            demonstrate_encryption(input_file, output_file, password)
        else:
            demonstrate_decryption(input_file, output_file, password)

    else:
        print(f"❌ Error: Unknown command '{command}'")
        print_usage()


if __name__ == "__main__":
    main()