#!/usr/bin/env python3
"""
Test script for verifying the installation of PyTorch, torchvision, torchaudio, and Triton.
Run this script after installing these packages to ensure they are working correctly.
"""

import sys
import platform

def print_separator(title):
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)

def check_import(package_name, import_name=None):
    """Try importing a package and return the module or None."""
    if import_name is None:
        import_name = package_name
    try:
        module = __import__(import_name)
        return module
    except ImportError as e:
        print(f"❌ Failed to import {package_name}: {e}")
        return None

def test_torch(torch):
    print_separator("Testing PyTorch")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Build configuration: {torch.__config__.show()[:200]}...")  # truncated

    # Basic tensor operations
    x = torch.tensor([1.0, 2.0, 3.0])
    y = x + 2
    print(f"Basic tensor addition: {x} + 2 = {y}")
    assert y.tolist() == [3.0, 4.0, 5.0], "Tensor addition failed"

    # Check CUDA / ROCm availability
    if torch.cuda.is_available():
        print(f"✅ CUDA is available. Device count: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.get_device_name(0)}")
        # Test moving tensor to GPU
        x_gpu = x.to('cuda')
        y_gpu = x_gpu + 2
        print(f"GPU tensor addition: {x_gpu} + 2 = {y_gpu}")
        assert y_gpu.cpu().tolist() == [3.0, 4.0, 5.0], "GPU tensor addition failed"
    else:
        print("⚠️ CUDA not available. Running on CPU only.")

    print("✅ PyTorch basic test passed.")

def test_torchvision(torchvision):
    print_separator("Testing torchvision")
    print(f"torchvision version: {torchvision.__version__}")

    # Test a simple transform
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    # Create a dummy PIL image (or tensor) – we'll use a random numpy array
    import numpy as np
    dummy_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    try:
        from PIL import Image
        dummy_pil = Image.fromarray(dummy_image)
        tensor_img = transform(dummy_pil)
        print(f"Transformed image shape: {tensor_img.shape}")
        assert tensor_img.shape == (3, 100, 100), "Unexpected tensor shape"
        print("✅ torchvision transform test passed.")
    except ImportError:
        print("⚠️ PIL not installed, skipping image transform test.")
    except Exception as e:
        print(f"❌ torchvision test failed: {e}")

def test_torchaudio(torchaudio):
    print_separator("Testing torchaudio")
    print(f"torchaudio version: {torchaudio.__version__}")

    # Create a synthetic waveform and test save/load if backend available
    import torch
    sample_rate = 16000
    waveform = torch.sin(2 * 3.14159 * 440 * torch.arange(0, 1, 1/sample_rate)).unsqueeze(0)  # 1 second of 440 Hz sine

    # Try saving and loading (using temporary file)
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        temp_path = f.name
    try:
        torchaudio.save(temp_path, waveform, sample_rate)
        waveform_loaded, sr_loaded = torchaudio.load(temp_path)
        assert sr_loaded == sample_rate, "Sample rate mismatch"
        assert torch.allclose(waveform, waveform_loaded, atol=1e-4), "Waveform mismatch after save/load"
        print(f"✅ torchaudio save/load test passed (temporary file: {temp_path})")
    except Exception as e:
        print(f"❌ torchaudio test failed: {e}")
    finally:
        os.unlink(temp_path)

def test_triton(triton, torch):
    print_separator("Testing Triton")
    print(f"Triton version: {triton.__version__}")

    # Check if a GPU is available (Triton requires a GPU)
    if not torch.cuda.is_available():
        print("⚠️ No CUDA device found. Triton tests require a GPU. Skipping Triton kernel test.")
        return

    # Define a simple Triton kernel (vector addition)
    try:
        import triton
        import triton.language as tl

        @triton.jit
        def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(axis=0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(x_ptr + offsets, mask=mask)
            y = tl.load(y_ptr + offsets, mask=mask)
            output = x + y
            tl.store(output_ptr + offsets, output, mask=mask)

        def add(x: torch.Tensor, y: torch.Tensor):
            output = torch.empty_like(x)
            assert x.is_cuda and y.is_cuda and output.is_cuda
            n_elements = output.numel()
            grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
            add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
            return output

        # Test the kernel
        size = 1024 * 10
        x = torch.rand(size, device='cuda')
        y = torch.rand(size, device='cuda')
        expected = x + y
        result = add(x, y)
        torch.testing.assert_close(result, expected)
        print("✅ Triton kernel (vector addition) test passed.")
    except Exception as e:
        print(f"❌ Triton test failed: {e}")

def main():
    print("Python version:", sys.version)
    print("Platform:", platform.platform())
    print("Starting installation tests...\n")

    # Import packages
    torch = check_import("torch")
    torchvision = check_import("torchvision")
    torchaudio = check_import("torchaudio")
    triton = check_import("triton")

    # Track overall success
    all_good = True

    if torch:
        test_torch(torch)
    else:
        all_good = False

    if torchvision:
        test_torchvision(torchvision)
    else:
        all_good = False

    if torchaudio:
        test_torchaudio(torchaudio)
    else:
        all_good = False

    if triton:
        test_triton(triton, torch)
    else:
        all_good = False

    print_separator("Summary")
    if all_good:
        print("✅ All packages imported successfully and passed basic tests.")
    else:
        print("❌ Some packages failed to import or passed tests. See errors above.")

if __name__ == "__main__":
    main()