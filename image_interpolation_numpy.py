"""
Image Interpolation Using NumPy Only (Loop-Based Implementation)

Implements Nearest Neighbor, Bilinear, and Bicubic interpolation using
explicit nested loops and only NumPy arrays for storage and basic ops.

Usage: run this file with Python. It demonstrates resizing two synthetic
grayscale images and compares visual quality, MSE, edge strength, and runtime.

Constraints honored:
- No use of image-processing libraries for interpolation (PIL/OpenCV/scipy/skimage)
- Pixel computations use explicit nested loops
- Boundary handling implemented via clamping

Author: GitHub Copilot (Raptor mini (Preview))
"""

import os
import time
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt


# ------------------------------- Utilities -------------------------------

def clamp(v: int, lo: int, hi: int) -> int:
    """Clamp integer index to [lo, hi]."""
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


def ensure_gray_uint8(img: np.ndarray) -> np.ndarray:
    """Ensure input is a 2D uint8 array (grayscale in 0-255 range).

    If input is float in [0,1] or [0,255], it will be converted.
    """
    if img.ndim == 3:
        # If has color channels, convert to simple luminance by averaging
        img = img.mean(axis=2)
    if img.dtype == np.float32 or img.dtype == np.float64:
        # If in 0-1 range
        if img.max() <= 1.0:
            img = (img * 255.0).clip(0, 255)
        else:
            img = img.clip(0, 255)
    img = img.astype(np.uint8)
    return img


def gen_gradient(h: int, w: int) -> np.ndarray:
    """Generate a horizontal gradient grayscale image."""
    img = np.zeros((h, w), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            img[y, x] = int(255.0 * x / (w - 1))
    return img


def gen_checkerboard(h: int, w: int, tile: int = 8) -> np.ndarray:
    """Generate a checkerboard pattern."""
    img = np.zeros((h, w), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            block_x = (x // tile) % 2
            block_y = (y // tile) % 2
            img[y, x] = 255 if (block_x ^ block_y) else 0
    return img


# --------------------------- Interpolation Algos -------------------------

def resize_nearest(src: np.ndarray, dst_shape: Tuple[int, int]) -> np.ndarray:
    """Nearest neighbor interpolation using explicit loops.

    src: 2D uint8 array
    dst_shape: (height, width)
    """
    src_h, src_w = src.shape
    dst_h, dst_w = dst_shape
    dst = np.zeros((dst_h, dst_w), dtype=np.uint8)

    scale_x = src_w / dst_w
    scale_y = src_h / dst_h

    for y in range(dst_h):
        for x in range(dst_w):
            src_x = int(round((x + 0.5) * scale_x - 0.5))
            src_y = int(round((y + 0.5) * scale_y - 0.5))
            sx = clamp(src_x, 0, src_w - 1)
            sy = clamp(src_y, 0, src_h - 1)
            dst[y, x] = src[sy, sx]

    return dst


def resize_bilinear(src: np.ndarray, dst_shape: Tuple[int, int]) -> np.ndarray:
    """Bilinear interpolation using explicit nested loops.

    For each destination pixel, compute corresponding continuous source
    coordinate, then bilinearly combine the four neighbors.
    """
    src_h, src_w = src.shape
    dst_h, dst_w = dst_shape
    dst = np.zeros((dst_h, dst_w), dtype=np.uint8)

    scale_x = src_w / dst_w
    scale_y = src_h / dst_h

    for y in range(dst_h):
        for x in range(dst_w):
            src_x = (x + 0.5) * scale_x - 0.5
            src_y = (y + 0.5) * scale_y - 0.5

            x0 = int(np.floor(src_x))
            y0 = int(np.floor(src_y))
            x1 = x0 + 1
            y1 = y0 + 1

            x0c = clamp(x0, 0, src_w - 1)
            x1c = clamp(x1, 0, src_w - 1)
            y0c = clamp(y0, 0, src_h - 1)
            y1c = clamp(y1, 0, src_h - 1)

            wx = src_x - x0
            wy = src_y - y0

            v00 = float(src[y0c, x0c])
            v10 = float(src[y0c, x1c])
            v01 = float(src[y1c, x0c])
            v11 = float(src[y1c, x1c])

            top = v00 * (1 - wx) + v10 * wx
            bottom = v01 * (1 - wx) + v11 * wx
            value = top * (1 - wy) + bottom * wy

            dst[y, x] = int(round(clamp(int(round(value)), 0, 255)))

    return dst


def _cubic_kernel(x: float, a: float = -0.5) -> float:
    """Cubic kernel (Keys) used for bicubic interpolation.

    a = -0.5 is Catmull-Rom like; -0.75 gives slightly different smoothing.
    """
    x = abs(x)
    if x <= 1:
        return (a + 2) * (x**3) - (a + 3) * (x**2) + 1
    elif 1 < x < 2:
        return a * (x**3) - 5 * a * (x**2) + 8 * a * x - 4 * a
    else:
        return 0.0


def resize_bicubic(src: np.ndarray, dst_shape: Tuple[int, int], a: float = -0.5) -> np.ndarray:
    """Bicubic interpolation using explicit nested loops and separable cubic kernel.

    For each destination pixel, map to source coords and sum over 4x4 neighborhood
    with cubic weights.
    """
    src_h, src_w = src.shape
    dst_h, dst_w = dst_shape
    dst = np.zeros((dst_h, dst_w), dtype=np.uint8)

    scale_x = src_w / dst_w
    scale_y = src_h / dst_h

    for y in range(dst_h):
        for x in range(dst_w):
            src_x = (x + 0.5) * scale_x - 0.5
            src_y = (y + 0.5) * scale_y - 0.5

            x0 = int(np.floor(src_x))
            y0 = int(np.floor(src_y))

            value = 0.0
            total_w = 0.0

            for j in range(-1, 3):  # y neighbors
                wy = _cubic_kernel(src_y - (y0 + j), a)
                y_idx = clamp(y0 + j, 0, src_h - 1)
                for i in range(-1, 3):  # x neighbors
                    wx = _cubic_kernel(src_x - (x0 + i), a)
                    x_idx = clamp(x0 + i, 0, src_w - 1)
                    w = wy * wx
                    total_w += w
                    value += float(src[y_idx, x_idx]) * w

            # Normalize by total weight (could be slightly different from 1 near borders)
            if total_w != 0:
                value = value / total_w
            value = min(max(int(round(value)), 0), 255)
            dst[y, x] = value

    return dst


# ---------------------------- Evaluation Metrics -------------------------

def mse_loop(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Mean Squared Error via explicit loops."""
    assert a.shape == b.shape
    h, w = a.shape
    s = 0.0
    for y in range(h):
        for x in range(w):
            d = float(a[y, x]) - float(b[y, x])
            s += d * d
    return s / (h * w)


def edge_strength_loop(img: np.ndarray) -> float:
    """Estimate edge strength by sum of absolute gradients using loops."""
    h, w = img.shape
    s = 0.0
    for y in range(h - 1):
        for x in range(w - 1):
            gx = abs(int(img[y, x + 1]) - int(img[y, x]))
            gy = abs(int(img[y + 1, x]) - int(img[y, x]))
            s += gx + gy
    # Normalize
    return s / (h * w)


# ----------------------------- Helper Runner -----------------------------

def compare_interpolations(src: np.ndarray, dst_shape: Tuple[int, int], prefix: str = "out") -> dict:
    """Apply all three methods, time them, compute MSE pairwise and edge strength.

    Returns dict with outputs and metrics.
    """
    results = {}

    # Nearest
    t0 = time.time()
    n = resize_nearest(src, dst_shape)
    t1 = time.time()

    # Bilinear
    b0 = time.time()
    bi = resize_bilinear(src, dst_shape)
    b1 = time.time()

    # Bicubic
    c0 = time.time()
    bc = resize_bicubic(src, dst_shape)
    c1 = time.time()

    results['nearest'] = {'img': n, 'time': t1 - t0}
    results['bilinear'] = {'img': bi, 'time': b1 - b0}
    results['bicubic'] = {'img': bc, 'time': c1 - c0}

    # Pairwise MSEs
    results['mse_nn_bilinear'] = mse_loop(n, bi)
    results['mse_nn_bicubic'] = mse_loop(n, bc)
    results['mse_bilinear_bicubic'] = mse_loop(bi, bc)

    # Edge strengths
    results['edge_nn'] = edge_strength_loop(n)
    results['edge_bilinear'] = edge_strength_loop(bi)
    results['edge_bicubic'] = edge_strength_loop(bc)

    # Save images
    out_dir = 'results'
    os.makedirs(out_dir, exist_ok=True)
    plt.imsave(os.path.join(out_dir, f"{prefix}_nearest.png"), n, cmap='gray', vmin=0, vmax=255)
    plt.imsave(os.path.join(out_dir, f"{prefix}_bilinear.png"), bi, cmap='gray', vmin=0, vmax=255)
    plt.imsave(os.path.join(out_dir, f"{prefix}_bicubic.png"), bc, cmap='gray', vmin=0, vmax=255)

    return results


# ------------------------------- Demo & CLI ------------------------------

def demo():
    """Demonstrate algorithms on two synthetic grayscale images and summarize results."""
    # Source images (two different spatial resolutions)
    src1 = gen_gradient(40, 60)
    src2 = gen_checkerboard(60, 40, tile=4)

    # Target resolution
    target = (200, 200)

    print("Resizing two synthetic grayscale images to", target)

    res1 = compare_interpolations(src1, target, prefix='gradient')
    res2 = compare_interpolations(src2, target, prefix='checker')

    # Print results summary table
    print("\nComparison table for 'gradient' image:\n")
    print("Method      | Time (s) | Edge strength | Notes")
    print("-----------------------------------------------------")
    print(f"Nearest     | {res1['nearest']['time']:.4f}   | {res1['edge_nn']:.2f}          | Fast, blocky")
    print(f"Bilinear    | {res1['bilinear']['time']:.4f}   | {res1['edge_bilinear']:.2f}          | Smoother")
    print(f"Bicubic     | {res1['bicubic']['time']:.4f}   | {res1['edge_bicubic']:.2f}          | Smoothest, preserves gradients")

    print("\nPairwise MSEs (gradient):")
    print(f"NN vs Bilinear: {res1['mse_nn_bilinear']:.2f}")
    print(f"NN vs Bicubic:  {res1['mse_nn_bicubic']:.2f}")
    print(f"Bilinear vs Bicubic: {res1['mse_bilinear_bicubic']:.2f}")

    print("\n---\n")

    print("Comparison table for 'checker' image:\n")
    print("Method      | Time (s) | Edge strength | Notes")
    print("-----------------------------------------------------")
    print(f"Nearest     | {res2['nearest']['time']:.4f}   | {res2['edge_nn']:.2f}          | Fast, blocky, jagged edges")
    print(f"Bilinear    | {res2['bilinear']['time']:.4f}   | {res2['edge_bilinear']:.2f}          | Smoother but blurs edges")
    print(f"Bicubic     | {res2['bicubic']['time']:.4f}   | {res2['edge_bicubic']:.2f}          | Better edge preservation vs bilinear")

    print("\nPairwise MSEs (checker):")
    print(f"NN vs Bilinear: {res2['mse_nn_bilinear']:.2f}")
    print(f"NN vs Bicubic:  {res2['mse_nn_bicubic']:.2f}")
    print(f"Bilinear vs Bicubic: {res2['mse_bilinear_bicubic']:.2f}")

print("\nResult images saved in ./results/*.png")
print("Summary: Bicubic typically gives the best visual smoothness with higher "
      "edge preservation compared to bilinear, at the cost of extra computation time.")


if __name__ == '__main__':
    demo()
