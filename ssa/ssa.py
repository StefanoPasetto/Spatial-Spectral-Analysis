# ssa.py

import numpy as np
import matplotlib.pyplot as plt
import re
from scipy.integrate import quad

##############################################################################
# Global plot settings
##############################################################################
figsize = (12, 8)
font_title = {'fontsize': 25}
font_label = {'fontsize': 25}
tick_font_size = 18

##############################################################################
# GPU Setup: Try to import CuPy; if unavailable or no GPU, fall back to NumPy
##############################################################################
cupy_available = False
gpu_info_message = "[INFO] CuPy not installed - Using CPU."

try:
    import cupy as cp
    from cupy.cuda.runtime import getDeviceCount, getDeviceProperties
    if cp.cuda.is_available() and getDeviceCount() > 0:
        cupy_available = True
        device_id = 0  # Use the first available device
        device_properties = getDeviceProperties(device_id)
        gpu_info_message = (
            f"[INFO] GPU found: {device_properties['name'].decode('utf-8')} - Using GPU."
        )
    else:
        cp = np  # Fallback: use NumPy under the alias cp
        gpu_info_message = "[INFO] No CUDA device found - Using CPU."
except ImportError:
    cp = np
except Exception as e:
    cp = np
    gpu_info_message = f"[INFO] An error occurred while trying to use CuPy: {str(e)} - Using CPU."

def print_gpu_info():
    """Prints out whether GPU or CPU is being used."""
    print(gpu_info_message)

##############################################################################
# CUDA Kernel (only defined if using CuPy)
##############################################################################
if cupy_available:
    calculate_pairwise_distances_kernel = cp.RawKernel(r'''
    extern "C" {
    __global__ void calculate_pairwise_distances(double* points_x, double* points_y,
                                                 double* distances, int N,
                                                 double box_size_x, double box_size_y) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < N * (N - 1) / 2) {
            int count = 0;
            for (int i = 0; i < N - 1; ++i) {
                for (int j = i + 1; j < N; ++j) {
                    if (count == idx) {
                        double dx = points_x[i] - points_x[j];
                        double dy = points_y[i] - points_y[j];
                        dx = dx - round(dx / box_size_x) * box_size_x;
                        dy = dy - round(dy / box_size_y) * box_size_y;
                        distances[idx] = sqrt(dx * dx + dy * dy);
                        return;
                    }
                    count++;
                }
            }
        }
    }
    }
    ''', 'calculate_pairwise_distances')

##############################################################################
# Function: Calculate pairwise distances (GPU or CPU)
##############################################################################
def calculate_pairwise_distances(points_x, points_y, distances, box_size):
    """
    Calculates pairwise distances between points_x, points_y arrays,
    accounting for periodic boundaries given by box_size.
    If CuPy is available, uses the CUDA kernel; otherwise uses NumPy.
    """
    print_gpu_info()  # Show which device is being used
    N = points_x.shape[0]
    total_pairs = (N * (N - 1)) // 2

    if cupy_available:
        try:
            print("[INFO] Executing CUDA kernel to calculate pairwise distances...")
            threads_per_block = 256
            blocks_per_grid = (total_pairs + threads_per_block - 1) // threads_per_block
            calculate_pairwise_distances_kernel(
                (blocks_per_grid,),
                (threads_per_block,),
                (
                    points_x, points_y, distances, np.int32(N),
                    np.float64(box_size[0]), np.float64(box_size[1])
                )
            )
            print("[INFO] CUDA kernel executed successfully.")
        except Exception as e:
            print(f"[ERROR] Failed to execute CUDA kernel: {str(e)}")
            raise
    else:
        print("[INFO] Using NumPy to calculate pairwise distances...")
        i_indices, j_indices = np.triu_indices(N, k=1)
        dx = points_x[i_indices] - points_x[j_indices]
        dy = points_y[i_indices] - points_y[j_indices]
        # Apply periodic boundary conditions
        dx = dx - np.round(dx / box_size[0]) * box_size[0]
        dy = dy - np.round(dy / box_size[1]) * box_size[1]
        distances[:] = np.sqrt(dx * dx + dy * dy)
        print("[INFO] NumPy computation completed.")

##############################################################################
# Optional kernel (for smoothing) used in two_point_correlation_function
##############################################################################
def kernel_function(x, kernel_type, bandwidth):
    """
    Kernel for optional smoothing of the correlation function.
    """
    if kernel_type == "Gaussian":
        return np.exp(-0.5 * (x / bandwidth)**2) / (bandwidth * np.sqrt(2 * np.pi))
    elif kernel_type == "Epanechnikov":
        return 0.75 * (1 - (x / bandwidth)**2) / bandwidth if np.abs(x) <= bandwidth else 0
    elif kernel_type == "Rectangular":
        return 0.5 / bandwidth if np.abs(x) <= bandwidth else 0
    else:
        raise ValueError(f"Unknown kernel: {kernel_type}")

##############################################################################
# Function: Compute Two-Point Correlation Function (2pCF)
##############################################################################
def two_point_correlation_function(points, r_min, r_max, bins,
                                   box_size=(1.0, 1.0),
                                   kernel_type=None,
                                   bandwidth=None):
    """
    Calculates the two-point correlation function (2pCF) for a set of 2D points
    within a box of size box_size, using periodic boundary conditions.
    Optionally applies kernel smoothing if kernel_type and bandwidth are provided.

    Returns:
      bin_centers (numpy array): center of each radial bin
      corr_function (numpy array): correlation function values in those bins
    """
    N = len(points)
    total_pairs = (N * (N - 1)) // 2

    # Convert to GPU or CPU array
    if cupy_available:
        points_x = cp.asarray(points[:, 0], dtype=cp.float64)
        points_y = cp.asarray(points[:, 1], dtype=cp.float64)
        distances = cp.zeros(total_pairs, dtype=cp.float64)
    else:
        points_x = points[:, 0].astype(np.float64)
        points_y = points[:, 1].astype(np.float64)
        distances = np.zeros(total_pairs, dtype=np.float64)

    # Calculate all pairwise distances
    calculate_pairwise_distances(points_x, points_y, distances, box_size)
    if cupy_available:
        # Move back to NumPy
        distances = distances.get()

    # Create histogram and compute bin centers
    hist, bin_edges = np.histogram(distances, bins=bins, range=(r_min, r_max), density=False)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Expected number of pairs per bin (assuming uniform density)
    areas = np.pi * (bin_edges[1:]**2 - bin_edges[:-1]**2)
    density = N / (box_size[0] * box_size[1])
    expected_pairs = density * areas * (N - 1) / 2

    corr_function = hist / expected_pairs

    # Optional kernel smoothing
    if kernel_type is not None and bandwidth is not None:
        smoothed_corr_function = np.zeros_like(corr_function)
        for i, r_val in enumerate(bin_centers):
            for j, r_prime in enumerate(bin_centers):
                smoothed_corr_function[i] += (
                    corr_function[j] * kernel_function(r_val - r_prime, kernel_type, bandwidth)
                )
        corr_function = smoothed_corr_function

    return bin_centers, corr_function

##############################################################################
# Function: Compute Power Spectral Density from 2pCF
##############################################################################
def compute_psd(xi_vals, klmin=0.001, klmax=1.0):
    """
    Given a 1D array xi_vals (the 2pCF over bins),
    compute the Power Spectral Density (PSD).
    Returns (k_values, PSD_vals).

    The code uses an internal definition of r_autocorr, wn, fPSD, integrand,
    so that it doesn't rely on global variables.
    """
    xi_data = xi_vals.copy()
    len_data = len(xi_data)
    mu = np.mean(xi_data)

    def r_autocorr(h):
        h = int(abs(h))
        if h >= len_data:
            return 0
        return (1.0 / len_data) * np.sum((xi_data[:len_data - h] - mu) *
                                         (xi_data[h:] - mu))

    def wn(x):
        x = np.asarray(x)
        result = np.zeros_like(x)
        mask = (-0.5 <= x) & (x <= 0.5)
        numerator = (88942
                     + 121849 * np.cos(2 * np.pi * x[mask])
                     + 36058 * np.cos(4 * np.pi * x[mask])
                     + 3151  * np.cos(6 * np.pi * x[mask]))
        result[mask] = numerator / 250000
        return result

    def fPSD(omega, c, a, b):
        factor = (b / (2 * np.pi)) ** ((1 - a) / 2)
        h_vals = np.arange(-c, c + 1)
        r_vals_autocorr = np.array([r_autocorr(h) for h in h_vals])
        wn_vals = wn(h_vals / (2 * c))
        exponentials = np.exp(-1j * b * np.outer(omega, h_vals))
        s = np.dot(exponentials, r_vals_autocorr * wn_vals)
        return factor * s

    def integrand(omega):
        return np.real(fPSD(omega, len_data - 1, 1, -1))

    # Normalization factor via integration
    nfPSD, _ = quad(integrand, klmin, klmax)

    # Generate k values and compute PSD
    k_values = np.logspace(np.log10(klmin), np.log10(klmax), num=100)
    fPSD_vals = fPSD(k_values, len_data - 1, 1, -1)
    PSD_vals = (1.0 / nfPSD) * np.real(fPSD_vals)

    return k_values, PSD_vals

##############################################################################
# Function: Read 2D points from a file (format: {x, y})
##############################################################################
def read_points_from_file(filename):
    """
    Reads points from a file whose lines contain {x, y}.
    Returns a NumPy array of shape (N, 2).
    """
    points = []
    with open(filename, 'r', encoding='utf-8') as f:
        data = f.read()
        matches = re.findall(r'\{([\d\.\-]+),\s*([\d\.\-]+)\}', data)
        for match in matches:
            x, y = float(match[0]), float(match[1])
            points.append((x, y))
    return np.array(points)

##############################################################################
# Main: Put the entire pipeline together
##############################################################################
def main():
    # -------------------------------------------------------------------------
    # Part 1 – Compute 2pCF from input points, write output file, and plot
    # -------------------------------------------------------------------------
    input_filename = 'data.dat'
    points = read_points_from_file(input_filename)

    # Determine box size and shift points so the minimum is at the origin
    x_min, y_min = points.min(axis=0)
    x_max, y_max = points.max(axis=0)
    box_size = (x_max - x_min, y_max - y_min)
    points_shifted = points - np.array([x_min, y_min])

    # Set parameters for 2pCF
    r_min = 0
    r_max = min(box_size) / 2
    bins = 60
    r_vals, xi_vals = two_point_correlation_function(points_shifted,
                                                     r_min, r_max,
                                                     bins,
                                                     box_size=box_size,
                                                     kernel_type=None,
                                                     bandwidth=None)

    # Write 2pCF output to file
    output_2pCF_filename = '2pCF.dat'
    with open(output_2pCF_filename, 'w', encoding='utf-8') as f:
        f.write("# r [μm]         ξ(r)\n")
        for r_val, xi_val_ in zip(r_vals, xi_vals):
            f.write(f"{r_val:.6e} {xi_val_:.6e}\n")

    # Plot the two-point correlation function
    plt.figure(figsize=figsize)
    plt.plot(r_vals, xi_vals, marker='o', linestyle='-')
    plt.xlabel("r [μm]", **font_label)
    plt.ylabel("ξ(r)", **font_label)
    plt.ylim(0, max(xi_vals)*1.1)
    plt.xlim(0, r_max * 0.5)
    plt.axhline(y=1, color='r', linestyle='--')
    plt.title("Two-Point Correlation Function (2pCF)", **font_title)
    plt.xticks(fontsize=tick_font_size)
    plt.yticks(fontsize=tick_font_size)
    plt.grid(True)
  
    # -------------------------------------------------------------------------
    # Part 2 – Compute PSD using the 2pCF result, write PSD.dat, and plot
    # -------------------------------------------------------------------------
    k_values, PSD_vals = compute_psd(xi_vals, klmin=0.001, klmax=1.0)

    # Write PSD output to file
    output_PSD_filename = 'PSD.dat'
    with open(output_PSD_filename, 'w', encoding='utf-8') as f:
        f.write("# k [μm^-1]         P [μm^3]\n")
        for k, P in zip(k_values, PSD_vals):
            f.write(f"{k:.6e} {P:.6e}\n")

    # Plot the PSD on a log-log scale
    plt.figure(figsize=figsize)
    plt.loglog(k_values, PSD_vals, color='red', linewidth=2)
    plt.grid(True, which="both", ls="--")
    plt.title('PSD - pre A - patient #20', **font_title)
    plt.xlabel('k [μm$^{-1}$]', **font_label)
    plt.ylabel('P [μm$^{3}$]', **font_label)
    plt.xticks([0.001, 0.01, 0.1, 1], fontsize=tick_font_size)
    plt.yticks(fontsize=tick_font_size)
    plt.tight_layout()
    plt.show()
    
    # ------------------------------------------------------------------------------
    # Add your spatial transcriptomics analysis code here!
    # Steps:
    # 1. Load spatial barcode and gene expression data.
    # 2. Create gene expression fields.
    # 3. Compute PSD of gene expression.
    # 4. Compare with cell PSD (already computed above).

# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
