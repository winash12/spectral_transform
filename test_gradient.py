import sys
import os
import numpy as np

# Add the build directory to the path so we can find the .so
sys.path.append(os.path.join(os.getcwd(), 'builddir'))

try:
    import sphere_ops
    print("✅ Module imported successfully!")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

# 1. Create a simple test grid (73x144)
# A simple constant + a sine wave
lats = np.linspace(-90, 90, 73)
lons = np.linspace(0, 360, 144, endpoint=False)
lon_grid, lat_grid = np.meshgrid(lons, lats)
original = 10.0 + np.sin(np.deg2rad(lon_grid)) 

# 2. Round-trip
sh = sphere_ops.to_spectral(original)
print("SH Coeffs (first 5x5):", sh[:5, :5])
recovered = sphere_ops.to_grid(sh, 73, 144)
# 3. Check
diff = np.abs(original - recovered).max()
print(f"Max Difference: {diff:.2e}")
if diff < 1e-10:
    print("🚀 Perfect Round-trip! The engine is ready.")
else:
    print("⚠️ Scaling or logic check needed.")
    
sys.exit()
    
# 1. Setup a dummy 73x144 grid of coefficients
# Rows = l (0 to 72), Cols = m (0 to 143)
n_l, n_m = 73, 144
coeffs = np.zeros((n_l, n_m), dtype=np.complex128)

# Set a single coefficient: l=10, m=5
l_test, m_test = 10, 5
coeffs[l_test, m_test] = 1.0 + 0j

radius = 6.371e6  # Earth Radius in meters

# 2. Call your new C++ backend
u_coeffs, v_coeffs = sphere_ops.get_gradient(coeffs, radius)

# 3. Verify the Zonal (u) math: (i * m * coeffs) / R
expected_u = (1j * m_test * coeffs[l_test, m_test]) / radius
actual_u = u_coeffs[l_test, m_test]

# 4. Verify the Meridional (v) math: sqrt(l*(l+1)) * coeffs / R
expected_v = (np.sqrt(l_test * (l_test + 1)) * coeffs[l_test, m_test]) / radius
actual_v = v_coeffs[l_test, m_test]

print(f"\n--- Testing l={l_test}, m={m_test} ---")
print(f"Zonal (u):    Expected {expected_u:.2e}, Actual {actual_u:.2e}")
print(f"Meridional (v): Expected {expected_v:.2e}, Actual {actual_v:.2e}")

if np.allclose(actual_u, expected_u) and np.allclose(actual_v, expected_v):
    print("\n🔥 MATH VERIFIED: The C++ backend is perfect.")
else:
    print("\n⚠️ MATH MISMATCH: Check the scaling factors.")
