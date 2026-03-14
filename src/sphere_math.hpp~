#pragma once
#include <xtensor/containers/xarray.hpp>
#include <xtensor/containers/xadapt.hpp>
#include <complex>
#include <vector>
#include <cmath>
#include <fftw3.h>

namespace sphere {

template <typename T>
struct VectorGradient {
    xt::xarray<std::complex<T>> zonal;
    xt::xarray<std::complex<T>> meridional;
};

  template <typename T>
  T compute_plm(int l, int m, T x) {
    if (m > l) return 0.0;
    
    // 1. Initial value P_00 = sqrt(1/4pi)
    T pmm = std::sqrt(1.0 / (4.0 * M_PI));
    
    // 2. Compute P_mm using recurrence
    T somx2 = std::sqrt(std::max(0.0, (1.0 - x) * (1.0 + x)));
    for (int i = 1; i <= m; i++) {
        pmm *= -somx2 * std::sqrt((2.0 * i + 1.0) / (2.0 * i));
    }
    if (l == m) return pmm;
    
    // 3. Compute P_m+1,m
    T pmmp1 = x * std::sqrt(2.0 * m + 3.0) * pmm;
    if (l == m + 1) return pmmp1;
    
    // 4. Stable recurrence for l > m + 1
    T pll = 0.0;
    for (int ll = m + 2; ll <= l; ll++) {
      T a = std::sqrt((4.0 * ll * ll - 1.0) / (ll * ll - m * m));
      T b = std::sqrt(((ll - 1.0) * (ll - 1.0) - m * m) / (4.0 * (ll - 1.0) * (ll - 1.0) - 1.0));
      pll = a * (x * pmmp1 - b * pmm);
      pmm = pmmp1;
      pmmp1 = pll;
    }
    return pll;
  }
  
// Use a template for the expression type (E) to accept adapted arrays
template <typename T, typename E>
VectorGradient<T> compute_grad_impl(const E& coeffs, T radius) {
    auto shape = coeffs.shape();
    size_t n_l = shape[0];
    size_t n_m = shape[1];

    VectorGradient<T> res;
    res.zonal = xt::zeros<std::complex<T>>({n_l, n_m});
    res.meridional = xt::zeros<std::complex<T>>({n_l, n_m});

    for (size_t l = 0; l < n_l; ++l) {
        T fl = static_cast<T>(l);
        T scale_v = std::sqrt(fl * (fl + 1.0)) / radius;
        for (size_t m = 0; m < n_m; ++m) {
            if (l == 0) continue;
            T fm = static_cast<T>(m);
            res.zonal(l, m) = coeffs(l, m) * std::complex<T>(0, fm) / radius;
            res.meridional(l, m) = coeffs(l, m) * scale_v;
        }
    }
    return res;
}

  template <typename T>
  void compute_grad(std::complex<T>* data, size_t rows, size_t cols, T radius, 
                    std::complex<T>* out_zonal, std::complex<T>* out_merid) {
    
    std::vector<size_t> shape = {rows, cols};
    
    // Wrap input and output pointers
    auto coeffs = xt::adapt(data, rows * cols, xt::no_ownership(), shape);
    auto zonal_view = xt::adapt(out_zonal, rows * cols, xt::no_ownership(), shape);
    auto merid_view = xt::adapt(out_merid, rows * cols, xt::no_ownership(), shape);
    
    // Now this call will work because 'E' will match the adapted type
    auto res = compute_grad_impl<T>(coeffs, radius);

    // Copy results into the NumPy-backed views
    zonal_view = res.zonal;
    merid_view = res.meridional;
}
  
  template <typename T>
  std::vector<T> compute_weights_fixed(int nlat) {
    std::vector<T> weights(nlat);
    T dphi = M_PI / (nlat - 1);
    for (int i = 0; i < nlat; ++i) {
        T phi = -M_PI/2.0 + i * dphi;
        // Basic trapezoidal weight for sin-theta integration
        weights[i] = std::cos(phi) * dphi; 
        if (i == 0 || i == nlat - 1) weights[i] *= 0.5;
    }
    return weights;
  }
  template <typename T>
  void zonal_forward_fft(const T* grid_data, std::complex<T>* spectral_out, int nlat, int nlon) {
    int n_complex = nlon / 2 + 1;
    for (int i = 0; i < nlat; ++i) {
      fftw_plan plan = fftw_plan_dft_r2c_1d(nlon, (double*)&grid_data[i * nlon], 
                                            (fftw_complex*)&spectral_out[i * n_complex], 
                                            FFTW_ESTIMATE);
      fftw_execute(plan);
      fftw_destroy_plan(plan);
      // REMOVED: spectral_out /= nlon;
    }
  }
  template <typename T>
  void meridional_forward_legendre(const std::complex<T>* zonal_coeffs, 
                                   std::complex<T>* sh_out, 
                                   int nlat, int n_l, int n_m) {
    auto weights = compute_weights_fixed<T>(nlat);
    T dphi = M_PI / (nlat - 1);
    
    for (int m = 0; m < n_m; ++m) {
        for (int l = m; l < n_l; ++l) {
            std::complex<T> sum(0, 0);
            for (int i = 0; i < nlat; ++i) {
                T phi = -M_PI/2.0 + i * dphi;
                T x = std::sin(phi);
                // Use your existing compute_plm function
                T plm = compute_plm(l, m, x);
                
                // Integrate: Zonal_Coeff * Plm * cos(phi) * dphi
                sum += zonal_coeffs[i * n_m + m] * plm * weights[i];
            }
            sh_out[l * n_m + m] = sum;
        }
    }
  }

  template <typename T>
  void meridional_inverse_legendre(const std::complex<T>* sh_coeffs, 
                                   std::complex<T>* zonal_out, 
                                   int nlat, int n_l, int n_m) {
    T dphi = M_PI / (nlat - 1);
    // Loop over each latitude to reconstruct Fourier coefficients
    for (int i = 0; i < nlat; ++i) {
      T phi = -M_PI/2.0 + i * dphi;
      T x = std::sin(phi);
      for (int m = 0; m < n_m; ++m) {
        std::complex<T> sum(0, 0);
        for (int l = m; l < n_l; ++l) {
          // Sum: Coeff * Plm
          sum += sh_coeffs[l * n_m + m] * compute_plm<T>(l, m, x);
        }
        zonal_out[i * n_m + m] = sum;
      }
    }
  }
  
  template <typename T>
  void zonal_inverse_fft(const std::complex<T>* spectral_in, T* grid_out, int nlat, int nlon) {
   int n_complex = nlon / 2 + 1;
   for (int i = 0; i < nlat; ++i) {
     // Create plan for THIS specific row
     fftw_plan plan = fftw_plan_dft_c2r_1d(nlon, 
                                           (fftw_complex*)&spectral_in[i * n_complex], 
                                           (double*)&grid_out[i * nlon], 
                                           FFTW_ESTIMATE);
     fftw_execute(plan);
     fftw_destroy_plan(plan);
   }
 }
} // namespace sphere
