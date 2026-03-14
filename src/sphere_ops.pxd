from libcpp.complex cimport complex

# Points to your specific header path
cdef extern from "xtensor/containers/xarray.hpp" namespace "xt":
    cdef cppclass xarray[T]:
        xarray() except +

# Typedef to hide brackets from the parser
ctypedef xarray[complex[double]] xarray_complex_double

cdef extern from "sphere_math.hpp" namespace "sphere":
    void compute_grad(complex[double]* data, size_t rows, size_t cols, double radius,
                      complex[double]* out_zonal, complex[double]* out_merid) except +
    void zonal_forward_fft(double* grid_data, complex[double]* spectral_out, int nlat, int nlon) except +


    # 3. Meridional Forward Legendre (Latitude)
    void meridional_forward_legendre(complex[double]* zonal_coeffs, 
                                     complex[double]* sh_out, 
                                     int nlat, int n_l, int n_m) except +
    void meridional_inverse_legendre(complex[double]* sh_coeffs, 
                                     complex[double]* zonal_out, 
                                     int nlat, int n_l, int n_m) except +
    void zonal_inverse_fft(complex[double]* spectral_in, double* grid_out, int nlat, int nlon) except +
