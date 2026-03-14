# Use xtensor-python headers to bridge NumPy and xtensor
from libcpp.complex cimport complex as c_complex

cdef extern from "sphere_math.hpp" namespace "sphere":
    # We tell Cython about our C++ function template
    # Using 'double' for the demo, can be templated later
    cdef cppclass xarray[T]:
        pass

    cdef xarray[c_complex[double]] apply_gradient_scale(xarray[c_complex[double]] coeffs, double radius)
