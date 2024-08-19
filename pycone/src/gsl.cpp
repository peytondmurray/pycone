#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_rng.h>
#include <boost/math/distributions/normal.hpp>

namespace py = pybind11;


py::array_t<double> halfnorm_rvs(double sigma = 1.0, size_t n = 1) {
    gsl_rng_env_setup();

    const gsl_rng_type * T = gsl_rng_default;
    gsl_rng * r = gsl_rng_alloc(T);

    py::array_t<double> result = py::array_t<double>(n);
    py::buffer_info result_info = result.request();
    double *ptr = static_cast<double *>(result_info.ptr);

    for (size_t i = 0; i<n; i++) {
        ptr[i] = std::abs(gsl_ran_gaussian(r, sigma));
    }
    gsl_rng_free(r);
    return result;
}


py::array_t<double> halfnorm_pdf(py::array_t<double> data, double sigma = 1.0) {
    auto arr = data.unchecked<1>();  // Access the NumPy array data in a C++-friendly way

    // Create a new NumPy array to store the PMF results
    // py::array_t<double> result(size);
    // auto result_mutable = result.mutable_unchecked<1>();

    py::array_t<double> result = py::array_t<double>(arr.shape(0));
    py::buffer_info result_info = result.request();
    double *ptr = static_cast<double *>(result_info.ptr);

    for (size_t i = 0; i < arr.shape(0); i++) {
        if (arr[i] >= 0) {
            ptr[i] = 2.0*gsl_ran_gaussian_pdf(arr[i], sigma);
        } else {
            ptr[i] = -1.0;
        }
    }

    return result;
}

py::array_t<double> halfnorm_pdf_boost(py::array_t<double> data, double sigma = 1.0) {
    auto arr = data.unchecked<1>();  // Access the NumPy array data in a C++-friendly way

    // Create a new NumPy array to store the PMF results
    // py::array_t<double> result(size);
    // auto result_mutable = result.mutable_unchecked<1>();

    py::array_t<double> result = py::array_t<double>(arr.shape(0));
    py::buffer_info result_info = result.request();
    double *ptr = static_cast<double *>(result_info.ptr);

    boost::math::normal dist = boost::math::normal_distribution(0.0, sigma);

    for (size_t i = 0; i < arr.shape(0); i++) {
        double value = arr[i];
        if (value >= 0) {
            ptr[i] = 2.0*boost::math::pdf(dist, arr[i]);
        } else {
            ptr[i] = -1.0;
        }
    }

    return result;
}



PYBIND11_MODULE(gsl, m) {
    m.doc() = "Bindings for fast random number operations.";

    m.def(
        "halfnorm_pdf_boost",
        &halfnorm_pdf_boost,
        py::arg("data"),
        py::arg("sigma") = 1,
        "Compute the halfnormal PDF for an array of random variables.",
        py::return_value_policy::take_ownership
    );
    m.def(
        "halfnorm_pdf",
        &halfnorm_pdf,
        py::arg("data"),
        py::arg("sigma") = 1,
        "Compute the halfnormal PDF for an array of random variables.",
        py::return_value_policy::take_ownership
    );

    m.def(
        "halfnorm_rvs",
        &halfnorm_rvs,
        py::arg("sigma") = 1.0,
        py::arg("n") = 1,
        "Generate samples from the halfnormal distribution.",
        py::return_value_policy::take_ownership
    );
}
