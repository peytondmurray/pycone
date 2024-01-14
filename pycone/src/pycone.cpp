#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

/**
 * @brief Calculate delta-T for a given site and duration.
 *
 * This function makes use of the buffer protocol; see
 * https://docs.python.org/3/c-api/buffer.html and
 * https://pybind11.readthedocs.io/en/stable/advanced/pycpp/numpy.html
 * for more information.
 *
 * @param start1 Interval start for year 1
 * @param mean_t1 Mean temperature for the interval for year 1
 * @param start2 Interval start for year 2
 * @param mean_t2 Mean temperature for the interval for year 2
 * @return Tuple of three arrays:
 *      (delta_t(year2 - year1), year1, year2)
 */
py::tuple fast_delta_t_site_duration(
    py::array_t<short> start1,
    py::array_t<float> mean_t1,
    py::array_t<short> start2,
    py::array_t<float> mean_t2
) {
    py::buffer_info start1_bufinfo = start1.request();
    py::buffer_info start2_bufinfo = start2.request();
    py::buffer_info mean_t1_bufinfo = mean_t1.request();
    py::buffer_info mean_t2_bufinfo = mean_t2.request();

    if (
        start1_bufinfo.ndim != 1 ||
        start2_bufinfo.ndim != 1 ||
        mean_t1_bufinfo.ndim != 1 ||
        mean_t2_bufinfo.ndim != 1
    ) {
        throw std::runtime_error("All inputs must be 1-dimensional");
    }

    short *pStart1 = static_cast<short *>(start1_bufinfo.ptr);
    short *pStart2 = static_cast<short *>(start2_bufinfo.ptr);
    float *pMeanT1 = static_cast<float *>(mean_t1_bufinfo.ptr);
    float *pMeanT2 = static_cast<float *>(mean_t2_bufinfo.ptr);

    py::ssize_t outLen = start1_bufinfo.shape[0]*start2_bufinfo.shape[0];

    // Let numpy allocate the new buffers
    py::array_t<float> arrDt = py::array_t<float>(outLen);
    py::array_t<short> arrStart1 = py::array_t<short>(outLen);
    py::array_t<short> arrStart2 = py::array_t<short>(outLen);

    // Grab the buffer info via the buffer protocol
    py::buffer_info buf_arrDt = arrDt.request();
    py::buffer_info buf_arrStart1 = arrStart1.request();
    py::buffer_info buf_arrStart2 = arrStart2.request();

    // Cast the poshorters for each array to the appropriate type
    float *pbuf_arrDt = static_cast<float *>(buf_arrDt.ptr);
    short *pbuf_arrStart1 = static_cast<short *>(buf_arrStart1.ptr);
    short *pbuf_arrStart2 = static_cast<short *>(buf_arrStart2.ptr);

    int k = 0;
    for (int i=0; i<start1_bufinfo.shape[0]; i++) {
        for (int j=0; j<start2_bufinfo.shape[0]; j++) {
            pbuf_arrDt[k] = pMeanT2[j] - pMeanT1[i];
            pbuf_arrStart1[k] = pStart1[i];
            pbuf_arrStart2[k] = pStart2[j];
            k++;
        }
    }

    return py::make_tuple(
        arrDt, arrStart1, arrStart2
    );
}


PYBIND11_MODULE(_pycone_main, m) {
    m.doc() = "Bindings for fast compiled pycone functions.";
    m.def(
        "fast_delta_t_site_duration",
        &fast_delta_t_site_duration,
        "A fast implementation of the delta-t calculation",
        py::return_value_policy::take_ownership
    );
}
