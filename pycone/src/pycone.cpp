#include <set>
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

    // Cast the pointers for each array to the appropriate type
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

// float getDeltaT() {
//     pSite, pYear, pStart, pDuration, pMean, s, y1, y2, s1, s2, d, site_bufinfo.size
//     return 0;
// }
//
// py::tuple fast_delta_t(
//     py::array_t<short> site,
//     py::array_t<short> year,
//     py::array_t<short> start,
//     py::array_t<short> duration,
//     py::array_t<float> mean_t,
//     int delta_t_year_gap,
//     int crop_year_gap
// ) {
//     py::buffer_info site_bufinfo = site.request();
//     py::buffer_info year_bufinfo = year.request();
//     py::buffer_info start_bufinfo = start.request();
//     py::buffer_info duration_bufinfo = duration.request();
//     py::buffer_info mean_t_bufinfo = mean_t.request();
//
//     short* pSite = static_cast<short *>(site_bufinfo.ptr);
//     short* pYear = static_cast<short *>(year_bufinfo.ptr);
//     short* pStart = static_cast<short *>(start_bufinfo.ptr);
//     short* pDuration = static_cast<short *>(duration_bufinfo.ptr);
//     float* pMean = static_cast<float *>(mean_t_bufinfo.ptr);
//
//     // Get the unique sites, years, starts, and durations
//     std::set<short> uniqueSites(pSite, pSite + site_bufinfo.size);
//     std::set<short> uniqueYears(pYear, pYear + year_bufinfo.size);
//     std::set<short> uniqueStarts(pStart, pStart + start_bufinfo.size);
//     std::set<short> uniqueDurations(pDuration, pDuration + duration_bufinfo.size);
//
//     // Compute the length:
//     // - For each unique site
//     // - Each pair of unique years (uniqueYears.size() - delta_t_year_gap)
//     // - Each unique duration
//     // - Each start1 and start2
//     //
//     // Some years will have fewer days; these will be nan-valued
//     py::ssize_t outLen = uniqueSites.size()
//         *(uniqueYears.size() - 1)
//         *uniqueDurations.size()
//         *uniqueStarts.size()
//         *uniqueStarts.size();
//
//     py::array_t<short> outSite(outLen);
//     py::array_t<short> outYear1(outLen);
//     py::array_t<short> outYear2(outLen);
//     py::array_t<short> outStart1(outLen);
//     py::array_t<short> outStart2(outLen);
//     py::array_t<short> outDuration(outLen);
//     py::array_t<float> outDeltaT(outLen);
//     py::array_t<short> outCropYear(outLen);
//
//     py::buffer_info outSite_bufinfo = outSite.request();
//     py::buffer_info outYear1_bufinfo = outYear1.request();
//     py::buffer_info outYear2_bufinfo = outYear2.request();
//     py::buffer_info outStart1_bufinfo = outStart1.request();
//     py::buffer_info outStart2_bufinfo = outStart2.request();
//     py::buffer_info outDuration_bufinfo = outDuration.request();
//     py::buffer_info outDeltaT_bufinfo = outDeltaT.request();
//     py::buffer_info outCropYear_bufinfo = outCropYear.request();
//
//     short* pOutSite = static_cast<short *>(outSite_bufinfo.ptr);
//     short* pOutYear1 = static_cast<short *>(outYear1_bufinfo.ptr);
//     short* pOutYear2 = static_cast<short *>(outYear2_bufinfo.ptr);
//     short* pOutStart1 = static_cast<short *>(outStart1_bufinfo.ptr);
//     short* pOutStart2 = static_cast<short *>(outStart2_bufinfo.ptr);
//     short* pOutDuration = static_cast<short *>(outDuration_bufinfo.ptr);
//     float* pOutDeltaT = static_cast<float *>(outDeltaT_bufinfo.ptr);
//     short* pOutCropYear = static_cast<short *>(outCropYear_bufinfo.ptr);
//
//
//     int k = 0;
//     for (short s: uniqueSites) {
//         // Assume that there is roughly equal data for each site
//         std::vector<int> indexBufSite = std::vector<int>(site_bufinfo.size/uniqueSites.size());
//
//         for (auto yearIt=uniqueYears.begin(); (*yearIt + delta_t_year_gap)!=*uniqueYears.end(); yearIt++) {
//             short y1 = *yearIt;
//             short y2 = y1 + delta_t_year_gap;
//             short cy = y2 + crop_year_gap;
//             for (short d: uniqueDurations) {
//                 for (short s1: uniqueStarts) {
//                     for (short s2: uniqueStarts) {
//
//                         getDeltaT(
//                             pSite, pYear, pStart, pDuration, pMean, s, y1, y2, s1, s2, d, site_bufinfo.size
//                         );
//
//
//                         pOutSite[k] = s;
//                         pOutYear1[k] = y1;
//                         pOutYear2[k] = y2;
//                         pOutStart1[k] = s1;
//                         pOutStart2[k] = s2;
//                         pOutDuration[k] = d;
//                         pOutDeltaT[k] = 0;
//                         pOutCropYear[k] = cy;
//                         k++;
//                     }
//                 }
//             }
//         }
//     }
//
//
//     return py::make_tuple(
//         outSite, outYear1, outYear2, outStart1, outStart2, outDuration, outDeltaT, outCropYear
//     );
// }


PYBIND11_MODULE(_pycone_main, m) {
    m.doc() = "Bindings for fast compiled pycone functions.";
    // m.def(
    //     "fast_delta_t",
    //     &fast_delta_t,
    //     "A fast implementation of the delta-t calculation",
    //     py::return_value_policy::take_ownership
    // );
    m.def(
        "fast_delta_t_site_duration",
        &fast_delta_t_site_duration,
        "A fast implementation of the delta-t calculation",
        py::return_value_policy::take_ownership
    );
}
