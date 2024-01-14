#include <Python.h>
#include "numpy/arrayobject.h"

static PyObject *fast_delta_t_site_duration(PyObject *NPY_UNUSED(self), PyObject *args) {

    PyArrayObject *start1, *mean_t1, *start2, *mean_t2;
    if (
        !PyArg_ParseTuple(
            args,
            "O!O!O!O!",
            &PyArray_Type,
            &start1,
            &PyArray_Type,
            &mean_t1,
            &PyArray_Type,
            &start2,
            &PyArray_Type,
            &mean_t2
        )
    ) {
        PyErr_SetString(
            PyExc_TypeError,
            "Unable to parse the array arguments."
        );
        return NULL;
    }

    if (start1->nd != 1 || mean_t2->nd != 1 || start2->nd != 1 || mean_t2->nd != 1) {
        PyErr_SetString(
            PyExc_TypeError,
            "Can only be called on 1-dimensional arrays."
        );
        return NULL;
    }

    if (
        start1->dimensions[0] != 1 ||
        mean_t1->dimensions[0] != 1 ||
        start2->dimensions[0] != 1 ||
        mean_t2->dimensions[0] != 1
    ) {
        PyErr_SetString(
            PyExc_TypeError,
            "All array dimensions must match."
        );
        return NULL;
    }

    int n_result = start1->dimensions[0]*start2->dimensions[0];
    int *start1_buf = (int *)malloc(sizeof(int)*n_result);
    int *start2_buf = (int *)malloc(sizeof(int)*n_result);
    double *dt_buf = (double *)malloc(sizeof(double)*n_result);

    for (int i=0; i<start1->dimensions[0]; i++) {
        for (int j=0; j<start2->dimensions[0]; j++) {
            start1_buf[];
        }
    }

    PyObject *out_start1 = PyArray_SimpleNewFromData(
        1, (npy_intp *)&n_result, NPY_DOUBLE, start1_buf
    );
    if (out_start1 == NULL) {
        PyErr_SetString(
            PyExc_TypeError,
            "Problem creating output start1 array."
        );
        return NULL;
    };
    PyObject *out_start2 = PyArray_SimpleNewFromData(
        1, (npy_intp *)&n_result, NPY_DOUBLE, start2_buf
    );
    if (out_start2 == NULL) {
        PyErr_SetString(
            PyExc_TypeError,
            "Problem creating output start2 array."
        );
        return NULL;
    };
    PyObject *out_dt = PyArray_SimpleNewFromData(
        1, (npy_intp *)&n_result, NPY_DOUBLE, dt_buf
    );
    if (out_dt == NULL) {
        PyErr_SetString(
            PyExc_TypeError,
            "Problem creating delta-T array."
        );
        return NULL;
    };

    PyObject *out = PyTuple_Pack(3, out_start1, out_start2, out_dt);
    if (out == NULL) {
        PyErr_SetString(
            PyExc_TypeError,
            "Problem packing the computed delta-t values into a tuple."
        );
        return NULL;
    }

    return out;
}


static PyMethodDef methods[] = {
    {
        "fast_delta_t_site_duration",
        fast_delta_t_site_duration,
        METH_VARARGS,
        "Compute delta-T for a given site and duration"
    },
    {NULL, NULL, 0, NULL} // Sentinel
};

static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        .m_name = "pycone_main",
        .m_size = -1,
        .m_methods = methods,
};

/* Module initialization function */
PyMODINIT_FUNC
init_pycone(void)
{
    import_array();

    PyObject *m = PyModule_Create(&moduledef);
    if (m == NULL) {
        return NULL;
    }
    return m;
}
