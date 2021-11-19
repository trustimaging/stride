
#include "profile.h"

#include "frameobject.h"


static PyObject*
is_suspended(PyObject *self, PyObject *args)
{
    PyFrameObject* frame;

    if (!PyArg_ParseTuple(args, "O", &frame)) {
        return NULL;
    }

    int result = 0;

#if PY_MAJOR_VERSION >= 3 && PY_MINOR_VERSION >= 10
    result = (frame->f_state == FRAME_SUSPENDED);
#else
    result = (frame->f_stacktop != NULL);
#endif

    return PyBool_FromLong(result);
}

static PyObject*
is_async(PyObject *self, PyObject *args)
{
    PyFrameObject* frame;

    if (!PyArg_ParseTuple(args, "O", &frame)) {
        return NULL;
    }

    int result = 0;

#if PY_MINOR_VERSION >= 4
    result = frame->f_code->co_flags & CO_COROUTINE ||
        frame->f_code->co_flags & CO_ITERABLE_COROUTINE;
#endif
#if PY_MINOR_VERSION >= 6
    result = result || frame->f_code->co_flags & CO_ASYNC_GENERATOR;
#endif

    return PyBool_FromLong(result);
}

static PyMethodDef profile_methods[] = {
    {"is_suspended", is_suspended, METH_VARARGS, NULL},
    {"is_async", is_async, METH_VARARGS, NULL},
    {NULL, NULL, 0, NULL}      /* sentinel */
};

static struct PyModuleDef _profile_module = {
    PyModuleDef_HEAD_INIT,
    "_profile",
    "C Utils for Mosaic profiling",
    -1,
    profile_methods,
    NULL,
    NULL,
    NULL,
    NULL
};

PyMODINIT_FUNC  PyInit__profile(void) {
    return PyModule_Create(&_profile_module);
}
