
#include "profile.h"

#include "frameobject.h"


// Functions IS_SUSPENDED and IS_ASYNC were copied from
// https://github.com/sumerc/yappi/blob/master/yappi/_yappi.c
#if PY_MAJOR_VERSION >= 3
#define IS_PY3K
#endif

static PyCodeObject *
FRAME2CODE(PyFrameObject *frame) {
#if PY_MAJOR_VERSION >= 3 && PY_MINOR_VERSION >= 10
    return PyFrame_GetCode(frame);
#else
    return frame->f_code;
#endif
}

int IS_SUSPENDED(PyFrameObject *frame) {
#if PY_MAJOR_VERSION >= 3 && PY_MINOR_VERSION == 11
    PyGenObject *gen = (PyGenObject *)PyFrame_GetGenerator(frame);
    if (gen == NULL) {
        return 0;
    }

    // -1 is FRAME_SUSPENDED. See internal/pycore_frame.h
    // TODO: Remove these after 3.12 make necessary public APIs.
    // See https://discuss.python.org/t/python-3-11-frame-structure-and-various-changes/17895
    return gen->gi_frame_state == -1;
#elif PY_MAJOR_VERSION >= 3 && PY_MINOR_VERSION == 10
    return (frame->f_state == FRAME_SUSPENDED);
#else
    return (frame->f_stacktop != NULL);
#endif
}

int IS_ASYNC(PyFrameObject *frame)
{
    int result = 0;

#if defined(IS_PY3K)
#if PY_MINOR_VERSION >= 4
    result = FRAME2CODE(frame)->co_flags & CO_COROUTINE ||
        FRAME2CODE(frame)->co_flags & CO_ITERABLE_COROUTINE;
#endif
#if PY_MINOR_VERSION >= 6
    result = result || FRAME2CODE(frame)->co_flags & CO_ASYNC_GENERATOR;
#endif
#endif

    return result;
}

static PyObject*
is_suspended(PyObject *self, PyObject *args)
{
    PyFrameObject* frame;
    if (!PyArg_ParseTuple(args, "O", &frame)) {
        return NULL;
    }

    int result = IS_SUSPENDED(frame);

    return PyBool_FromLong(result);
}

static PyObject*
is_async(PyObject *self, PyObject *args)
{
    PyFrameObject* frame;

    if (!PyArg_ParseTuple(args, "O", &frame)) {
        return NULL;
    }

    int result = IS_ASYNC(frame);

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
