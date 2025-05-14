#include <Python.h>

#include <numpy/arrayobject.h>
#include <numpy/numpyconfig.h>

#include "_tricub.h"

/*****************************************************************

    This file is part of the eqtools package.

    EqTools is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    EqTools is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with EqTools.  If not, see <http://www.gnu.org/licenses/>.

    Copyright 2025 Ian C. Faust

******************************************************************/

#define LENGTH_ASSERT(a, i)                                                    \
  if (check_1d_array(a, i))                                                    \
  return 1

#define ARRAY_SHAPE(a, i) PyArray_DIM((PyArrayObject *)a, i)
#define ARRAY_DATA(a) PyArray_DATA((PyArrayObject *)a)

#define CHECK_ARRAY(a)                                                         \
  if (to_array(&a))                                                            \
  return 1

struct input {
  double *x0;
  double *x1;
  double *x2;
  double *f;
  double *fx0;
  double *fx1;
  double *fx2;
  int ix0;
  int ix1;
  int ix2;
  int ix;
  int d0;
  int d1;
  int d2;
};

static inline int
to_array(PyObject **obj) { /* Check in numpy array, dtype is double, and if the
                            * number of dimensions of the array is correct, then
                            * return the numpy C-contiguous array. Otherwise,
                            * raise a specified Python error and return NULL.
                            */
  PyArrayObject *array = NULL;

  if (!((*obj) && PyArray_Check(*obj))) {
    PyErr_SetString(PyExc_TypeError, "Input is not a numpy.ndarray subtype");
    return 1;
  }
  array = (PyArrayObject *)*obj;

  if (PyArray_TYPE(array) != NPY_DOUBLE) {
    PyErr_SetString(PyExc_TypeError, "array must be dtype double");
    return 1;
  }

  if (!PyArray_ISCARRAY_RO(array))
    array = PyArray_GETCONTIGUOUS(*obj);
    *obj = (PyObject*) array;

  return 0;
}

static inline int check_1d_array(PyObject *array, int length) {
  if (array && ARRAY_SHAPE(array, 0) != length) {
    PyErr_SetString(PyExc_TypeError, "1-d array has incorrect length");
    return 1;
  }
  return 0;
}

static int to_scalar(PyObject *obj, void *output) {
  // if not supplied do nothing
  if (!obj)
    return 0;

  if (PyArray_CheckScalar(obj)) {
    PyErr_SetString(PyExc_TypeError, "Input is not a numpy scalar");
    return 1;
  }

  if (PyArray_TYPE((PyArrayObject *)obj) != NPY_INT) {
    PyErr_SetString(PyExc_TypeError, "scalar must be dtype int");
    return 1;
  }

  PyArray_ScalarAsCtype(obj, output);
  return 0;
}

static inline int parse_input(PyObject *args, struct input *data) {
  PyObject *x0obj, *x1obj, *x2obj, *fobj, *fx0obj, *fx1obj, *fx2obj;
  PyObject *d0obj = NULL, *d1obj = NULL, *d2obj = NULL;
  if (!PyArg_ParseTuple(args, "OOOOOOO|OOO", &x0obj, &x1obj, &x2obj, &fobj,
                       &fx0obj, &fx1obj, &fx2obj, &d0obj, &d1obj, &d2obj))
    return 1;

  CHECK_ARRAY(x0obj);
  CHECK_ARRAY(x1obj);
  CHECK_ARRAY(x2obj);
  CHECK_ARRAY(fobj);
  CHECK_ARRAY(fx0obj);
  CHECK_ARRAY(fx1obj);
  CHECK_ARRAY(fx2obj);

  // get dimension for testing
  data->ix = ARRAY_SHAPE(x0obj, 0);

  LENGTH_ASSERT(x1obj, data->ix);
  LENGTH_ASSERT(x2obj, data->ix);

  // assert f is 3 dimensions
  if (!fobj || PyArray_NDIM((PyArrayObject *)fobj) != 3) {
    PyErr_SetString(PyExc_TypeError, "f array is not 3 dimensional");
    return 1;
  }

  data->ix0 = ARRAY_SHAPE(fobj, 2);
  data->ix1 = ARRAY_SHAPE(fobj, 1);
  data->ix2 = ARRAY_SHAPE(fobj, 0);

  LENGTH_ASSERT(fx0obj, data->ix0);
  LENGTH_ASSERT(fx1obj, data->ix1);
  LENGTH_ASSERT(fx2obj, data->ix2);
  // set values for C code
  data->x0 = (double *)ARRAY_DATA(x0obj);
  data->x1 = (double *)ARRAY_DATA(x1obj);
  data->x2 = (double *)ARRAY_DATA(x2obj);
  data->f = (double *)ARRAY_DATA(fobj);
  data->fx0 = (double *)ARRAY_DATA(fx0obj);
  data->fx1 = (double *)ARRAY_DATA(fx1obj);
  data->fx2 = (double *)ARRAY_DATA(fx2obj);

  if (d0obj && to_scalar(d0obj, &data->d0))
    return 1;
  if (d1obj && to_scalar(d0obj, &data->d1))
    return 1;
  if (d2obj && to_scalar(d0obj, &data->d2))
    return 1;

  return 0;
}

static PyObject *python_reg_ev(
    PyObject *self,
    PyObject *args) { /* If the above function returns -1, an
                       * appropriate Python exception will      have been
                       * set, and the function simply returns NULL
                       */
  double *val;
  struct input s;
  npy_intp length;
  PyObject *output;

  if (parse_input(args, &s))
    return NULL;
  length = (npy_intp)s.ix;
  output = PyArray_SimpleNew(1, &length, NPY_DOUBLE);
  val = (double *)ARRAY_DATA(output);

  reg_ev(val, s.x0, s.x1, s.x2, s.f, s.fx0, s.fx1, s.fx2, s.ix0, s.ix1, s.ix2,
         s.ix);
  return output;
}

static PyObject *python_reg_ev_full(
    PyObject *self,
    PyObject *args) { /* If the above function returns -1, an appropriate Python
                       * exception will have been set, and the function simply
                       * returns NULL
                       */
  double *val;
  struct input s;
  npy_intp length;
  PyObject *output;

  if (parse_input(args, &s))
    return NULL;
  length = (npy_intp)s.ix;
  output = PyArray_SimpleNew(1, &length, NPY_DOUBLE);
  val = (double *)ARRAY_DATA(output);

  reg_ev_full(val, s.x0, s.x1, s.x2, s.f, s.fx0, s.fx1, s.fx2, s.ix0, s.ix1,
              s.ix2, s.ix, s.d0, s.d1, s.d2);
  return output;
}

static PyObject *python_nonreg_ev(
    PyObject *self,
    PyObject *args) { /* If the above function returns -1, an appropriate Python
                       * exception will have been set, and the function simply
                       * returns NULL
                       */
  double *val;
  struct input s;
  npy_intp length;
  PyObject *output;

  if (parse_input(args, &s))
    return NULL;
  length = (npy_intp)s.ix;
  output = PyArray_SimpleNew(1, &length, NPY_DOUBLE);
  val = (double *)ARRAY_DATA(output);

  nonreg_ev(val, s.x0, s.x1, s.x2, s.f, s.fx0, s.fx1, s.fx2, s.ix0, s.ix1,
            s.ix2, s.ix);
  return output;
}

static PyObject *python_nonreg_ev_full(
    PyObject *self,
    PyObject *args) { /* If the above function returns -1, an appropriate Python
                       * exception will have been set, and the function simply
                       * returns NULL
                       */
  double *val;
  struct input s;
  npy_intp length;
  PyObject *output;

  if (parse_input(args, &s))
    return NULL;
  length = (npy_intp)s.ix;
  output = PyArray_SimpleNew(1, &length, NPY_DOUBLE);
  val = (double *)ARRAY_DATA(output);

  nonreg_ev_full(val, s.x0, s.x1, s.x2, s.f, s.fx0, s.fx1, s.fx2, s.ix0, s.ix1,
                 s.ix2, s.ix, s.d0, s.d1, s.d2);
  return output;
}

static PyObject *
python_ev(PyObject *self,
          PyObject *args) { /* If the above function returns -1, an appropriate
                             * Python exception will have been set, and the
                             * function simply returns NULL
                             */
  double *val;
  struct input s;
  npy_intp length;
  PyObject *output;

  if (parse_input(args, &s))
    return NULL;
  length = (npy_intp)s.ix;
  output = PyArray_SimpleNew(1, &length, NPY_DOUBLE);
  val = (double *)ARRAY_DATA(output);

  ev(val, s.x0, s.x1, s.x2, s.f, s.fx0, s.fx1, s.fx2, s.ix0, s.ix1, s.ix2,
     s.ix);
  return output;
}

static PyObject *python_ismonotonic(PyObject *self, PyObject *arg) {
  int ix;
  double *data;

  if (to_array(&arg))
    return NULL;
  ix = ARRAY_SHAPE(arg, 0);
  data = (double *)ARRAY_DATA(arg);
  return PyBool_FromLong(ismonotonic(data, ix));
}

static PyObject *python_isregular(PyObject *self, PyObject *arg) {
  int ix;
  double *data;
  if (to_array(&arg))
    return NULL;

  ix = ARRAY_SHAPE(arg, 0);
  data = (double *)ARRAY_DATA(arg);
  return PyBool_FromLong(isregular(data, ix));
}

static PyMethodDef TricubMethods[] = {
    {"reg_ev", python_reg_ev, METH_VARARGS, ""},
    {"reg_ev_full", python_reg_ev_full, METH_VARARGS, ""},
    {"nonreg_ev", python_nonreg_ev, METH_VARARGS, ""},
    {"nonreg_ev_full", python_nonreg_ev_full, METH_VARARGS, ""},
    {"ev", python_ev, METH_VARARGS, ""},
    {"ismonotonic", python_ismonotonic, METH_O, ""},
    {"isregular", python_isregular, METH_O, ""},
    {NULL, NULL, 0, NULL} /* Sentinel */
};

static struct PyModuleDef _tricubStruct = {PyModuleDef_HEAD_INIT,
                                           "_tricub",
                                           "",
                                           -1,
                                           TricubMethods,
                                           NULL,
                                           NULL,
                                           NULL,
                                           NULL};

/* Module initialization */
PyObject *PyInit__tricub(void) {
  import_array();
  return PyModule_Create(&_tricubStruct);
}
