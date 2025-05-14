import unittest
import numpy

import eqtools
import os
import warnings

try:
    shot = 1120914027
    # Run tests with both of these to be sure that tspline does everything right:
    e = eqtools.CModEFITTree(shot)
    if eqtools.core._has_trispline:
        et = eqtools.CModEFITTree(shot, tspline=True)
    else:
        warnings.warn(
            "Trispline is not compiled and will not be tested.",
            RuntimeWarning
        )
        et = e
except:
    warnings.warn(
        "Could not access MDSplus data. Defaulting to pickled data. You may want "
        "to modify unittests.py to use your own local data system to ensure "
        "consistency for your use case.",
        RuntimeWarning
    )
    try:
        import cPickle as pkl
    except:
        import pickle as pkl
    dir_path = os.path.dirname(os.path.realpath(__file__))    
    with open(os.sep.join([dir_path, 'test_data.pkl']), 'rb') as f:
        shot, e, et = pkl.load(f, encoding='latin1')
        if not eqtools.core._has_trispline:
            et = e

scalar_R = 0.75
scalar_Z = 0.1
scalar_t = e.getTimeBase()[45]
vector_R = e.getRGrid()
vector_Z = e.getZGrid()
vector_t = e.getTimeBase()[10:-10]
matrix_R, matrix_Z = numpy.meshgrid(vector_R, vector_Z)
matrix_t = numpy.linspace(
    vector_t.min(),
    vector_t.max(),
    len(vector_R) * len(vector_Z)
).reshape(matrix_R.shape)
# psinorm may be used as a stand-in for any normalized variable.
scalar_psinorm = 0.5
vector_psinorm = numpy.linspace(0.0, 1.0, 100)
matrix_psinorm = numpy.linspace(
    0, 1.0, len(vector_R) * len(vector_Z)
).reshape(matrix_R.shape)
scalar_Rmid = 0.75
vector_Rmid = e.getRmidPsi()[10, :]
matrix_Rmid = e.getRmidPsi()[:len(vector_R), :len(vector_Z)]

tol = 0.01
places = 4


class rz2psiTestCase(unittest.TestCase):
    # The following tests focus primarily on the consistency of the input/output
    # shapes. More tests should be added to check the consistency between the
    # different mapping routines.

    # scalar R, scalar Z, scalar t:
    def test_rz2psi_scalar_R_scalar_Z_scalar_t_make_grid_False_each_t_True(self):
        out = e.rz2psi(scalar_R, scalar_Z, scalar_t)
        out_t = et.rz2psi(scalar_R, scalar_Z, scalar_t)

        with self.assertRaises(TypeError):
            iter(out)
        self.assertFalse(numpy.isnan(out))
        self.assertFalse(numpy.isinf(out))

        with self.assertRaises(TypeError):
            iter(out_t)
        self.assertFalse(numpy.isnan(out_t))
        self.assertFalse(numpy.isinf(out_t))

        diff = numpy.absolute(out - out_t) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    def test_rz2psi_scalar_R_scalar_Z_scalar_t_make_grid_True_each_t_True(self):
        with self.assertRaises(ValueError):
            out = e.rz2psi(scalar_R, scalar_Z, scalar_t, make_grid=True)
        with self.assertRaises(ValueError):
            out_t = et.rz2psi(scalar_R, scalar_Z, scalar_t, make_grid=True)

    def test_rz2psi_scalar_R_scalar_Z_scalar_t_make_grid_False_each_t_False(self):
        out = e.rz2psi(scalar_R, scalar_Z, scalar_t, each_t=False)
        out_t = et.rz2psi(scalar_R, scalar_Z, scalar_t, each_t=False)

        with self.assertRaises(TypeError):
            iter(out)
        self.assertFalse(numpy.isnan(out))
        self.assertFalse(numpy.isinf(out))

        with self.assertRaises(TypeError):
            iter(out_t)
        self.assertFalse(numpy.isnan(out_t))
        self.assertFalse(numpy.isinf(out_t))

        diff = numpy.absolute(out - out_t) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    def test_rz2psi_scalar_R_scalar_Z_scalar_t_make_grid_True_each_t_False(self):
        with self.assertRaises(ValueError):
            out = e.rz2psi(scalar_R, scalar_Z, scalar_t, make_grid=True, each_t=False)
        with self.assertRaises(ValueError):
            out_t = et.rz2psi(scalar_R, scalar_Z, scalar_t, make_grid=True, each_t=False)

    # vector R, vector Z, scalar t:
    def test_rz2psi_vector_R_vector_Z_scalar_t_make_grid_False_each_t_True(self):
        out = e.rz2psi(vector_R, vector_Z, scalar_t)
        out_t = et.rz2psi(vector_R, vector_Z, scalar_t)

        self.assertEqual(out.shape, (len(vector_R),))
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, (len(vector_R),))
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    def test_rz2psi_vector_R_vector_Z_scalar_t_make_grid_True_each_t_True(self):
        out = e.rz2psi(vector_R, vector_Z, scalar_t, make_grid=True)
        out_t = et.rz2psi(vector_R, vector_Z, scalar_t, make_grid=True)

        self.assertEqual(out.shape, (len(vector_R), len(vector_Z)))
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, (len(vector_R), len(vector_Z)))
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    def test_rz2psi_vector_R_vector_Z_scalar_t_make_grid_False_each_t_False(self):
        out = e.rz2psi(vector_R, vector_Z, scalar_t, each_t=False)
        out_t = et.rz2psi(vector_R, vector_Z, scalar_t, each_t=False)

        self.assertEqual(out.shape, (len(vector_R),))
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, (len(vector_R),))
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    def test_rz2psi_vector_R_vector_Z_scalar_t_make_grid_True_each_t_False(self):
        out = e.rz2psi(vector_R, vector_Z, scalar_t, make_grid=True, each_t=False)
        out_t = et.rz2psi(vector_R, vector_Z, scalar_t, make_grid=True, each_t=False)

        self.assertEqual(out.shape, (len(vector_R), len(vector_Z)))
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, (len(vector_R), len(vector_Z)))
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    # matrix R, matrix Z, scalar t:
    def test_rz2psi_matrix_R_matrix_Z_scalar_t_make_grid_False_each_t_True(self):
        out = e.rz2psi(matrix_R, matrix_Z, scalar_t)
        out_t = et.rz2psi(matrix_R, matrix_Z, scalar_t)

        self.assertEqual(out.shape, matrix_R.shape)
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, matrix_R.shape)
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    def test_rz2psi_matrix_R_matrix_Z_scalar_t_make_grid_True_each_t_True(self):
        with self.assertRaises(ValueError):
            out = e.rz2psi(matrix_R, matrix_Z, scalar_t, make_grid=True)
        with self.assertRaises(ValueError):
            out_t = et.rz2psi(matrix_R, matrix_Z, scalar_t, make_grid=True)

    def test_rz2psi_matrix_R_matrix_Z_scalar_t_make_grid_False_each_t_False(self):
        out = e.rz2psi(matrix_R, matrix_Z, scalar_t, each_t=False)
        out_t = et.rz2psi(matrix_R, matrix_Z, scalar_t, each_t=False)

        self.assertEqual(out.shape, matrix_R.shape)
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, matrix_R.shape)
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    def test_rz2psi_matrix_R_matrix_Z_scalar_t_make_grid_True_each_t_False(self):
        with self.assertRaises(ValueError):
            out = e.rz2psi(matrix_R, matrix_Z, scalar_t, make_grid=True, each_t=False)
        with self.assertRaises(ValueError):
            out_t = et.rz2psi(matrix_R, matrix_Z, scalar_t, make_grid=True, each_t=False)

    # scalar R, scalar Z, vector t:
    def test_rz2psi_scalar_R_scalar_Z_vector_t_make_grid_False_each_t_True(self):
        out = e.rz2psi(scalar_R, scalar_Z, vector_t)
        out_t = et.rz2psi(scalar_R, scalar_Z, vector_t)

        self.assertEqual(out.shape, (len(vector_t),))
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, (len(vector_t),))
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    def test_rz2psi_scalar_R_scalar_Z_vector_t_make_grid_True_each_t_True(self):
        with self.assertRaises(ValueError):
            out = e.rz2psi(scalar_R, scalar_Z, vector_t, make_grid=True)
        with self.assertRaises(ValueError):
            out_t = et.rz2psi(scalar_R, scalar_Z, vector_t, make_grid=True)

    def test_rz2psi_scalar_R_scalar_Z_vector_t_make_grid_False_each_t_False(self):
        with self.assertRaises(ValueError):
            out = e.rz2psi(scalar_R, scalar_Z, vector_t, each_t=False)
        with self.assertRaises(ValueError):
            out_t = et.rz2psi(scalar_R, scalar_Z, vector_t, each_t=False)

    def test_rz2psi_scalar_R_scalar_Z_vector_t_make_grid_True_each_t_False(self):
        with self.assertRaises(ValueError):
            out = e.rz2psi(scalar_R, scalar_Z, vector_t, make_grid=True, each_t=False)
        with self.assertRaises(ValueError):
            out_t = et.rz2psi(scalar_R, scalar_Z, vector_t, make_grid=True, each_t=False)

    # vector R, vector Z, vector t:
    def test_rz2psi_vector_R_vector_Z_vector_t_make_grid_False_each_t_True(self):
        out = e.rz2psi(vector_R, vector_Z, vector_t)
        out_t = et.rz2psi(vector_R, vector_Z, vector_t)

        self.assertEqual(out.shape, (len(vector_t), len(vector_R)))
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, (len(vector_t), len(vector_R)))
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    def test_rz2psi_vector_R_vector_Z_vector_t_make_grid_True_each_t_True(self):
        out = e.rz2psi(vector_R, vector_Z, vector_t, make_grid=True)
        out_t = et.rz2psi(vector_R, vector_Z, vector_t, make_grid=True)

        self.assertEqual(out.shape, (len(vector_t), len(vector_R), len(vector_Z)))
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, (len(vector_t), len(vector_R), len(vector_Z)))
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    def test_rz2psi_vector_R_vector_Z_vector_t_make_grid_False_each_t_False(self):
        with self.assertRaises(ValueError):
            out = e.rz2psi(vector_R, vector_Z, vector_t, each_t=False)
        with self.assertRaises(ValueError):
            out_t = et.rz2psi(vector_R, vector_Z, vector_t, each_t=False)

    def test_rz2psi_vector_R_vector_Z_vector_t_make_grid_True_each_t_False(self):
        with self.assertRaises(ValueError):
            out = e.rz2psi(vector_R, vector_Z, vector_t, make_grid=True, each_t=False)
        with self.assertRaises(ValueError):
            out_t = et.rz2psi(vector_R, vector_Z, vector_t, make_grid=True, each_t=False)

    # matrix R, matrix Z, vector t:
    def test_rz2psi_matrix_R_matrix_Z_vector_t_make_grid_False_each_t_True(self):
        out = e.rz2psi(matrix_R, matrix_Z, vector_t)
        out_t = et.rz2psi(matrix_R, matrix_Z, vector_t)

        self.assertEqual(out.shape, tuple([len(vector_t),] + list(matrix_R.shape)))
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, tuple([len(vector_t),] + list(matrix_R.shape)))
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    def test_rz2psi_matrix_R_matrix_Z_vector_t_make_grid_True_each_t_True(self):
        with self.assertRaises(ValueError):
            out = e.rz2psi(matrix_R, matrix_Z, vector_t, make_grid=True)
        with self.assertRaises(ValueError):
            out_t = et.rz2psi(matrix_R, matrix_Z, vector_t, make_grid=True)

    def test_rz2psi_matrix_R_matrix_Z_vector_t_make_grid_False_each_t_False(self):
        with self.assertRaises(ValueError):
            out = e.rz2psi(matrix_R, matrix_Z, vector_t, each_t=False)
        with self.assertRaises(ValueError):
            out_t = et.rz2psi(matrix_R, matrix_Z, vector_t, each_t=False)

    def test_rz2psi_matrix_R_matrix_Z_vector_t_make_grid_True_each_t_False(self):
        with self.assertRaises(ValueError):
            out = e.rz2psi(matrix_R, matrix_Z, vector_t, make_grid=True, each_t=False)
        with self.assertRaises(ValueError):
            out_t = et.rz2psi(matrix_R, matrix_Z, vector_t, make_grid=True, each_t=False)

    # matrix R, matrix Z, matrix t:
    def test_rz2psi_matrix_R_matrix_Z_matrix_t_make_grid_False_each_t_True(self):
        with self.assertRaises(ValueError):
            out = e.rz2psi(matrix_R, matrix_Z, matrix_t)
        with self.assertRaises(ValueError):
            out_t = et.rz2psi(matrix_R, matrix_Z, matrix_t)

    def test_rz2psi_matrix_R_matrix_Z_matrix_t_make_grid_True_each_t_True(self):
        with self.assertRaises(ValueError):
            out = e.rz2psi(matrix_R, matrix_Z, matrix_t, make_grid=True)
        with self.assertRaises(ValueError):
            out_t = et.rz2psi(matrix_R, matrix_Z, matrix_t, make_grid=True)

    def test_rz2psi_matrix_R_matrix_Z_matrix_t_make_grid_False_each_t_False(self):
        out = e.rz2psi(matrix_R, matrix_Z, matrix_t, each_t=False)
        out_t = et.rz2psi(matrix_R, matrix_Z, matrix_t, each_t=False)

        self.assertEqual(out.shape, matrix_R.shape)
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, matrix_R.shape)
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        # TODO: Only good to 2%.
        diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertLessEqual(diff, 0.02)

    def test_rz2psi_matrix_R_matrix_Z_matrix_t_make_grid_True_each_t_False(self):
        with self.assertRaises(ValueError):
            out = e.rz2psi(matrix_R, matrix_Z, matrix_t, make_grid=True, each_t=False)
        with self.assertRaises(ValueError):
            out_t = et.rz2psi(matrix_R, matrix_Z, matrix_t, make_grid=True, each_t=False)

class rz2psinormTestCase(unittest.TestCase):
    # The following tests focus primarily on the consistency of the input/output
    # shapes. More tests should be added to check the consistency between the
    # different mapping routines.

    # scalar R, scalar Z, scalar t:
    def test_rz2psinorm_scalar_R_scalar_Z_scalar_t_make_grid_False_each_t_True(self):
        out = e.rz2psinorm(scalar_R, scalar_Z, scalar_t)
        out_t = et.rz2psinorm(scalar_R, scalar_Z, scalar_t)

        with self.assertRaises(TypeError):
            iter(out)
        self.assertFalse(numpy.isnan(out))
        self.assertFalse(numpy.isinf(out))

        with self.assertRaises(TypeError):
            iter(out_t)
        self.assertFalse(numpy.isnan(out_t))
        self.assertFalse(numpy.isinf(out_t))

        diff = numpy.absolute(out - out_t) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    def test_rz2psinorm_scalar_R_scalar_Z_scalar_t_make_grid_True_each_t_True(self):
        with self.assertRaises(ValueError):
            out = e.rz2psinorm(scalar_R, scalar_Z, scalar_t, make_grid=True)
        with self.assertRaises(ValueError):
            out_t = et.rz2psinorm(scalar_R, scalar_Z, scalar_t, make_grid=True)

    def test_rz2psinorm_scalar_R_scalar_Z_scalar_t_make_grid_False_each_t_False(self):
        out = e.rz2psinorm(scalar_R, scalar_Z, scalar_t, each_t=False)
        out_t = et.rz2psinorm(scalar_R, scalar_Z, scalar_t, each_t=False)

        with self.assertRaises(TypeError):
            iter(out)
        self.assertFalse(numpy.isnan(out))
        self.assertFalse(numpy.isinf(out))

        with self.assertRaises(TypeError):
            iter(out_t)
        self.assertFalse(numpy.isnan(out_t))
        self.assertFalse(numpy.isinf(out_t))

        diff = numpy.absolute(out - out_t) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    def test_rz2psinorm_scalar_R_scalar_Z_scalar_t_make_grid_True_each_t_False(self):
        with self.assertRaises(ValueError):
            out = e.rz2psinorm(scalar_R, scalar_Z, scalar_t, make_grid=True, each_t=False)
        with self.assertRaises(ValueError):
            out_t = et.rz2psinorm(scalar_R, scalar_Z, scalar_t, make_grid=True, each_t=False)

    # vector R, vector Z, scalar t:
    def test_rz2psinorm_vector_R_vector_Z_scalar_t_make_grid_False_each_t_True(self):
        out = e.rz2psinorm(vector_R, vector_Z, scalar_t)
        out_t = et.rz2psinorm(vector_R, vector_Z, scalar_t)

        self.assertEqual(out.shape, (len(vector_R),))
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, (len(vector_R),))
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    def test_rz2psinorm_vector_R_vector_Z_scalar_t_make_grid_True_each_t_True(self):
        out = e.rz2psinorm(vector_R, vector_Z, scalar_t, make_grid=True)
        out_t = et.rz2psinorm(vector_R, vector_Z, scalar_t, make_grid=True)

        self.assertEqual(out.shape, (len(vector_R), len(vector_Z)))
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, (len(vector_R), len(vector_Z)))
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    def test_rz2psinorm_vector_R_vector_Z_scalar_t_make_grid_False_each_t_False(self):
        out = e.rz2psinorm(vector_R, vector_Z, scalar_t, each_t=False)
        out_t = et.rz2psinorm(vector_R, vector_Z, scalar_t, each_t=False)

        self.assertEqual(out.shape, (len(vector_R),))
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, (len(vector_R),))
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    def test_rz2psinorm_vector_R_vector_Z_scalar_t_make_grid_True_each_t_False(self):
        out = e.rz2psinorm(vector_R, vector_Z, scalar_t, make_grid=True, each_t=False)
        out_t = et.rz2psinorm(vector_R, vector_Z, scalar_t, make_grid=True, each_t=False)

        self.assertEqual(out.shape, (len(vector_R), len(vector_Z)))
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, (len(vector_R), len(vector_Z)))
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    # matrix R, matrix Z, scalar t:
    def test_rz2psinorm_matrix_R_matrix_Z_scalar_t_make_grid_False_each_t_True(self):
        out = e.rz2psinorm(matrix_R, matrix_Z, scalar_t)
        out_t = et.rz2psinorm(matrix_R, matrix_Z, scalar_t)

        self.assertEqual(out.shape, matrix_R.shape)
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, matrix_R.shape)
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    def test_rz2psinorm_matrix_R_matrix_Z_scalar_t_make_grid_True_each_t_True(self):
        with self.assertRaises(ValueError):
            out = e.rz2psinorm(matrix_R, matrix_Z, scalar_t, make_grid=True)
        with self.assertRaises(ValueError):
            out_t = et.rz2psinorm(matrix_R, matrix_Z, scalar_t, make_grid=True)

    def test_rz2psinorm_matrix_R_matrix_Z_scalar_t_make_grid_False_each_t_False(self):
        out = e.rz2psinorm(matrix_R, matrix_Z, scalar_t, each_t=False)
        out_t = et.rz2psinorm(matrix_R, matrix_Z, scalar_t, each_t=False)

        self.assertEqual(out.shape, matrix_R.shape)
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, matrix_R.shape)
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    def test_rz2psinorm_matrix_R_matrix_Z_scalar_t_make_grid_True_each_t_False(self):
        with self.assertRaises(ValueError):
            out = e.rz2psinorm(matrix_R, matrix_Z, scalar_t, make_grid=True, each_t=False)
        with self.assertRaises(ValueError):
            out_t = et.rz2psinorm(matrix_R, matrix_Z, scalar_t, make_grid=True, each_t=False)

    # scalar R, scalar Z, vector t:
    def test_rz2psinorm_scalar_R_scalar_Z_vector_t_make_grid_False_each_t_True(self):
        out = e.rz2psinorm(scalar_R, scalar_Z, vector_t)
        out_t = et.rz2psinorm(scalar_R, scalar_Z, vector_t)

        self.assertEqual(out.shape, (len(vector_t),))
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, (len(vector_t),))
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    def test_rz2psinorm_scalar_R_scalar_Z_vector_t_make_grid_True_each_t_True(self):
        with self.assertRaises(ValueError):
            out = e.rz2psinorm(scalar_R, scalar_Z, vector_t, make_grid=True)
        with self.assertRaises(ValueError):
            out_t = et.rz2psinorm(scalar_R, scalar_Z, vector_t, make_grid=True)

    def test_rz2psinorm_scalar_R_scalar_Z_vector_t_make_grid_False_each_t_False(self):
        with self.assertRaises(ValueError):
            out = e.rz2psinorm(scalar_R, scalar_Z, vector_t, each_t=False)
        with self.assertRaises(ValueError):
            out_t = et.rz2psinorm(scalar_R, scalar_Z, vector_t, each_t=False)

    def test_rz2psinorm_scalar_R_scalar_Z_vector_t_make_grid_True_each_t_False(self):
        with self.assertRaises(ValueError):
            out = e.rz2psinorm(scalar_R, scalar_Z, vector_t, make_grid=True, each_t=False)
        with self.assertRaises(ValueError):
            out_t = et.rz2psinorm(scalar_R, scalar_Z, vector_t, make_grid=True, each_t=False)

    # vector R, vector Z, vector t:
    def test_rz2psinorm_vector_R_vector_Z_vector_t_make_grid_False_each_t_True(self):
        out = e.rz2psinorm(vector_R, vector_Z, vector_t)
        out_t = et.rz2psinorm(vector_R, vector_Z, vector_t)

        self.assertEqual(out.shape, (len(vector_t), len(vector_R)))
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, (len(vector_t), len(vector_R)))
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    def test_rz2psinorm_vector_R_vector_Z_vector_t_make_grid_True_each_t_True(self):
        out = e.rz2psinorm(vector_R, vector_Z, vector_t, make_grid=True)
        out_t = et.rz2psinorm(vector_R, vector_Z, vector_t, make_grid=True)

        self.assertEqual(out.shape, (len(vector_t), len(vector_R), len(vector_Z)))
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, (len(vector_t), len(vector_R), len(vector_Z)))
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    def test_rz2psinorm_vector_R_vector_Z_vector_t_make_grid_False_each_t_False(self):
        with self.assertRaises(ValueError):
            out = e.rz2psinorm(vector_R, vector_Z, vector_t, each_t=False)
        with self.assertRaises(ValueError):
            out_t = et.rz2psinorm(vector_R, vector_Z, vector_t, each_t=False)

    def test_rz2psinorm_vector_R_vector_Z_vector_t_make_grid_True_each_t_False(self):
        with self.assertRaises(ValueError):
            out = e.rz2psinorm(vector_R, vector_Z, vector_t, make_grid=True, each_t=False)
        with self.assertRaises(ValueError):
            out_t = et.rz2psinorm(vector_R, vector_Z, vector_t, make_grid=True, each_t=False)

    # matrix R, matrix Z, vector t:
    def test_rz2psinorm_matrix_R_matrix_Z_vector_t_make_grid_False_each_t_True(self):
        out = e.rz2psinorm(matrix_R, matrix_Z, vector_t)
        out_t = et.rz2psinorm(matrix_R, matrix_Z, vector_t)

        self.assertEqual(out.shape, tuple([len(vector_t),] + list(matrix_R.shape)))
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, tuple([len(vector_t),] + list(matrix_R.shape)))
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    def test_rz2psinorm_matrix_R_matrix_Z_vector_t_make_grid_True_each_t_True(self):
        with self.assertRaises(ValueError):
            out = e.rz2psinorm(matrix_R, matrix_Z, vector_t, make_grid=True)
        with self.assertRaises(ValueError):
            out_t = et.rz2psinorm(matrix_R, matrix_Z, vector_t, make_grid=True)

    def test_rz2psinorm_matrix_R_matrix_Z_vector_t_make_grid_False_each_t_False(self):
        with self.assertRaises(ValueError):
            out = e.rz2psinorm(matrix_R, matrix_Z, vector_t, each_t=False)
        with self.assertRaises(ValueError):
            out_t = et.rz2psinorm(matrix_R, matrix_Z, vector_t, each_t=False)

    def test_rz2psinorm_matrix_R_matrix_Z_vector_t_make_grid_True_each_t_False(self):
        with self.assertRaises(ValueError):
            out = e.rz2psinorm(matrix_R, matrix_Z, vector_t, make_grid=True, each_t=False)
        with self.assertRaises(ValueError):
            out_t = et.rz2psinorm(matrix_R, matrix_Z, vector_t, make_grid=True, each_t=False)

    # matrix R, matrix Z, matrix t:
    def test_rz2psinorm_matrix_R_matrix_Z_matrix_t_make_grid_False_each_t_True(self):
        with self.assertRaises(ValueError):
            out = e.rz2psinorm(matrix_R, matrix_Z, matrix_t)
        with self.assertRaises(ValueError):
            out_t = et.rz2psinorm(matrix_R, matrix_Z, matrix_t)

    def test_rz2psinorm_matrix_R_matrix_Z_matrix_t_make_grid_True_each_t_True(self):
        with self.assertRaises(ValueError):
            out = e.rz2psinorm(matrix_R, matrix_Z, matrix_t, make_grid=True)
        with self.assertRaises(ValueError):
            out_t = et.rz2psinorm(matrix_R, matrix_Z, matrix_t, make_grid=True)

    def test_rz2psinorm_matrix_R_matrix_Z_matrix_t_make_grid_False_each_t_False(self):
        out = e.rz2psinorm(matrix_R, matrix_Z, matrix_t, each_t=False)
        out_t = et.rz2psinorm(matrix_R, matrix_Z, matrix_t, each_t=False)

        self.assertEqual(out.shape, matrix_R.shape)
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, matrix_R.shape)
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        # TODO: Only good to 2%.
        self.assertLessEqual(diff, 0.02)

    def test_rz2psinorm_matrix_R_matrix_Z_matrix_t_make_grid_True_each_t_False(self):
        with self.assertRaises(ValueError):
            out = e.rz2psinorm(matrix_R, matrix_Z, matrix_t, make_grid=True, each_t=False)
        with self.assertRaises(ValueError):
            out_t = et.rz2psinorm(matrix_R, matrix_Z, matrix_t, make_grid=True, each_t=False)

class rz2phinormTestCase(unittest.TestCase):
    # The following tests focus primarily on the consistency of the input/output
    # shapes. More tests should be added to check the consistency between the
    # different mapping routines.

    # scalar R, scalar Z, scalar t:
    def test_rz2phinorm_scalar_R_scalar_Z_scalar_t_make_grid_False_each_t_True(self):
        out = e.rz2phinorm(scalar_R, scalar_Z, scalar_t)
        out_t = et.rz2phinorm(scalar_R, scalar_Z, scalar_t)

        with self.assertRaises(TypeError):
            iter(out)
        self.assertFalse(numpy.isnan(out))
        self.assertFalse(numpy.isinf(out))

        with self.assertRaises(TypeError):
            iter(out_t)
        self.assertFalse(numpy.isnan(out_t))
        self.assertFalse(numpy.isinf(out_t))

        diff = numpy.absolute(out - out_t) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    def test_rz2phinorm_scalar_R_scalar_Z_scalar_t_make_grid_True_each_t_True(self):
        with self.assertRaises(ValueError):
            out = e.rz2phinorm(scalar_R, scalar_Z, scalar_t, make_grid=True)
        with self.assertRaises(ValueError):
            out_t = et.rz2phinorm(scalar_R, scalar_Z, scalar_t, make_grid=True)

    def test_rz2phinorm_scalar_R_scalar_Z_scalar_t_make_grid_False_each_t_False(self):
        out = e.rz2phinorm(scalar_R, scalar_Z, scalar_t, each_t=False)
        out_t = et.rz2phinorm(scalar_R, scalar_Z, scalar_t, each_t=False)

        with self.assertRaises(TypeError):
            iter(out)
        self.assertFalse(numpy.isnan(out))
        self.assertFalse(numpy.isinf(out))

        with self.assertRaises(TypeError):
            iter(out_t)
        self.assertFalse(numpy.isnan(out_t))
        self.assertFalse(numpy.isinf(out_t))

        diff = numpy.absolute(out - out_t) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    def test_rz2phinorm_scalar_R_scalar_Z_scalar_t_make_grid_True_each_t_False(self):
        with self.assertRaises(ValueError):
            out = e.rz2phinorm(scalar_R, scalar_Z, scalar_t, make_grid=True, each_t=False)
        with self.assertRaises(ValueError):
            out_t = et.rz2phinorm(scalar_R, scalar_Z, scalar_t, make_grid=True, each_t=False)

    # vector R, vector Z, scalar t:
    def test_rz2phinorm_vector_R_vector_Z_scalar_t_make_grid_False_each_t_True(self):
        out = e.rz2phinorm(vector_R, vector_Z, scalar_t)
        out_t = et.rz2phinorm(vector_R, vector_Z, scalar_t)

        self.assertEqual(out.shape, (len(vector_R),))
        self.assertTrue(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, (len(vector_R),))
        self.assertTrue(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        self.assertTrue((numpy.isnan(out) == numpy.isnan(out_t)).all())
        diff = numpy.sqrt(((out[~numpy.isnan(out)] - out_t[~numpy.isnan(out)])**2).max()) / numpy.absolute(out[~numpy.isnan(out)]).max()
        self.assertLessEqual(diff, tol)

    def test_rz2phinorm_vector_R_vector_Z_scalar_t_make_grid_True_each_t_True(self):
        out = e.rz2phinorm(vector_R, vector_Z, scalar_t, make_grid=True)
        out_t = et.rz2phinorm(vector_R, vector_Z, scalar_t, make_grid=True)

        self.assertEqual(out.shape, (len(vector_R), len(vector_Z)))
        self.assertTrue(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, (len(vector_R), len(vector_Z)))
        self.assertTrue(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        self.assertTrue((numpy.isnan(out) == numpy.isnan(out_t)).all())
        diff = numpy.sqrt(((out[~numpy.isnan(out)] - out_t[~numpy.isnan(out)])**2).max()) / numpy.absolute(out[~numpy.isnan(out)]).max()
        self.assertLessEqual(diff, tol)

    def test_rz2phinorm_vector_R_vector_Z_scalar_t_make_grid_False_each_t_False(self):
        out = e.rz2phinorm(vector_R, vector_Z, scalar_t, each_t=False)
        out_t = et.rz2phinorm(vector_R, vector_Z, scalar_t, each_t=False)

        self.assertEqual(out.shape, (len(vector_R),))
        self.assertTrue(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, (len(vector_R),))
        self.assertTrue(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        self.assertTrue((numpy.isnan(out) == numpy.isnan(out_t)).all())
        diff = numpy.sqrt(((out[~numpy.isnan(out)] - out_t[~numpy.isnan(out)])**2).max()) / numpy.absolute(out[~numpy.isnan(out)]).max()
        self.assertLessEqual(diff, tol)

    def test_rz2phinorm_vector_R_vector_Z_scalar_t_make_grid_True_each_t_False(self):
        out = e.rz2phinorm(vector_R, vector_Z, scalar_t, make_grid=True, each_t=False)
        out_t = et.rz2phinorm(vector_R, vector_Z, scalar_t, make_grid=True, each_t=False)

        self.assertEqual(out.shape, (len(vector_R), len(vector_Z)))
        self.assertTrue(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, (len(vector_R), len(vector_Z)))
        self.assertTrue(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        self.assertTrue((numpy.isnan(out) == numpy.isnan(out_t)).all())
        diff = numpy.sqrt(((out[~numpy.isnan(out)] - out_t[~numpy.isnan(out)])**2).max()) / numpy.absolute(out[~numpy.isnan(out)]).max()
        self.assertLessEqual(diff, tol)

    # matrix R, matrix Z, scalar t:
    def test_rz2phinorm_matrix_R_matrix_Z_scalar_t_make_grid_False_each_t_True(self):
        out = e.rz2phinorm(matrix_R, matrix_Z, scalar_t)
        out_t = et.rz2phinorm(matrix_R, matrix_Z, scalar_t)

        self.assertEqual(out.shape, matrix_R.shape)
        self.assertTrue(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, matrix_R.shape)
        self.assertTrue(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        self.assertTrue((numpy.isnan(out) == numpy.isnan(out_t)).all())
        diff = numpy.sqrt(((out[~numpy.isnan(out)] - out_t[~numpy.isnan(out)])**2).max()) / numpy.absolute(out[~numpy.isnan(out)]).max()
        self.assertLessEqual(diff, tol)

    def test_rz2phinorm_matrix_R_matrix_Z_scalar_t_make_grid_True_each_t_True(self):
        with self.assertRaises(ValueError):
            out = e.rz2phinorm(matrix_R, matrix_Z, scalar_t, make_grid=True)
        with self.assertRaises(ValueError):
            out_t = et.rz2phinorm(matrix_R, matrix_Z, scalar_t, make_grid=True)

    def test_rz2phinorm_matrix_R_matrix_Z_scalar_t_make_grid_False_each_t_False(self):
        out = e.rz2phinorm(matrix_R, matrix_Z, scalar_t, each_t=False)
        out_t = et.rz2phinorm(matrix_R, matrix_Z, scalar_t, each_t=False)

        self.assertEqual(out.shape, matrix_R.shape)
        self.assertTrue(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, matrix_R.shape)
        self.assertTrue(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        self.assertTrue((numpy.isnan(out) == numpy.isnan(out_t)).all())
        diff = numpy.sqrt(((out[~numpy.isnan(out)] - out_t[~numpy.isnan(out)])**2).max()) / numpy.absolute(out[~numpy.isnan(out)]).max()
        self.assertLessEqual(diff, tol)

    def test_rz2phinorm_matrix_R_matrix_Z_scalar_t_make_grid_True_each_t_False(self):
        with self.assertRaises(ValueError):
            out = e.rz2phinorm(matrix_R, matrix_Z, scalar_t, make_grid=True, each_t=False)
        with self.assertRaises(ValueError):
            out_t = et.rz2phinorm(matrix_R, matrix_Z, scalar_t, make_grid=True, each_t=False)

    # scalar R, scalar Z, vector t:
    def test_rz2phinorm_scalar_R_scalar_Z_vector_t_make_grid_False_each_t_True(self):
        out = e.rz2phinorm(scalar_R, scalar_Z, vector_t)
        out_t = et.rz2phinorm(scalar_R, scalar_Z, vector_t)

        self.assertEqual(out.shape, (len(vector_t),))
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, (len(vector_t),))
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    def test_rz2phinorm_scalar_R_scalar_Z_vector_t_make_grid_True_each_t_True(self):
        with self.assertRaises(ValueError):
            out = e.rz2phinorm(scalar_R, scalar_Z, vector_t, make_grid=True)
        with self.assertRaises(ValueError):
            out_t = et.rz2phinorm(scalar_R, scalar_Z, vector_t, make_grid=True)

    def test_rz2phinorm_scalar_R_scalar_Z_vector_t_make_grid_False_each_t_False(self):
        with self.assertRaises(ValueError):
            out = e.rz2phinorm(scalar_R, scalar_Z, vector_t, each_t=False)
        with self.assertRaises(ValueError):
            out_t = et.rz2phinorm(scalar_R, scalar_Z, vector_t, each_t=False)

    def test_rz2phinorm_scalar_R_scalar_Z_vector_t_make_grid_True_each_t_False(self):
        with self.assertRaises(ValueError):
            out = e.rz2phinorm(scalar_R, scalar_Z, vector_t, make_grid=True, each_t=False)
        with self.assertRaises(ValueError):
            out_t = et.rz2phinorm(scalar_R, scalar_Z, vector_t, make_grid=True, each_t=False)

    # vector R, vector Z, vector t:
    def test_rz2phinorm_vector_R_vector_Z_vector_t_make_grid_False_each_t_True(self):
        out = e.rz2phinorm(vector_R, vector_Z, vector_t)
        out_t = et.rz2phinorm(vector_R, vector_Z, vector_t)

        self.assertEqual(out.shape, (len(vector_t), len(vector_R)))
        self.assertTrue(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, (len(vector_t), len(vector_R)))
        self.assertTrue(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        self.assertTrue((numpy.isnan(out) == numpy.isnan(out_t)).all())
        diff = numpy.sqrt(((out[~numpy.isnan(out)] - out_t[~numpy.isnan(out)])**2).max()) / numpy.absolute(out[~numpy.isnan(out)]).max()
        self.assertLessEqual(diff, tol)

    def test_rz2phinorm_vector_R_vector_Z_vector_t_make_grid_True_each_t_True(self):
        out = e.rz2phinorm(vector_R, vector_Z, vector_t, make_grid=True)
        out_t = et.rz2phinorm(vector_R, vector_Z, vector_t, make_grid=True)

        self.assertEqual(out.shape, (len(vector_t), len(vector_R), len(vector_Z)))
        self.assertTrue(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, (len(vector_t), len(vector_R), len(vector_Z)))
        self.assertTrue(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        self.assertTrue((numpy.isnan(out) == numpy.isnan(out_t)).all())
        diff = numpy.sqrt(((out[~numpy.isnan(out)] - out_t[~numpy.isnan(out)])**2).max()) / numpy.absolute(out[~numpy.isnan(out)]).max()
        self.assertLessEqual(diff, tol)

    def test_rz2phinorm_vector_R_vector_Z_vector_t_make_grid_False_each_t_False(self):
        with self.assertRaises(ValueError):
            out = e.rz2phinorm(vector_R, vector_Z, vector_t, each_t=False)
        with self.assertRaises(ValueError):
            out_t = et.rz2phinorm(vector_R, vector_Z, vector_t, each_t=False)

    def test_rz2phinorm_vector_R_vector_Z_vector_t_make_grid_True_each_t_False(self):
        with self.assertRaises(ValueError):
            out = e.rz2phinorm(vector_R, vector_Z, vector_t, make_grid=True, each_t=False)
        with self.assertRaises(ValueError):
            out_t = et.rz2phinorm(vector_R, vector_Z, vector_t, make_grid=True, each_t=False)

    # matrix R, matrix Z, vector t:
    def test_rz2phinorm_matrix_R_matrix_Z_vector_t_make_grid_False_each_t_True(self):
        out = e.rz2phinorm(matrix_R, matrix_Z, vector_t)
        out_t = et.rz2phinorm(matrix_R, matrix_Z, vector_t)

        self.assertEqual(out.shape, tuple([len(vector_t),] + list(matrix_R.shape)))
        self.assertTrue(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, tuple([len(vector_t),] + list(matrix_R.shape)))
        self.assertTrue(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        self.assertTrue((numpy.isnan(out) == numpy.isnan(out_t)).all())
        diff = numpy.sqrt(((out[~numpy.isnan(out)] - out_t[~numpy.isnan(out)])**2).max()) / numpy.absolute(out[~numpy.isnan(out)]).max()
        self.assertLessEqual(diff, tol)

    def test_rz2phinorm_matrix_R_matrix_Z_vector_t_make_grid_True_each_t_True(self):
        with self.assertRaises(ValueError):
            out = e.rz2phinorm(matrix_R, matrix_Z, vector_t, make_grid=True)
        with self.assertRaises(ValueError):
            out_t = et.rz2phinorm(matrix_R, matrix_Z, vector_t, make_grid=True)

    def test_rz2phinorm_matrix_R_matrix_Z_vector_t_make_grid_False_each_t_False(self):
        with self.assertRaises(ValueError):
            out = e.rz2phinorm(matrix_R, matrix_Z, vector_t, each_t=False)
        with self.assertRaises(ValueError):
            out_t = et.rz2phinorm(matrix_R, matrix_Z, vector_t, each_t=False)

    def test_rz2phinorm_matrix_R_matrix_Z_vector_t_make_grid_True_each_t_False(self):
        with self.assertRaises(ValueError):
            out = e.rz2phinorm(matrix_R, matrix_Z, vector_t, make_grid=True, each_t=False)
        with self.assertRaises(ValueError):
            out_t = et.rz2phinorm(matrix_R, matrix_Z, vector_t, make_grid=True, each_t=False)

    # matrix R, matrix Z, matrix t:
    def test_rz2phinorm_matrix_R_matrix_Z_matrix_t_make_grid_False_each_t_True(self):
        with self.assertRaises(ValueError):
            out = e.rz2phinorm(matrix_R, matrix_Z, matrix_t)
        with self.assertRaises(ValueError):
            out_t = et.rz2phinorm(matrix_R, matrix_Z, matrix_t)

    def test_rz2phinorm_matrix_R_matrix_Z_matrix_t_make_grid_True_each_t_True(self):
        with self.assertRaises(ValueError):
            out = e.rz2phinorm(matrix_R, matrix_Z, matrix_t, make_grid=True)
        with self.assertRaises(ValueError):
            out_t = et.rz2phinorm(matrix_R, matrix_Z, matrix_t, make_grid=True)

    def test_rz2phinorm_matrix_R_matrix_Z_matrix_t_make_grid_False_each_t_False(self):
        out = e.rz2phinorm(matrix_R, matrix_Z, matrix_t, each_t=False)
        out_t = et.rz2phinorm(matrix_R, matrix_Z, matrix_t, each_t=False)

        self.assertEqual(out.shape, matrix_R.shape)
        self.assertTrue(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, matrix_R.shape)
        self.assertTrue(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        # TODO: The trispline case has an extra non-NaN point here, appears to
        # be near a coil or something.
        # TODO: Only good to 2%
        # self.assertTrue((numpy.isnan(out) == numpy.isnan(out_t)).all())
        res2 = (out - out_t)**2
        diff = numpy.sqrt((res2[~numpy.isnan(res2)]).max()) / numpy.absolute(out[~numpy.isnan(out)]).max()
        self.assertLessEqual(diff, 0.02)

    def test_rz2phinorm_matrix_R_matrix_Z_matrix_t_make_grid_True_each_t_False(self):
        with self.assertRaises(ValueError):
            out = e.rz2phinorm(matrix_R, matrix_Z, matrix_t, make_grid=True, each_t=False)
        with self.assertRaises(ValueError):
            out_t = et.rz2phinorm(matrix_R, matrix_Z, matrix_t, make_grid=True, each_t=False)

class rz2volnormTestCase(unittest.TestCase):
    # The following tests focus primarily on the consistency of the input/output
    # shapes. More tests should be added to check the consistency between the
    # different mapping routines.

    # scalar R, scalar Z, scalar t:
    def test_rz2volnorm_scalar_R_scalar_Z_scalar_t_make_grid_False_each_t_True(self):
        out = e.rz2volnorm(scalar_R, scalar_Z, scalar_t)
        out_t = et.rz2volnorm(scalar_R, scalar_Z, scalar_t)

        with self.assertRaises(TypeError):
            iter(out)
        self.assertFalse(numpy.isnan(out))
        self.assertFalse(numpy.isinf(out))

        with self.assertRaises(TypeError):
            iter(out_t)
        self.assertFalse(numpy.isnan(out_t))
        self.assertFalse(numpy.isinf(out_t))

        diff = numpy.absolute(out - out_t) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    def test_rz2volnorm_scalar_R_scalar_Z_scalar_t_make_grid_True_each_t_True(self):
        with self.assertRaises(ValueError):
            out = e.rz2volnorm(scalar_R, scalar_Z, scalar_t, make_grid=True)
        with self.assertRaises(ValueError):
            out_t = et.rz2volnorm(scalar_R, scalar_Z, scalar_t, make_grid=True)

    def test_rz2volnorm_scalar_R_scalar_Z_scalar_t_make_grid_False_each_t_False(self):
        out = e.rz2volnorm(scalar_R, scalar_Z, scalar_t, each_t=False)
        out_t = et.rz2volnorm(scalar_R, scalar_Z, scalar_t, each_t=False)

        with self.assertRaises(TypeError):
            iter(out)
        self.assertFalse(numpy.isnan(out))
        self.assertFalse(numpy.isinf(out))

        with self.assertRaises(TypeError):
            iter(out_t)
        self.assertFalse(numpy.isnan(out_t))
        self.assertFalse(numpy.isinf(out_t))

        diff = numpy.absolute(out - out_t) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    def test_rz2volnorm_scalar_R_scalar_Z_scalar_t_make_grid_True_each_t_False(self):
        with self.assertRaises(ValueError):
            out = e.rz2volnorm(scalar_R, scalar_Z, scalar_t, make_grid=True, each_t=False)
        with self.assertRaises(ValueError):
            out_t = et.rz2volnorm(scalar_R, scalar_Z, scalar_t, make_grid=True, each_t=False)

    # vector R, vector Z, scalar t:
    def test_rz2volnorm_vector_R_vector_Z_scalar_t_make_grid_False_each_t_True(self):
        out = e.rz2volnorm(vector_R, vector_Z, scalar_t)
        out_t = et.rz2volnorm(vector_R, vector_Z, scalar_t)

        self.assertEqual(out.shape, (len(vector_R),))
        self.assertTrue(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, (len(vector_R),))
        self.assertTrue(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        self.assertTrue((numpy.isnan(out) == numpy.isnan(out_t)).all())
        diff = numpy.sqrt(((out[~numpy.isnan(out)] - out_t[~numpy.isnan(out)])**2).max()) / numpy.absolute(out[~numpy.isnan(out)]).max()
        self.assertLessEqual(diff, tol)

    def test_rz2volnorm_vector_R_vector_Z_scalar_t_make_grid_True_each_t_True(self):
        out = e.rz2volnorm(vector_R, vector_Z, scalar_t, make_grid=True)
        out_t = et.rz2volnorm(vector_R, vector_Z, scalar_t, make_grid=True)

        self.assertEqual(out.shape, (len(vector_R), len(vector_Z)))
        self.assertTrue(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, (len(vector_R), len(vector_Z)))
        self.assertTrue(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        self.assertTrue((numpy.isnan(out) == numpy.isnan(out_t)).all())
        diff = numpy.sqrt(((out[~numpy.isnan(out)] - out_t[~numpy.isnan(out)])**2).max()) / numpy.absolute(out[~numpy.isnan(out)]).max()
        self.assertLessEqual(diff, tol)

    def test_rz2volnorm_vector_R_vector_Z_scalar_t_make_grid_False_each_t_False(self):
        out = e.rz2volnorm(vector_R, vector_Z, scalar_t, each_t=False)
        out_t = et.rz2volnorm(vector_R, vector_Z, scalar_t, each_t=False)

        self.assertEqual(out.shape, (len(vector_R),))
        self.assertTrue(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, (len(vector_R),))
        self.assertTrue(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        self.assertTrue((numpy.isnan(out) == numpy.isnan(out_t)).all())
        diff = numpy.sqrt(((out[~numpy.isnan(out)] - out_t[~numpy.isnan(out)])**2).max()) / numpy.absolute(out[~numpy.isnan(out)]).max()
        self.assertLessEqual(diff, tol)

    def test_rz2volnorm_vector_R_vector_Z_scalar_t_make_grid_True_each_t_False(self):
        out = e.rz2volnorm(vector_R, vector_Z, scalar_t, make_grid=True, each_t=False)
        out_t = et.rz2volnorm(vector_R, vector_Z, scalar_t, make_grid=True, each_t=False)

        self.assertEqual(out.shape, (len(vector_R), len(vector_Z)))
        self.assertTrue(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, (len(vector_R), len(vector_Z)))
        self.assertTrue(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        self.assertTrue((numpy.isnan(out) == numpy.isnan(out_t)).all())
        diff = numpy.sqrt(((out[~numpy.isnan(out)] - out_t[~numpy.isnan(out)])**2).max()) / numpy.absolute(out[~numpy.isnan(out)]).max()
        self.assertLessEqual(diff, tol)

    # matrix R, matrix Z, scalar t:
    def test_rz2volnorm_matrix_R_matrix_Z_scalar_t_make_grid_False_each_t_True(self):
        out = e.rz2volnorm(matrix_R, matrix_Z, scalar_t)
        out_t = et.rz2volnorm(matrix_R, matrix_Z, scalar_t)

        self.assertEqual(out.shape, matrix_R.shape)
        self.assertTrue(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, matrix_R.shape)
        self.assertTrue(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        self.assertTrue((numpy.isnan(out) == numpy.isnan(out_t)).all())
        diff = numpy.sqrt(((out[~numpy.isnan(out)] - out_t[~numpy.isnan(out)])**2).max()) / numpy.absolute(out[~numpy.isnan(out)]).max()
        self.assertLessEqual(diff, tol)

    def test_rz2volnorm_matrix_R_matrix_Z_scalar_t_make_grid_True_each_t_True(self):
        with self.assertRaises(ValueError):
            out = e.rz2volnorm(matrix_R, matrix_Z, scalar_t, make_grid=True)
        with self.assertRaises(ValueError):
            out_t = et.rz2volnorm(matrix_R, matrix_Z, scalar_t, make_grid=True)

    def test_rz2volnorm_matrix_R_matrix_Z_scalar_t_make_grid_False_each_t_False(self):
        out = e.rz2volnorm(matrix_R, matrix_Z, scalar_t, each_t=False)
        out_t = et.rz2volnorm(matrix_R, matrix_Z, scalar_t, each_t=False)

        self.assertEqual(out.shape, matrix_R.shape)
        self.assertTrue(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, matrix_R.shape)
        self.assertTrue(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        self.assertTrue((numpy.isnan(out) == numpy.isnan(out_t)).all())
        diff = numpy.sqrt(((out[~numpy.isnan(out)] - out_t[~numpy.isnan(out)])**2).max()) / numpy.absolute(out[~numpy.isnan(out)]).max()
        self.assertLessEqual(diff, tol)

    def test_rz2volnorm_matrix_R_matrix_Z_scalar_t_make_grid_True_each_t_False(self):
        with self.assertRaises(ValueError):
            out = e.rz2volnorm(matrix_R, matrix_Z, scalar_t, make_grid=True, each_t=False)
        with self.assertRaises(ValueError):
            out_t = et.rz2volnorm(matrix_R, matrix_Z, scalar_t, make_grid=True, each_t=False)

    # scalar R, scalar Z, vector t:
    def test_rz2volnorm_scalar_R_scalar_Z_vector_t_make_grid_False_each_t_True(self):
        out = e.rz2volnorm(scalar_R, scalar_Z, vector_t)
        out_t = et.rz2volnorm(scalar_R, scalar_Z, vector_t)

        self.assertEqual(out.shape, (len(vector_t),))
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, (len(vector_t),))
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    def test_rz2volnorm_scalar_R_scalar_Z_vector_t_make_grid_True_each_t_True(self):
        with self.assertRaises(ValueError):
            out = e.rz2volnorm(scalar_R, scalar_Z, vector_t, make_grid=True)
        with self.assertRaises(ValueError):
            out_t = et.rz2volnorm(scalar_R, scalar_Z, vector_t, make_grid=True)

    def test_rz2volnorm_scalar_R_scalar_Z_vector_t_make_grid_False_each_t_False(self):
        with self.assertRaises(ValueError):
            out = e.rz2volnorm(scalar_R, scalar_Z, vector_t, each_t=False)
        with self.assertRaises(ValueError):
            out_t = et.rz2volnorm(scalar_R, scalar_Z, vector_t, each_t=False)

    def test_rz2volnorm_scalar_R_scalar_Z_vector_t_make_grid_True_each_t_False(self):
        with self.assertRaises(ValueError):
            out = e.rz2volnorm(scalar_R, scalar_Z, vector_t, make_grid=True, each_t=False)
        with self.assertRaises(ValueError):
            out_t = et.rz2volnorm(scalar_R, scalar_Z, vector_t, make_grid=True, each_t=False)

    # vector R, vector Z, vector t:
    def test_rz2volnorm_vector_R_vector_Z_vector_t_make_grid_False_each_t_True(self):
        out = e.rz2volnorm(vector_R, vector_Z, vector_t)
        out_t = et.rz2volnorm(vector_R, vector_Z, vector_t)

        self.assertEqual(out.shape, (len(vector_t), len(vector_R)))
        self.assertTrue(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, (len(vector_t), len(vector_R)))
        self.assertTrue(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        self.assertTrue((numpy.isnan(out) == numpy.isnan(out_t)).all())
        diff = numpy.sqrt(((out[~numpy.isnan(out)] - out_t[~numpy.isnan(out)])**2).max()) / numpy.absolute(out[~numpy.isnan(out)]).max()
        self.assertLessEqual(diff, tol)

    def test_rz2volnorm_vector_R_vector_Z_vector_t_make_grid_True_each_t_True(self):
        out = e.rz2volnorm(vector_R, vector_Z, vector_t, make_grid=True)
        out_t = et.rz2volnorm(vector_R, vector_Z, vector_t, make_grid=True)

        self.assertEqual(out.shape, (len(vector_t), len(vector_R), len(vector_Z)))
        self.assertTrue(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, (len(vector_t), len(vector_R), len(vector_Z)))
        self.assertTrue(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        self.assertTrue((numpy.isnan(out) == numpy.isnan(out_t)).all())
        diff = numpy.sqrt(((out[~numpy.isnan(out)] - out_t[~numpy.isnan(out)])**2).max()) / numpy.absolute(out[~numpy.isnan(out)]).max()
        self.assertLessEqual(diff, tol)

    def test_rz2volnorm_vector_R_vector_Z_vector_t_make_grid_False_each_t_False(self):
        with self.assertRaises(ValueError):
            out = e.rz2volnorm(vector_R, vector_Z, vector_t, each_t=False)
        with self.assertRaises(ValueError):
            out_t = et.rz2volnorm(vector_R, vector_Z, vector_t, each_t=False)

    def test_rz2volnorm_vector_R_vector_Z_vector_t_make_grid_True_each_t_False(self):
        with self.assertRaises(ValueError):
            out = e.rz2volnorm(vector_R, vector_Z, vector_t, make_grid=True, each_t=False)
        with self.assertRaises(ValueError):
            out_t = et.rz2volnorm(vector_R, vector_Z, vector_t, make_grid=True, each_t=False)

    # matrix R, matrix Z, vector t:
    def test_rz2volnorm_matrix_R_matrix_Z_vector_t_make_grid_False_each_t_True(self):
        out = e.rz2volnorm(matrix_R, matrix_Z, vector_t)
        out_t = et.rz2volnorm(matrix_R, matrix_Z, vector_t)

        self.assertEqual(out.shape, tuple([len(vector_t),] + list(matrix_R.shape)))
        self.assertTrue(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, tuple([len(vector_t),] + list(matrix_R.shape)))
        self.assertTrue(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        self.assertTrue((numpy.isnan(out) == numpy.isnan(out_t)).all())
        diff = numpy.sqrt(((out[~numpy.isnan(out)] - out_t[~numpy.isnan(out)])**2).max()) / numpy.absolute(out[~numpy.isnan(out)]).max()
        self.assertLessEqual(diff, tol)

    def test_rz2volnorm_matrix_R_matrix_Z_vector_t_make_grid_True_each_t_True(self):
        with self.assertRaises(ValueError):
            out = e.rz2volnorm(matrix_R, matrix_Z, vector_t, make_grid=True)
        with self.assertRaises(ValueError):
            out_t = et.rz2volnorm(matrix_R, matrix_Z, vector_t, make_grid=True)

    def test_rz2volnorm_matrix_R_matrix_Z_vector_t_make_grid_False_each_t_False(self):
        with self.assertRaises(ValueError):
            out = e.rz2volnorm(matrix_R, matrix_Z, vector_t, each_t=False)
        with self.assertRaises(ValueError):
            out_t = et.rz2volnorm(matrix_R, matrix_Z, vector_t, each_t=False)

    def test_rz2volnorm_matrix_R_matrix_Z_vector_t_make_grid_True_each_t_False(self):
        with self.assertRaises(ValueError):
            out = e.rz2volnorm(matrix_R, matrix_Z, vector_t, make_grid=True, each_t=False)
        with self.assertRaises(ValueError):
            out_t = et.rz2volnorm(matrix_R, matrix_Z, vector_t, make_grid=True, each_t=False)

    # matrix R, matrix Z, matrix t:
    def test_rz2volnorm_matrix_R_matrix_Z_matrix_t_make_grid_False_each_t_True(self):
        with self.assertRaises(ValueError):
            out = e.rz2volnorm(matrix_R, matrix_Z, matrix_t)
        with self.assertRaises(ValueError):
            out_t = et.rz2volnorm(matrix_R, matrix_Z, matrix_t)

    def test_rz2volnorm_matrix_R_matrix_Z_matrix_t_make_grid_True_each_t_True(self):
        with self.assertRaises(ValueError):
            out = e.rz2volnorm(matrix_R, matrix_Z, matrix_t, make_grid=True)
        with self.assertRaises(ValueError):
            out_t = et.rz2volnorm(matrix_R, matrix_Z, matrix_t, make_grid=True)

    def test_rz2volnorm_matrix_R_matrix_Z_matrix_t_make_grid_False_each_t_False(self):
        out = e.rz2volnorm(matrix_R, matrix_Z, matrix_t, each_t=False)
        out_t = et.rz2volnorm(matrix_R, matrix_Z, matrix_t, each_t=False)

        self.assertEqual(out.shape, matrix_R.shape)
        self.assertTrue(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, matrix_R.shape)
        self.assertTrue(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        # TODO: The trispline case has an extra non-NaN point here, appears to
        # be near a coil or something.
        # TODO: Only good to 2%.
        # self.assertTrue((numpy.isnan(out) == numpy.isnan(out_t)).all())
        res2 = (out - out_t)**2
        diff = numpy.sqrt((res2[~numpy.isnan(res2)]).max()) / numpy.absolute(out[~numpy.isnan(out)]).max()
        self.assertLessEqual(diff, 0.02)

    def test_rz2volnorm_matrix_R_matrix_Z_matrix_t_make_grid_True_each_t_False(self):
        with self.assertRaises(ValueError):
            out = e.rz2volnorm(matrix_R, matrix_Z, matrix_t, make_grid=True, each_t=False)
        with self.assertRaises(ValueError):
            out_t = et.rz2volnorm(matrix_R, matrix_Z, matrix_t, make_grid=True, each_t=False)

class rz2rmidTestCase(unittest.TestCase):
    # The following tests focus primarily on the consistency of the input/output
    # shapes. More tests should be added to check the consistency between the
    # different mapping routines.

    # scalar R, scalar Z, scalar t:
    def test_rz2rmid_scalar_R_scalar_Z_scalar_t_make_grid_False_each_t_True_rho_False(self):
        out = e.rz2rmid(scalar_R, scalar_Z, scalar_t)
        out_t = et.rz2rmid(scalar_R, scalar_Z, scalar_t)

        with self.assertRaises(TypeError):
            iter(out)
        self.assertFalse(numpy.isnan(out))
        self.assertFalse(numpy.isinf(out))

        with self.assertRaises(TypeError):
            iter(out_t)
        self.assertFalse(numpy.isnan(out_t))
        self.assertFalse(numpy.isinf(out_t))

        diff = numpy.absolute(out - out_t) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    def test_rz2rmid_scalar_R_scalar_Z_scalar_t_make_grid_True_each_t_True_rho_False(self):
        with self.assertRaises(ValueError):
            out = e.rz2rmid(scalar_R, scalar_Z, scalar_t, make_grid=True)
        with self.assertRaises(ValueError):
            out_t = et.rz2rmid(scalar_R, scalar_Z, scalar_t, make_grid=True)

    def test_rz2rmid_scalar_R_scalar_Z_scalar_t_make_grid_False_each_t_False_rho_False(self):
        out = e.rz2rmid(scalar_R, scalar_Z, scalar_t, each_t=False)
        out_t = et.rz2rmid(scalar_R, scalar_Z, scalar_t, each_t=False)

        with self.assertRaises(TypeError):
            iter(out)
        self.assertFalse(numpy.isnan(out))
        self.assertFalse(numpy.isinf(out))

        with self.assertRaises(TypeError):
            iter(out_t)
        self.assertFalse(numpy.isnan(out_t))
        self.assertFalse(numpy.isinf(out_t))

        diff = numpy.absolute(out - out_t) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    def test_rz2rmid_scalar_R_scalar_Z_scalar_t_make_grid_True_each_t_False_rho_False(self):
        with self.assertRaises(ValueError):
            out = e.rz2rmid(scalar_R, scalar_Z, scalar_t, make_grid=True, each_t=False)
        with self.assertRaises(ValueError):
            out_t = et.rz2rmid(scalar_R, scalar_Z, scalar_t, make_grid=True, each_t=False)

    def test_rz2rmid_scalar_R_scalar_Z_scalar_t_make_grid_False_each_t_True_rho_True(self):
        out = e.rz2rmid(scalar_R, scalar_Z, scalar_t, rho=True)
        out_t = et.rz2rmid(scalar_R, scalar_Z, scalar_t, rho=True)

        with self.assertRaises(TypeError):
            iter(out)
        self.assertFalse(numpy.isnan(out))
        self.assertFalse(numpy.isinf(out))

        with self.assertRaises(TypeError):
            iter(out_t)
        self.assertFalse(numpy.isnan(out_t))
        self.assertFalse(numpy.isinf(out_t))

        diff = numpy.absolute(out - out_t) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    def test_rz2rmid_scalar_R_scalar_Z_scalar_t_make_grid_True_each_t_True_rho_True(self):
        with self.assertRaises(ValueError):
            out = e.rz2rmid(scalar_R, scalar_Z, scalar_t, make_grid=True, rho=True)
        with self.assertRaises(ValueError):
            out_t = et.rz2rmid(scalar_R, scalar_Z, scalar_t, make_grid=True, rho=True)

    def test_rz2rmid_scalar_R_scalar_Z_scalar_t_make_grid_False_each_t_False_rho_True(self):
        out = e.rz2rmid(scalar_R, scalar_Z, scalar_t, each_t=False, rho=True)
        out_t = et.rz2rmid(scalar_R, scalar_Z, scalar_t, each_t=False, rho=True)

        with self.assertRaises(TypeError):
            iter(out)
        self.assertFalse(numpy.isnan(out))
        self.assertFalse(numpy.isinf(out))

        with self.assertRaises(TypeError):
            iter(out_t)
        self.assertFalse(numpy.isnan(out_t))
        self.assertFalse(numpy.isinf(out_t))

        diff = numpy.absolute(out - out_t) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    def test_rz2rmid_scalar_R_scalar_Z_scalar_t_make_grid_True_each_t_False_rho_True(self):
        with self.assertRaises(ValueError):
            out = e.rz2rmid(scalar_R, scalar_Z, scalar_t, make_grid=True, each_t=False, rho=True)
        with self.assertRaises(ValueError):
            out_t = et.rz2rmid(scalar_R, scalar_Z, scalar_t, make_grid=True, each_t=False, rho=True)

    # vector R, vector Z, scalar t:
    def test_rz2rmid_vector_R_vector_Z_scalar_t_make_grid_False_each_t_True_rho_False(self):
        out = e.rz2rmid(vector_R, vector_Z, scalar_t)
        out_t = et.rz2rmid(vector_R, vector_Z, scalar_t)

        self.assertEqual(out.shape, (len(vector_R),))
        self.assertTrue(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, (len(vector_R),))
        self.assertTrue(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        self.assertTrue((numpy.isnan(out) == numpy.isnan(out_t)).all())
        diff = numpy.sqrt(((out[~numpy.isnan(out)] - out_t[~numpy.isnan(out)])**2).max()) / numpy.absolute(out[~numpy.isnan(out)]).max()
        self.assertLessEqual(diff, tol)

    def test_rz2rmid_vector_R_vector_Z_scalar_t_make_grid_True_each_t_True_rho_False(self):
        out = e.rz2rmid(vector_R, vector_Z, scalar_t, make_grid=True)
        out_t = et.rz2rmid(vector_R, vector_Z, scalar_t, make_grid=True)

        self.assertEqual(out.shape, (len(vector_R), len(vector_Z)))
        self.assertTrue(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, (len(vector_R), len(vector_Z)))
        self.assertTrue(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        # TODO: The trispline case has an extra non-NaN two points here.
        # self.assertTrue((numpy.isnan(out) == numpy.isnan(out_t)).all())
        res2 = (out - out_t)**2
        diff = numpy.sqrt((res2[~numpy.isnan(res2)]).max()) / numpy.absolute(out[~numpy.isnan(out)]).max()
        self.assertLessEqual(diff, tol)

    def test_rz2rmid_vector_R_vector_Z_scalar_t_make_grid_False_each_t_False_rho_False(self):
        out = e.rz2rmid(vector_R, vector_Z, scalar_t, each_t=False)
        out_t = et.rz2rmid(vector_R, vector_Z, scalar_t, each_t=False)

        self.assertEqual(out.shape, (len(vector_R),))
        self.assertTrue(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, (len(vector_R),))
        self.assertTrue(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        self.assertTrue((numpy.isnan(out) == numpy.isnan(out_t)).all())
        diff = numpy.sqrt(((out[~numpy.isnan(out)] - out_t[~numpy.isnan(out)])**2).max()) / numpy.absolute(out[~numpy.isnan(out)]).max()
        self.assertLessEqual(diff, tol)

    def test_rz2rmid_vector_R_vector_Z_scalar_t_make_grid_True_each_t_False_rho_False(self):
        out = e.rz2rmid(vector_R, vector_Z, scalar_t, make_grid=True, each_t=False)
        out_t = et.rz2rmid(vector_R, vector_Z, scalar_t, make_grid=True, each_t=False)

        self.assertEqual(out.shape, (len(vector_R), len(vector_Z)))
        self.assertTrue(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, (len(vector_R), len(vector_Z)))
        self.assertTrue(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        # TODO: The trispline case has an extra non-NaN two points here.
        # self.assertTrue((numpy.isnan(out) == numpy.isnan(out_t)).all())
        res2 = (out - out_t)**2
        diff = numpy.sqrt((res2[~numpy.isnan(res2)]).max()) / numpy.absolute(out[~numpy.isnan(out)]).max()
        self.assertLessEqual(diff, tol)

    def test_rz2rmid_vector_R_vector_Z_scalar_t_make_grid_False_each_t_True_rho_True(self):
        out = e.rz2rmid(vector_R, vector_Z, scalar_t, rho=True)
        out_t = et.rz2rmid(vector_R, vector_Z, scalar_t, rho=True)

        self.assertEqual(out.shape, (len(vector_R),))
        self.assertTrue(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, (len(vector_R),))
        self.assertTrue(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        self.assertTrue((numpy.isnan(out) == numpy.isnan(out_t)).all())
        diff = numpy.sqrt(((out[~numpy.isnan(out)] - out_t[~numpy.isnan(out)])**2).max()) / numpy.absolute(out[~numpy.isnan(out)]).max()
        self.assertLessEqual(diff, tol)

    def test_rz2rmid_vector_R_vector_Z_scalar_t_make_grid_True_each_t_True_rho_True(self):
        out = e.rz2rmid(vector_R, vector_Z, scalar_t, make_grid=True, rho=True)
        out_t = et.rz2rmid(vector_R, vector_Z, scalar_t, make_grid=True, rho=True)

        self.assertEqual(out.shape, (len(vector_R), len(vector_Z)))
        self.assertTrue(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, (len(vector_R), len(vector_Z)))
        self.assertTrue(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        # TODO: The trispline case has an extra non-NaN two points here.
        # self.assertTrue((numpy.isnan(out) == numpy.isnan(out_t)).all())
        res2 = (out - out_t)**2
        diff = numpy.sqrt((res2[~numpy.isnan(res2)]).max()) / numpy.absolute(out[~numpy.isnan(out)]).max()
        self.assertLessEqual(diff, tol)

    def test_rz2rmid_vector_R_vector_Z_scalar_t_make_grid_False_each_t_False_rho_True(self):
        out = e.rz2rmid(vector_R, vector_Z, scalar_t, each_t=False, rho=True)
        out_t = et.rz2rmid(vector_R, vector_Z, scalar_t, each_t=False, rho=True)

        self.assertEqual(out.shape, (len(vector_R),))
        self.assertTrue(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, (len(vector_R),))
        self.assertTrue(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        self.assertTrue((numpy.isnan(out) == numpy.isnan(out_t)).all())
        diff = numpy.sqrt(((out[~numpy.isnan(out)] - out_t[~numpy.isnan(out)])**2).max()) / numpy.absolute(out[~numpy.isnan(out)]).max()
        self.assertLessEqual(diff, tol)

    def test_rz2rmid_vector_R_vector_Z_scalar_t_make_grid_True_each_t_False_rho_True(self):
        out = e.rz2rmid(vector_R, vector_Z, scalar_t, make_grid=True, each_t=False, rho=True)
        out_t = et.rz2rmid(vector_R, vector_Z, scalar_t, make_grid=True, each_t=False, rho=True)

        self.assertEqual(out.shape, (len(vector_R), len(vector_Z)))
        self.assertTrue(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, (len(vector_R), len(vector_Z)))
        self.assertTrue(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        # TODO: The trispline case has an extra non-NaN two points here.
        # self.assertTrue((numpy.isnan(out) == numpy.isnan(out_t)).all())
        res2 = (out - out_t)**2
        diff = numpy.sqrt((res2[~numpy.isnan(res2)]).max()) / numpy.absolute(out[~numpy.isnan(out)]).max()
        self.assertLessEqual(diff, tol)

    # matrix R, matrix Z, scalar t:
    def test_rz2rmid_matrix_R_matrix_Z_scalar_t_make_grid_False_each_t_True_rho_False(self):
        out = e.rz2rmid(matrix_R, matrix_Z, scalar_t)
        out_t = et.rz2rmid(matrix_R, matrix_Z, scalar_t)

        self.assertEqual(out.shape, matrix_R.shape)
        self.assertTrue(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, matrix_R.shape)
        self.assertTrue(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        # TODO: The trispline case has an extra non-NaN two points here.
        # self.assertTrue((numpy.isnan(out) == numpy.isnan(out_t)).all())
        res2 = (out - out_t)**2
        diff = numpy.sqrt((res2[~numpy.isnan(res2)]).max()) / numpy.absolute(out[~numpy.isnan(out)]).max()
        self.assertLessEqual(diff, tol)

    def test_rz2rmid_matrix_R_matrix_Z_scalar_t_make_grid_True_each_t_True_rho_False(self):
        with self.assertRaises(ValueError):
            out = e.rz2rmid(matrix_R, matrix_Z, scalar_t, make_grid=True)
        with self.assertRaises(ValueError):
            out_t = et.rz2rmid(matrix_R, matrix_Z, scalar_t, make_grid=True)

    def test_rz2rmid_matrix_R_matrix_Z_scalar_t_make_grid_False_each_t_False_rho_False(self):
        out = e.rz2rmid(matrix_R, matrix_Z, scalar_t, each_t=False)
        out_t = et.rz2rmid(matrix_R, matrix_Z, scalar_t, each_t=False)

        self.assertEqual(out.shape, matrix_R.shape)
        self.assertTrue(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, matrix_R.shape)
        self.assertTrue(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        # TODO: The trispline case has an extra non-NaN two points here.
        # self.assertTrue((numpy.isnan(out) == numpy.isnan(out_t)).all())
        res2 = (out - out_t)**2
        diff = numpy.sqrt((res2[~numpy.isnan(res2)]).max()) / numpy.absolute(out[~numpy.isnan(out)]).max()
        self.assertLessEqual(diff, tol)

    def test_rz2rmid_matrix_R_matrix_Z_scalar_t_make_grid_True_each_t_False_rho_False(self):
        with self.assertRaises(ValueError):
            out = e.rz2rmid(matrix_R, matrix_Z, scalar_t, make_grid=True, each_t=False)
        with self.assertRaises(ValueError):
            out_t = et.rz2rmid(matrix_R, matrix_Z, scalar_t, make_grid=True, each_t=False)

    def test_rz2rmid_matrix_R_matrix_Z_scalar_t_make_grid_False_each_t_True_rho_True(self):
        out = e.rz2rmid(matrix_R, matrix_Z, scalar_t, rho=True)
        out_t = et.rz2rmid(matrix_R, matrix_Z, scalar_t, rho=True)

        self.assertEqual(out.shape, matrix_R.shape)
        self.assertTrue(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, matrix_R.shape)
        self.assertTrue(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        # TODO: The trispline case has an extra non-NaN two points here.
        # self.assertTrue((numpy.isnan(out) == numpy.isnan(out_t)).all())
        res2 = (out - out_t)**2
        diff = numpy.sqrt((res2[~numpy.isnan(res2)]).max()) / numpy.absolute(out[~numpy.isnan(out)]).max()
        self.assertLessEqual(diff, tol)

    def test_rz2rmid_matrix_R_matrix_Z_scalar_t_make_grid_True_each_t_True_rho_True(self):
        with self.assertRaises(ValueError):
            out = e.rz2rmid(matrix_R, matrix_Z, scalar_t, make_grid=True, rho=True)
        with self.assertRaises(ValueError):
            out_t = et.rz2rmid(matrix_R, matrix_Z, scalar_t, make_grid=True, rho=True)

    def test_rz2rmid_matrix_R_matrix_Z_scalar_t_make_grid_False_each_t_False_rho_True(self):
        out = e.rz2rmid(matrix_R, matrix_Z, scalar_t, each_t=False, rho=True)
        out_t = et.rz2rmid(matrix_R, matrix_Z, scalar_t, each_t=False, rho=True)

        self.assertEqual(out.shape, matrix_R.shape)
        self.assertTrue(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, matrix_R.shape)
        self.assertTrue(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        # TODO: The trispline case has an extra non-NaN two points here.
        # self.assertTrue((numpy.isnan(out) == numpy.isnan(out_t)).all())
        res2 = (out - out_t)**2
        diff = numpy.sqrt((res2[~numpy.isnan(res2)]).max()) / numpy.absolute(out[~numpy.isnan(out)]).max()
        self.assertLessEqual(diff, tol)

    def test_rz2rmid_matrix_R_matrix_Z_scalar_t_make_grid_True_each_t_False_rho_True(self):
        with self.assertRaises(ValueError):
            out = e.rz2rmid(matrix_R, matrix_Z, scalar_t, make_grid=True, each_t=False, rho=True)
        with self.assertRaises(ValueError):
            out_t = et.rz2rmid(matrix_R, matrix_Z, scalar_t, make_grid=True, each_t=False, rho=True)

    # scalar R, scalar Z, vector t:
    def test_rz2rmid_scalar_R_scalar_Z_vector_t_make_grid_False_each_t_True_rho_False(self):
        out = e.rz2rmid(scalar_R, scalar_Z, vector_t)
        out_t = et.rz2rmid(scalar_R, scalar_Z, vector_t)

        self.assertEqual(out.shape, (len(vector_t),))
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, (len(vector_t),))
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    def test_rz2rmid_scalar_R_scalar_Z_vector_t_make_grid_True_each_t_True_rho_False(self):
        with self.assertRaises(ValueError):
            out = e.rz2rmid(scalar_R, scalar_Z, vector_t, make_grid=True)
        with self.assertRaises(ValueError):
            out_t = et.rz2rmid(scalar_R, scalar_Z, vector_t, make_grid=True)

    def test_rz2rmid_scalar_R_scalar_Z_vector_t_make_grid_False_each_t_False_rho_False(self):
        with self.assertRaises(ValueError):
            out = e.rz2rmid(scalar_R, scalar_Z, vector_t, each_t=False)
        with self.assertRaises(ValueError):
            out_t = et.rz2rmid(scalar_R, scalar_Z, vector_t, each_t=False)

    def test_rz2rmid_scalar_R_scalar_Z_vector_t_make_grid_True_each_t_False_rho_False(self):
        with self.assertRaises(ValueError):
            out = e.rz2rmid(scalar_R, scalar_Z, vector_t, make_grid=True, each_t=False)
        with self.assertRaises(ValueError):
            out_t = et.rz2rmid(scalar_R, scalar_Z, vector_t, make_grid=True, each_t=False)

    def test_rz2rmid_scalar_R_scalar_Z_vector_t_make_grid_False_each_t_True_rho_True(self):
        out = e.rz2rmid(scalar_R, scalar_Z, vector_t, rho=True)
        out_t = et.rz2rmid(scalar_R, scalar_Z, vector_t, rho=True)

        self.assertEqual(out.shape, (len(vector_t),))
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, (len(vector_t),))
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    def test_rz2rmid_scalar_R_scalar_Z_vector_t_make_grid_True_each_t_True_rho_True(self):
        with self.assertRaises(ValueError):
            out = e.rz2rmid(scalar_R, scalar_Z, vector_t, make_grid=True, rho=True)
        with self.assertRaises(ValueError):
            out_t = et.rz2rmid(scalar_R, scalar_Z, vector_t, make_grid=True, rho=True)

    def test_rz2rmid_scalar_R_scalar_Z_vector_t_make_grid_False_each_t_False_rho_True(self):
        with self.assertRaises(ValueError):
            out = e.rz2rmid(scalar_R, scalar_Z, vector_t, each_t=False, rho=True)
        with self.assertRaises(ValueError):
            out_t = et.rz2rmid(scalar_R, scalar_Z, vector_t, each_t=False, rho=True)

    def test_rz2rmid_scalar_R_scalar_Z_vector_t_make_grid_True_each_t_False_rho_True(self):
        with self.assertRaises(ValueError):
            out = e.rz2rmid(scalar_R, scalar_Z, vector_t, make_grid=True, each_t=False, rho=True)
        with self.assertRaises(ValueError):
            out_t = et.rz2rmid(scalar_R, scalar_Z, vector_t, make_grid=True, each_t=False, rho=True)

    # vector R, vector Z, vector t:
    def test_rz2rmid_vector_R_vector_Z_vector_t_make_grid_False_each_t_True_rho_False(self):
        out = e.rz2rmid(vector_R, vector_Z, vector_t)
        out_t = et.rz2rmid(vector_R, vector_Z, vector_t)

        self.assertEqual(out.shape, (len(vector_t), len(vector_R)))
        self.assertTrue(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, (len(vector_t), len(vector_R)))
        self.assertTrue(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        # TODO: The trispline case has an extra non-NaN ten points here.
        # self.assertTrue((numpy.isnan(out) == numpy.isnan(out_t)).all())
        res2 = (out - out_t)**2
        diff = numpy.sqrt((res2[~numpy.isnan(res2)]).max()) / numpy.absolute(out[~numpy.isnan(out)]).max()
        self.assertLessEqual(diff, tol)

    def test_rz2rmid_vector_R_vector_Z_vector_t_make_grid_True_each_t_True_rho_False(self):
        out = e.rz2rmid(vector_R, vector_Z, vector_t, make_grid=True)
        out_t = et.rz2rmid(vector_R, vector_Z, vector_t, make_grid=True)

        self.assertEqual(out.shape, (len(vector_t), len(vector_R), len(vector_Z)))
        self.assertTrue(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, (len(vector_t), len(vector_R), len(vector_Z)))
        self.assertTrue(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        # TODO: The trispline case has an extra non-NaN 289 points here.
        # self.assertTrue((numpy.isnan(out) == numpy.isnan(out_t)).all())
        res2 = (out - out_t)**2
        diff = numpy.sqrt((res2[~numpy.isnan(res2)]).max()) / numpy.absolute(out[~numpy.isnan(out)]).max()
        self.assertLessEqual(diff, tol)

    def test_rz2rmid_vector_R_vector_Z_vector_t_make_grid_False_each_t_False_rho_False(self):
        with self.assertRaises(ValueError):
            out = e.rz2rmid(vector_R, vector_Z, vector_t, each_t=False)
        with self.assertRaises(ValueError):
            out_t = et.rz2rmid(vector_R, vector_Z, vector_t, each_t=False)

    def test_rz2rmid_vector_R_vector_Z_vector_t_make_grid_True_each_t_False_rho_False(self):
        with self.assertRaises(ValueError):
            out = e.rz2rmid(vector_R, vector_Z, vector_t, make_grid=True, each_t=False)
        with self.assertRaises(ValueError):
            out_t = et.rz2rmid(vector_R, vector_Z, vector_t, make_grid=True, each_t=False)

    def test_rz2rmid_vector_R_vector_Z_vector_t_make_grid_False_each_t_True_rho_True(self):
        out = e.rz2rmid(vector_R, vector_Z, vector_t, rho=True)
        out_t = et.rz2rmid(vector_R, vector_Z, vector_t, rho=True)

        self.assertEqual(out.shape, (len(vector_t), len(vector_R)))
        self.assertTrue(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, (len(vector_t), len(vector_R)))
        self.assertTrue(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        # TODO: The trispline case has an extra non-NaN ten points here.
        # self.assertTrue((numpy.isnan(out) == numpy.isnan(out_t)).all())
        res2 = (out - out_t)**2
        diff = numpy.sqrt((res2[~numpy.isnan(res2)]).max()) / numpy.absolute(out[~numpy.isnan(out)]).max()
        self.assertLessEqual(diff, tol)

    def test_rz2rmid_vector_R_vector_Z_vector_t_make_grid_True_each_t_True_rho_True(self):
        out = e.rz2rmid(vector_R, vector_Z, vector_t, make_grid=True, rho=True)
        out_t = et.rz2rmid(vector_R, vector_Z, vector_t, make_grid=True, rho=True)

        self.assertEqual(out.shape, (len(vector_t), len(vector_R), len(vector_Z)))
        self.assertTrue(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, (len(vector_t), len(vector_R), len(vector_Z)))
        self.assertTrue(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        # TODO: The trispline case has an extra non-NaN 289 points here.
        # self.assertTrue((numpy.isnan(out) == numpy.isnan(out_t)).all())
        res2 = (out - out_t)**2
        diff = numpy.sqrt((res2[~numpy.isnan(res2)]).max()) / numpy.absolute(out[~numpy.isnan(out)]).max()
        self.assertLessEqual(diff, tol)

    def test_rz2rmid_vector_R_vector_Z_vector_t_make_grid_False_each_t_False_rho_True(self):
        with self.assertRaises(ValueError):
            out = e.rz2rmid(vector_R, vector_Z, vector_t, each_t=False, rho=True)
        with self.assertRaises(ValueError):
            out_t = et.rz2rmid(vector_R, vector_Z, vector_t, each_t=False, rho=True)

    def test_rz2rmid_vector_R_vector_Z_vector_t_make_grid_True_each_t_False_rho_True(self):
        with self.assertRaises(ValueError):
            out = e.rz2rmid(vector_R, vector_Z, vector_t, make_grid=True, each_t=False, rho=True)
        with self.assertRaises(ValueError):
            out_t = et.rz2rmid(vector_R, vector_Z, vector_t, make_grid=True, each_t=False, rho=True)

    # matrix R, matrix Z, vector t:
    def test_rz2rmid_matrix_R_matrix_Z_vector_t_make_grid_False_each_t_True_rho_False(self):
        out = e.rz2rmid(matrix_R, matrix_Z, vector_t)
        out_t = et.rz2rmid(matrix_R, matrix_Z, vector_t)

        self.assertEqual(out.shape, tuple([len(vector_t),] + list(matrix_R.shape)))
        self.assertTrue(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, tuple([len(vector_t),] + list(matrix_R.shape)))
        self.assertTrue(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        # TODO: The trispline case has an extra non-NaN 289 points here.
        # self.assertTrue((numpy.isnan(out) == numpy.isnan(out_t)).all())
        res2 = (out - out_t)**2
        diff = numpy.sqrt((res2[~numpy.isnan(res2)]).max()) / numpy.absolute(out[~numpy.isnan(out)]).max()
        self.assertLessEqual(diff, tol)

    def test_rz2rmid_matrix_R_matrix_Z_vector_t_make_grid_True_each_t_True_rho_False(self):
        with self.assertRaises(ValueError):
            out = e.rz2rmid(matrix_R, matrix_Z, vector_t, make_grid=True)
        with self.assertRaises(ValueError):
            out_t = et.rz2rmid(matrix_R, matrix_Z, vector_t, make_grid=True)

    def test_rz2rmid_matrix_R_matrix_Z_vector_t_make_grid_False_each_t_False_rho_False(self):
        with self.assertRaises(ValueError):
            out = e.rz2rmid(matrix_R, matrix_Z, vector_t, each_t=False)
        with self.assertRaises(ValueError):
            out_t = et.rz2rmid(matrix_R, matrix_Z, vector_t, each_t=False)

    def test_rz2rmid_matrix_R_matrix_Z_vector_t_make_grid_True_each_t_False_rho_False(self):
        with self.assertRaises(ValueError):
            out = e.rz2rmid(matrix_R, matrix_Z, vector_t, make_grid=True, each_t=False)
        with self.assertRaises(ValueError):
            out_t = et.rz2rmid(matrix_R, matrix_Z, vector_t, make_grid=True, each_t=False)

    def test_rz2rmid_matrix_R_matrix_Z_vector_t_make_grid_False_each_t_True_rho_True(self):
        out = e.rz2rmid(matrix_R, matrix_Z, vector_t, rho=True)
        out_t = et.rz2rmid(matrix_R, matrix_Z, vector_t, rho=True)

        self.assertEqual(out.shape, tuple([len(vector_t),] + list(matrix_R.shape)))
        self.assertTrue(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, tuple([len(vector_t),] + list(matrix_R.shape)))
        self.assertTrue(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        # TODO: The trispline case has an extra non-NaN 289 points here.
        # self.assertTrue((numpy.isnan(out) == numpy.isnan(out_t)).all())
        res2 = (out - out_t)**2
        diff = numpy.sqrt((res2[~numpy.isnan(res2)]).max()) / numpy.absolute(out[~numpy.isnan(out)]).max()
        self.assertLessEqual(diff, tol)

    def test_rz2rmid_matrix_R_matrix_Z_vector_t_make_grid_True_each_t_True_rho_True(self):
        with self.assertRaises(ValueError):
            out = e.rz2rmid(matrix_R, matrix_Z, vector_t, make_grid=True, rho=True)
        with self.assertRaises(ValueError):
            out_t = et.rz2rmid(matrix_R, matrix_Z, vector_t, make_grid=True, rho=True)

    def test_rz2rmid_matrix_R_matrix_Z_vector_t_make_grid_False_each_t_False_rho_True(self):
        with self.assertRaises(ValueError):
            out = e.rz2rmid(matrix_R, matrix_Z, vector_t, each_t=False, rho=True)
        with self.assertRaises(ValueError):
            out_t = et.rz2rmid(matrix_R, matrix_Z, vector_t, each_t=False, rho=True)

    def test_rz2rmid_matrix_R_matrix_Z_vector_t_make_grid_True_each_t_False_rho_True(self):
        with self.assertRaises(ValueError):
            out = e.rz2rmid(matrix_R, matrix_Z, vector_t, make_grid=True, each_t=False, rho=True)
        with self.assertRaises(ValueError):
            out_t = et.rz2rmid(matrix_R, matrix_Z, vector_t, make_grid=True, each_t=False, rho=True)

    # matrix R, matrix Z, matrix t:
    def test_rz2rmid_matrix_R_matrix_Z_matrix_t_make_grid_False_each_t_True_rho_False(self):
        with self.assertRaises(ValueError):
            out = e.rz2rmid(matrix_R, matrix_Z, matrix_t)
        with self.assertRaises(ValueError):
            out_t = et.rz2rmid(matrix_R, matrix_Z, matrix_t)

    def test_rz2rmid_matrix_R_matrix_Z_matrix_t_make_grid_True_each_t_True_rho_False(self):
        with self.assertRaises(ValueError):
            out = e.rz2rmid(matrix_R, matrix_Z, matrix_t, make_grid=True)
        with self.assertRaises(ValueError):
            out_t = et.rz2rmid(matrix_R, matrix_Z, matrix_t, make_grid=True)

    def test_rz2rmid_matrix_R_matrix_Z_matrix_t_make_grid_False_each_t_False_rho_False(self):
        out = e.rz2rmid(matrix_R, matrix_Z, matrix_t, each_t=False)
        out_t = et.rz2rmid(matrix_R, matrix_Z, matrix_t, each_t=False)

        self.assertEqual(out.shape, matrix_R.shape)
        self.assertTrue(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, matrix_R.shape)
        self.assertTrue(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        # TODO: The trispline case has an extra non-NaN seven points here.
        # self.assertTrue((numpy.isnan(out) == numpy.isnan(out_t)).all())
        res2 = (out - out_t)**2
        diff = numpy.sqrt((res2[~numpy.isnan(res2)]).max()) / numpy.absolute(out[~numpy.isnan(out)]).max()
        self.assertLessEqual(diff, tol)

    def test_rz2rmid_matrix_R_matrix_Z_matrix_t_make_grid_True_each_t_False_rho_False(self):
        with self.assertRaises(ValueError):
            out = e.rz2rmid(matrix_R, matrix_Z, matrix_t, make_grid=True, each_t=False)
        with self.assertRaises(ValueError):
            out_t = et.rz2rmid(matrix_R, matrix_Z, matrix_t, make_grid=True, each_t=False)

    def test_rz2rmid_matrix_R_matrix_Z_matrix_t_make_grid_False_each_t_True_rho_True(self):
        with self.assertRaises(ValueError):
            out = e.rz2rmid(matrix_R, matrix_Z, matrix_t, rho=True)
        with self.assertRaises(ValueError):
            out_t = et.rz2rmid(matrix_R, matrix_Z, matrix_t, rho=True)

    def test_rz2rmid_matrix_R_matrix_Z_matrix_t_make_grid_True_each_t_True_rho_True(self):
        with self.assertRaises(ValueError):
            out = e.rz2rmid(matrix_R, matrix_Z, matrix_t, make_grid=True, rho=True)
        with self.assertRaises(ValueError):
            out_t = et.rz2rmid(matrix_R, matrix_Z, matrix_t, make_grid=True, rho=True)

    def test_rz2rmid_matrix_R_matrix_Z_matrix_t_make_grid_False_each_t_False_rho_True(self):
        out = e.rz2rmid(matrix_R, matrix_Z, matrix_t, each_t=False, rho=True)
        out_t = et.rz2rmid(matrix_R, matrix_Z, matrix_t, each_t=False, rho=True)

        self.assertEqual(out.shape, matrix_R.shape)
        self.assertTrue(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, matrix_R.shape)
        self.assertTrue(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        # TODO: The trispline case has an extra non-NaN seven points here.
        # TODO: Why is this only good to 3%?
        # self.assertTrue((numpy.isnan(out) == numpy.isnan(out_t)).all())
        res2 = (out - out_t)**2
        diff = numpy.sqrt((res2[~numpy.isnan(res2)]).max()) / numpy.absolute(out[~numpy.isnan(out)]).max()
        self.assertLessEqual(diff, 0.03)

    def test_rz2rmid_matrix_R_matrix_Z_matrix_t_make_grid_True_each_t_False_rho_True(self):
        with self.assertRaises(ValueError):
            out = e.rz2rmid(matrix_R, matrix_Z, matrix_t, make_grid=True, each_t=False, rho=True)
        with self.assertRaises(ValueError):
            out_t = et.rz2rmid(matrix_R, matrix_Z, matrix_t, make_grid=True, each_t=False, rho=True)

class rz2roaTestCase(unittest.TestCase):
    # The following tests focus primarily on the consistency of the input/output
    # shapes. More tests should be added to check the consistency between the
    # different mapping routines.

    # scalar R, scalar Z, scalar t:
    def test_rz2roa_scalar_R_scalar_Z_scalar_t_make_grid_False_each_t_True(self):
        out = e.rz2roa(scalar_R, scalar_Z, scalar_t)
        out_t = et.rz2roa(scalar_R, scalar_Z, scalar_t)

        with self.assertRaises(TypeError):
            iter(out)
        self.assertFalse(numpy.isnan(out))
        self.assertFalse(numpy.isinf(out))

        with self.assertRaises(TypeError):
            iter(out_t)
        self.assertFalse(numpy.isnan(out_t))
        self.assertFalse(numpy.isinf(out_t))

        diff = numpy.absolute(out - out_t) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    def test_rz2roa_scalar_R_scalar_Z_scalar_t_make_grid_True_each_t_True(self):
        with self.assertRaises(ValueError):
            out = e.rz2roa(scalar_R, scalar_Z, scalar_t, make_grid=True)
        with self.assertRaises(ValueError):
            out_t = et.rz2roa(scalar_R, scalar_Z, scalar_t, make_grid=True)

    def test_rz2roa_scalar_R_scalar_Z_scalar_t_make_grid_False_each_t_False(self):
        out = e.rz2roa(scalar_R, scalar_Z, scalar_t, each_t=False)
        out_t = et.rz2roa(scalar_R, scalar_Z, scalar_t, each_t=False)

        with self.assertRaises(TypeError):
            iter(out)
        self.assertFalse(numpy.isnan(out))
        self.assertFalse(numpy.isinf(out))

        with self.assertRaises(TypeError):
            iter(out_t)
        self.assertFalse(numpy.isnan(out_t))
        self.assertFalse(numpy.isinf(out_t))

        diff = numpy.absolute(out - out_t) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    def test_rz2roa_scalar_R_scalar_Z_scalar_t_make_grid_True_each_t_False(self):
        with self.assertRaises(ValueError):
            out = e.rz2roa(scalar_R, scalar_Z, scalar_t, make_grid=True, each_t=False)
        with self.assertRaises(ValueError):
            out_t = et.rz2roa(scalar_R, scalar_Z, scalar_t, make_grid=True, each_t=False)

    # vector R, vector Z, scalar t:
    def test_rz2roa_vector_R_vector_Z_scalar_t_make_grid_False_each_t_True(self):
        out = e.rz2roa(vector_R, vector_Z, scalar_t)
        out_t = et.rz2roa(vector_R, vector_Z, scalar_t)

        self.assertEqual(out.shape, (len(vector_R),))
        self.assertTrue(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, (len(vector_R),))
        self.assertTrue(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        self.assertTrue((numpy.isnan(out) == numpy.isnan(out_t)).all())
        diff = numpy.sqrt(((out[~numpy.isnan(out)] - out_t[~numpy.isnan(out)])**2).max()) / numpy.absolute(out[~numpy.isnan(out)]).max()
        self.assertLessEqual(diff, tol)

    def test_rz2roa_vector_R_vector_Z_scalar_t_make_grid_True_each_t_True(self):
        out = e.rz2roa(vector_R, vector_Z, scalar_t, make_grid=True)
        out_t = et.rz2roa(vector_R, vector_Z, scalar_t, make_grid=True)

        self.assertEqual(out.shape, (len(vector_R), len(vector_Z)))
        self.assertTrue(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, (len(vector_R), len(vector_Z)))
        self.assertTrue(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        # TODO: The trispline case has an extra non-NaN two points here.
        # self.assertTrue((numpy.isnan(out) == numpy.isnan(out_t)).all())
        res2 = (out - out_t)**2
        diff = numpy.sqrt((res2[~numpy.isnan(res2)]).max()) / numpy.absolute(out[~numpy.isnan(out)]).max()
        self.assertLessEqual(diff, tol)

    def test_rz2roa_vector_R_vector_Z_scalar_t_make_grid_False_each_t_False(self):
        out = e.rz2roa(vector_R, vector_Z, scalar_t, each_t=False)
        out_t = et.rz2roa(vector_R, vector_Z, scalar_t, each_t=False)

        self.assertEqual(out.shape, (len(vector_R),))
        self.assertTrue(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, (len(vector_R),))
        self.assertTrue(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        self.assertTrue((numpy.isnan(out) == numpy.isnan(out_t)).all())
        diff = numpy.sqrt(((out[~numpy.isnan(out)] - out_t[~numpy.isnan(out)])**2).max()) / numpy.absolute(out[~numpy.isnan(out)]).max()
        self.assertLessEqual(diff, tol)

    def test_rz2roa_vector_R_vector_Z_scalar_t_make_grid_True_each_t_False(self):
        out = e.rz2roa(vector_R, vector_Z, scalar_t, make_grid=True, each_t=False)
        out_t = et.rz2roa(vector_R, vector_Z, scalar_t, make_grid=True, each_t=False)

        self.assertEqual(out.shape, (len(vector_R), len(vector_Z)))
        self.assertTrue(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, (len(vector_R), len(vector_Z)))
        self.assertTrue(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        # TODO: The trispline case has an extra non-NaN two points here.
        # self.assertTrue((numpy.isnan(out) == numpy.isnan(out_t)).all())
        res2 = (out - out_t)**2
        diff = numpy.sqrt((res2[~numpy.isnan(res2)]).max()) / numpy.absolute(out[~numpy.isnan(out)]).max()
        self.assertLessEqual(diff, tol)

    # matrix R, matrix Z, scalar t:
    def test_rz2roa_matrix_R_matrix_Z_scalar_t_make_grid_False_each_t_True(self):
        out = e.rz2roa(matrix_R, matrix_Z, scalar_t)
        out_t = et.rz2roa(matrix_R, matrix_Z, scalar_t)

        self.assertEqual(out.shape, matrix_R.shape)
        self.assertTrue(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, matrix_R.shape)
        self.assertTrue(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        # TODO: The trispline case has an extra non-NaN two points here.
        # self.assertTrue((numpy.isnan(out) == numpy.isnan(out_t)).all())
        res2 = (out - out_t)**2
        diff = numpy.sqrt((res2[~numpy.isnan(res2)]).max()) / numpy.absolute(out[~numpy.isnan(out)]).max()
        self.assertLessEqual(diff, tol)

    def test_rz2roa_matrix_R_matrix_Z_scalar_t_make_grid_True_each_t_True(self):
        with self.assertRaises(ValueError):
            out = e.rz2roa(matrix_R, matrix_Z, scalar_t, make_grid=True)
        with self.assertRaises(ValueError):
            out_t = et.rz2roa(matrix_R, matrix_Z, scalar_t, make_grid=True)

    def test_rz2roa_matrix_R_matrix_Z_scalar_t_make_grid_False_each_t_False(self):
        out = e.rz2roa(matrix_R, matrix_Z, scalar_t, each_t=False)
        out_t = et.rz2roa(matrix_R, matrix_Z, scalar_t, each_t=False)

        self.assertEqual(out.shape, matrix_R.shape)
        self.assertTrue(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, matrix_R.shape)
        self.assertTrue(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        # TODO: The trispline case has an extra non-NaN two points here.
        # self.assertTrue((numpy.isnan(out) == numpy.isnan(out_t)).all())
        res2 = (out - out_t)**2
        diff = numpy.sqrt((res2[~numpy.isnan(res2)]).max()) / numpy.absolute(out[~numpy.isnan(out)]).max()
        self.assertLessEqual(diff, tol)

    def test_rz2roa_matrix_R_matrix_Z_scalar_t_make_grid_True_each_t_False(self):
        with self.assertRaises(ValueError):
            out = e.rz2roa(matrix_R, matrix_Z, scalar_t, make_grid=True, each_t=False)
        with self.assertRaises(ValueError):
            out_t = et.rz2roa(matrix_R, matrix_Z, scalar_t, make_grid=True, each_t=False)

    # scalar R, scalar Z, vector t:
    def test_rz2roa_scalar_R_scalar_Z_vector_t_make_grid_False_each_t_True(self):
        out = e.rz2roa(scalar_R, scalar_Z, vector_t)
        out_t = et.rz2roa(scalar_R, scalar_Z, vector_t)

        self.assertEqual(out.shape, (len(vector_t),))
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, (len(vector_t),))
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    def test_rz2roa_scalar_R_scalar_Z_vector_t_make_grid_True_each_t_True(self):
        with self.assertRaises(ValueError):
            out = e.rz2roa(scalar_R, scalar_Z, vector_t, make_grid=True)
        with self.assertRaises(ValueError):
            out_t = et.rz2roa(scalar_R, scalar_Z, vector_t, make_grid=True)

    def test_rz2roa_scalar_R_scalar_Z_vector_t_make_grid_False_each_t_False(self):
        with self.assertRaises(ValueError):
            out = e.rz2roa(scalar_R, scalar_Z, vector_t, each_t=False)
        with self.assertRaises(ValueError):
            out_t = et.rz2roa(scalar_R, scalar_Z, vector_t, each_t=False)

    def test_rz2roa_scalar_R_scalar_Z_vector_t_make_grid_True_each_t_False(self):
        with self.assertRaises(ValueError):
            out = e.rz2roa(scalar_R, scalar_Z, vector_t, make_grid=True, each_t=False)
        with self.assertRaises(ValueError):
            out_t = et.rz2roa(scalar_R, scalar_Z, vector_t, make_grid=True, each_t=False)

    # vector R, vector Z, vector t:
    def test_rz2roa_vector_R_vector_Z_vector_t_make_grid_False_each_t_True(self):
        out = e.rz2roa(vector_R, vector_Z, vector_t)
        out_t = et.rz2roa(vector_R, vector_Z, vector_t)

        self.assertEqual(out.shape, (len(vector_t), len(vector_R)))
        self.assertTrue(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, (len(vector_t), len(vector_R)))
        self.assertTrue(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        # TODO: The trispline case has an extra non-NaN ten points here.
        # self.assertTrue((numpy.isnan(out) == numpy.isnan(out_t)).all())
        res2 = (out - out_t)**2
        diff = numpy.sqrt((res2[~numpy.isnan(res2)]).max()) / numpy.absolute(out[~numpy.isnan(out)]).max()
        self.assertLessEqual(diff, tol)

    def test_rz2roa_vector_R_vector_Z_vector_t_make_grid_True_each_t_True(self):
        out = e.rz2roa(vector_R, vector_Z, vector_t, make_grid=True)
        out_t = et.rz2roa(vector_R, vector_Z, vector_t, make_grid=True)

        self.assertEqual(out.shape, (len(vector_t), len(vector_R), len(vector_Z)))
        self.assertTrue(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, (len(vector_t), len(vector_R), len(vector_Z)))
        self.assertTrue(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        # TODO: The trispline case has an extra non-NaN 289 points here.
        # self.assertTrue((numpy.isnan(out) == numpy.isnan(out_t)).all())
        res2 = (out - out_t)**2
        diff = numpy.sqrt((res2[~numpy.isnan(res2)]).max()) / numpy.absolute(out[~numpy.isnan(out)]).max()
        self.assertLessEqual(diff, tol)

    def test_rz2roa_vector_R_vector_Z_vector_t_make_grid_False_each_t_False(self):
        with self.assertRaises(ValueError):
            out = e.rz2roa(vector_R, vector_Z, vector_t, each_t=False)
        with self.assertRaises(ValueError):
            out_t = et.rz2roa(vector_R, vector_Z, vector_t, each_t=False)

    def test_rz2roa_vector_R_vector_Z_vector_t_make_grid_True_each_t_False(self):
        with self.assertRaises(ValueError):
            out = e.rz2roa(vector_R, vector_Z, vector_t, make_grid=True, each_t=False)
        with self.assertRaises(ValueError):
            out_t = et.rz2roa(vector_R, vector_Z, vector_t, make_grid=True, each_t=False)

    # matrix R, matrix Z, vector t:
    def test_rz2roa_matrix_R_matrix_Z_vector_t_make_grid_False_each_t_True(self):
        out = e.rz2roa(matrix_R, matrix_Z, vector_t)
        out_t = et.rz2roa(matrix_R, matrix_Z, vector_t)

        self.assertEqual(out.shape, tuple([len(vector_t),] + list(matrix_R.shape)))
        self.assertTrue(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, tuple([len(vector_t),] + list(matrix_R.shape)))
        self.assertTrue(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        # TODO: The trispline case has an extra non-NaN 289 points here.
        # self.assertTrue((numpy.isnan(out) == numpy.isnan(out_t)).all())
        res2 = (out - out_t)**2
        diff = numpy.sqrt((res2[~numpy.isnan(res2)]).max()) / numpy.absolute(out[~numpy.isnan(out)]).max()
        self.assertLessEqual(diff, tol)

    def test_rz2roa_matrix_R_matrix_Z_vector_t_make_grid_True_each_t_True(self):
        with self.assertRaises(ValueError):
            out = e.rz2roa(matrix_R, matrix_Z, vector_t, make_grid=True)
        with self.assertRaises(ValueError):
            out_t = et.rz2roa(matrix_R, matrix_Z, vector_t, make_grid=True)

    def test_rz2roa_matrix_R_matrix_Z_vector_t_make_grid_False_each_t_False(self):
        with self.assertRaises(ValueError):
            out = e.rz2roa(matrix_R, matrix_Z, vector_t, each_t=False)
        with self.assertRaises(ValueError):
            out_t = et.rz2roa(matrix_R, matrix_Z, vector_t, each_t=False)

    def test_rz2roa_matrix_R_matrix_Z_vector_t_make_grid_True_each_t_False(self):
        with self.assertRaises(ValueError):
            out = e.rz2roa(matrix_R, matrix_Z, vector_t, make_grid=True, each_t=False)
        with self.assertRaises(ValueError):
            out_t = et.rz2roa(matrix_R, matrix_Z, vector_t, make_grid=True, each_t=False)

    # matrix R, matrix Z, matrix t:
    def test_rz2roa_matrix_R_matrix_Z_matrix_t_make_grid_False_each_t_True(self):
        with self.assertRaises(ValueError):
            out = e.rz2roa(matrix_R, matrix_Z, matrix_t)
        with self.assertRaises(ValueError):
            out_t = et.rz2roa(matrix_R, matrix_Z, matrix_t)

    def test_rz2roa_matrix_R_matrix_Z_matrix_t_make_grid_True_each_t_True(self):
        with self.assertRaises(ValueError):
            out = e.rz2roa(matrix_R, matrix_Z, matrix_t, make_grid=True)
        with self.assertRaises(ValueError):
            out_t = et.rz2roa(matrix_R, matrix_Z, matrix_t, make_grid=True)

    def test_rz2roa_matrix_R_matrix_Z_matrix_t_make_grid_False_each_t_False(self):
        out = e.rz2roa(matrix_R, matrix_Z, matrix_t, each_t=False)
        out_t = et.rz2roa(matrix_R, matrix_Z, matrix_t, each_t=False)

        self.assertEqual(out.shape, matrix_R.shape)
        self.assertTrue(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, matrix_R.shape)
        self.assertTrue(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        # TODO: The trispline case has an extra non-NaN 7 points here.
        # self.assertTrue((numpy.isnan(out) == numpy.isnan(out_t)).all())
        # TODO: Why is this only good to 3%?
        res2 = (out - out_t)**2
        diff = numpy.sqrt((res2[~numpy.isnan(res2)]).max()) / numpy.absolute(out[~numpy.isnan(out)]).max()
        self.assertLessEqual(diff, 0.03)

    def test_rz2roa_matrix_R_matrix_Z_matrix_t_make_grid_True_each_t_False(self):
        with self.assertRaises(ValueError):
            out = e.rz2roa(matrix_R, matrix_Z, matrix_t, make_grid=True, each_t=False)
        with self.assertRaises(ValueError):
            out_t = et.rz2roa(matrix_R, matrix_Z, matrix_t, make_grid=True, each_t=False)

class rmid2roaTestCase(unittest.TestCase):
    # The following tests focus primarily on the consistency of the input/output
    # shapes. More tests should be added to check the consistency between the
    # different mapping routines.

    # scalar Rmid, scalar t:
    def test_rmid2roa_scalar_Rmid_scalar_t_each_t_True(self):
        out = e.rmid2roa(scalar_Rmid, scalar_t)
        out_t = et.rmid2roa(scalar_Rmid, scalar_t)

        with self.assertRaises(TypeError):
            iter(out)
        self.assertFalse(numpy.isnan(out))
        self.assertFalse(numpy.isinf(out))

        with self.assertRaises(TypeError):
            iter(out_t)
        self.assertFalse(numpy.isnan(out_t))
        self.assertFalse(numpy.isinf(out_t))

        diff = numpy.absolute(out - out_t) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    def test_rmid2roa_scalar_Rmid_scalar_t_each_t_False(self):
        out = e.rmid2roa(scalar_Rmid, scalar_t, each_t=False)
        out_t = et.rmid2roa(scalar_Rmid, scalar_t, each_t=False)

        with self.assertRaises(TypeError):
            iter(out)
        self.assertFalse(numpy.isnan(out))
        self.assertFalse(numpy.isinf(out))

        with self.assertRaises(TypeError):
            iter(out_t)
        self.assertFalse(numpy.isnan(out_t))
        self.assertFalse(numpy.isinf(out_t))

        diff = numpy.absolute(out - out_t) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    # vector Rmid, scalar t:
    def test_rmid2roa_vector_Rmid_scalar_t_each_t_True(self):
        out = e.rmid2roa(vector_Rmid, scalar_t)
        out_t = et.rmid2roa(vector_Rmid, scalar_t)

        self.assertEqual(out.shape, (len(vector_Rmid),))
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, (len(vector_Rmid),))
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    def test_rmid2roa_vector_Rmid_scalar_t_each_t_False(self):
        out = e.rmid2roa(vector_Rmid, scalar_t, each_t=False)
        out_t = et.rmid2roa(vector_Rmid, scalar_t, each_t=False)

        self.assertEqual(out.shape, (len(vector_Rmid),))
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, (len(vector_Rmid),))
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    # matrix Rmid, scalar t:
    def test_rmid2roa_matrix_Rmid_scalar_t_each_t_True(self):
        out = e.rmid2roa(matrix_Rmid, scalar_t)
        out_t = et.rmid2roa(matrix_Rmid, scalar_t)

        self.assertEqual(out.shape, matrix_Rmid.shape)
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, matrix_Rmid.shape)
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    def test_rmid2roa_matrix_Rmid_scalar_t_each_t_False(self):
        out = e.rmid2roa(matrix_Rmid, scalar_t, each_t=False)
        out_t = et.rmid2roa(matrix_Rmid, scalar_t, each_t=False)

        self.assertEqual(out.shape, matrix_Rmid.shape)
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, matrix_Rmid.shape)
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    # scalar Rmid, vector t:
    def test_rmid2roa_scalar_Rmid_vector_t_each_t_True(self):
        out = e.rmid2roa(scalar_Rmid, vector_t)
        out_t = et.rmid2roa(scalar_Rmid, vector_t)

        self.assertEqual(out.shape, (len(vector_t),))
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, (len(vector_t),))
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    def test_rmid2roa_scalar_Rmid_vector_t_each_t_False(self):
        with self.assertRaises(ValueError):
            out = e.rmid2roa(scalar_Rmid, vector_t, each_t=False)
        with self.assertRaises(ValueError):
            out_t = et.rmid2roa(scalar_Rmid, vector_t, each_t=False)

    # vector Rmid, vector t:
    def test_rmid2roa_vector_Rmid_vector_t_each_t_True(self):
        out = e.rmid2roa(vector_Rmid, vector_t)
        out_t = et.rmid2roa(vector_Rmid, vector_t)

        self.assertEqual(out.shape, (len(vector_t), len(vector_Rmid)))
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, (len(vector_t), len(vector_Rmid)))
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    def test_rmid2roa_vector_Rmid_vector_t_each_t_False(self):
        with self.assertRaises(ValueError):
            out = e.rmid2roa(vector_Rmid, vector_t, each_t=False)
        with self.assertRaises(ValueError):
            out_t = et.rmid2roa(vector_Rmid, vector_t, each_t=False)

    # matrix Rmid, vector t:
    def test_rmid2roa_matrix_Rmid_vector_t_each_t_True(self):
        out = e.rmid2roa(matrix_Rmid, vector_t)
        out_t = et.rmid2roa(matrix_Rmid, vector_t)

        self.assertEqual(out.shape, tuple([len(vector_t),] + list(matrix_Rmid.shape)))
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, tuple([len(vector_t),] + list(matrix_Rmid.shape)))
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    def test_rmid2roa_matrix_Rmid_vector_t_each_t_False(self):
        with self.assertRaises(ValueError):
            out = e.rmid2roa(matrix_Rmid, vector_t, each_t=False)
        with self.assertRaises(ValueError):
            out_t = et.rmid2roa(matrix_Rmid, vector_t, each_t=False)

    # matrix Rmid, matrix t:
    def test_rmid2roa_matrix_Rmid_matrix_t_each_t_True(self):
        with self.assertRaises(ValueError):
            out = e.rmid2roa(matrix_Rmid, matrix_t)
        with self.assertRaises(ValueError):
            out_t = et.rmid2roa(matrix_Rmid, matrix_t)

    def test_rmid2roa_matrix_Rmid_matrix_t_each_t_False(self):
        out = e.rmid2roa(matrix_Rmid, matrix_t, each_t=False)
        out_t = et.rmid2roa(matrix_Rmid, matrix_t, each_t=False)

        self.assertEqual(out.shape, matrix_Rmid.shape)
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, matrix_Rmid.shape)
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        # TODO: Only good to 2%.
        diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertLessEqual(diff, 0.02)

class rmid2psinormTestCase(unittest.TestCase):
    # The following tests focus primarily on the consistency of the input/output
    # shapes. More tests should be added to check the consistency between the
    # different mapping routines.

    # scalar Rmid, scalar t:
    def test_rmid2psinorm_scalar_psinorm_scalar_t_each_t_True(self):
        out = e.rmid2psinorm(scalar_Rmid, scalar_t)
        out_t = et.rmid2psinorm(scalar_Rmid, scalar_t)

        with self.assertRaises(TypeError):
            iter(out)
        self.assertFalse(numpy.isnan(out))
        self.assertFalse(numpy.isinf(out))

        with self.assertRaises(TypeError):
            iter(out_t)
        self.assertFalse(numpy.isnan(out_t))
        self.assertFalse(numpy.isinf(out_t))

        diff = numpy.absolute(out - out_t) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    def test_rmid2psinorm_scalar_psinorm_scalar_t_each_t_False(self):
        out = e.rmid2psinorm(scalar_Rmid, scalar_t, each_t=False)
        out_t = et.rmid2psinorm(scalar_Rmid, scalar_t, each_t=False)

        with self.assertRaises(TypeError):
            iter(out)
        self.assertFalse(numpy.isnan(out))
        self.assertFalse(numpy.isinf(out))

        with self.assertRaises(TypeError):
            iter(out_t)
        self.assertFalse(numpy.isnan(out_t))
        self.assertFalse(numpy.isinf(out_t))

        diff = numpy.absolute(out - out_t) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    # vector Rmid, scalar t:
    def test_rmid2psinorm_vector_psinorm_scalar_t_each_t_True(self):
        out = e.rmid2psinorm(vector_Rmid, scalar_t)
        out_t = et.rmid2psinorm(vector_Rmid, scalar_t)

        self.assertEqual(out.shape, (len(vector_Rmid),))
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, (len(vector_Rmid),))
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    def test_rmid2psinorm_vector_psinorm_scalar_t_each_t_False(self):
        out = e.rmid2psinorm(vector_Rmid, scalar_t, each_t=False)
        out_t = et.rmid2psinorm(vector_Rmid, scalar_t, each_t=False)

        self.assertEqual(out.shape, (len(vector_Rmid),))
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, (len(vector_Rmid),))
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    # matrix Rmid, scalar t:
    def test_rmid2psinorm_matrix_psinorm_scalar_t_each_t_True(self):
        out = e.rmid2psinorm(matrix_Rmid, scalar_t)
        out_t = et.rmid2psinorm(matrix_Rmid, scalar_t)

        self.assertEqual(out.shape, matrix_Rmid.shape)
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, matrix_Rmid.shape)
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    def test_rmid2psinorm_matrix_psinorm_scalar_t_each_t_False(self):
        out = e.rmid2psinorm(matrix_Rmid, scalar_t, each_t=False)
        out_t = et.rmid2psinorm(matrix_Rmid, scalar_t, each_t=False)

        self.assertEqual(out.shape, matrix_Rmid.shape)
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, matrix_Rmid.shape)
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    # scalar Rmid, vector t:
    def test_rmid2psinorm_scalar_psinorm_vector_t_each_t_True(self):
        out = e.rmid2psinorm(scalar_Rmid, vector_t)
        out_t = et.rmid2psinorm(scalar_Rmid, vector_t)

        self.assertEqual(out.shape, (len(vector_t),))
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, (len(vector_t),))
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    def test_rmid2psinorm_scalar_psinorm_vector_t_each_t_False(self):
        with self.assertRaises(ValueError):
            out = e.rmid2psinorm(scalar_Rmid, vector_t, each_t=False)
        with self.assertRaises(ValueError):
            out_t = et.rmid2psinorm(scalar_Rmid, vector_t, each_t=False)

    # vector Rmid, vector t:
    def test_rmid2psinorm_vector_psinorm_vector_t_each_t_True(self):
        out = e.rmid2psinorm(vector_Rmid, vector_t)
        out_t = et.rmid2psinorm(vector_Rmid, vector_t)

        self.assertEqual(out.shape, (len(vector_t), len(vector_Rmid)))
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, (len(vector_t), len(vector_Rmid)))
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    def test_rmid2psinorm_vector_psinorm_vector_t_each_t_False(self):
        with self.assertRaises(ValueError):
            out = e.rmid2psinorm(vector_Rmid, vector_t, each_t=False)
        with self.assertRaises(ValueError):
            out_t = et.rmid2psinorm(vector_Rmid, vector_t, each_t=False)

    # matrix Rmid, vector t:
    def test_rmid2psinorm_matrix_psinorm_vector_t_each_t_True(self):
        out = e.rmid2psinorm(matrix_Rmid, vector_t)
        out_t = et.rmid2psinorm(matrix_Rmid, vector_t)

        self.assertEqual(out.shape, tuple([len(vector_t),] + list(matrix_Rmid.shape)))
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, tuple([len(vector_t),] + list(matrix_Rmid.shape)))
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    def test_rmid2psinorm_matrix_psinorm_vector_t_each_t_False(self):
        with self.assertRaises(ValueError):
            out = e.rmid2psinorm(matrix_Rmid, vector_t, each_t=False)
        with self.assertRaises(ValueError):
            out_t = et.rmid2psinorm(matrix_Rmid, vector_t, each_t=False)

    # matrix Rmid, matrix t:
    def test_rmid2psinorm_matrix_psinorm_matrix_t_each_t_True(self):
        with self.assertRaises(ValueError):
            out = e.rmid2psinorm(matrix_Rmid, matrix_t)
        with self.assertRaises(ValueError):
            out_t = et.rmid2psinorm(matrix_Rmid, matrix_t)

    def test_rmid2psinorm_matrix_psinorm_matrix_t_each_t_False(self):
        out = e.rmid2psinorm(matrix_Rmid, matrix_t, each_t=False)
        out_t = et.rmid2psinorm(matrix_Rmid, matrix_t, each_t=False)

        self.assertEqual(out.shape, matrix_Rmid.shape)
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, matrix_Rmid.shape)
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        # TODO: Only good to 2%.
        diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertLessEqual(diff, 0.02)

class rmid2phinormTestCase(unittest.TestCase):
    # The following tests focus primarily on the consistency of the input/output
    # shapes. More tests should be added to check the consistency between the
    # different mapping routines.

    # scalar Rmid, scalar t:
    def test_rmid2phinorm_scalar_psinorm_scalar_t_each_t_True(self):
        out = e.rmid2phinorm(scalar_Rmid, scalar_t)
        out_t = et.rmid2phinorm(scalar_Rmid, scalar_t)

        with self.assertRaises(TypeError):
            iter(out)
        self.assertFalse(numpy.isnan(out))
        self.assertFalse(numpy.isinf(out))

        with self.assertRaises(TypeError):
            iter(out_t)
        self.assertFalse(numpy.isnan(out_t))
        self.assertFalse(numpy.isinf(out_t))

        diff = numpy.absolute(out - out_t) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    def test_rmid2phinorm_scalar_psinorm_scalar_t_each_t_False(self):
        out = e.rmid2phinorm(scalar_Rmid, scalar_t, each_t=False)
        out_t = et.rmid2phinorm(scalar_Rmid, scalar_t, each_t=False)

        with self.assertRaises(TypeError):
            iter(out)
        self.assertFalse(numpy.isnan(out))
        self.assertFalse(numpy.isinf(out))

        with self.assertRaises(TypeError):
            iter(out_t)
        self.assertFalse(numpy.isnan(out_t))
        self.assertFalse(numpy.isinf(out_t))

        diff = numpy.absolute(out - out_t) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    # vector Rmid, scalar t:
    def test_rmid2phinorm_vector_psinorm_scalar_t_each_t_True(self):
        out = e.rmid2phinorm(vector_Rmid, scalar_t)
        out_t = et.rmid2phinorm(vector_Rmid, scalar_t)

        self.assertEqual(out.shape, (len(vector_Rmid),))
        self.assertTrue(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, (len(vector_Rmid),))
        self.assertTrue(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        self.assertTrue((numpy.isnan(out) == numpy.isnan(out_t)).all())
        diff = numpy.sqrt(((out[~numpy.isnan(out)] - out_t[~numpy.isnan(out)])**2).max()) / numpy.absolute(out[~numpy.isnan(out)]).max()
        self.assertLessEqual(diff, tol)

    def test_rmid2phinorm_vector_psinorm_scalar_t_each_t_False(self):
        out = e.rmid2phinorm(vector_Rmid, scalar_t, each_t=False)
        out_t = et.rmid2phinorm(vector_Rmid, scalar_t, each_t=False)

        self.assertEqual(out.shape, (len(vector_Rmid),))
        self.assertTrue(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, (len(vector_Rmid),))
        self.assertTrue(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        self.assertTrue((numpy.isnan(out) == numpy.isnan(out_t)).all())
        diff = numpy.sqrt(((out[~numpy.isnan(out)] - out_t[~numpy.isnan(out)])**2).max()) / numpy.absolute(out[~numpy.isnan(out)]).max()
        self.assertLessEqual(diff, tol)

    # matrix Rmid, scalar t:
    def test_rmid2phinorm_matrix_psinorm_scalar_t_each_t_True(self):
        out = e.rmid2phinorm(matrix_Rmid, scalar_t)
        out_t = et.rmid2phinorm(matrix_Rmid, scalar_t)

        self.assertEqual(out.shape, matrix_Rmid.shape)
        self.assertTrue(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out.shape, matrix_Rmid.shape)
        self.assertTrue(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertTrue((numpy.isnan(out) == numpy.isnan(out_t)).all())
        diff = numpy.sqrt(((out[~numpy.isnan(out)] - out_t[~numpy.isnan(out)])**2).max()) / numpy.absolute(out[~numpy.isnan(out)]).max()
        self.assertLessEqual(diff, tol)

    def test_rmid2phinorm_matrix_psinorm_scalar_t_each_t_False(self):
        out = e.rmid2phinorm(matrix_Rmid, scalar_t, each_t=False)
        out_t = et.rmid2phinorm(matrix_Rmid, scalar_t, each_t=False)

        self.assertEqual(out.shape, matrix_Rmid.shape)
        self.assertTrue(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, matrix_Rmid.shape)
        self.assertTrue(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        self.assertTrue((numpy.isnan(out) == numpy.isnan(out_t)).all())
        diff = numpy.sqrt(((out[~numpy.isnan(out)] - out_t[~numpy.isnan(out)])**2).max()) / numpy.absolute(out[~numpy.isnan(out)]).max()
        self.assertLessEqual(diff, tol)

    # scalar Rmid, vector t:
    def test_rmid2phinorm_scalar_psinorm_vector_t_each_t_True(self):
        out = e.rmid2phinorm(scalar_Rmid, vector_t)
        out_t = et.rmid2phinorm(scalar_Rmid, vector_t)

        self.assertEqual(out.shape, (len(vector_t),))
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, (len(vector_t),))
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    def test_rmid2phinorm_scalar_psinorm_vector_t_each_t_False(self):
        with self.assertRaises(ValueError):
            out = e.rmid2phinorm(scalar_Rmid, vector_t, each_t=False)
        with self.assertRaises(ValueError):
            out_t = et.rmid2phinorm(scalar_Rmid, vector_t, each_t=False)

    # vector Rmid, vector t:
    def test_rmid2phinorm_vector_psinorm_vector_t_each_t_True(self):
        out = e.rmid2phinorm(vector_Rmid, vector_t)
        out_t = et.rmid2phinorm(vector_Rmid, vector_t)

        self.assertEqual(out.shape, (len(vector_t), len(vector_Rmid)))
        self.assertTrue(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, (len(vector_t), len(vector_Rmid)))
        self.assertTrue(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        # TODO: The trispline case has an extra 1 non-NaN points here.
        # self.assertTrue((numpy.isnan(out) == numpy.isnan(out_t)).all())
        res2 = (out - out_t)**2
        diff = numpy.sqrt((res2[~numpy.isnan(res2)]).max()) / numpy.absolute(out[~numpy.isnan(out)]).max()
        self.assertLessEqual(diff, tol)

    def test_rmid2phinorm_vector_psinorm_vector_t_each_t_False(self):
        with self.assertRaises(ValueError):
            out = e.rmid2phinorm(vector_Rmid, vector_t, each_t=False)
        with self.assertRaises(ValueError):
            out_t = et.rmid2phinorm(vector_Rmid, vector_t, each_t=False)

    # matrix Rmid, vector t:
    def test_rmid2phinorm_matrix_psinorm_vector_t_each_t_True(self):
        out = e.rmid2phinorm(matrix_Rmid, vector_t)
        out_t = et.rmid2phinorm(matrix_Rmid, vector_t)

        self.assertEqual(out.shape, tuple([len(vector_t),] + list(matrix_Rmid.shape)))
        self.assertTrue(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, tuple([len(vector_t),] + list(matrix_Rmid.shape)))
        self.assertTrue(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        # TODO: The trispline case has an extra 25 non-NaN points here.
        # self.assertTrue((numpy.isnan(out) == numpy.isnan(out_t)).all())
        res2 = (out - out_t)**2
        diff = numpy.sqrt((res2[~numpy.isnan(res2)]).max()) / numpy.absolute(out[~numpy.isnan(out)]).max()
        self.assertLessEqual(diff, tol)

    def test_rmid2phinorm_matrix_psinorm_vector_t_each_t_False(self):
        with self.assertRaises(ValueError):
            out = e.rmid2phinorm(matrix_Rmid, vector_t, each_t=False)
        with self.assertRaises(ValueError):
            out_t = et.rmid2phinorm(matrix_Rmid, vector_t, each_t=False)

    # matrix Rmid, matrix t:
    def test_rmid2phinorm_matrix_psinorm_matrix_t_each_t_True(self):
        with self.assertRaises(ValueError):
            out = e.rmid2phinorm(matrix_Rmid, matrix_t)
        with self.assertRaises(ValueError):
            out_t = et.rmid2phinorm(matrix_Rmid, matrix_t)

    def test_rmid2phinorm_matrix_psinorm_matrix_t_each_t_False(self):
        out = e.rmid2phinorm(matrix_Rmid, matrix_t, each_t=False)
        out_t = et.rmid2phinorm(matrix_Rmid, matrix_t, each_t=False)

        self.assertEqual(out.shape, matrix_Rmid.shape)
        self.assertTrue(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, matrix_Rmid.shape)
        self.assertTrue(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        # TODO: The trispline case has an extra 1 non-NaN points here.
        # self.assertTrue((numpy.isnan(out) == numpy.isnan(out_t)).all())
        # TODO: Only good to 2%.
        res2 = (out - out_t)**2
        diff = numpy.sqrt((res2[~numpy.isnan(res2)]).max()) / numpy.absolute(out[~numpy.isnan(out)]).max()
        self.assertLessEqual(diff, 0.02)

class rmid2volnormTestCase(unittest.TestCase):
    # The following tests focus primarily on the consistency of the input/output
    # shapes. More tests should be added to check the consistency between the
    # different mapping routines.

    # scalar Rmid, scalar t:
    def test_rmid2volnorm_scalar_psinorm_scalar_t_each_t_True(self):
        out = e.rmid2volnorm(scalar_Rmid, scalar_t)
        out_t = et.rmid2volnorm(scalar_Rmid, scalar_t)

        with self.assertRaises(TypeError):
            iter(out)
        self.assertFalse(numpy.isnan(out))
        self.assertFalse(numpy.isinf(out))

        with self.assertRaises(TypeError):
            iter(out_t)
        self.assertFalse(numpy.isnan(out_t))
        self.assertFalse(numpy.isinf(out_t))

        diff = numpy.absolute(out - out_t) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    def test_rmid2volnorm_scalar_psinorm_scalar_t_each_t_False(self):
        out = e.rmid2volnorm(scalar_Rmid, scalar_t, each_t=False)
        out_t = et.rmid2volnorm(scalar_Rmid, scalar_t, each_t=False)

        with self.assertRaises(TypeError):
            iter(out)
        self.assertFalse(numpy.isnan(out))
        self.assertFalse(numpy.isinf(out))

        with self.assertRaises(TypeError):
            iter(out_t)
        self.assertFalse(numpy.isnan(out_t))
        self.assertFalse(numpy.isinf(out_t))

        diff = numpy.absolute(out - out_t) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    # vector Rmid, scalar t:
    def test_rmid2volnorm_vector_psinorm_scalar_t_each_t_True(self):
        out = e.rmid2volnorm(vector_Rmid, scalar_t)
        out_t = et.rmid2volnorm(vector_Rmid, scalar_t)

        self.assertEqual(out.shape, (len(vector_Rmid),))
        self.assertTrue(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, (len(vector_Rmid),))
        self.assertTrue(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        self.assertTrue((numpy.isnan(out) == numpy.isnan(out_t)).all())
        diff = numpy.sqrt(((out[~numpy.isnan(out)] - out_t[~numpy.isnan(out)])**2).max()) / numpy.absolute(out[~numpy.isnan(out)]).max()
        self.assertLessEqual(diff, tol)

    def test_rmid2volnorm_vector_psinorm_scalar_t_each_t_False(self):
        out = e.rmid2volnorm(vector_Rmid, scalar_t, each_t=False)
        out_t = et.rmid2volnorm(vector_Rmid, scalar_t, each_t=False)

        self.assertEqual(out.shape, (len(vector_Rmid),))
        self.assertTrue(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, (len(vector_Rmid),))
        self.assertTrue(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        self.assertTrue((numpy.isnan(out) == numpy.isnan(out_t)).all())
        diff = numpy.sqrt(((out[~numpy.isnan(out)] - out_t[~numpy.isnan(out)])**2).max()) / numpy.absolute(out[~numpy.isnan(out)]).max()
        self.assertLessEqual(diff, tol)

    # matrix Rmid, scalar t:
    def test_rmid2volnorm_matrix_psinorm_scalar_t_each_t_True(self):
        out = e.rmid2volnorm(matrix_Rmid, scalar_t)
        out_t = et.rmid2volnorm(matrix_Rmid, scalar_t)

        self.assertEqual(out.shape, matrix_Rmid.shape)
        self.assertTrue(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, matrix_Rmid.shape)
        self.assertTrue(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        self.assertTrue((numpy.isnan(out) == numpy.isnan(out_t)).all())
        diff = numpy.sqrt(((out[~numpy.isnan(out)] - out_t[~numpy.isnan(out)])**2).max()) / numpy.absolute(out[~numpy.isnan(out)]).max()
        self.assertLessEqual(diff, tol)

    def test_rmid2volnorm_matrix_psinorm_scalar_t_each_t_False(self):
        out = e.rmid2volnorm(matrix_Rmid, scalar_t, each_t=False)
        out_t = et.rmid2volnorm(matrix_Rmid, scalar_t, each_t=False)

        self.assertEqual(out.shape, matrix_Rmid.shape)
        self.assertTrue(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, matrix_Rmid.shape)
        self.assertTrue(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        self.assertTrue((numpy.isnan(out) == numpy.isnan(out_t)).all())
        diff = numpy.sqrt(((out[~numpy.isnan(out)] - out_t[~numpy.isnan(out)])**2).max()) / numpy.absolute(out[~numpy.isnan(out)]).max()
        self.assertLessEqual(diff, tol)

    # scalar Rmid, vector t:
    def test_rmid2volnorm_scalar_psinorm_vector_t_each_t_True(self):
        out = e.rmid2volnorm(scalar_Rmid, vector_t)
        out_t = et.rmid2volnorm(scalar_Rmid, vector_t)

        self.assertEqual(out.shape, (len(vector_t),))
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, (len(vector_t),))
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    def test_rmid2volnorm_scalar_psinorm_vector_t_each_t_False(self):
        with self.assertRaises(ValueError):
            out = e.rmid2volnorm(scalar_Rmid, vector_t, each_t=False)
        with self.assertRaises(ValueError):
            out_t = et.rmid2volnorm(scalar_Rmid, vector_t, each_t=False)

    # vector Rmid, vector t:
    def test_rmid2volnorm_vector_psinorm_vector_t_each_t_True(self):
        out = e.rmid2volnorm(vector_Rmid, vector_t)
        out_t = et.rmid2volnorm(vector_Rmid, vector_t)

        self.assertEqual(out.shape, (len(vector_t), len(vector_Rmid)))
        self.assertTrue(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, (len(vector_t), len(vector_Rmid)))
        self.assertTrue(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        # TODO: The trispline case has an extra 1 non-NaN points here.
        # self.assertTrue((numpy.isnan(out) == numpy.isnan(out_t)).all())
        res2 = (out - out_t)**2
        diff = numpy.sqrt((res2[~numpy.isnan(res2)]).max()) / numpy.absolute(out[~numpy.isnan(out)]).max()
        self.assertLessEqual(diff, tol)

    def test_rmid2volnorm_vector_psinorm_vector_t_each_t_False(self):
        with self.assertRaises(ValueError):
            out = e.rmid2volnorm(vector_Rmid, vector_t, each_t=False)
        with self.assertRaises(ValueError):
            out_t = et.rmid2volnorm(vector_Rmid, vector_t, each_t=False)

    # matrix Rmid, vector t:
    def test_rmid2volnorm_matrix_psinorm_vector_t_each_t_True(self):
        out = e.rmid2volnorm(matrix_Rmid, vector_t)
        out_t = et.rmid2volnorm(matrix_Rmid, vector_t)

        self.assertEqual(out.shape, tuple([len(vector_t),] + list(matrix_Rmid.shape)))
        self.assertTrue(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, tuple([len(vector_t),] + list(matrix_Rmid.shape)))
        self.assertTrue(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        # TODO: The trispline case has an extra 25 non-NaN points here.
        # self.assertTrue((numpy.isnan(out) == numpy.isnan(out_t)).all())
        res2 = (out - out_t)**2
        diff = numpy.sqrt((res2[~numpy.isnan(res2)]).max()) / numpy.absolute(out[~numpy.isnan(out)]).max()
        self.assertLessEqual(diff, tol)

    def test_rmid2volnorm_matrix_psinorm_vector_t_each_t_False(self):
        with self.assertRaises(ValueError):
            out = e.rmid2volnorm(matrix_Rmid, vector_t, each_t=False)
        with self.assertRaises(ValueError):
            out_t = et.rmid2volnorm(matrix_Rmid, vector_t, each_t=False)

    # matrix Rmid, matrix t:
    def test_rmid2volnorm_matrix_psinorm_matrix_t_each_t_True(self):
        with self.assertRaises(ValueError):
            out = e.rmid2volnorm(matrix_Rmid, matrix_t)
        with self.assertRaises(ValueError):
            out_t = et.rmid2volnorm(matrix_Rmid, matrix_t)

    def test_rmid2volnorm_matrix_psinorm_matrix_t_each_t_False(self):
        out = e.rmid2volnorm(matrix_Rmid, matrix_t, each_t=False)
        out_t = et.rmid2volnorm(matrix_Rmid, matrix_t, each_t=False)

        self.assertEqual(out.shape, matrix_Rmid.shape)
        self.assertTrue(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, matrix_Rmid.shape)
        self.assertTrue(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        # TODO: The trispline case has an extra 1 non-NaN points here.
        # self.assertTrue((numpy.isnan(out) == numpy.isnan(out_t)).all())
        # TODO: Only good to 2%.
        res2 = (out - out_t)**2
        diff = numpy.sqrt((res2[~numpy.isnan(res2)]).max()) / numpy.absolute(out[~numpy.isnan(out)]).max()
        self.assertLessEqual(diff, 0.02)

class roa2rmidTestCase(unittest.TestCase):
    # The following tests focus primarily on the consistency of the input/output
    # shapes. More tests should be added to check the consistency between the
    # different mapping routines.

    # scalar r/a, scalar t:
    def test_roa2rmid_scalar_psinorm_scalar_t_each_t_True(self):
        out = e.roa2rmid(scalar_psinorm, scalar_t)
        out_t = et.roa2rmid(scalar_psinorm, scalar_t)

        with self.assertRaises(TypeError):
            iter(out)
        self.assertFalse(numpy.isnan(out))
        self.assertFalse(numpy.isinf(out))

        with self.assertRaises(TypeError):
            iter(out_t)
        self.assertFalse(numpy.isnan(out_t))
        self.assertFalse(numpy.isinf(out_t))

        diff = numpy.absolute(out - out_t) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    def test_roa2rmid_scalar_psinorm_scalar_t_each_t_False(self):
        out = e.roa2rmid(scalar_psinorm, scalar_t, each_t=False)
        out_t = et.roa2rmid(scalar_psinorm, scalar_t, each_t=False)

        with self.assertRaises(TypeError):
            iter(out)
        self.assertFalse(numpy.isnan(out))
        self.assertFalse(numpy.isinf(out))

        with self.assertRaises(TypeError):
            iter(out_t)
        self.assertFalse(numpy.isnan(out_t))
        self.assertFalse(numpy.isinf(out_t))

        diff = numpy.absolute(out - out_t) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    # vector r/a, scalar t:
    def test_roa2rmid_vector_psinorm_scalar_t_each_t_True(self):
        out = e.roa2rmid(vector_psinorm, scalar_t)
        out_t = et.roa2rmid(vector_psinorm, scalar_t)

        self.assertEqual(out.shape, (len(vector_psinorm),))
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, (len(vector_psinorm),))
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    def test_roa2rmid_vector_psinorm_scalar_t_each_t_False(self):
        out = e.roa2rmid(vector_psinorm, scalar_t, each_t=False)
        out_t = et.roa2rmid(vector_psinorm, scalar_t, each_t=False)

        self.assertEqual(out.shape, (len(vector_psinorm),))
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, (len(vector_psinorm),))
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    # matrix r/a, scalar t:
    def test_roa2rmid_matrix_psinorm_scalar_t_each_t_True(self):
        out = e.roa2rmid(matrix_psinorm, scalar_t)
        out_t = et.roa2rmid(matrix_psinorm, scalar_t)

        self.assertEqual(out.shape, matrix_psinorm.shape)
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, matrix_psinorm.shape)
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    def test_roa2rmid_matrix_psinorm_scalar_t_each_t_False(self):
        out = e.roa2rmid(matrix_psinorm, scalar_t, each_t=False)
        out_t = et.roa2rmid(matrix_psinorm, scalar_t, each_t=False)

        self.assertEqual(out.shape, matrix_psinorm.shape)
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, matrix_psinorm.shape)
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    # scalar r/a, vector t:
    def test_roa2rmid_scalar_psinorm_vector_t_each_t_True(self):
        out = e.roa2rmid(scalar_psinorm, vector_t)
        out_t = et.roa2rmid(scalar_psinorm, vector_t)

        self.assertEqual(out.shape, (len(vector_t),))
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, (len(vector_t),))
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    def test_roa2rmid_scalar_psinorm_vector_t_each_t_False(self):
        with self.assertRaises(ValueError):
            out = e.roa2rmid(scalar_psinorm, vector_t, each_t=False)
        with self.assertRaises(ValueError):
            out_t = et.roa2rmid(scalar_psinorm, vector_t, each_t=False)

    # vector r/a, vector t:
    def test_roa2rmid_vector_psinorm_vector_t_each_t_True(self):
        out = e.roa2rmid(vector_psinorm, vector_t)
        out_t = et.roa2rmid(vector_psinorm, vector_t)

        self.assertEqual(out.shape, (len(vector_t), len(vector_psinorm)))
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, (len(vector_t), len(vector_psinorm)))
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    def test_roa2rmid_vector_psinorm_vector_t_each_t_False(self):
        with self.assertRaises(ValueError):
            out = e.roa2rmid(vector_psinorm, vector_t, each_t=False)
        with self.assertRaises(ValueError):
            out_t = et.roa2rmid(vector_psinorm, vector_t, each_t=False)

    # matrix r/a, vector t:
    def test_roa2rmid_matrix_psinorm_vector_t_each_t_True(self):
        out = e.roa2rmid(matrix_psinorm, vector_t)
        out_t = et.roa2rmid(matrix_psinorm, vector_t)

        self.assertEqual(out.shape, tuple([len(vector_t),] + list(matrix_psinorm.shape)))
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, tuple([len(vector_t),] + list(matrix_psinorm.shape)))
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    def test_roa2rmid_matrix_psinorm_vector_t_each_t_False(self):
        with self.assertRaises(ValueError):
            out = e.roa2rmid(matrix_psinorm, vector_t, each_t=False)
        with self.assertRaises(ValueError):
            out = e.roa2rmid(matrix_psinorm, vector_t, each_t=False)

    # matrix r/a, matrix t:
    def test_roa2rmid_matrix_psinorm_matrix_t_each_t_True(self):
        with self.assertRaises(ValueError):
            out = e.roa2rmid(matrix_psinorm, matrix_t)
        with self.assertRaises(ValueError):
            out = e.roa2rmid(matrix_psinorm, matrix_t)

    def test_roa2rmid_matrix_psinorm_matrix_t_each_t_False(self):
        out = e.roa2rmid(matrix_psinorm, matrix_t, each_t=False)
        out_t = et.roa2rmid(matrix_psinorm, matrix_t, each_t=False)

        self.assertEqual(out.shape, matrix_psinorm.shape)
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, matrix_psinorm.shape)
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

class roa2psinormTestCase(unittest.TestCase):
    # The following tests focus primarily on the consistency of the input/output
    # shapes. More tests should be added to check the consistency between the
    # different mapping routines.

    # scalar r/a, scalar t:
    def test_roa2psinorm_scalar_psinorm_scalar_t_each_t_True(self):
        out = e.roa2psinorm(scalar_psinorm, scalar_t)
        out_t = et.roa2psinorm(scalar_psinorm, scalar_t)

        with self.assertRaises(TypeError):
            iter(out)
        self.assertFalse(numpy.isnan(out))
        self.assertFalse(numpy.isinf(out))

        with self.assertRaises(TypeError):
            iter(out_t)
        self.assertFalse(numpy.isnan(out_t))
        self.assertFalse(numpy.isinf(out_t))

        diff = numpy.absolute(out - out_t) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    def test_roa2psinorm_scalar_psinorm_scalar_t_each_t_False(self):
        out = e.roa2psinorm(scalar_psinorm, scalar_t, each_t=False)
        out_t = et.roa2psinorm(scalar_psinorm, scalar_t, each_t=False)

        with self.assertRaises(TypeError):
            iter(out)
        self.assertFalse(numpy.isnan(out))
        self.assertFalse(numpy.isinf(out))

        with self.assertRaises(TypeError):
            iter(out_t)
        self.assertFalse(numpy.isnan(out_t))
        self.assertFalse(numpy.isinf(out_t))

        diff = numpy.absolute(out - out_t) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    # vector r/a, scalar t:
    def test_roa2psinorm_vector_psinorm_scalar_t_each_t_True(self):
        out = e.roa2psinorm(vector_psinorm, scalar_t)
        out_t = et.roa2psinorm(vector_psinorm, scalar_t)

        self.assertEqual(out.shape, (len(vector_psinorm),))
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, (len(vector_psinorm),))
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    def test_roa2psinorm_vector_psinorm_scalar_t_each_t_False(self):
        out = e.roa2psinorm(vector_psinorm, scalar_t, each_t=False)
        out_t = et.roa2psinorm(vector_psinorm, scalar_t, each_t=False)

        self.assertEqual(out.shape, (len(vector_psinorm),))
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, (len(vector_psinorm),))
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    # matrix r/a, scalar t:
    def test_roa2psinorm_matrix_psinorm_scalar_t_each_t_True(self):
        out = e.roa2psinorm(matrix_psinorm, scalar_t)
        out_t = et.roa2psinorm(matrix_psinorm, scalar_t)

        self.assertEqual(out.shape, matrix_psinorm.shape)
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, matrix_psinorm.shape)
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    def test_roa2psinorm_matrix_psinorm_scalar_t_each_t_False(self):
        out = e.roa2psinorm(matrix_psinorm, scalar_t, each_t=False)
        out_t = et.roa2psinorm(matrix_psinorm, scalar_t, each_t=False)

        self.assertEqual(out.shape, matrix_psinorm.shape)
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, matrix_psinorm.shape)
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    # scalar r/a, vector t:
    def test_roa2psinorm_scalar_psinorm_vector_t_each_t_True(self):
        out = e.roa2psinorm(scalar_psinorm, vector_t)
        out_t = et.roa2psinorm(scalar_psinorm, vector_t)

        self.assertEqual(out.shape, (len(vector_t),))
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, (len(vector_t),))
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    def test_roa2psinorm_scalar_psinorm_vector_t_each_t_False(self):
        with self.assertRaises(ValueError):
            out = e.roa2psinorm(scalar_psinorm, vector_t, each_t=False)
        with self.assertRaises(ValueError):
            out = e.roa2psinorm(scalar_psinorm, vector_t, each_t=False)

    # vector r/a, vector t:
    def test_roa2psinorm_vector_psinorm_vector_t_each_t_True(self):
        out = e.roa2psinorm(vector_psinorm, vector_t)
        out_t = et.roa2psinorm(vector_psinorm, vector_t)

        self.assertEqual(out.shape, (len(vector_t), len(vector_psinorm)))
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, (len(vector_t), len(vector_psinorm)))
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    def test_roa2psinorm_vector_psinorm_vector_t_each_t_False(self):
        with self.assertRaises(ValueError):
            out = e.roa2psinorm(vector_psinorm, vector_t, each_t=False)
        with self.assertRaises(ValueError):
            out_t = et.roa2psinorm(vector_psinorm, vector_t, each_t=False)

    # matrix r/a, vector t:
    def test_roa2psinorm_matrix_psinorm_vector_t_each_t_True(self):
        out = e.roa2psinorm(matrix_psinorm, vector_t)
        out_t = et.roa2psinorm(matrix_psinorm, vector_t)

        self.assertEqual(out.shape, tuple([len(vector_t),] + list(matrix_psinorm.shape)))
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, tuple([len(vector_t),] + list(matrix_psinorm.shape)))
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    def test_roa2psinorm_matrix_psinorm_vector_t_each_t_False(self):
        with self.assertRaises(ValueError):
            out = e.roa2psinorm(matrix_psinorm, vector_t, each_t=False)
        with self.assertRaises(ValueError):
            out_t = et.roa2psinorm(matrix_psinorm, vector_t, each_t=False)

    # matrix r/a, matrix t:
    def test_roa2psinorm_matrix_psinorm_matrix_t_each_t_True(self):
        with self.assertRaises(ValueError):
            out = e.roa2psinorm(matrix_psinorm, matrix_t)
        with self.assertRaises(ValueError):
            out_t = et.roa2psinorm(matrix_psinorm, matrix_t)

    def test_roa2psinorm_matrix_psinorm_matrix_t_each_t_False(self):
        out = e.roa2psinorm(matrix_psinorm, matrix_t, each_t=False)
        out_t = et.roa2psinorm(matrix_psinorm, matrix_t, each_t=False)

        self.assertEqual(out.shape, matrix_psinorm.shape)
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, matrix_psinorm.shape)
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

class roa2phinormTestCase(unittest.TestCase):
    # The following tests focus primarily on the consistency of the input/output
    # shapes. More tests should be added to check the consistency between the
    # different mapping routines.

    # scalar r/a, scalar t:
    def test_roa2phinorm_scalar_psinorm_scalar_t_each_t_True(self):
        out = e.roa2phinorm(scalar_psinorm, scalar_t)
        out_t = et.roa2phinorm(scalar_psinorm, scalar_t)

        with self.assertRaises(TypeError):
            iter(out)
        self.assertFalse(numpy.isnan(out))
        self.assertFalse(numpy.isinf(out))

        with self.assertRaises(TypeError):
            iter(out_t)
        self.assertFalse(numpy.isnan(out_t))
        self.assertFalse(numpy.isinf(out_t))

        diff = numpy.absolute(out - out_t) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    def test_roa2phinorm_scalar_psinorm_scalar_t_each_t_False(self):
        out = e.roa2phinorm(scalar_psinorm, scalar_t, each_t=False)
        out_t = et.roa2phinorm(scalar_psinorm, scalar_t, each_t=False)

        with self.assertRaises(TypeError):
            iter(out)
        self.assertFalse(numpy.isnan(out))
        self.assertFalse(numpy.isinf(out))

        with self.assertRaises(TypeError):
            iter(out_t)
        self.assertFalse(numpy.isnan(out_t))
        self.assertFalse(numpy.isinf(out_t))

        diff = numpy.absolute(out - out_t) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    # vector r/a, scalar t:
    def test_roa2phinorm_vector_psinorm_scalar_t_each_t_True(self):
        out = e.roa2phinorm(vector_psinorm, scalar_t)
        out_t = et.roa2phinorm(vector_psinorm, scalar_t)

        self.assertEqual(out.shape, (len(vector_psinorm),))
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, (len(vector_psinorm),))
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    def test_roa2phinorm_vector_psinorm_scalar_t_each_t_False(self):
        out = e.roa2phinorm(vector_psinorm, scalar_t, each_t=False)
        out_t = et.roa2phinorm(vector_psinorm, scalar_t, each_t=False)

        self.assertEqual(out.shape, (len(vector_psinorm),))
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, (len(vector_psinorm),))
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    # matrix r/a, scalar t:
    def test_roa2phinorm_matrix_psinorm_scalar_t_each_t_True(self):
        out = e.roa2phinorm(matrix_psinorm, scalar_t)
        out_t = et.roa2phinorm(matrix_psinorm, scalar_t)

        self.assertEqual(out.shape, matrix_psinorm.shape)
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, matrix_psinorm.shape)
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    def test_roa2phinorm_matrix_psinorm_scalar_t_each_t_False(self):
        out = e.roa2phinorm(matrix_psinorm, scalar_t, each_t=False)
        out_t = et.roa2phinorm(matrix_psinorm, scalar_t, each_t=False)

        self.assertEqual(out.shape, matrix_psinorm.shape)
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, matrix_psinorm.shape)
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    # scalar r/a, vector t:
    def test_roa2phinorm_scalar_psinorm_vector_t_each_t_True(self):
        out = e.roa2phinorm(scalar_psinorm, vector_t)
        out_t = et.roa2phinorm(scalar_psinorm, vector_t)

        self.assertEqual(out.shape, (len(vector_t),))
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, (len(vector_t),))
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    def test_roa2phinorm_scalar_psinorm_vector_t_each_t_False(self):
        with self.assertRaises(ValueError):
            out = e.roa2phinorm(scalar_psinorm, vector_t, each_t=False)
        with self.assertRaises(ValueError):
            out_t = et.roa2phinorm(scalar_psinorm, vector_t, each_t=False)

    # vector r/a, vector t:
    def test_roa2phinorm_vector_psinorm_vector_t_each_t_True(self):
        out = e.roa2phinorm(vector_psinorm, vector_t)
        out_t = et.roa2phinorm(vector_psinorm, vector_t)

        self.assertEqual(out.shape, (len(vector_t), len(vector_psinorm)))
        # TODO: Fix boundaries!
        # self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, (len(vector_t), len(vector_psinorm)))
        # TODO: Fix boundaries!
        # self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        # diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        # self.assertTrue((numpy.isnan(out) == numpy.isnan(out_t)).all())
        res2 = (out - out_t)**2
        diff = numpy.sqrt((res2[~numpy.isnan(res2)]).max()) / numpy.absolute(out[~numpy.isnan(out)]).max()
        self.assertLessEqual(diff, tol)

    def test_roa2phinorm_vector_psinorm_vector_t_each_t_False(self):
        with self.assertRaises(ValueError):
            out = e.roa2phinorm(vector_psinorm, vector_t, each_t=False)
        with self.assertRaises(ValueError):
            out_t = et.roa2phinorm(vector_psinorm, vector_t, each_t=False)

    # matrix r/a, vector t:
    def test_roa2phinorm_matrix_psinorm_vector_t_each_t_True(self):
        out = e.roa2phinorm(matrix_psinorm, vector_t)
        out_t = et.roa2phinorm(matrix_psinorm, vector_t)

        self.assertEqual(out.shape, tuple([len(vector_t),] + list(matrix_psinorm.shape)))
        # TODO: Fix boundaries!
        # self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, tuple([len(vector_t),] + list(matrix_psinorm.shape)))
        # TODO: Fix boundaries!
        # self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        # diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        # self.assertTrue((numpy.isnan(out) == numpy.isnan(out_t)).all())
        res2 = (out - out_t)**2
        diff = numpy.sqrt((res2[~numpy.isnan(res2)]).max()) / numpy.absolute(out[~numpy.isnan(out)]).max()
        self.assertLessEqual(diff, tol)

    def test_roa2phinorm_matrix_psinorm_vector_t_each_t_False(self):
        with self.assertRaises(ValueError):
            out = e.roa2phinorm(matrix_psinorm, vector_t, each_t=False)
        with self.assertRaises(ValueError):
            out_t = et.roa2phinorm(matrix_psinorm, vector_t, each_t=False)

    # matrix r/a, matrix t:
    def test_roa2phinorm_matrix_psinorm_matrix_t_each_t_True(self):
        with self.assertRaises(ValueError):
            out = e.roa2phinorm(matrix_psinorm, matrix_t)
        with self.assertRaises(ValueError):
            out_t = et.roa2phinorm(matrix_psinorm, matrix_t)

    def test_roa2phinorm_matrix_psinorm_matrix_t_each_t_False(self):
        out = e.roa2phinorm(matrix_psinorm, matrix_t, each_t=False)
        out_t = et.roa2phinorm(matrix_psinorm, matrix_t, each_t=False)

        self.assertEqual(out.shape, matrix_psinorm.shape)
        # TODO: Fix boundaries!
        # self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, matrix_psinorm.shape)
        # TODO: Fix boundaries!
        # self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        # diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        # self.assertTrue((numpy.isnan(out) == numpy.isnan(out_t)).all())
        res2 = (out - out_t)**2
        diff = numpy.sqrt((res2[~numpy.isnan(res2)]).max()) / numpy.absolute(out[~numpy.isnan(out)]).max()
        self.assertLessEqual(diff, tol)

class roa2volnormTestCase(unittest.TestCase):
    # The following tests focus primarily on the consistency of the input/output
    # shapes. More tests should be added to check the consistency between the
    # different mapping routines.

    # scalar r/a, scalar t:
    def test_roa2volnorm_scalar_psinorm_scalar_t_each_t_True(self):
        out = e.roa2volnorm(scalar_psinorm, scalar_t)
        out_t = et.roa2volnorm(scalar_psinorm, scalar_t)

        with self.assertRaises(TypeError):
            iter(out)
        self.assertFalse(numpy.isnan(out))
        self.assertFalse(numpy.isinf(out))

        with self.assertRaises(TypeError):
            iter(out_t)
        self.assertFalse(numpy.isnan(out_t))
        self.assertFalse(numpy.isinf(out_t))

        diff = numpy.absolute(out - out_t) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    def test_roa2volnorm_scalar_psinorm_scalar_t_each_t_False(self):
        out = e.roa2volnorm(scalar_psinorm, scalar_t, each_t=False)
        out_t = et.roa2volnorm(scalar_psinorm, scalar_t, each_t=False)

        with self.assertRaises(TypeError):
            iter(out)
        self.assertFalse(numpy.isnan(out))
        self.assertFalse(numpy.isinf(out))

        with self.assertRaises(TypeError):
            iter(out_t)
        self.assertFalse(numpy.isnan(out_t))
        self.assertFalse(numpy.isinf(out_t))

        diff = numpy.absolute(out - out_t) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    # vector r/a, scalar t:
    def test_roa2volnorm_vector_psinorm_scalar_t_each_t_True(self):
        out = e.roa2volnorm(vector_psinorm, scalar_t)
        out_t = et.roa2volnorm(vector_psinorm, scalar_t)

        self.assertEqual(out.shape, (len(vector_psinorm),))
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, (len(vector_psinorm),))
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    def test_roa2volnorm_vector_psinorm_scalar_t_each_t_False(self):
        out = e.roa2volnorm(vector_psinorm, scalar_t, each_t=False)
        out_t = et.roa2volnorm(vector_psinorm, scalar_t, each_t=False)

        self.assertEqual(out.shape, (len(vector_psinorm),))
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, (len(vector_psinorm),))
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    # matrix r/a, scalar t:
    def test_roa2volnorm_matrix_psinorm_scalar_t_each_t_True(self):
        out = e.roa2volnorm(matrix_psinorm, scalar_t)
        out_t = et.roa2volnorm(matrix_psinorm, scalar_t)

        self.assertEqual(out.shape, matrix_psinorm.shape)
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, matrix_psinorm.shape)
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    def test_roa2volnorm_matrix_psinorm_scalar_t_each_t_False(self):
        out = e.roa2volnorm(matrix_psinorm, scalar_t, each_t=False)
        out_t = et.roa2volnorm(matrix_psinorm, scalar_t, each_t=False)

        self.assertEqual(out.shape, matrix_psinorm.shape)
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, matrix_psinorm.shape)
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    # scalar r/a, vector t:
    def test_roa2volnorm_scalar_psinorm_vector_t_each_t_True(self):
        out = e.roa2volnorm(scalar_psinorm, vector_t)
        out_t = et.roa2volnorm(scalar_psinorm, vector_t)

        self.assertEqual(out.shape, (len(vector_t),))
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, (len(vector_t),))
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    def test_roa2volnorm_scalar_psinorm_vector_t_each_t_False(self):
        with self.assertRaises(ValueError):
            out = e.roa2volnorm(scalar_psinorm, vector_t, each_t=False)
        with self.assertRaises(ValueError):
            out_t = et.roa2volnorm(scalar_psinorm, vector_t, each_t=False)

    # vector r/a, vector t:
    def test_roa2volnorm_vector_psinorm_vector_t_each_t_True(self):
        out = e.roa2volnorm(vector_psinorm, vector_t)
        out_t = et.roa2volnorm(vector_psinorm, vector_t)

        self.assertEqual(out.shape, (len(vector_t), len(vector_psinorm)))
        # TODO: Fix boundaries!
        # self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, (len(vector_t), len(vector_psinorm)))
        # TODO: Fix boundaries!
        # self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        # diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        # self.assertTrue((numpy.isnan(out) == numpy.isnan(out_t)).all())
        res2 = (out - out_t)**2
        diff = numpy.sqrt((res2[~numpy.isnan(res2)]).max()) / numpy.absolute(out[~numpy.isnan(out)]).max()
        self.assertLessEqual(diff, tol)

    def test_roa2volnorm_vector_psinorm_vector_t_each_t_False(self):
        with self.assertRaises(ValueError):
            out = e.roa2volnorm(vector_psinorm, vector_t, each_t=False)
        with self.assertRaises(ValueError):
            out_t = et.roa2volnorm(vector_psinorm, vector_t, each_t=False)

    # matrix r/a, vector t:
    def test_roa2volnorm_matrix_psinorm_vector_t_each_t_True(self):
        out = e.roa2volnorm(matrix_psinorm, vector_t)
        out_t = et.roa2volnorm(matrix_psinorm, vector_t)

        self.assertEqual(out.shape, tuple([len(vector_t),] + list(matrix_psinorm.shape)))
        # TODO: Fix boundaries!
        # self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, tuple([len(vector_t),] + list(matrix_psinorm.shape)))
        # TODO: Fix boundaries!
        # self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        # diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        # self.assertTrue((numpy.isnan(out) == numpy.isnan(out_t)).all())
        res2 = (out - out_t)**2
        diff = numpy.sqrt((res2[~numpy.isnan(res2)]).max()) / numpy.absolute(out[~numpy.isnan(out)]).max()
        self.assertLessEqual(diff, tol)

    def test_roa2volnorm_matrix_psinorm_vector_t_each_t_False(self):
        with self.assertRaises(ValueError):
            out = e.roa2volnorm(matrix_psinorm, vector_t, each_t=False)
        with self.assertRaises(ValueError):
            out_t = et.roa2volnorm(matrix_psinorm, vector_t, each_t=False)

    # matrix r/a, matrix t:
    def test_roa2volnorm_matrix_psinorm_matrix_t_each_t_True(self):
        with self.assertRaises(ValueError):
            out = e.roa2volnorm(matrix_psinorm, matrix_t)
        with self.assertRaises(ValueError):
            out_t = et.roa2volnorm(matrix_psinorm, matrix_t)

    def test_roa2volnorm_matrix_psinorm_matrix_t_each_t_False(self):
        out = e.roa2volnorm(matrix_psinorm, matrix_t, each_t=False)
        out_t = et.roa2volnorm(matrix_psinorm, matrix_t, each_t=False)

        self.assertEqual(out.shape, matrix_psinorm.shape)
        # TODO: Fix boundaries!
        # self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, matrix_psinorm.shape)
        # TODO: Fix boundaries!
        # self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        # diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        # self.assertTrue((numpy.isnan(out) == numpy.isnan(out_t)).all())
        res2 = (out - out_t)**2
        diff = numpy.sqrt((res2[~numpy.isnan(res2)]).max()) / numpy.absolute(out[~numpy.isnan(out)]).max()
        self.assertLessEqual(diff, tol)

class psinorm2rmidTestCase(unittest.TestCase):
    # The following tests focus primarily on the consistency of the input/output
    # shapes. More tests should be added to check the consistency between the
    # different mapping routines.

    # scalar psinorm, scalar t:
    def test_psinorm2rmid_scalar_psinorm_scalar_t_each_t_True_rho_False(self):
        out = e.psinorm2rmid(scalar_psinorm, scalar_t)
        out_t = et.psinorm2rmid(scalar_psinorm, scalar_t)

        with self.assertRaises(TypeError):
            iter(out)
        self.assertFalse(numpy.isnan(out))
        self.assertFalse(numpy.isinf(out))

        with self.assertRaises(TypeError):
            iter(out_t)
        self.assertFalse(numpy.isnan(out_t))
        self.assertFalse(numpy.isinf(out_t))

        diff = numpy.absolute(out - out_t) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    def test_psinorm2rmid_scalar_psinorm_scalar_t_each_t_False_rho_False(self):
        out = e.psinorm2rmid(scalar_psinorm, scalar_t, each_t=False)
        out_t = et.psinorm2rmid(scalar_psinorm, scalar_t, each_t=False)

        with self.assertRaises(TypeError):
            iter(out)
        self.assertFalse(numpy.isnan(out))
        self.assertFalse(numpy.isinf(out))

        with self.assertRaises(TypeError):
            iter(out_t)
        self.assertFalse(numpy.isnan(out_t))
        self.assertFalse(numpy.isinf(out_t))

        diff = numpy.absolute(out - out_t) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    def test_psinorm2rmid_scalar_psinorm_scalar_t_each_t_True_rho_True(self):
        out = e.psinorm2rmid(scalar_psinorm, scalar_t, rho=True)
        out_t = et.psinorm2rmid(scalar_psinorm, scalar_t, rho=True)

        with self.assertRaises(TypeError):
            iter(out)
        self.assertFalse(numpy.isnan(out))
        self.assertFalse(numpy.isinf(out))

        with self.assertRaises(TypeError):
            iter(out_t)
        self.assertFalse(numpy.isnan(out_t))
        self.assertFalse(numpy.isinf(out_t))

        diff = numpy.absolute(out - out_t) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    def test_psinorm2rmid_scalar_psinorm_scalar_t_each_t_False_rho_True(self):
        out = e.psinorm2rmid(scalar_psinorm, scalar_t, each_t=False, rho=True)
        out_t = et.psinorm2rmid(scalar_psinorm, scalar_t, each_t=False, rho=True)

        with self.assertRaises(TypeError):
            iter(out)
        self.assertFalse(numpy.isnan(out))
        self.assertFalse(numpy.isinf(out))

        with self.assertRaises(TypeError):
            iter(out_t)
        self.assertFalse(numpy.isnan(out_t))
        self.assertFalse(numpy.isinf(out_t))

        diff = numpy.absolute(out - out_t) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    # vector psinorm, scalar t:
    def test_psinorm2rmid_vector_psinorm_scalar_t_each_t_True_rho_False(self):
        out = e.psinorm2rmid(vector_psinorm, scalar_t)
        out_t = et.psinorm2rmid(vector_psinorm, scalar_t)

        self.assertEqual(out.shape, (len(vector_psinorm),))
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, (len(vector_psinorm),))
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        # diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertTrue((numpy.isnan(out) == numpy.isnan(out_t)).all())
        res2 = (out - out_t)**2
        diff = numpy.sqrt((res2[~numpy.isnan(res2)]).max()) / numpy.absolute(out[~numpy.isnan(out)]).max()
        self.assertLessEqual(diff, tol)

    def test_psinorm2rmid_vector_psinorm_scalar_t_each_t_False_rho_False(self):
        out = e.psinorm2rmid(vector_psinorm, scalar_t, each_t=False)
        out_t = et.psinorm2rmid(vector_psinorm, scalar_t, each_t=False)

        self.assertEqual(out.shape, (len(vector_psinorm),))
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, (len(vector_psinorm),))
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        # diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertTrue((numpy.isnan(out) == numpy.isnan(out_t)).all())
        res2 = (out - out_t)**2
        diff = numpy.sqrt((res2[~numpy.isnan(res2)]).max()) / numpy.absolute(out[~numpy.isnan(out)]).max()
        self.assertLessEqual(diff, tol)

    def test_psinorm2rmid_vector_psinorm_scalar_t_each_t_True_rho_True(self):
        out = e.psinorm2rmid(vector_psinorm, scalar_t, rho=True)
        out_t = et.psinorm2rmid(vector_psinorm, scalar_t, rho=True)

        self.assertEqual(out.shape, (len(vector_psinorm),))
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, (len(vector_psinorm),))
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        # diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertTrue((numpy.isnan(out) == numpy.isnan(out_t)).all())
        res2 = (out - out_t)**2
        diff = numpy.sqrt((res2[~numpy.isnan(res2)]).max()) / numpy.absolute(out[~numpy.isnan(out)]).max()
        self.assertLessEqual(diff, tol)

    def test_psinorm2rmid_vector_psinorm_scalar_t_each_t_False_rho_True(self):
        out = e.psinorm2rmid(vector_psinorm, scalar_t, each_t=False, rho=True)
        out_t = et.psinorm2rmid(vector_psinorm, scalar_t, each_t=False, rho=True)

        self.assertEqual(out.shape, (len(vector_psinorm),))
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, (len(vector_psinorm),))
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        # diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertTrue((numpy.isnan(out) == numpy.isnan(out_t)).all())
        res2 = (out - out_t)**2
        diff = numpy.sqrt((res2[~numpy.isnan(res2)]).max()) / numpy.absolute(out[~numpy.isnan(out)]).max()
        self.assertLessEqual(diff, tol)

    # matrix psinorm, scalar t:
    def test_psinorm2rmid_matrix_psinorm_scalar_t_each_t_True_rho_False(self):
        out = e.psinorm2rmid(matrix_psinorm, scalar_t)
        out_t = et.psinorm2rmid(matrix_psinorm, scalar_t)

        self.assertEqual(out.shape, matrix_psinorm.shape)
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, matrix_psinorm.shape)
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        # diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertTrue((numpy.isnan(out) == numpy.isnan(out_t)).all())
        res2 = (out - out_t)**2
        diff = numpy.sqrt((res2[~numpy.isnan(res2)]).max()) / numpy.absolute(out[~numpy.isnan(out)]).max()
        self.assertLessEqual(diff, tol)

    def test_psinorm2rmid_matrix_psinorm_scalar_t_each_t_False_rho_False(self):
        out = e.psinorm2rmid(matrix_psinorm, scalar_t, each_t=False)
        out_t = et.psinorm2rmid(matrix_psinorm, scalar_t, each_t=False)

        self.assertEqual(out.shape, matrix_psinorm.shape)
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, matrix_psinorm.shape)
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        # diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertTrue((numpy.isnan(out) == numpy.isnan(out_t)).all())
        res2 = (out - out_t)**2
        diff = numpy.sqrt((res2[~numpy.isnan(res2)]).max()) / numpy.absolute(out[~numpy.isnan(out)]).max()
        self.assertLessEqual(diff, tol)

    def test_psinorm2rmid_matrix_psinorm_scalar_t_each_t_True_rho_True(self):
        out = e.psinorm2rmid(matrix_psinorm, scalar_t, rho=True)
        out_t = et.psinorm2rmid(matrix_psinorm, scalar_t, rho=True)

        self.assertEqual(out.shape, matrix_psinorm.shape)
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, matrix_psinorm.shape)
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        # diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertTrue((numpy.isnan(out) == numpy.isnan(out_t)).all())
        res2 = (out - out_t)**2
        diff = numpy.sqrt((res2[~numpy.isnan(res2)]).max()) / numpy.absolute(out[~numpy.isnan(out)]).max()
        self.assertLessEqual(diff, tol)

    def test_psinorm2rmid_matrix_psinorm_scalar_t_each_t_False_rho_True(self):
        out = e.psinorm2rmid(matrix_psinorm, scalar_t, each_t=False, rho=True)
        out_t = et.psinorm2rmid(matrix_psinorm, scalar_t, each_t=False, rho=True)

        self.assertEqual(out.shape, matrix_psinorm.shape)
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, matrix_psinorm.shape)
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        # diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertTrue((numpy.isnan(out) == numpy.isnan(out_t)).all())
        res2 = (out - out_t)**2
        diff = numpy.sqrt((res2[~numpy.isnan(res2)]).max()) / numpy.absolute(out[~numpy.isnan(out)]).max()
        self.assertLessEqual(diff, tol)

    # scalar psinorm, vector t:
    def test_psinorm2rmid_scalar_psinorm_vector_t_each_t_True_rho_False(self):
        out = e.psinorm2rmid(scalar_psinorm, vector_t)
        out_t = et.psinorm2rmid(scalar_psinorm, vector_t)

        self.assertEqual(out.shape, (len(vector_t),))
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, (len(vector_t),))
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    def test_psinorm2rmid_scalar_psinorm_vector_t_each_t_False_rho_False(self):
        with self.assertRaises(ValueError):
            out = e.psinorm2rmid(scalar_psinorm, vector_t, each_t=False)
        with self.assertRaises(ValueError):
            out_t = et.psinorm2rmid(scalar_psinorm, vector_t, each_t=False)

    def test_psinorm2rmid_scalar_psinorm_vector_t_each_t_True_rho_True(self):
        out = e.psinorm2rmid(scalar_psinorm, vector_t, rho=True)
        out_t = et.psinorm2rmid(scalar_psinorm, vector_t, rho=True)

        self.assertEqual(out.shape, (len(vector_t),))
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, (len(vector_t),))
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    def test_psinorm2rmid_scalar_psinorm_vector_t_each_t_False_rho_True(self):
        with self.assertRaises(ValueError):
            out = e.psinorm2rmid(scalar_psinorm, vector_t, each_t=False, rho=True)
        with self.assertRaises(ValueError):
            out_t = et.psinorm2rmid(scalar_psinorm, vector_t, each_t=False, rho=True)

    # vector psinorm, vector t:
    def test_psinorm2rmid_vector_psinorm_vector_t_each_t_True_rho_False(self):
        out = e.psinorm2rmid(vector_psinorm, vector_t)
        out_t = et.psinorm2rmid(vector_psinorm, vector_t)

        self.assertEqual(out.shape, (len(vector_t), len(vector_psinorm)))
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, (len(vector_t), len(vector_psinorm)))
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        # diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertTrue((numpy.isnan(out) == numpy.isnan(out_t)).all())
        res2 = (out - out_t)**2
        diff = numpy.sqrt((res2[~numpy.isnan(res2)]).max()) / numpy.absolute(out[~numpy.isnan(out)]).max()
        self.assertLessEqual(diff, tol)

    def test_psinorm2rmid_vector_psinorm_vector_t_each_t_False_rho_False(self):
        with self.assertRaises(ValueError):
            out = e.psinorm2rmid(vector_psinorm, vector_t, each_t=False)
        with self.assertRaises(ValueError):
            out_t = et.psinorm2rmid(vector_psinorm, vector_t, each_t=False)

    def test_psinorm2rmid_vector_psinorm_vector_t_each_t_True_rho_True(self):
        out = e.psinorm2rmid(vector_psinorm, vector_t, rho=True)
        out_t = et.psinorm2rmid(vector_psinorm, vector_t, rho=True)

        self.assertEqual(out.shape, (len(vector_t), len(vector_psinorm)))
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, (len(vector_t), len(vector_psinorm)))
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        # diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertTrue((numpy.isnan(out) == numpy.isnan(out_t)).all())
        res2 = (out - out_t)**2
        diff = numpy.sqrt((res2[~numpy.isnan(res2)]).max()) / numpy.absolute(out[~numpy.isnan(out)]).max()
        self.assertLessEqual(diff, tol)

    def test_psinorm2rmid_vector_psinorm_vector_t_each_t_False_rho_True(self):
        with self.assertRaises(ValueError):
            out = e.psinorm2rmid(vector_psinorm, vector_t, each_t=False, rho=True)
        with self.assertRaises(ValueError):
            out_t = et.psinorm2rmid(vector_psinorm, vector_t, each_t=False, rho=True)

    # matrix psinorm, vector t:
    def test_psinorm2rmid_matrix_psinorm_vector_t_each_t_True_rho_False(self):
        out = e.psinorm2rmid(matrix_psinorm, vector_t)
        out_t = et.psinorm2rmid(matrix_psinorm, vector_t)

        self.assertEqual(out.shape, tuple([len(vector_t),] + list(matrix_psinorm.shape)))
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, tuple([len(vector_t),] + list(matrix_psinorm.shape)))
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        # diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertTrue((numpy.isnan(out) == numpy.isnan(out_t)).all())
        res2 = (out - out_t)**2
        diff = numpy.sqrt((res2[~numpy.isnan(res2)]).max()) / numpy.absolute(out[~numpy.isnan(out)]).max()
        self.assertLessEqual(diff, tol)

    def test_psinorm2rmid_matrix_psinorm_vector_t_each_t_False_rho_False(self):
        with self.assertRaises(ValueError):
            out = e.psinorm2rmid(matrix_psinorm, vector_t, each_t=False)
        with self.assertRaises(ValueError):
            out_t = et.psinorm2rmid(matrix_psinorm, vector_t, each_t=False)

    def test_psinorm2rmid_matrix_psinorm_vector_t_each_t_True_rho_True(self):
        out = e.psinorm2rmid(matrix_psinorm, vector_t, rho=True)
        out_t = et.psinorm2rmid(matrix_psinorm, vector_t, rho=True)

        self.assertEqual(out.shape, tuple([len(vector_t),] + list(matrix_psinorm.shape)))
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, tuple([len(vector_t),] + list(matrix_psinorm.shape)))
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        # diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertTrue((numpy.isnan(out) == numpy.isnan(out_t)).all())
        res2 = (out - out_t)**2
        diff = numpy.sqrt((res2[~numpy.isnan(res2)]).max()) / numpy.absolute(out[~numpy.isnan(out)]).max()
        self.assertLessEqual(diff, tol)

    def test_psinorm2rmid_matrix_psinorm_vector_t_each_t_False_rho_True(self):
        with self.assertRaises(ValueError):
            out = e.psinorm2rmid(matrix_psinorm, vector_t, each_t=False, rho=True)
        with self.assertRaises(ValueError):
            out_t = et.psinorm2rmid(matrix_psinorm, vector_t, each_t=False, rho=True)

    # matrix psinorm, matrix t:
    def test_psinorm2rmid_matrix_psinorm_matrix_t_each_t_True_rho_False(self):
        with self.assertRaises(ValueError):
            out = e.psinorm2rmid(matrix_psinorm, matrix_t)
        with self.assertRaises(ValueError):
            out_t = et.psinorm2rmid(matrix_psinorm, matrix_t)

    def test_psinorm2rmid_matrix_psinorm_matrix_t_each_t_False_rho_False(self):
        out = e.psinorm2rmid(matrix_psinorm, matrix_t, each_t=False)
        out_t = et.psinorm2rmid(matrix_psinorm, matrix_t, each_t=False)

        self.assertEqual(out.shape, matrix_psinorm.shape)
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, matrix_psinorm.shape)
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        # diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertTrue((numpy.isnan(out) == numpy.isnan(out_t)).all())
        res2 = (out - out_t)**2
        diff = numpy.sqrt((res2[~numpy.isnan(res2)]).max()) / numpy.absolute(out[~numpy.isnan(out)]).max()
        self.assertLessEqual(diff, tol)

    def test_psinorm2rmid_matrix_psinorm_matrix_t_each_t_True_rho_True(self):
        with self.assertRaises(ValueError):
            out = e.psinorm2rmid(matrix_psinorm, matrix_t, rho=True)
        with self.assertRaises(ValueError):
            out_t = et.psinorm2rmid(matrix_psinorm, matrix_t, rho=True)

    def test_psinorm2rmid_matrix_psinorm_matrix_t_each_t_False_rho_True(self):
        out = e.psinorm2rmid(matrix_psinorm, matrix_t, each_t=False, rho=True)
        out_t = et.psinorm2rmid(matrix_psinorm, matrix_t, each_t=False, rho=True)

        self.assertEqual(out.shape, matrix_psinorm.shape)
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, matrix_psinorm.shape)
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        # diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertTrue((numpy.isnan(out) == numpy.isnan(out_t)).all())
        res2 = (out - out_t)**2
        diff = numpy.sqrt((res2[~numpy.isnan(res2)]).max()) / numpy.absolute(out[~numpy.isnan(out)]).max()
        self.assertLessEqual(diff, tol)

class psinorm2roaTestCase(unittest.TestCase):
    # The following tests focus primarily on the consistency of the input/output
    # shapes. More tests should be added to check the consistency between the
    # different mapping routines.

    # scalar psinorm, scalar t:
    def test_psinorm2roa_scalar_psinorm_scalar_t_each_t_True(self):
        out = e.psinorm2roa(scalar_psinorm, scalar_t)
        out_t = et.psinorm2roa(scalar_psinorm, scalar_t)

        with self.assertRaises(TypeError):
            iter(out)
        self.assertFalse(numpy.isnan(out))
        self.assertFalse(numpy.isinf(out))

        with self.assertRaises(TypeError):
            iter(out_t)
        self.assertFalse(numpy.isnan(out_t))
        self.assertFalse(numpy.isinf(out_t))

        diff = numpy.absolute(out - out_t) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    def test_psinorm2roa_scalar_psinorm_scalar_t_each_t_False(self):
        out = e.psinorm2roa(scalar_psinorm, scalar_t, each_t=False)
        out_t = et.psinorm2roa(scalar_psinorm, scalar_t, each_t=False)

        with self.assertRaises(TypeError):
            iter(out)
        self.assertFalse(numpy.isnan(out))
        self.assertFalse(numpy.isinf(out))

        with self.assertRaises(TypeError):
            iter(out_t)
        self.assertFalse(numpy.isnan(out_t))
        self.assertFalse(numpy.isinf(out_t))

        diff = numpy.absolute(out - out_t) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    # vector psinorm, scalar t:
    def test_psinorm2roa_vector_psinorm_scalar_t_each_t_True(self):
        out = e.psinorm2roa(vector_psinorm, scalar_t)
        out_t = et.psinorm2roa(vector_psinorm, scalar_t)

        self.assertEqual(out.shape, (len(vector_psinorm),))
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, (len(vector_psinorm),))
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        # diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertTrue((numpy.isnan(out) == numpy.isnan(out_t)).all())
        res2 = (out - out_t)**2
        diff = numpy.sqrt((res2[~numpy.isnan(res2)]).max()) / numpy.absolute(out[~numpy.isnan(out)]).max()
        self.assertLessEqual(diff, tol)

    def test_psinorm2roa_vector_psinorm_scalar_t_each_t_False(self):
        out = e.psinorm2roa(vector_psinorm, scalar_t, each_t=False)
        out_t = et.psinorm2roa(vector_psinorm, scalar_t, each_t=False)

        self.assertEqual(out.shape, (len(vector_psinorm),))
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, (len(vector_psinorm),))
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        # diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertTrue((numpy.isnan(out) == numpy.isnan(out_t)).all())
        res2 = (out - out_t)**2
        diff = numpy.sqrt((res2[~numpy.isnan(res2)]).max()) / numpy.absolute(out[~numpy.isnan(out)]).max()
        self.assertLessEqual(diff, tol)

    # matrix psinorm, scalar t:
    def test_psinorm2roa_matrix_psinorm_scalar_t_each_t_True(self):
        out = e.psinorm2roa(matrix_psinorm, scalar_t)
        out_t = et.psinorm2roa(matrix_psinorm, scalar_t)

        self.assertEqual(out.shape, matrix_psinorm.shape)
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, matrix_psinorm.shape)
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        # diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertTrue((numpy.isnan(out) == numpy.isnan(out_t)).all())
        res2 = (out - out_t)**2
        diff = numpy.sqrt((res2[~numpy.isnan(res2)]).max()) / numpy.absolute(out[~numpy.isnan(out)]).max()
        self.assertLessEqual(diff, tol)

    def test_psinorm2roa_matrix_psinorm_scalar_t_each_t_False(self):
        out = e.psinorm2roa(matrix_psinorm, scalar_t, each_t=False)
        out_t = et.psinorm2roa(matrix_psinorm, scalar_t, each_t=False)

        self.assertEqual(out.shape, matrix_psinorm.shape)
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, matrix_psinorm.shape)
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        # diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertTrue((numpy.isnan(out) == numpy.isnan(out_t)).all())
        res2 = (out - out_t)**2
        diff = numpy.sqrt((res2[~numpy.isnan(res2)]).max()) / numpy.absolute(out[~numpy.isnan(out)]).max()
        self.assertLessEqual(diff, tol)

    # scalar psinorm, vector t:
    def test_psinorm2roa_scalar_psinorm_vector_t_each_t_True(self):
        out = e.psinorm2roa(scalar_psinorm, vector_t)
        out_t = et.psinorm2roa(scalar_psinorm, vector_t)

        self.assertEqual(out.shape, (len(vector_t),))
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, (len(vector_t),))
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    def test_psinorm2roa_scalar_psinorm_vector_t_each_t_False(self):
        with self.assertRaises(ValueError):
            out = e.psinorm2roa(scalar_psinorm, vector_t, each_t=False)
        with self.assertRaises(ValueError):
            out_t = et.psinorm2roa(scalar_psinorm, vector_t, each_t=False)

    # vector psinorm, vector t:
    def test_psinorm2roa_vector_psinorm_vector_t_each_t_True(self):
        out = e.psinorm2roa(vector_psinorm, vector_t)
        out_t = et.psinorm2roa(vector_psinorm, vector_t)

        self.assertEqual(out.shape, (len(vector_t), len(vector_psinorm)))
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, (len(vector_t), len(vector_psinorm)))
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        # diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertTrue((numpy.isnan(out) == numpy.isnan(out_t)).all())
        res2 = (out - out_t)**2
        diff = numpy.sqrt((res2[~numpy.isnan(res2)]).max()) / numpy.absolute(out[~numpy.isnan(out)]).max()
        self.assertLessEqual(diff, tol)

    def test_psinorm2roa_vector_psinorm_vector_t_each_t_False(self):
        with self.assertRaises(ValueError):
            out = e.psinorm2roa(vector_psinorm, vector_t, each_t=False)
        with self.assertRaises(ValueError):
            out_t = et.psinorm2roa(vector_psinorm, vector_t, each_t=False)

    # matrix psinorm, vector t:
    def test_psinorm2roa_matrix_psinorm_vector_t_each_t_True(self):
        out = e.psinorm2roa(matrix_psinorm, vector_t)
        out_t = et.psinorm2roa(matrix_psinorm, vector_t)

        self.assertEqual(out.shape, tuple([len(vector_t),] + list(matrix_psinorm.shape)))
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, tuple([len(vector_t),] + list(matrix_psinorm.shape)))
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        # diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertTrue((numpy.isnan(out) == numpy.isnan(out_t)).all())
        res2 = (out - out_t)**2
        diff = numpy.sqrt((res2[~numpy.isnan(res2)]).max()) / numpy.absolute(out[~numpy.isnan(out)]).max()
        self.assertLessEqual(diff, tol)

    def test_psinorm2roa_matrix_psinorm_vector_t_each_t_False(self):
        with self.assertRaises(ValueError):
            out = e.psinorm2roa(matrix_psinorm, vector_t, each_t=False)
        with self.assertRaises(ValueError):
            out_t = et.psinorm2roa(matrix_psinorm, vector_t, each_t=False)

    # matrix psinorm, matrix t:
    def test_psinorm2roa_matrix_psinorm_matrix_t_each_t_True(self):
        with self.assertRaises(ValueError):
            out = e.psinorm2roa(matrix_psinorm, matrix_t)
        with self.assertRaises(ValueError):
            out_t = et.psinorm2roa(matrix_psinorm, matrix_t)

    def test_psinorm2roa_matrix_psinorm_matrix_t_each_t_False(self):
        out = e.psinorm2roa(matrix_psinorm, matrix_t, each_t=False)
        out_t = et.psinorm2roa(matrix_psinorm, matrix_t, each_t=False)

        self.assertEqual(out.shape, matrix_psinorm.shape)
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, matrix_psinorm.shape)
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        # diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertTrue((numpy.isnan(out) == numpy.isnan(out_t)).all())
        res2 = (out - out_t)**2
        diff = numpy.sqrt((res2[~numpy.isnan(res2)]).max()) / numpy.absolute(out[~numpy.isnan(out)]).max()
        self.assertLessEqual(diff, tol)

class psinorm2volnormTestCase(unittest.TestCase):
    # The following tests focus primarily on the consistency of the input/output
    # shapes. More tests should be added to check the consistency between the
    # different mapping routines.

    # scalar psinorm, scalar t:
    def test_psinorm2volnorm_scalar_psinorm_scalar_t_each_t_True(self):
        out = e.psinorm2volnorm(scalar_psinorm, scalar_t)
        out_t = et.psinorm2volnorm(scalar_psinorm, scalar_t)

        with self.assertRaises(TypeError):
            iter(out)
        self.assertFalse(numpy.isnan(out))
        self.assertFalse(numpy.isinf(out))

        with self.assertRaises(TypeError):
            iter(out_t)
        self.assertFalse(numpy.isnan(out_t))
        self.assertFalse(numpy.isinf(out_t))

        diff = numpy.absolute(out - out_t) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    def test_psinorm2volnorm_scalar_psinorm_scalar_t_each_t_False(self):
        out = e.psinorm2volnorm(scalar_psinorm, scalar_t, each_t=False)
        out_t = et.psinorm2volnorm(scalar_psinorm, scalar_t, each_t=False)

        with self.assertRaises(TypeError):
            iter(out)
        self.assertFalse(numpy.isnan(out))
        self.assertFalse(numpy.isinf(out))

        with self.assertRaises(TypeError):
            iter(out_t)
        self.assertFalse(numpy.isnan(out_t))
        self.assertFalse(numpy.isinf(out_t))

        diff = numpy.absolute(out - out_t) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    # vector psinorm, scalar t:
    def test_psinorm2volnorm_vector_psinorm_scalar_t_each_t_True(self):
        out = e.psinorm2volnorm(vector_psinorm, scalar_t)
        out_t = et.psinorm2volnorm(vector_psinorm, scalar_t)

        self.assertEqual(out.shape, (len(vector_psinorm),))
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out.shape, (len(vector_psinorm),))
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    def test_psinorm2volnorm_vector_psinorm_scalar_t_each_t_False(self):
        out = e.psinorm2volnorm(vector_psinorm, scalar_t, each_t=False)
        out_t = et.psinorm2volnorm(vector_psinorm, scalar_t, each_t=False)

        self.assertEqual(out.shape, (len(vector_psinorm),))
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, (len(vector_psinorm),))
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    # matrix psinorm, scalar t:
    def test_psinorm2volnorm_matrix_psinorm_scalar_t_each_t_True(self):
        out = e.psinorm2volnorm(matrix_psinorm, scalar_t)
        out_t = et.psinorm2volnorm(matrix_psinorm, scalar_t)

        self.assertEqual(out.shape, matrix_psinorm.shape)
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, matrix_psinorm.shape)
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    def test_psinorm2volnorm_matrix_psinorm_scalar_t_each_t_False(self):
        out = e.psinorm2volnorm(matrix_psinorm, scalar_t, each_t=False)
        out_t = et.psinorm2volnorm(matrix_psinorm, scalar_t, each_t=False)

        self.assertEqual(out.shape, matrix_psinorm.shape)
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, matrix_psinorm.shape)
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    # scalar psinorm, vector t:
    def test_psinorm2volnorm_scalar_psinorm_vector_t_each_t_True(self):
        out = e.psinorm2volnorm(scalar_psinorm, vector_t)
        out_t = et.psinorm2volnorm(scalar_psinorm, vector_t)

        self.assertEqual(out.shape, (len(vector_t),))
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, (len(vector_t),))
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    def test_psinorm2volnorm_scalar_psinorm_vector_t_each_t_False(self):
        with self.assertRaises(ValueError):
            out = e.psinorm2volnorm(scalar_psinorm, vector_t, each_t=False)
        with self.assertRaises(ValueError):
            out_t = et.psinorm2volnorm(scalar_psinorm, vector_t, each_t=False)

    # vector psinorm, vector t:
    def test_psinorm2volnorm_vector_psinorm_vector_t_each_t_True(self):
        out = e.psinorm2volnorm(vector_psinorm, vector_t)
        out_t = et.psinorm2volnorm(vector_psinorm, vector_t)

        self.assertEqual(out.shape, (len(vector_t), len(vector_psinorm)))
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, (len(vector_t), len(vector_psinorm)))
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    def test_psinorm2volnorm_vector_psinorm_vector_t_each_t_False(self):
        with self.assertRaises(ValueError):
            out = e.psinorm2volnorm(vector_psinorm, vector_t, each_t=False)
        with self.assertRaises(ValueError):
            out_t = et.psinorm2volnorm(vector_psinorm, vector_t, each_t=False)

    # matrix psinorm, vector t:
    def test_psinorm2volnorm_matrix_psinorm_vector_t_each_t_True(self):
        out = e.psinorm2volnorm(matrix_psinorm, vector_t)
        out_t = et.psinorm2volnorm(matrix_psinorm, vector_t)

        self.assertEqual(out.shape, tuple([len(vector_t),] + list(matrix_psinorm.shape)))
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, tuple([len(vector_t),] + list(matrix_psinorm.shape)))
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    def test_psinorm2volnorm_matrix_psinorm_vector_t_each_t_False(self):
        with self.assertRaises(ValueError):
            out = e.psinorm2volnorm(matrix_psinorm, vector_t, each_t=False)
        with self.assertRaises(ValueError):
            out_t = et.psinorm2volnorm(matrix_psinorm, vector_t, each_t=False)

    # matrix psinorm, matrix t:
    def test_psinorm2volnorm_matrix_psinorm_matrix_t_each_t_True(self):
        with self.assertRaises(ValueError):
            out = e.psinorm2volnorm(matrix_psinorm, matrix_t)
        with self.assertRaises(ValueError):
            out_t = et.psinorm2volnorm(matrix_psinorm, matrix_t)

    def test_psinorm2volnorm_matrix_psinorm_matrix_t_each_t_False(self):
        out = e.psinorm2volnorm(matrix_psinorm, matrix_t, each_t=False)
        out_t = et.psinorm2volnorm(matrix_psinorm, matrix_t, each_t=False)

        self.assertEqual(out.shape, matrix_psinorm.shape)
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, matrix_psinorm.shape)
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

class psinorm2phinormTestCase(unittest.TestCase):
    # The following tests focus primarily on the consistency of the input/output
    # shapes. More tests should be added to check the consistency between the
    # different mapping routines.

    # scalar psinorm, scalar t:
    def test_psinorm2phinorm_scalar_psinorm_scalar_t_each_t_True(self):
        out = e.psinorm2phinorm(scalar_psinorm, scalar_t)
        out_t = et.psinorm2phinorm(scalar_psinorm, scalar_t)

        with self.assertRaises(TypeError):
            iter(out)
        self.assertFalse(numpy.isnan(out))
        self.assertFalse(numpy.isinf(out))

        with self.assertRaises(TypeError):
            iter(out_t)
        self.assertFalse(numpy.isnan(out_t))
        self.assertFalse(numpy.isinf(out_t))

        diff = numpy.absolute(out - out_t) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    def test_psinorm2phinorm_scalar_psinorm_scalar_t_each_t_False(self):
        out = e.psinorm2phinorm(scalar_psinorm, scalar_t, each_t=False)
        out_t = et.psinorm2phinorm(scalar_psinorm, scalar_t, each_t=False)

        with self.assertRaises(TypeError):
            iter(out)
        self.assertFalse(numpy.isnan(out))
        self.assertFalse(numpy.isinf(out))

        with self.assertRaises(TypeError):
            iter(out_t)
        self.assertFalse(numpy.isnan(out_t))
        self.assertFalse(numpy.isinf(out_t))

        diff = numpy.absolute(out - out_t) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    # vector psinorm, scalar t:
    def test_psinorm2phinorm_vector_psinorm_scalar_t_each_t_True(self):
        out = e.psinorm2phinorm(vector_psinorm, scalar_t)
        out_t = e.psinorm2phinorm(vector_psinorm, scalar_t)

        self.assertEqual(out.shape, (len(vector_psinorm),))
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, (len(vector_psinorm),))
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    def test_psinorm2phinorm_vector_psinorm_scalar_t_each_t_False(self):
        out = e.psinorm2phinorm(vector_psinorm, scalar_t, each_t=False)
        out_t = et.psinorm2phinorm(vector_psinorm, scalar_t, each_t=False)

        self.assertEqual(out.shape, (len(vector_psinorm),))
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, (len(vector_psinorm),))
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    # matrix psinorm, scalar t:
    def test_psinorm2phinorm_matrix_psinorm_scalar_t_each_t_True(self):
        out = e.psinorm2phinorm(matrix_psinorm, scalar_t)
        out_t = et.psinorm2phinorm(matrix_psinorm, scalar_t)

        self.assertEqual(out.shape, matrix_psinorm.shape)
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, matrix_psinorm.shape)
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    def test_psinorm2phinorm_matrix_psinorm_scalar_t_each_t_False(self):
        out = e.psinorm2phinorm(matrix_psinorm, scalar_t, each_t=False)
        out_t = et.psinorm2phinorm(matrix_psinorm, scalar_t, each_t=False)

        self.assertEqual(out.shape, matrix_psinorm.shape)
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, matrix_psinorm.shape)
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    # scalar psinorm, vector t:
    def test_psinorm2phinorm_scalar_psinorm_vector_t_each_t_True(self):
        out = e.psinorm2phinorm(scalar_psinorm, vector_t)
        out_t = et.psinorm2phinorm(scalar_psinorm, vector_t)

        self.assertEqual(out.shape, (len(vector_t),))
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, (len(vector_t),))
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    def test_psinorm2phinorm_scalar_psinorm_vector_t_each_t_False(self):
        with self.assertRaises(ValueError):
            out = e.psinorm2phinorm(scalar_psinorm, vector_t, each_t=False)
        with self.assertRaises(ValueError):
            out_t = et.psinorm2phinorm(scalar_psinorm, vector_t, each_t=False)

    # vector psinorm, vector t:
    def test_psinorm2phinorm_vector_psinorm_vector_t_each_t_True(self):
        out = e.psinorm2phinorm(vector_psinorm, vector_t)
        out_t = et.psinorm2phinorm(vector_psinorm, vector_t)

        self.assertEqual(out.shape, (len(vector_t), len(vector_psinorm)))
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, (len(vector_t), len(vector_psinorm)))
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    def test_psinorm2phinorm_vector_psinorm_vector_t_each_t_False(self):
        with self.assertRaises(ValueError):
            out = e.psinorm2phinorm(vector_psinorm, vector_t, each_t=False)
        with self.assertRaises(ValueError):
            out_t = et.psinorm2phinorm(vector_psinorm, vector_t, each_t=False)

    # matrix psinorm, vector t:
    def test_psinorm2phinorm_matrix_psinorm_vector_t_each_t_True(self):
        out = e.psinorm2phinorm(matrix_psinorm, vector_t)
        out_t = et.psinorm2phinorm(matrix_psinorm, vector_t)

        self.assertEqual(out.shape, tuple([len(vector_t),] + list(matrix_psinorm.shape)))
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, tuple([len(vector_t),] + list(matrix_psinorm.shape)))
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    def test_psinorm2phinorm_matrix_psinorm_vector_t_each_t_False(self):
        with self.assertRaises(ValueError):
            out = e.psinorm2phinorm(matrix_psinorm, vector_t, each_t=False)
        with self.assertRaises(ValueError):
            out_t = et.psinorm2phinorm(matrix_psinorm, vector_t, each_t=False)

    # matrix psinorm, matrix t:
    def test_psinorm2phinorm_matrix_psinorm_matrix_t_each_t_True(self):
        with self.assertRaises(ValueError):
            out = e.psinorm2phinorm(matrix_psinorm, matrix_t)
        with self.assertRaises(ValueError):
            out_t = et.psinorm2phinorm(matrix_psinorm, matrix_t)

    def test_psinorm2phinorm_matrix_psinorm_matrix_t_each_t_False(self):
        out = e.psinorm2phinorm(matrix_psinorm, matrix_t, each_t=False)
        out_t = et.psinorm2phinorm(matrix_psinorm, matrix_t, each_t=False)

        self.assertEqual(out.shape, matrix_psinorm.shape)
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, matrix_psinorm.shape)
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

class phinorm2psinormTestCase(unittest.TestCase):
    # The following tests focus primarily on the consistency of the input/output
    # shapes. More tests should be added to check the consistency between the
    # different mapping routines.

    # scalar psinorm, scalar t:
    def test_phinorm2psinorm_scalar_psinorm_scalar_t_each_t_True(self):
        out = e.phinorm2psinorm(scalar_psinorm, scalar_t)
        out_t = et.phinorm2psinorm(scalar_psinorm, scalar_t)

        with self.assertRaises(TypeError):
            iter(out)
        self.assertFalse(numpy.isnan(out))
        self.assertFalse(numpy.isinf(out))

        with self.assertRaises(TypeError):
            iter(out_t)
        self.assertFalse(numpy.isnan(out_t))
        self.assertFalse(numpy.isinf(out_t))

        diff = numpy.absolute(out - out_t) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    def test_phinorm2psinorm_scalar_psinorm_scalar_t_each_t_False(self):
        out = e.phinorm2psinorm(scalar_psinorm, scalar_t, each_t=False)
        out_t = et.phinorm2psinorm(scalar_psinorm, scalar_t, each_t=False)

        with self.assertRaises(TypeError):
            iter(out)
        self.assertFalse(numpy.isnan(out))
        self.assertFalse(numpy.isinf(out))

        with self.assertRaises(TypeError):
            iter(out_t)
        self.assertFalse(numpy.isnan(out_t))
        self.assertFalse(numpy.isinf(out_t))

        diff = numpy.absolute(out - out_t) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    # vector psinorm, scalar t:
    def test_phinorm2psinorm_vector_psinorm_scalar_t_each_t_True(self):
        out = e.phinorm2psinorm(vector_psinorm, scalar_t)
        out_t = et.phinorm2psinorm(vector_psinorm, scalar_t)

        self.assertEqual(out.shape, (len(vector_psinorm),))
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, (len(vector_psinorm),))
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    def test_phinorm2psinorm_vector_psinorm_scalar_t_each_t_False(self):
        out = e.phinorm2psinorm(vector_psinorm, scalar_t, each_t=False)
        out_t = et.phinorm2psinorm(vector_psinorm, scalar_t, each_t=False)

        self.assertEqual(out.shape, (len(vector_psinorm),))
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, (len(vector_psinorm),))
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    # matrix psinorm, scalar t:
    def test_phinorm2psinorm_matrix_psinorm_scalar_t_each_t_True(self):
        out = e.phinorm2psinorm(matrix_psinorm, scalar_t)
        out_t = et.phinorm2psinorm(matrix_psinorm, scalar_t)

        self.assertEqual(out.shape, matrix_psinorm.shape)
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, matrix_psinorm.shape)
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    def test_phinorm2psinorm_matrix_psinorm_scalar_t_each_t_False(self):
        out = e.phinorm2psinorm(matrix_psinorm, scalar_t, each_t=False)
        out_t = et.phinorm2psinorm(matrix_psinorm, scalar_t, each_t=False)

        self.assertEqual(out.shape, matrix_psinorm.shape)
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, matrix_psinorm.shape)
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    # scalar psinorm, vector t:
    def test_phinorm2psinorm_scalar_psinorm_vector_t_each_t_True(self):
        out = e.phinorm2psinorm(scalar_psinorm, vector_t)
        out_t = et.phinorm2psinorm(scalar_psinorm, vector_t)

        self.assertEqual(out.shape, (len(vector_t),))
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, (len(vector_t),))
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    def test_phinorm2psinorm_scalar_psinorm_vector_t_each_t_False(self):
        with self.assertRaises(ValueError):
            out = e.phinorm2psinorm(scalar_psinorm, vector_t, each_t=False)
        with self.assertRaises(ValueError):
            out_t = et.phinorm2psinorm(scalar_psinorm, vector_t, each_t=False)

    # vector psinorm, vector t:
    def test_phinorm2psinorm_vector_psinorm_vector_t_each_t_True(self):
        out = e.phinorm2psinorm(vector_psinorm, vector_t)
        out_t = et.phinorm2psinorm(vector_psinorm, vector_t)

        self.assertEqual(out.shape, (len(vector_t), len(vector_psinorm)))
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, (len(vector_t), len(vector_psinorm)))
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    def test_phinorm2psinorm_vector_psinorm_vector_t_each_t_False(self):
        with self.assertRaises(ValueError):
            out = e.phinorm2psinorm(vector_psinorm, vector_t, each_t=False)
        with self.assertRaises(ValueError):
            out_t = et.phinorm2psinorm(vector_psinorm, vector_t, each_t=False)

    # matrix psinorm, vector t:
    def test_phinorm2psinorm_matrix_psinorm_vector_t_each_t_True(self):
        out = e.phinorm2psinorm(matrix_psinorm, vector_t)
        out_t = et.phinorm2psinorm(matrix_psinorm, vector_t)

        self.assertEqual(out.shape, tuple([len(vector_t),] + list(matrix_psinorm.shape)))
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, tuple([len(vector_t),] + list(matrix_psinorm.shape)))
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    def test_phinorm2psinorm_matrix_psinorm_vector_t_each_t_False(self):
        with self.assertRaises(ValueError):
            out = e.phinorm2psinorm(matrix_psinorm, vector_t, each_t=False)
        with self.assertRaises(ValueError):
            out_t = et.phinorm2psinorm(matrix_psinorm, vector_t, each_t=False)

    # matrix psinorm, matrix t:
    def test_phinorm2psinorm_matrix_psinorm_matrix_t_each_t_True(self):
        with self.assertRaises(ValueError):
            out = e.phinorm2psinorm(matrix_psinorm, matrix_t)
        with self.assertRaises(ValueError):
            out_t = et.phinorm2psinorm(matrix_psinorm, matrix_t)

    def test_phinorm2psinorm_matrix_psinorm_matrix_t_each_t_False(self):
        out = e.phinorm2psinorm(matrix_psinorm, matrix_t, each_t=False)
        out_t = et.phinorm2psinorm(matrix_psinorm, matrix_t, each_t=False)

        self.assertEqual(out.shape, matrix_psinorm.shape)
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, matrix_psinorm.shape)
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

class phinorm2volnormTestCase(unittest.TestCase):
    # The following tests focus primarily on the consistency of the input/output
    # shapes. More tests should be added to check the consistency between the
    # different mapping routines.

    # scalar psinorm, scalar t:
    def test_phinorm2volnorm_scalar_psinorm_scalar_t_each_t_True(self):
        out = e.phinorm2volnorm(scalar_psinorm, scalar_t)
        out_t = et.phinorm2volnorm(scalar_psinorm, scalar_t)

        with self.assertRaises(TypeError):
            iter(out)
        self.assertFalse(numpy.isnan(out))
        self.assertFalse(numpy.isinf(out))

        with self.assertRaises(TypeError):
            iter(out_t)
        self.assertFalse(numpy.isnan(out_t))
        self.assertFalse(numpy.isinf(out_t))

        diff = numpy.absolute(out - out_t) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    def test_phinorm2volnorm_scalar_psinorm_scalar_t_each_t_False(self):
        out = e.phinorm2volnorm(scalar_psinorm, scalar_t, each_t=False)
        out_t = et.phinorm2volnorm(scalar_psinorm, scalar_t, each_t=False)

        with self.assertRaises(TypeError):
            iter(out)
        self.assertFalse(numpy.isnan(out))
        self.assertFalse(numpy.isinf(out))

        with self.assertRaises(TypeError):
            iter(out_t)
        self.assertFalse(numpy.isnan(out_t))
        self.assertFalse(numpy.isinf(out_t))

        diff = numpy.absolute(out - out_t) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    # vector psinorm, scalar t:
    def test_phinorm2volnorm_vector_psinorm_scalar_t_each_t_True(self):
        out = e.phinorm2volnorm(vector_psinorm, scalar_t)
        out_t = et.phinorm2volnorm(vector_psinorm, scalar_t)

        self.assertEqual(out.shape, (len(vector_psinorm),))
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, (len(vector_psinorm),))
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertTrue((numpy.isnan(out) == numpy.isnan(out_t)).all())
        # res2 = (out - out_t)**2
        # diff = numpy.sqrt((res2[~numpy.isnan(res2)]).max()) / numpy.absolute(out[~numpy.isnan(out)]).max()
        self.assertLessEqual(diff, tol)

    def test_phinorm2volnorm_vector_psinorm_scalar_t_each_t_False(self):
        out = e.phinorm2volnorm(vector_psinorm, scalar_t, each_t=False)
        out_t = et.phinorm2volnorm(vector_psinorm, scalar_t, each_t=False)

        self.assertEqual(out.shape, (len(vector_psinorm),))
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, (len(vector_psinorm),))
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertTrue((numpy.isnan(out) == numpy.isnan(out_t)).all())
        # res2 = (out - out_t)**2
        # diff = numpy.sqrt((res2[~numpy.isnan(res2)]).max()) / numpy.absolute(out[~numpy.isnan(out)]).max()
        self.assertLessEqual(diff, tol)

    # matrix psinorm, scalar t:
    def test_phinorm2volnorm_matrix_psinorm_scalar_t_each_t_True(self):
        out = e.phinorm2volnorm(matrix_psinorm, scalar_t)
        out_t = et.phinorm2volnorm(matrix_psinorm, scalar_t)

        self.assertEqual(out.shape, matrix_psinorm.shape)
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, matrix_psinorm.shape)
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertTrue((numpy.isnan(out) == numpy.isnan(out_t)).all())
        # res2 = (out - out_t)**2
        # diff = numpy.sqrt((res2[~numpy.isnan(res2)]).max()) / numpy.absolute(out[~numpy.isnan(out)]).max()
        self.assertLessEqual(diff, tol)

    def test_phinorm2volnorm_matrix_psinorm_scalar_t_each_t_False(self):
        out = e.phinorm2volnorm(matrix_psinorm, scalar_t, each_t=False)
        out_t = et.phinorm2volnorm(matrix_psinorm, scalar_t, each_t=False)

        self.assertEqual(out.shape, matrix_psinorm.shape)
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, matrix_psinorm.shape)
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertTrue((numpy.isnan(out) == numpy.isnan(out_t)).all())
        # res2 = (out - out_t)**2
        # diff = numpy.sqrt((res2[~numpy.isnan(res2)]).max()) / numpy.absolute(out[~numpy.isnan(out)]).max()
        self.assertLessEqual(diff, tol)

    # scalar psinorm, vector t:
    def test_phinorm2volnorm_scalar_psinorm_vector_t_each_t_True(self):
        out = e.phinorm2volnorm(scalar_psinorm, vector_t)
        out_t = et.phinorm2volnorm(scalar_psinorm, vector_t)

        self.assertEqual(out.shape, (len(vector_t),))
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, (len(vector_t),))
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    def test_phinorm2volnorm_scalar_psinorm_vector_t_each_t_False(self):
        with self.assertRaises(ValueError):
            out = e.phinorm2volnorm(scalar_psinorm, vector_t, each_t=False)
        with self.assertRaises(ValueError):
            out_t = et.phinorm2volnorm(scalar_psinorm, vector_t, each_t=False)

    # vector psinorm, vector t:
    def test_phinorm2volnorm_vector_psinorm_vector_t_each_t_True(self):
        out = e.phinorm2volnorm(vector_psinorm, vector_t)
        out_t = et.phinorm2volnorm(vector_psinorm, vector_t)

        self.assertEqual(out.shape, (len(vector_t), len(vector_psinorm)))
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, (len(vector_t), len(vector_psinorm)))
        # TODO: Fix boundary!
        # self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        # diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        # self.assertTrue((numpy.isnan(out) == numpy.isnan(out_t)).all())
        res2 = (out - out_t)**2
        diff = numpy.sqrt((res2[~numpy.isnan(res2)]).max()) / numpy.absolute(out[~numpy.isnan(out)]).max()
        self.assertLessEqual(diff, tol)

    def test_phinorm2volnorm_vector_psinorm_vector_t_each_t_False(self):
        with self.assertRaises(ValueError):
            out = e.phinorm2volnorm(vector_psinorm, vector_t, each_t=False)
        with self.assertRaises(ValueError):
            out_t = et.phinorm2volnorm(vector_psinorm, vector_t, each_t=False)

    # matrix psinorm, vector t:
    def test_phinorm2volnorm_matrix_psinorm_vector_t_each_t_True(self):
        out = e.phinorm2volnorm(matrix_psinorm, vector_t)
        out_t = et.phinorm2volnorm(matrix_psinorm, vector_t)

        self.assertEqual(out.shape, tuple([len(vector_t),] + list(matrix_psinorm.shape)))
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, tuple([len(vector_t),] + list(matrix_psinorm.shape)))
        # TODO: Fix boundary!
        # self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        # diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        # self.assertTrue((numpy.isnan(out) == numpy.isnan(out_t)).all())
        res2 = (out - out_t)**2
        diff = numpy.sqrt((res2[~numpy.isnan(res2)]).max()) / numpy.absolute(out[~numpy.isnan(out)]).max()
        self.assertLessEqual(diff, tol)

    def test_phinorm2volnorm_matrix_psinorm_vector_t_each_t_False(self):
        with self.assertRaises(ValueError):
            out = e.phinorm2volnorm(matrix_psinorm, vector_t, each_t=False)
        with self.assertRaises(ValueError):
            out_t = et.phinorm2volnorm(matrix_psinorm, vector_t, each_t=False)

    # matrix psinorm, matrix t:
    def test_phinorm2volnorm_matrix_psinorm_matrix_t_each_t_True(self):
        with self.assertRaises(ValueError):
            out = e.phinorm2volnorm(matrix_psinorm, matrix_t)
        with self.assertRaises(ValueError):
            out_t = et.phinorm2volnorm(matrix_psinorm, matrix_t)

    def test_phinorm2volnorm_matrix_psinorm_matrix_t_each_t_False(self):
        out = e.phinorm2volnorm(matrix_psinorm, matrix_t, each_t=False)
        out_t = et.phinorm2volnorm(matrix_psinorm, matrix_t, each_t=False)

        self.assertEqual(out.shape, matrix_psinorm.shape)
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, matrix_psinorm.shape)
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        # diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertTrue((numpy.isnan(out) == numpy.isnan(out_t)).all())
        res2 = (out - out_t)**2
        diff = numpy.sqrt((res2[~numpy.isnan(res2)]).max()) / numpy.absolute(out[~numpy.isnan(out)]).max()
        self.assertLessEqual(diff, tol)

class phinorm2rmidTestCase(unittest.TestCase):
    # The following tests focus primarily on the consistency of the input/output
    # shapes. More tests should be added to check the consistency between the
    # different mapping routines.

    # scalar psinorm, scalar t:
    def test_phinorm2rmid_scalar_psinorm_scalar_t_each_t_True_rho_False(self):
        out = e.phinorm2rmid(scalar_psinorm, scalar_t)
        out_t = et.phinorm2rmid(scalar_psinorm, scalar_t)

        with self.assertRaises(TypeError):
            iter(out)
        self.assertFalse(numpy.isnan(out))
        self.assertFalse(numpy.isinf(out))

        with self.assertRaises(TypeError):
            iter(out_t)
        self.assertFalse(numpy.isnan(out_t))
        self.assertFalse(numpy.isinf(out_t))

        diff = numpy.absolute(out - out_t) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    def test_phinorm2rmid_scalar_psinorm_scalar_t_each_t_False_rho_False(self):
        out = e.phinorm2rmid(scalar_psinorm, scalar_t, each_t=False)
        out_t = et.phinorm2rmid(scalar_psinorm, scalar_t, each_t=False)

        with self.assertRaises(TypeError):
            iter(out)
        self.assertFalse(numpy.isnan(out))
        self.assertFalse(numpy.isinf(out))

        with self.assertRaises(TypeError):
            iter(out_t)
        self.assertFalse(numpy.isnan(out_t))
        self.assertFalse(numpy.isinf(out_t))

        diff = numpy.absolute(out - out_t) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    def test_phinorm2rmid_scalar_psinorm_scalar_t_each_t_True_rho_True(self):
        out = e.phinorm2rmid(scalar_psinorm, scalar_t, rho=True)
        out_t = et.phinorm2rmid(scalar_psinorm, scalar_t, rho=True)

        with self.assertRaises(TypeError):
            iter(out)
        self.assertFalse(numpy.isnan(out))
        self.assertFalse(numpy.isinf(out))

        with self.assertRaises(TypeError):
            iter(out_t)
        self.assertFalse(numpy.isnan(out_t))
        self.assertFalse(numpy.isinf(out_t))

        diff = numpy.absolute(out - out_t) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    def test_phinorm2rmid_scalar_psinorm_scalar_t_each_t_False_rho_True(self):
        out = e.phinorm2rmid(scalar_psinorm, scalar_t, each_t=False, rho=True)
        out_t = et.phinorm2rmid(scalar_psinorm, scalar_t, each_t=False, rho=True)

        with self.assertRaises(TypeError):
            iter(out)
        self.assertFalse(numpy.isnan(out))
        self.assertFalse(numpy.isinf(out))

        with self.assertRaises(TypeError):
            iter(out_t)
        self.assertFalse(numpy.isnan(out_t))
        self.assertFalse(numpy.isinf(out_t))

        diff = numpy.absolute(out - out_t) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    # vector psinorm, scalar t:
    def test_phinorm2rmid_vector_psinorm_scalar_t_each_t_True_rho_False(self):
        out = e.phinorm2rmid(vector_psinorm, scalar_t)
        out_t = et.phinorm2rmid(vector_psinorm, scalar_t)

        self.assertEqual(out.shape, (len(vector_psinorm),))
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, (len(vector_psinorm),))
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        # diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertTrue((numpy.isnan(out) == numpy.isnan(out_t)).all())
        res2 = (out - out_t)**2
        diff = numpy.sqrt((res2[~numpy.isnan(res2)]).max()) / numpy.absolute(out[~numpy.isnan(out)]).max()
        self.assertLessEqual(diff, tol)

    def test_phinorm2rmid_vector_psinorm_scalar_t_each_t_False_rho_False(self):
        out = e.phinorm2rmid(vector_psinorm, scalar_t, each_t=False)
        out_t = et.phinorm2rmid(vector_psinorm, scalar_t, each_t=False)

        self.assertEqual(out.shape, (len(vector_psinorm),))
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, (len(vector_psinorm),))
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        # diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertTrue((numpy.isnan(out) == numpy.isnan(out_t)).all())
        res2 = (out - out_t)**2
        diff = numpy.sqrt((res2[~numpy.isnan(res2)]).max()) / numpy.absolute(out[~numpy.isnan(out)]).max()
        self.assertLessEqual(diff, tol)

    def test_phinorm2rmid_vector_psinorm_scalar_t_each_t_True_rho_True(self):
        out = e.phinorm2rmid(vector_psinorm, scalar_t, rho=True)
        out_t = et.phinorm2rmid(vector_psinorm, scalar_t, rho=True)

        self.assertEqual(out.shape, (len(vector_psinorm),))
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, (len(vector_psinorm),))
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        # diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertTrue((numpy.isnan(out) == numpy.isnan(out_t)).all())
        res2 = (out - out_t)**2
        diff = numpy.sqrt((res2[~numpy.isnan(res2)]).max()) / numpy.absolute(out[~numpy.isnan(out)]).max()
        self.assertLessEqual(diff, tol)

    def test_phinorm2rmid_vector_psinorm_scalar_t_each_t_False_rho_True(self):
        out = e.phinorm2rmid(vector_psinorm, scalar_t, each_t=False, rho=True)
        out_t = et.phinorm2rmid(vector_psinorm, scalar_t, each_t=False, rho=True)

        self.assertEqual(out.shape, (len(vector_psinorm),))
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, (len(vector_psinorm),))
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        # diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertTrue((numpy.isnan(out) == numpy.isnan(out_t)).all())
        res2 = (out - out_t)**2
        diff = numpy.sqrt((res2[~numpy.isnan(res2)]).max()) / numpy.absolute(out[~numpy.isnan(out)]).max()
        self.assertLessEqual(diff, tol)

    # matrix psinorm, scalar t:
    def test_phinorm2rmid_matrix_psinorm_scalar_t_each_t_True_rho_False(self):
        out = e.phinorm2rmid(matrix_psinorm, scalar_t)
        out_t = et.phinorm2rmid(matrix_psinorm, scalar_t)

        self.assertEqual(out.shape, matrix_psinorm.shape)
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, matrix_psinorm.shape)
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        # diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertTrue((numpy.isnan(out) == numpy.isnan(out_t)).all())
        res2 = (out - out_t)**2
        diff = numpy.sqrt((res2[~numpy.isnan(res2)]).max()) / numpy.absolute(out[~numpy.isnan(out)]).max()
        self.assertLessEqual(diff, tol)

    def test_phinorm2rmid_matrix_psinorm_scalar_t_each_t_False_rho_False(self):
        out = e.phinorm2rmid(matrix_psinorm, scalar_t, each_t=False)
        out_t = et.phinorm2rmid(matrix_psinorm, scalar_t, each_t=False)

        self.assertEqual(out.shape, matrix_psinorm.shape)
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, matrix_psinorm.shape)
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        # diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertTrue((numpy.isnan(out) == numpy.isnan(out_t)).all())
        res2 = (out - out_t)**2
        diff = numpy.sqrt((res2[~numpy.isnan(res2)]).max()) / numpy.absolute(out[~numpy.isnan(out)]).max()
        self.assertLessEqual(diff, tol)

    def test_phinorm2rmid_matrix_psinorm_scalar_t_each_t_True_rho_True(self):
        out = e.phinorm2rmid(matrix_psinorm, scalar_t, rho=True)
        out_t = et.phinorm2rmid(matrix_psinorm, scalar_t, rho=True)

        self.assertEqual(out.shape, matrix_psinorm.shape)
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, matrix_psinorm.shape)
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        # diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertTrue((numpy.isnan(out) == numpy.isnan(out_t)).all())
        res2 = (out - out_t)**2
        diff = numpy.sqrt((res2[~numpy.isnan(res2)]).max()) / numpy.absolute(out[~numpy.isnan(out)]).max()
        self.assertLessEqual(diff, tol)

    def test_phinorm2rmid_matrix_psinorm_scalar_t_each_t_False_rho_True(self):
        out = e.phinorm2rmid(matrix_psinorm, scalar_t, each_t=False, rho=True)
        out_t = et.phinorm2rmid(matrix_psinorm, scalar_t, each_t=False, rho=True)

        self.assertEqual(out.shape, matrix_psinorm.shape)
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, matrix_psinorm.shape)
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        # diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertTrue((numpy.isnan(out) == numpy.isnan(out_t)).all())
        res2 = (out - out_t)**2
        diff = numpy.sqrt((res2[~numpy.isnan(res2)]).max()) / numpy.absolute(out[~numpy.isnan(out)]).max()
        self.assertLessEqual(diff, tol)

    # scalar psinorm, vector t:
    def test_phinorm2rmid_scalar_psinorm_vector_t_each_t_True_rho_False(self):
        out = e.phinorm2rmid(scalar_psinorm, vector_t)
        out_t = et.phinorm2rmid(scalar_psinorm, vector_t)

        self.assertEqual(out.shape, (len(vector_t),))
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, (len(vector_t),))
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    def test_phinorm2rmid_scalar_psinorm_vector_t_each_t_False_rho_False(self):
        with self.assertRaises(ValueError):
            out = e.phinorm2rmid(scalar_psinorm, vector_t, each_t=False)
        with self.assertRaises(ValueError):
            out_t = et.phinorm2rmid(scalar_psinorm, vector_t, each_t=False)

    def test_phinorm2rmid_scalar_psinorm_vector_t_each_t_True_rho_True(self):
        out = e.phinorm2rmid(scalar_psinorm, vector_t, rho=True)
        out_t = et.phinorm2rmid(scalar_psinorm, vector_t, rho=True)

        self.assertEqual(out.shape, (len(vector_t),))
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, (len(vector_t),))
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    def test_phinorm2rmid_scalar_psinorm_vector_t_each_t_False_rho_True(self):
        with self.assertRaises(ValueError):
            out = e.phinorm2rmid(scalar_psinorm, vector_t, each_t=False, rho=True)
        with self.assertRaises(ValueError):
            out_t = et.phinorm2rmid(scalar_psinorm, vector_t, each_t=False, rho=True)

    # vector psinorm, vector t:
    def test_phinorm2rmid_vector_psinorm_vector_t_each_t_True_rho_False(self):
        out = e.phinorm2rmid(vector_psinorm, vector_t)
        out_t = et.phinorm2rmid(vector_psinorm, vector_t)

        self.assertEqual(out.shape, (len(vector_t), len(vector_psinorm)))
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, (len(vector_t), len(vector_psinorm)))
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertTrue((numpy.isnan(out) == numpy.isnan(out_t)).all())
        # res2 = (out - out_t)**2
        # diff = numpy.sqrt((res2[~numpy.isnan(res2)]).max()) / numpy.absolute(out[~numpy.isnan(out)]).max()
        self.assertLessEqual(diff, tol)

    def test_phinorm2rmid_vector_psinorm_vector_t_each_t_False_rho_False(self):
        with self.assertRaises(ValueError):
            out = e.phinorm2rmid(vector_psinorm, vector_t, each_t=False)
        with self.assertRaises(ValueError):
            out_t = et.phinorm2rmid(vector_psinorm, vector_t, each_t=False)

    def test_phinorm2rmid_vector_psinorm_vector_t_each_t_True_rho_True(self):
        out = e.phinorm2rmid(vector_psinorm, vector_t, rho=True)
        out_t = et.phinorm2rmid(vector_psinorm, vector_t, rho=True)

        self.assertEqual(out.shape, (len(vector_t), len(vector_psinorm)))
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, (len(vector_t), len(vector_psinorm)))
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertTrue((numpy.isnan(out) == numpy.isnan(out_t)).all())
        # res2 = (out - out_t)**2
        # diff = numpy.sqrt((res2[~numpy.isnan(res2)]).max()) / numpy.absolute(out[~numpy.isnan(out)]).max()
        self.assertLessEqual(diff, tol)

    def test_phinorm2rmid_vector_psinorm_vector_t_each_t_False_rho_True(self):
        with self.assertRaises(ValueError):
            out = e.phinorm2rmid(vector_psinorm, vector_t, each_t=False, rho=True)
        with self.assertRaises(ValueError):
            out_t = et.phinorm2rmid(vector_psinorm, vector_t, each_t=False, rho=True)

    # matrix psinorm, vector t:
    def test_phinorm2rmid_matrix_psinorm_vector_t_each_t_True_rho_False(self):
        out = e.phinorm2rmid(matrix_psinorm, vector_t)
        out_t = et.phinorm2rmid(matrix_psinorm, vector_t)

        self.assertEqual(out.shape, tuple([len(vector_t),] + list(matrix_psinorm.shape)))
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, tuple([len(vector_t),] + list(matrix_psinorm.shape)))
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertTrue((numpy.isnan(out) == numpy.isnan(out_t)).all())
        # res2 = (out - out_t)**2
        # diff = numpy.sqrt((res2[~numpy.isnan(res2)]).max()) / numpy.absolute(out[~numpy.isnan(out)]).max()
        self.assertLessEqual(diff, tol)

    def test_phinorm2rmid_matrix_psinorm_vector_t_each_t_False_rho_False(self):
        with self.assertRaises(ValueError):
            out = e.phinorm2rmid(matrix_psinorm, vector_t, each_t=False)
        with self.assertRaises(ValueError):
            out_t = et.phinorm2rmid(matrix_psinorm, vector_t, each_t=False)

    def test_phinorm2rmid_matrix_psinorm_vector_t_each_t_True_rho_True(self):
        out = e.phinorm2rmid(matrix_psinorm, vector_t, rho=True)
        out_t = et.phinorm2rmid(matrix_psinorm, vector_t, rho=True)

        self.assertEqual(out.shape, tuple([len(vector_t),] + list(matrix_psinorm.shape)))
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, tuple([len(vector_t),] + list(matrix_psinorm.shape)))
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertTrue((numpy.isnan(out) == numpy.isnan(out_t)).all())
        # res2 = (out - out_t)**2
        # diff = numpy.sqrt((res2[~numpy.isnan(res2)]).max()) / numpy.absolute(out[~numpy.isnan(out)]).max()
        self.assertLessEqual(diff, tol)

    def test_phinorm2rmid_matrix_psinorm_vector_t_each_t_False_rho_True(self):
        with self.assertRaises(ValueError):
            out = e.phinorm2rmid(matrix_psinorm, vector_t, each_t=False, rho=True)
        with self.assertRaises(ValueError):
            out_t = et.phinorm2rmid(matrix_psinorm, vector_t, each_t=False, rho=True)

    # matrix psinorm, matrix t:
    def test_phinorm2rmid_matrix_psinorm_matrix_t_each_t_True_rho_False(self):
        with self.assertRaises(ValueError):
            out = e.phinorm2rmid(matrix_psinorm, matrix_t)
        with self.assertRaises(ValueError):
            out_t = et.phinorm2rmid(matrix_psinorm, matrix_t)

    def test_phinorm2rmid_matrix_psinorm_matrix_t_each_t_False_rho_False(self):
        out = e.phinorm2rmid(matrix_psinorm, matrix_t, each_t=False)
        out_t = et.phinorm2rmid(matrix_psinorm, matrix_t, each_t=False)

        self.assertEqual(out.shape, matrix_psinorm.shape)
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, matrix_psinorm.shape)
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        # diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertTrue((numpy.isnan(out) == numpy.isnan(out_t)).all())
        res2 = (out - out_t)**2
        diff = numpy.sqrt((res2[~numpy.isnan(res2)]).max()) / numpy.absolute(out[~numpy.isnan(out)]).max()
        self.assertLessEqual(diff, tol)

    def test_phinorm2rmid_matrix_psinorm_matrix_t_each_t_True_rho_True(self):
        with self.assertRaises(ValueError):
            out = e.phinorm2rmid(matrix_psinorm, matrix_t, rho=True)
        with self.assertRaises(ValueError):
            out_t = et.phinorm2rmid(matrix_psinorm, matrix_t, rho=True)

    def test_phinorm2rmid_matrix_psinorm_matrix_t_each_t_False_rho_True(self):
        out = e.phinorm2rmid(matrix_psinorm, matrix_t, each_t=False, rho=True)
        out_t = et.phinorm2rmid(matrix_psinorm, matrix_t, each_t=False, rho=True)

        self.assertEqual(out.shape, matrix_psinorm.shape)
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, matrix_psinorm.shape)
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        # diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertTrue((numpy.isnan(out) == numpy.isnan(out_t)).all())
        res2 = (out - out_t)**2
        diff = numpy.sqrt((res2[~numpy.isnan(res2)]).max()) / numpy.absolute(out[~numpy.isnan(out)]).max()
        self.assertLessEqual(diff, tol)

class phinorm2roaTestCase(unittest.TestCase):
    # The following tests focus primarily on the consistency of the input/output
    # shapes. More tests should be added to check the consistency between the
    # different mapping routines.

    # scalar psinorm, scalar t:
    def test_phinorm2roa_scalar_psinorm_scalar_t_each_t_True(self):
        out = e.phinorm2roa(scalar_psinorm, scalar_t)
        out_t = et.phinorm2roa(scalar_psinorm, scalar_t)

        with self.assertRaises(TypeError):
            iter(out)
        self.assertFalse(numpy.isnan(out))
        self.assertFalse(numpy.isinf(out))

        with self.assertRaises(TypeError):
            iter(out_t)
        self.assertFalse(numpy.isnan(out_t))
        self.assertFalse(numpy.isinf(out_t))

        diff = numpy.absolute(out - out_t) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    def test_phinorm2roa_scalar_psinorm_scalar_t_each_t_False(self):
        out = e.phinorm2roa(scalar_psinorm, scalar_t, each_t=False)
        out_t = et.phinorm2roa(scalar_psinorm, scalar_t, each_t=False)

        with self.assertRaises(TypeError):
            iter(out)
        self.assertFalse(numpy.isnan(out))
        self.assertFalse(numpy.isinf(out))

        with self.assertRaises(TypeError):
            iter(out_t)
        self.assertFalse(numpy.isnan(out_t))
        self.assertFalse(numpy.isinf(out_t))

        diff = numpy.absolute(out - out_t) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    # vector psinorm, scalar t:
    def test_phinorm2roa_vector_psinorm_scalar_t_each_t_True(self):
        out = e.phinorm2roa(vector_psinorm, scalar_t)
        out_t = et.phinorm2roa(vector_psinorm, scalar_t)

        self.assertEqual(out.shape, (len(vector_psinorm),))
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, (len(vector_psinorm),))
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        # diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertTrue((numpy.isnan(out) == numpy.isnan(out_t)).all())
        res2 = (out - out_t)**2
        diff = numpy.sqrt((res2[~numpy.isnan(res2)]).max()) / numpy.absolute(out[~numpy.isnan(out)]).max()
        self.assertLessEqual(diff, tol)

    def test_phinorm2roa_vector_psinorm_scalar_t_each_t_False(self):
        out = e.phinorm2roa(vector_psinorm, scalar_t, each_t=False)
        out_t = et.phinorm2roa(vector_psinorm, scalar_t, each_t=False)

        self.assertEqual(out.shape, (len(vector_psinorm),))
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, (len(vector_psinorm),))
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        # diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertTrue((numpy.isnan(out) == numpy.isnan(out_t)).all())
        res2 = (out - out_t)**2
        diff = numpy.sqrt((res2[~numpy.isnan(res2)]).max()) / numpy.absolute(out[~numpy.isnan(out)]).max()
        self.assertLessEqual(diff, tol)

    # matrix psinorm, scalar t:
    def test_phinorm2roa_matrix_psinorm_scalar_t_each_t_True(self):
        out = e.phinorm2roa(matrix_psinorm, scalar_t)
        out_t = et.phinorm2roa(matrix_psinorm, scalar_t)

        self.assertEqual(out.shape, matrix_psinorm.shape)
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, matrix_psinorm.shape)
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        # diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertTrue((numpy.isnan(out) == numpy.isnan(out_t)).all())
        res2 = (out - out_t)**2
        diff = numpy.sqrt((res2[~numpy.isnan(res2)]).max()) / numpy.absolute(out[~numpy.isnan(out)]).max()
        self.assertLessEqual(diff, tol)

    def test_phinorm2roa_matrix_psinorm_scalar_t_each_t_False(self):
        out = e.phinorm2roa(matrix_psinorm, scalar_t, each_t=False)
        out_t = et.phinorm2roa(matrix_psinorm, scalar_t, each_t=False)

        self.assertEqual(out.shape, matrix_psinorm.shape)
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, matrix_psinorm.shape)
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        # diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertTrue((numpy.isnan(out) == numpy.isnan(out_t)).all())
        res2 = (out - out_t)**2
        diff = numpy.sqrt((res2[~numpy.isnan(res2)]).max()) / numpy.absolute(out[~numpy.isnan(out)]).max()
        self.assertLessEqual(diff, tol)

    # scalar psinorm, vector t:
    def test_phinorm2roa_scalar_psinorm_vector_t_each_t_True(self):
        out = e.phinorm2roa(scalar_psinorm, vector_t)
        out_t = et.phinorm2roa(scalar_psinorm, vector_t)

        self.assertEqual(out.shape, (len(vector_t),))
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, (len(vector_t),))
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    def test_phinorm2roa_scalar_psinorm_vector_t_each_t_False(self):
        with self.assertRaises(ValueError):
            out = e.phinorm2roa(scalar_psinorm, vector_t, each_t=False)
        with self.assertRaises(ValueError):
            out_t = et.phinorm2roa(scalar_psinorm, vector_t, each_t=False)

    # vector psinorm, vector t:
    def test_phinorm2roa_vector_psinorm_vector_t_each_t_True(self):
        out = e.phinorm2roa(vector_psinorm, vector_t)
        out_t = et.phinorm2roa(vector_psinorm, vector_t)

        self.assertEqual(out.shape, (len(vector_t), len(vector_psinorm)))
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, (len(vector_t), len(vector_psinorm)))
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertTrue((numpy.isnan(out) == numpy.isnan(out_t)).all())
        # res2 = (out - out_t)**2
        # diff = numpy.sqrt((res2[~numpy.isnan(res2)]).max()) / numpy.absolute(out[~numpy.isnan(out)]).max()
        self.assertLessEqual(diff, tol)

    def test_phinorm2roa_vector_psinorm_vector_t_each_t_False(self):
        with self.assertRaises(ValueError):
            out = e.phinorm2roa(vector_psinorm, vector_t, each_t=False)
        with self.assertRaises(ValueError):
            out_t = et.phinorm2roa(vector_psinorm, vector_t, each_t=False)

    # matrix psinorm, vector t:
    def test_phinorm2roa_matrix_psinorm_vector_t_each_t_True(self):
        out = e.phinorm2roa(matrix_psinorm, vector_t)
        out_t = et.phinorm2roa(matrix_psinorm, vector_t)

        self.assertEqual(out.shape, tuple([len(vector_t),] + list(matrix_psinorm.shape)))
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, tuple([len(vector_t),] + list(matrix_psinorm.shape)))
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertTrue((numpy.isnan(out) == numpy.isnan(out_t)).all())
        # res2 = (out - out_t)**2
        # diff = numpy.sqrt((res2[~numpy.isnan(res2)]).max()) / numpy.absolute(out[~numpy.isnan(out)]).max()
        self.assertLessEqual(diff, tol)

    def test_phinorm2roa_matrix_psinorm_vector_t_each_t_False(self):
        with self.assertRaises(ValueError):
            out = e.phinorm2roa(matrix_psinorm, vector_t, each_t=False)
        with self.assertRaises(ValueError):
            out_t = et.phinorm2roa(matrix_psinorm, vector_t, each_t=False)

    # matrix psinorm, matrix t:
    def test_phinorm2roa_matrix_psinorm_matrix_t_each_t_True(self):
        with self.assertRaises(ValueError):
            out = e.phinorm2roa(matrix_psinorm, matrix_t)
        with self.assertRaises(ValueError):
            out_t = et.phinorm2roa(matrix_psinorm, matrix_t)

    def test_phinorm2roa_matrix_psinorm_matrix_t_each_t_False(self):
        out = e.phinorm2roa(matrix_psinorm, matrix_t, each_t=False)
        out_t = et.phinorm2roa(matrix_psinorm, matrix_t, each_t=False)

        self.assertEqual(out.shape, matrix_psinorm.shape)
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, matrix_psinorm.shape)
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        # diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertTrue((numpy.isnan(out) == numpy.isnan(out_t)).all())
        res2 = (out - out_t)**2
        diff = numpy.sqrt((res2[~numpy.isnan(res2)]).max()) / numpy.absolute(out[~numpy.isnan(out)]).max()
        self.assertLessEqual(diff, tol)

class volnorm2psinormTestCase(unittest.TestCase):
    # The following tests focus primarily on the consistency of the input/output
    # shapes. More tests should be added to check the consistency between the
    # different mapping routines.

    # scalar psinorm, scalar t:
    def test_volnorm2psinorm_scalar_psinorm_scalar_t_each_t_True(self):
        out = e.volnorm2psinorm(scalar_psinorm, scalar_t)
        out_t = et.volnorm2psinorm(scalar_psinorm, scalar_t)

        with self.assertRaises(TypeError):
            iter(out)
        self.assertFalse(numpy.isnan(out))
        self.assertFalse(numpy.isinf(out))

        with self.assertRaises(TypeError):
            iter(out_t)
        self.assertFalse(numpy.isnan(out_t))
        self.assertFalse(numpy.isinf(out_t))

        diff = numpy.absolute(out - out_t) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    def test_volnorm2psinorm_scalar_psinorm_scalar_t_each_t_False(self):
        out = e.volnorm2psinorm(scalar_psinorm, scalar_t, each_t=False)
        out_t = et.volnorm2psinorm(scalar_psinorm, scalar_t, each_t=False)

        with self.assertRaises(TypeError):
            iter(out)
        self.assertFalse(numpy.isnan(out))
        self.assertFalse(numpy.isinf(out))

        with self.assertRaises(TypeError):
            iter(out_t)
        self.assertFalse(numpy.isnan(out_t))
        self.assertFalse(numpy.isinf(out_t))

        diff = numpy.absolute(out - out_t) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    # vector psinorm, scalar t:
    def test_volnorm2psinorm_vector_psinorm_scalar_t_each_t_True(self):
        out = e.volnorm2psinorm(vector_psinorm, scalar_t)
        out_t = et.volnorm2psinorm(vector_psinorm, scalar_t)

        self.assertEqual(out.shape, (len(vector_psinorm),))
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, (len(vector_psinorm),))
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    def test_volnorm2psinorm_vector_psinorm_scalar_t_each_t_False(self):
        out = e.volnorm2psinorm(vector_psinorm, scalar_t, each_t=False)
        out_t = et.volnorm2psinorm(vector_psinorm, scalar_t, each_t=False)

        self.assertEqual(out.shape, (len(vector_psinorm),))
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, (len(vector_psinorm),))
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    # matrix psinorm, scalar t:
    def test_volnorm2psinorm_matrix_psinorm_scalar_t_each_t_True(self):
        out = e.volnorm2psinorm(matrix_psinorm, scalar_t)
        out_t = et.volnorm2psinorm(matrix_psinorm, scalar_t)

        self.assertEqual(out.shape, matrix_psinorm.shape)
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, matrix_psinorm.shape)
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    def test_volnorm2psinorm_matrix_psinorm_scalar_t_each_t_False(self):
        out = e.volnorm2psinorm(matrix_psinorm, scalar_t, each_t=False)
        out_t = et.volnorm2psinorm(matrix_psinorm, scalar_t, each_t=False)

        self.assertEqual(out.shape, matrix_psinorm.shape)
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, matrix_psinorm.shape)
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    # scalar psinorm, vector t:
    def test_volnorm2psinorm_scalar_psinorm_vector_t_each_t_True(self):
        out = e.volnorm2psinorm(scalar_psinorm, vector_t)
        out_t = et.volnorm2psinorm(scalar_psinorm, vector_t)

        self.assertEqual(out.shape, (len(vector_t),))
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, (len(vector_t),))
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    def test_volnorm2psinorm_scalar_psinorm_vector_t_each_t_False(self):
        with self.assertRaises(ValueError):
            out = e.volnorm2psinorm(scalar_psinorm, vector_t, each_t=False)
        with self.assertRaises(ValueError):
            out_t = et.volnorm2psinorm(scalar_psinorm, vector_t, each_t=False)

    # vector psinorm, vector t:
    def test_volnorm2psinorm_vector_psinorm_vector_t_each_t_True(self):
        out = e.volnorm2psinorm(vector_psinorm, vector_t)
        out_t = et.volnorm2psinorm(vector_psinorm, vector_t)

        self.assertEqual(out.shape, (len(vector_t), len(vector_psinorm)))
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, (len(vector_t), len(vector_psinorm)))
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    def test_volnorm2psinorm_vector_psinorm_vector_t_each_t_False(self):
        with self.assertRaises(ValueError):
            out = e.volnorm2psinorm(vector_psinorm, vector_t, each_t=False)
        with self.assertRaises(ValueError):
            out_t = et.volnorm2psinorm(vector_psinorm, vector_t, each_t=False)

    # matrix psinorm, vector t:
    def test_volnorm2psinorm_matrix_psinorm_vector_t_each_t_True(self):
        out = e.volnorm2psinorm(matrix_psinorm, vector_t)
        out_t = et.volnorm2psinorm(matrix_psinorm, vector_t)

        self.assertEqual(out.shape, tuple([len(vector_t),] + list(matrix_psinorm.shape)))
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, tuple([len(vector_t),] + list(matrix_psinorm.shape)))
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    def test_volnorm2psinorm_matrix_psinorm_vector_t_each_t_False(self):
        with self.assertRaises(ValueError):
            out = e.volnorm2psinorm(matrix_psinorm, vector_t, each_t=False)
        with self.assertRaises(ValueError):
            out_t = et.volnorm2psinorm(matrix_psinorm, vector_t, each_t=False)

    # matrix psinorm, matrix t:
    def test_volnorm2psinorm_matrix_psinorm_matrix_t_each_t_True(self):
        with self.assertRaises(ValueError):
            out = e.volnorm2psinorm(matrix_psinorm, matrix_t)
        with self.assertRaises(ValueError):
            out_t = et.volnorm2psinorm(matrix_psinorm, matrix_t)

    def test_volnorm2psinorm_matrix_psinorm_matrix_t_each_t_False(self):
        out = e.volnorm2psinorm(matrix_psinorm, matrix_t, each_t=False)
        out_t = et.volnorm2psinorm(matrix_psinorm, matrix_t, each_t=False)

        self.assertEqual(out.shape, matrix_psinorm.shape)
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, matrix_psinorm.shape)
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

class volnorm2phinormTestCase(unittest.TestCase):
    # The following tests focus primarily on the consistency of the input/output
    # shapes. More tests should be added to check the consistency between the
    # different mapping routines.

    # scalar psinorm, scalar t:
    def test_volnorm2phinorm_scalar_psinorm_scalar_t_each_t_True(self):
        out = e.volnorm2phinorm(scalar_psinorm, scalar_t)
        out_t = et.volnorm2phinorm(scalar_psinorm, scalar_t)

        with self.assertRaises(TypeError):
            iter(out)
        self.assertFalse(numpy.isnan(out))
        self.assertFalse(numpy.isinf(out))

        with self.assertRaises(TypeError):
            iter(out_t)
        self.assertFalse(numpy.isnan(out_t))
        self.assertFalse(numpy.isinf(out_t))

        diff = numpy.absolute(out - out_t) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    def test_volnorm2phinorm_scalar_psinorm_scalar_t_each_t_False(self):
        out = e.volnorm2phinorm(scalar_psinorm, scalar_t, each_t=False)
        out_t = et.volnorm2phinorm(scalar_psinorm, scalar_t, each_t=False)

        with self.assertRaises(TypeError):
            iter(out)
        self.assertFalse(numpy.isnan(out))
        self.assertFalse(numpy.isinf(out))

        with self.assertRaises(TypeError):
            iter(out_t)
        self.assertFalse(numpy.isnan(out_t))
        self.assertFalse(numpy.isinf(out_t))

        diff = numpy.absolute(out - out_t) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    # vector psinorm, scalar t:
    def test_volnorm2phinorm_vector_psinorm_scalar_t_each_t_True(self):
        out = e.volnorm2phinorm(vector_psinorm, scalar_t)
        out_t = et.volnorm2phinorm(vector_psinorm, scalar_t)

        self.assertEqual(out.shape, (len(vector_psinorm),))
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, (len(vector_psinorm),))
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertTrue((numpy.isnan(out) == numpy.isnan(out_t)).all())
        # res2 = (out - out_t)**2
        # diff = numpy.sqrt((res2[~numpy.isnan(res2)]).max()) / numpy.absolute(out[~numpy.isnan(out)]).max()
        self.assertLessEqual(diff, tol)

    def test_volnorm2phinorm_vector_psinorm_scalar_t_each_t_False(self):
        out = e.volnorm2phinorm(vector_psinorm, scalar_t, each_t=False)
        out_t = et.volnorm2phinorm(vector_psinorm, scalar_t, each_t=False)

        self.assertEqual(out.shape, (len(vector_psinorm),))
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, (len(vector_psinorm),))
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertTrue((numpy.isnan(out) == numpy.isnan(out_t)).all())
        # res2 = (out - out_t)**2
        # diff = numpy.sqrt((res2[~numpy.isnan(res2)]).max()) / numpy.absolute(out[~numpy.isnan(out)]).max()
        self.assertLessEqual(diff, tol)

    # matrix psinorm, scalar t:
    def test_volnorm2phinorm_matrix_psinorm_scalar_t_each_t_True(self):
        out = e.volnorm2phinorm(matrix_psinorm, scalar_t)
        out_t = et.volnorm2phinorm(matrix_psinorm, scalar_t)

        self.assertEqual(out.shape, matrix_psinorm.shape)
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, matrix_psinorm.shape)
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertTrue((numpy.isnan(out) == numpy.isnan(out_t)).all())
        # res2 = (out - out_t)**2
        # diff = numpy.sqrt((res2[~numpy.isnan(res2)]).max()) / numpy.absolute(out[~numpy.isnan(out)]).max()
        self.assertLessEqual(diff, tol)

    def test_volnorm2phinorm_matrix_psinorm_scalar_t_each_t_False(self):
        out = e.volnorm2phinorm(matrix_psinorm, scalar_t, each_t=False)
        out_t = et.volnorm2phinorm(matrix_psinorm, scalar_t, each_t=False)

        self.assertEqual(out.shape, matrix_psinorm.shape)
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, matrix_psinorm.shape)
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertTrue((numpy.isnan(out) == numpy.isnan(out_t)).all())
        # res2 = (out - out_t)**2
        # diff = numpy.sqrt((res2[~numpy.isnan(res2)]).max()) / numpy.absolute(out[~numpy.isnan(out)]).max()
        self.assertLessEqual(diff, tol)

    # scalar psinorm, vector t:
    def test_volnorm2phinorm_scalar_psinorm_vector_t_each_t_True(self):
        out = e.volnorm2phinorm(scalar_psinorm, vector_t)
        out_t = et.volnorm2phinorm(scalar_psinorm, vector_t)

        self.assertEqual(out.shape, (len(vector_t),))
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, (len(vector_t),))
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    def test_volnorm2phinorm_scalar_psinorm_vector_t_each_t_False(self):
        with self.assertRaises(ValueError):
            out = e.volnorm2phinorm(scalar_psinorm, vector_t, each_t=False)
        with self.assertRaises(ValueError):
            out_t = et.volnorm2phinorm(scalar_psinorm, vector_t, each_t=False)

    # vector psinorm, vector t:
    def test_volnorm2phinorm_vector_psinorm_vector_t_each_t_True(self):
        out = e.volnorm2phinorm(vector_psinorm, vector_t)
        out_t = et.volnorm2phinorm(vector_psinorm, vector_t)

        self.assertEqual(out.shape, (len(vector_t), len(vector_psinorm)))
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, (len(vector_t), len(vector_psinorm)))
        # TODO: Fix boundary!
        # self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        # diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        # self.assertTrue((numpy.isnan(out) == numpy.isnan(out_t)).all())
        res2 = (out - out_t)**2
        diff = numpy.sqrt((res2[~numpy.isnan(res2)]).max()) / numpy.absolute(out[~numpy.isnan(out)]).max()
        self.assertLessEqual(diff, tol)

    def test_volnorm2phinorm_vector_psinorm_vector_t_each_t_False(self):
        with self.assertRaises(ValueError):
            out = e.volnorm2phinorm(vector_psinorm, vector_t, each_t=False)
        with self.assertRaises(ValueError):
            out_t = et.volnorm2phinorm(vector_psinorm, vector_t, each_t=False)

    # matrix psinorm, vector t:
    def test_volnorm2phinorm_matrix_psinorm_vector_t_each_t_True(self):
        out = e.volnorm2phinorm(matrix_psinorm, vector_t)
        out_t = et.volnorm2phinorm(matrix_psinorm, vector_t)

        self.assertEqual(out.shape, tuple([len(vector_t),] + list(matrix_psinorm.shape)))
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, tuple([len(vector_t),] + list(matrix_psinorm.shape)))
        # TODO: Fix boundary!
        # self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        # diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        # self.assertTrue((numpy.isnan(out) == numpy.isnan(out_t)).all())
        res2 = (out - out_t)**2
        diff = numpy.sqrt((res2[~numpy.isnan(res2)]).max()) / numpy.absolute(out[~numpy.isnan(out)]).max()
        self.assertLessEqual(diff, tol)

    def test_volnorm2phinorm_matrix_psinorm_vector_t_each_t_False(self):
        with self.assertRaises(ValueError):
            out = e.volnorm2phinorm(matrix_psinorm, vector_t, each_t=False)
        with self.assertRaises(ValueError):
            out_t = et.volnorm2phinorm(matrix_psinorm, vector_t, each_t=False)

    # matrix psinorm, matrix t:
    def test_volnorm2phinorm_matrix_psinorm_matrix_t_each_t_True(self):
        with self.assertRaises(ValueError):
            out = e.volnorm2phinorm(matrix_psinorm, matrix_t)
        with self.assertRaises(ValueError):
            out_t = et.volnorm2phinorm(matrix_psinorm, matrix_t)

    def test_volnorm2phinorm_matrix_psinorm_matrix_t_each_t_False(self):
        out = e.volnorm2phinorm(matrix_psinorm, matrix_t, each_t=False)
        out_t = et.volnorm2phinorm(matrix_psinorm, matrix_t, each_t=False)

        self.assertEqual(out.shape, matrix_psinorm.shape)
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, matrix_psinorm.shape)
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertTrue((numpy.isnan(out) == numpy.isnan(out_t)).all())
        # res2 = (out - out_t)**2
        # diff = numpy.sqrt((res2[~numpy.isnan(res2)]).max()) / numpy.absolute(out[~numpy.isnan(out)]).max()
        self.assertLessEqual(diff, tol)

class volnorm2rmidTestCase(unittest.TestCase):
    # The following tests focus primarily on the consistency of the input/output
    # shapes. More tests should be added to check the consistency between the
    # different mapping routines.

    # scalar psinorm, scalar t:
    def test_volnorm2rmid_scalar_psinorm_scalar_t_each_t_True_rho_False(self):
        out = e.volnorm2rmid(scalar_psinorm, scalar_t)
        out_t = et.volnorm2rmid(scalar_psinorm, scalar_t)

        with self.assertRaises(TypeError):
            iter(out)
        self.assertFalse(numpy.isnan(out))
        self.assertFalse(numpy.isinf(out))

        with self.assertRaises(TypeError):
            iter(out_t)
        self.assertFalse(numpy.isnan(out_t))
        self.assertFalse(numpy.isinf(out_t))

        diff = numpy.absolute(out - out_t) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    def test_volnorm2rmid_scalar_psinorm_scalar_t_each_t_False_rho_False(self):
        out = e.volnorm2rmid(scalar_psinorm, scalar_t, each_t=False)
        out_t = et.volnorm2rmid(scalar_psinorm, scalar_t, each_t=False)

        with self.assertRaises(TypeError):
            iter(out)
        self.assertFalse(numpy.isnan(out))
        self.assertFalse(numpy.isinf(out))

        with self.assertRaises(TypeError):
            iter(out_t)
        self.assertFalse(numpy.isnan(out_t))
        self.assertFalse(numpy.isinf(out_t))

        diff = numpy.absolute(out - out_t) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    def test_volnorm2rmid_scalar_psinorm_scalar_t_each_t_True_rho_True(self):
        out = e.volnorm2rmid(scalar_psinorm, scalar_t, rho=True)
        out_t = et.volnorm2rmid(scalar_psinorm, scalar_t, rho=True)

        with self.assertRaises(TypeError):
            iter(out)
        self.assertFalse(numpy.isnan(out))
        self.assertFalse(numpy.isinf(out))

        with self.assertRaises(TypeError):
            iter(out_t)
        self.assertFalse(numpy.isnan(out_t))
        self.assertFalse(numpy.isinf(out_t))

        diff = numpy.absolute(out - out_t) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    def test_volnorm2rmid_scalar_psinorm_scalar_t_each_t_False_rho_True(self):
        out = e.volnorm2rmid(scalar_psinorm, scalar_t, each_t=False, rho=True)
        out_t = et.volnorm2rmid(scalar_psinorm, scalar_t, each_t=False, rho=True)

        with self.assertRaises(TypeError):
            iter(out)
        self.assertFalse(numpy.isnan(out))
        self.assertFalse(numpy.isinf(out))

        with self.assertRaises(TypeError):
            iter(out_t)
        self.assertFalse(numpy.isnan(out_t))
        self.assertFalse(numpy.isinf(out_t))

        diff = numpy.absolute(out - out_t) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    # vector psinorm, scalar t:
    def test_volnorm2rmid_vector_psinorm_scalar_t_each_t_True_rho_False(self):
        out = e.volnorm2rmid(vector_psinorm, scalar_t)
        out_t = et.volnorm2rmid(vector_psinorm, scalar_t)

        self.assertEqual(out.shape, (len(vector_psinorm),))
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, (len(vector_psinorm),))
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        # diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertTrue((numpy.isnan(out) == numpy.isnan(out_t)).all())
        res2 = (out - out_t)**2
        diff = numpy.sqrt((res2[~numpy.isnan(res2)]).max()) / numpy.absolute(out[~numpy.isnan(out)]).max()
        self.assertLessEqual(diff, tol)

    def test_volnorm2rmid_vector_psinorm_scalar_t_each_t_False_rho_False(self):
        out = e.volnorm2rmid(vector_psinorm, scalar_t, each_t=False)
        out_t = et.volnorm2rmid(vector_psinorm, scalar_t, each_t=False)

        self.assertEqual(out.shape, (len(vector_psinorm),))
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, (len(vector_psinorm),))
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        # diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertTrue((numpy.isnan(out) == numpy.isnan(out_t)).all())
        res2 = (out - out_t)**2
        diff = numpy.sqrt((res2[~numpy.isnan(res2)]).max()) / numpy.absolute(out[~numpy.isnan(out)]).max()
        self.assertLessEqual(diff, tol)

    def test_volnorm2rmid_vector_psinorm_scalar_t_each_t_True_rho_True(self):
        out = e.volnorm2rmid(vector_psinorm, scalar_t, rho=True)
        out_t = et.volnorm2rmid(vector_psinorm, scalar_t, rho=True)

        self.assertEqual(out.shape, (len(vector_psinorm),))
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, (len(vector_psinorm),))
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        # diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertTrue((numpy.isnan(out) == numpy.isnan(out_t)).all())
        res2 = (out - out_t)**2
        diff = numpy.sqrt((res2[~numpy.isnan(res2)]).max()) / numpy.absolute(out[~numpy.isnan(out)]).max()
        self.assertLessEqual(diff, tol)

    def test_volnorm2rmid_vector_psinorm_scalar_t_each_t_False_rho_True(self):
        out = e.volnorm2rmid(vector_psinorm, scalar_t, each_t=False, rho=True)
        out_t = et.volnorm2rmid(vector_psinorm, scalar_t, each_t=False, rho=True)

        self.assertEqual(out.shape, (len(vector_psinorm),))
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, (len(vector_psinorm),))
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        # diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertTrue((numpy.isnan(out) == numpy.isnan(out_t)).all())
        res2 = (out - out_t)**2
        diff = numpy.sqrt((res2[~numpy.isnan(res2)]).max()) / numpy.absolute(out[~numpy.isnan(out)]).max()
        self.assertLessEqual(diff, tol)

    # matrix psinorm, scalar t:
    def test_volnorm2rmid_matrix_psinorm_scalar_t_each_t_True_rho_False(self):
        out = e.volnorm2rmid(matrix_psinorm, scalar_t)
        out_t = et.volnorm2rmid(matrix_psinorm, scalar_t)

        self.assertEqual(out.shape, matrix_psinorm.shape)
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, matrix_psinorm.shape)
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        # diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertTrue((numpy.isnan(out) == numpy.isnan(out_t)).all())
        res2 = (out - out_t)**2
        diff = numpy.sqrt((res2[~numpy.isnan(res2)]).max()) / numpy.absolute(out[~numpy.isnan(out)]).max()
        self.assertLessEqual(diff, tol)

    def test_volnorm2rmid_matrix_psinorm_scalar_t_each_t_False_rho_False(self):
        out = e.volnorm2rmid(matrix_psinorm, scalar_t, each_t=False)
        out_t = et.volnorm2rmid(matrix_psinorm, scalar_t, each_t=False)

        self.assertEqual(out.shape, matrix_psinorm.shape)
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, matrix_psinorm.shape)
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        # diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertTrue((numpy.isnan(out) == numpy.isnan(out_t)).all())
        res2 = (out - out_t)**2
        diff = numpy.sqrt((res2[~numpy.isnan(res2)]).max()) / numpy.absolute(out[~numpy.isnan(out)]).max()
        self.assertLessEqual(diff, tol)

    def test_volnorm2rmid_matrix_psinorm_scalar_t_each_t_True_rho_True(self):
        out = e.volnorm2rmid(matrix_psinorm, scalar_t, rho=True)
        out_t = et.volnorm2rmid(matrix_psinorm, scalar_t, rho=True)

        self.assertEqual(out.shape, matrix_psinorm.shape)
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, matrix_psinorm.shape)
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        # diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertTrue((numpy.isnan(out) == numpy.isnan(out_t)).all())
        res2 = (out - out_t)**2
        diff = numpy.sqrt((res2[~numpy.isnan(res2)]).max()) / numpy.absolute(out[~numpy.isnan(out)]).max()
        self.assertLessEqual(diff, tol)

    def test_volnorm2rmid_matrix_psinorm_scalar_t_each_t_False_rho_True(self):
        out = e.volnorm2rmid(matrix_psinorm, scalar_t, each_t=False, rho=True)
        out_t = et.volnorm2rmid(matrix_psinorm, scalar_t, each_t=False, rho=True)

        self.assertEqual(out.shape, matrix_psinorm.shape)
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, matrix_psinorm.shape)
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        # diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertTrue((numpy.isnan(out) == numpy.isnan(out_t)).all())
        res2 = (out - out_t)**2
        diff = numpy.sqrt((res2[~numpy.isnan(res2)]).max()) / numpy.absolute(out[~numpy.isnan(out)]).max()
        self.assertLessEqual(diff, tol)

    # scalar psinorm, vector t:
    def test_volnorm2rmid_scalar_psinorm_vector_t_each_t_True_rho_False(self):
        out = e.volnorm2rmid(scalar_psinorm, vector_t)
        out_t = et.volnorm2rmid(scalar_psinorm, vector_t)

        self.assertEqual(out.shape, (len(vector_t),))
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, (len(vector_t),))
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    def test_volnorm2rmid_scalar_psinorm_vector_t_each_t_False_rho_False(self):
        with self.assertRaises(ValueError):
            out = e.volnorm2rmid(scalar_psinorm, vector_t, each_t=False)
        with self.assertRaises(ValueError):
            out_t = et.volnorm2rmid(scalar_psinorm, vector_t, each_t=False)

    def test_volnorm2rmid_scalar_psinorm_vector_t_each_t_True_rho_True(self):
        out = e.volnorm2rmid(scalar_psinorm, vector_t, rho=True)
        out_t = et.volnorm2rmid(scalar_psinorm, vector_t, rho=True)

        self.assertEqual(out.shape, (len(vector_t),))
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, (len(vector_t),))
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    def test_volnorm2rmid_scalar_psinorm_vector_t_each_t_False_rho_True(self):
        with self.assertRaises(ValueError):
            out = e.volnorm2rmid(scalar_psinorm, vector_t, each_t=False, rho=True)
        with self.assertRaises(ValueError):
            out_t = et.volnorm2rmid(scalar_psinorm, vector_t, each_t=False, rho=True)

    # vector psinorm, vector t:
    def test_volnorm2rmid_vector_psinorm_vector_t_each_t_True_rho_False(self):
        out = e.volnorm2rmid(vector_psinorm, vector_t)
        out_t = et.volnorm2rmid(vector_psinorm, vector_t)

        self.assertEqual(out.shape, (len(vector_t), len(vector_psinorm)))
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, (len(vector_t), len(vector_psinorm)))
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertTrue((numpy.isnan(out) == numpy.isnan(out_t)).all())
        # res2 = (out - out_t)**2
        # diff = numpy.sqrt((res2[~numpy.isnan(res2)]).max()) / numpy.absolute(out[~numpy.isnan(out)]).max()
        self.assertLessEqual(diff, tol)

    def test_volnorm2rmid_vector_psinorm_vector_t_each_t_False_rho_False(self):
        with self.assertRaises(ValueError):
            out = e.volnorm2rmid(vector_psinorm, vector_t, each_t=False)
        with self.assertRaises(ValueError):
            out_t = et.volnorm2rmid(vector_psinorm, vector_t, each_t=False)

    def test_volnorm2rmid_vector_psinorm_vector_t_each_t_True_rho_True(self):
        out = e.volnorm2rmid(vector_psinorm, vector_t, rho=True)
        out_t = et.volnorm2rmid(vector_psinorm, vector_t, rho=True)

        self.assertEqual(out.shape, (len(vector_t), len(vector_psinorm)))
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, (len(vector_t), len(vector_psinorm)))
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertTrue((numpy.isnan(out) == numpy.isnan(out_t)).all())
        # res2 = (out - out_t)**2
        # diff = numpy.sqrt((res2[~numpy.isnan(res2)]).max()) / numpy.absolute(out[~numpy.isnan(out)]).max()
        self.assertLessEqual(diff, tol)

    def test_volnorm2rmid_vector_psinorm_vector_t_each_t_False_rho_True(self):
        with self.assertRaises(ValueError):
            out = e.volnorm2rmid(vector_psinorm, vector_t, each_t=False, rho=True)
        with self.assertRaises(ValueError):
            out_t = et.volnorm2rmid(vector_psinorm, vector_t, each_t=False, rho=True)

    # matrix psinorm, vector t:
    def test_volnorm2rmid_matrix_psinorm_vector_t_each_t_True_rho_False(self):
        out = e.volnorm2rmid(matrix_psinorm, vector_t)
        out_t = et.volnorm2rmid(matrix_psinorm, vector_t)

        self.assertEqual(out.shape, tuple([len(vector_t),] + list(matrix_psinorm.shape)))
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, tuple([len(vector_t),] + list(matrix_psinorm.shape)))
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertTrue((numpy.isnan(out) == numpy.isnan(out_t)).all())
        # res2 = (out - out_t)**2
        # diff = numpy.sqrt((res2[~numpy.isnan(res2)]).max()) / numpy.absolute(out[~numpy.isnan(out)]).max()
        self.assertLessEqual(diff, tol)

    def test_volnorm2rmid_matrix_psinorm_vector_t_each_t_False_rho_False(self):
        with self.assertRaises(ValueError):
            out = e.volnorm2rmid(matrix_psinorm, vector_t, each_t=False)
        with self.assertRaises(ValueError):
            out_t = et.volnorm2rmid(matrix_psinorm, vector_t, each_t=False)

    def test_volnorm2rmid_matrix_psinorm_vector_t_each_t_True_rho_True(self):
        out = e.volnorm2rmid(matrix_psinorm, vector_t, rho=True)
        out_t = et.volnorm2rmid(matrix_psinorm, vector_t, rho=True)

        self.assertEqual(out.shape, tuple([len(vector_t),] + list(matrix_psinorm.shape)))
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, tuple([len(vector_t),] + list(matrix_psinorm.shape)))
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertTrue((numpy.isnan(out) == numpy.isnan(out_t)).all())
        # res2 = (out - out_t)**2
        # diff = numpy.sqrt((res2[~numpy.isnan(res2)]).max()) / numpy.absolute(out[~numpy.isnan(out)]).max()
        self.assertLessEqual(diff, tol)

    def test_volnorm2rmid_matrix_psinorm_vector_t_each_t_False_rho_True(self):
        with self.assertRaises(ValueError):
            out = e.volnorm2rmid(matrix_psinorm, vector_t, each_t=False, rho=True)
        with self.assertRaises(ValueError):
            out_t = et.volnorm2rmid(matrix_psinorm, vector_t, each_t=False, rho=True)

    # matrix psinorm, matrix t:
    def test_volnorm2rmid_matrix_psinorm_matrix_t_each_t_True_rho_False(self):
        with self.assertRaises(ValueError):
            out = e.volnorm2rmid(matrix_psinorm, matrix_t)
        with self.assertRaises(ValueError):
            out_t = et.volnorm2rmid(matrix_psinorm, matrix_t)

    def test_volnorm2rmid_matrix_psinorm_matrix_t_each_t_False_rho_False(self):
        out = e.volnorm2rmid(matrix_psinorm, matrix_t, each_t=False)
        out_t = et.volnorm2rmid(matrix_psinorm, matrix_t, each_t=False)

        self.assertEqual(out.shape, matrix_psinorm.shape)
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, matrix_psinorm.shape)
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        # diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertTrue((numpy.isnan(out) == numpy.isnan(out_t)).all())
        res2 = (out - out_t)**2
        diff = numpy.sqrt((res2[~numpy.isnan(res2)]).max()) / numpy.absolute(out[~numpy.isnan(out)]).max()
        self.assertLessEqual(diff, tol)

    def test_volnorm2rmid_matrix_psinorm_matrix_t_each_t_True_rho_True(self):
        with self.assertRaises(ValueError):
            out = e.volnorm2rmid(matrix_psinorm, matrix_t, rho=True)
        with self.assertRaises(ValueError):
            out_t = et.volnorm2rmid(matrix_psinorm, matrix_t, rho=True)

    def test_volnorm2rmid_matrix_psinorm_matrix_t_each_t_False_rho_True(self):
        out = e.volnorm2rmid(matrix_psinorm, matrix_t, each_t=False, rho=True)
        out_t = et.volnorm2rmid(matrix_psinorm, matrix_t, each_t=False, rho=True)

        self.assertEqual(out.shape, matrix_psinorm.shape)
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, matrix_psinorm.shape)
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        # diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertTrue((numpy.isnan(out) == numpy.isnan(out_t)).all())
        res2 = (out - out_t)**2
        diff = numpy.sqrt((res2[~numpy.isnan(res2)]).max()) / numpy.absolute(out[~numpy.isnan(out)]).max()
        self.assertLessEqual(diff, tol)

class volnorm2roaTestCase(unittest.TestCase):
    # The following tests focus primarily on the consistency of the input/output
    # shapes. More tests should be added to check the consistency between the
    # different mapping routines.

    # scalar psinorm, scalar t:
    def test_volnorm2roa_scalar_psinorm_scalar_t_each_t_True(self):
        out = e.volnorm2roa(scalar_psinorm, scalar_t)
        out_t = et.volnorm2roa(scalar_psinorm, scalar_t)

        with self.assertRaises(TypeError):
            iter(out)
        self.assertFalse(numpy.isnan(out))
        self.assertFalse(numpy.isinf(out))

        with self.assertRaises(TypeError):
            iter(out_t)
        self.assertFalse(numpy.isnan(out_t))
        self.assertFalse(numpy.isinf(out_t))

        diff = numpy.absolute(out - out_t) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    def test_volnorm2roa_scalar_psinorm_scalar_t_each_t_False(self):
        out = e.volnorm2roa(scalar_psinorm, scalar_t, each_t=False)
        out_t = et.volnorm2roa(scalar_psinorm, scalar_t, each_t=False)

        with self.assertRaises(TypeError):
            iter(out)
        self.assertFalse(numpy.isnan(out))
        self.assertFalse(numpy.isinf(out))

        with self.assertRaises(TypeError):
            iter(out_t)
        self.assertFalse(numpy.isnan(out_t))
        self.assertFalse(numpy.isinf(out_t))

        diff = numpy.absolute(out - out_t) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    # vector psinorm, scalar t:
    def test_volnorm2roa_vector_psinorm_scalar_t_each_t_True(self):
        out = e.volnorm2roa(vector_psinorm, scalar_t)
        out_t = et.volnorm2roa(vector_psinorm, scalar_t)

        self.assertEqual(out.shape, (len(vector_psinorm),))
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, (len(vector_psinorm),))
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        # diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertTrue((numpy.isnan(out) == numpy.isnan(out_t)).all())
        res2 = (out - out_t)**2
        diff = numpy.sqrt((res2[~numpy.isnan(res2)]).max()) / numpy.absolute(out[~numpy.isnan(out)]).max()
        self.assertLessEqual(diff, tol)

    def test_volnorm2roa_vector_psinorm_scalar_t_each_t_False(self):
        out = e.volnorm2roa(vector_psinorm, scalar_t, each_t=False)
        out_t = et.volnorm2roa(vector_psinorm, scalar_t, each_t=False)

        self.assertEqual(out.shape, (len(vector_psinorm),))
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, (len(vector_psinorm),))
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        # diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertTrue((numpy.isnan(out) == numpy.isnan(out_t)).all())
        res2 = (out - out_t)**2
        diff = numpy.sqrt((res2[~numpy.isnan(res2)]).max()) / numpy.absolute(out[~numpy.isnan(out)]).max()
        self.assertLessEqual(diff, tol)

    # matrix psinorm, scalar t:
    def test_volnorm2roa_matrix_psinorm_scalar_t_each_t_True(self):
        out = e.volnorm2roa(matrix_psinorm, scalar_t)
        out_t = et.volnorm2roa(matrix_psinorm, scalar_t)

        self.assertEqual(out.shape, matrix_psinorm.shape)
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, matrix_psinorm.shape)
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        # diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertTrue((numpy.isnan(out) == numpy.isnan(out_t)).all())
        res2 = (out - out_t)**2
        diff = numpy.sqrt((res2[~numpy.isnan(res2)]).max()) / numpy.absolute(out[~numpy.isnan(out)]).max()
        self.assertLessEqual(diff, tol)

    def test_volnorm2roa_matrix_psinorm_scalar_t_each_t_False(self):
        out = e.volnorm2roa(matrix_psinorm, scalar_t, each_t=False)
        out_t = et.volnorm2roa(matrix_psinorm, scalar_t, each_t=False)

        self.assertEqual(out.shape, matrix_psinorm.shape)
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, matrix_psinorm.shape)
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        # diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertTrue((numpy.isnan(out) == numpy.isnan(out_t)).all())
        res2 = (out - out_t)**2
        diff = numpy.sqrt((res2[~numpy.isnan(res2)]).max()) / numpy.absolute(out[~numpy.isnan(out)]).max()
        self.assertLessEqual(diff, tol)

    # scalar psinorm, vector t:
    def test_volnorm2roa_scalar_psinorm_vector_t_each_t_True(self):
        out = e.volnorm2roa(scalar_psinorm, vector_t)
        out_t = et.volnorm2roa(scalar_psinorm, vector_t)

        self.assertEqual(out.shape, (len(vector_t),))
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, (len(vector_t),))
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertLessEqual(diff, tol)

    def test_volnorm2roa_scalar_psinorm_vector_t_each_t_False(self):
        with self.assertRaises(ValueError):
            out = e.volnorm2roa(scalar_psinorm, vector_t, each_t=False)
        with self.assertRaises(ValueError):
            out_t = et.volnorm2roa(scalar_psinorm, vector_t, each_t=False)

    # vector psinorm, vector t:
    def test_volnorm2roa_vector_psinorm_vector_t_each_t_True(self):
        out = e.volnorm2roa(vector_psinorm, vector_t)
        out_t = et.volnorm2roa(vector_psinorm, vector_t)

        self.assertEqual(out.shape, (len(vector_t), len(vector_psinorm)))
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, (len(vector_t), len(vector_psinorm)))
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertTrue((numpy.isnan(out) == numpy.isnan(out_t)).all())
        # res2 = (out - out_t)**2
        # diff = numpy.sqrt((res2[~numpy.isnan(res2)]).max()) / numpy.absolute(out[~numpy.isnan(out)]).max()
        self.assertLessEqual(diff, tol)

    def test_volnorm2roa_vector_psinorm_vector_t_each_t_False(self):
        with self.assertRaises(ValueError):
            out = e.volnorm2roa(vector_psinorm, vector_t, each_t=False)
        with self.assertRaises(ValueError):
            out_t = et.volnorm2roa(vector_psinorm, vector_t, each_t=False)

    # matrix psinorm, vector t:
    def test_volnorm2roa_matrix_psinorm_vector_t_each_t_True(self):
        out = e.volnorm2roa(matrix_psinorm, vector_t)
        out_t = et.volnorm2roa(matrix_psinorm, vector_t)

        self.assertEqual(out.shape, tuple([len(vector_t),] + list(matrix_psinorm.shape)))
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, tuple([len(vector_t),] + list(matrix_psinorm.shape)))
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        # diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertTrue((numpy.isnan(out) == numpy.isnan(out_t)).all())
        res2 = (out - out_t)**2
        diff = numpy.sqrt((res2[~numpy.isnan(res2)]).max()) / numpy.absolute(out[~numpy.isnan(out)]).max()
        self.assertLessEqual(diff, tol)

    def test_volnorm2roa_matrix_psinorm_vector_t_each_t_False(self):
        with self.assertRaises(ValueError):
            out = e.volnorm2roa(matrix_psinorm, vector_t, each_t=False)
        with self.assertRaises(ValueError):
            out_t = et.volnorm2roa(matrix_psinorm, vector_t, each_t=False)

    # matrix psinorm, matrix t:
    def test_volnorm2roa_matrix_psinorm_matrix_t_each_t_True(self):
        with self.assertRaises(ValueError):
            out = e.volnorm2roa(matrix_psinorm, matrix_t)
        with self.assertRaises(ValueError):
            out_t = et.volnorm2roa(matrix_psinorm, matrix_t)

    def test_volnorm2roa_matrix_psinorm_matrix_t_each_t_False(self):
        out = e.volnorm2roa(matrix_psinorm, matrix_t, each_t=False)
        out_t = et.volnorm2roa(matrix_psinorm, matrix_t, each_t=False)

        self.assertEqual(out.shape, matrix_psinorm.shape)
        self.assertFalse(numpy.isnan(out).any())
        self.assertFalse(numpy.isinf(out).any())

        self.assertEqual(out_t.shape, matrix_psinorm.shape)
        self.assertFalse(numpy.isnan(out_t).any())
        self.assertFalse(numpy.isinf(out_t).any())

        # diff = numpy.sqrt(((out - out_t)**2).max()) / numpy.absolute(out).max()
        self.assertTrue((numpy.isnan(out) == numpy.isnan(out_t)).all())
        res2 = (out - out_t)**2
        diff = numpy.sqrt((res2[~numpy.isnan(res2)]).max()) / numpy.absolute(out[~numpy.isnan(out)]).max()
        self.assertLessEqual(diff, tol)

class SqrtTestCase(unittest.TestCase):
    """This is a regression test to make sure the bug in the sqrt=True case
    doesn't pop up again.
    """

    # RZ -> *
    # TODO: rz2psi doesn't support sqrt (probably OK...)
    # def test_rz2psi_sqrt(self):
    #     self.assertAlmostEqual(
    #         e.rz2psi(scalar_R, scalar_Z, scalar_t),
    #         e.rz2psi(scalar_R, scalar_Z, scalar_t, sqrt=True)**2,
    #     )
    #     self.assertAlmostEqual(
    #         et.rz2psi(scalar_R, scalar_Z, scalar_t),
    #         et.rz2psi(scalar_R, scalar_Z, scalar_t, sqrt=True)**2,
    #     )

    def test_rz2psinorm_sqrt(self):
        self.assertAlmostEqual(
            e.rz2psinorm(scalar_R, scalar_Z, scalar_t),
            e.rz2psinorm(scalar_R, scalar_Z, scalar_t, sqrt=True)**2,
        )
        self.assertAlmostEqual(
            et.rz2psinorm(scalar_R, scalar_Z, scalar_t),
            et.rz2psinorm(scalar_R, scalar_Z, scalar_t, sqrt=True)**2,
        )

    def test_rz2phinorm_sqrt(self):
        self.assertAlmostEqual(
            e.rz2phinorm(scalar_R, scalar_Z, scalar_t),
            e.rz2phinorm(scalar_R, scalar_Z, scalar_t, sqrt=True)**2,
        )
        self.assertAlmostEqual(
            et.rz2phinorm(scalar_R, scalar_Z, scalar_t),
            et.rz2phinorm(scalar_R, scalar_Z, scalar_t, sqrt=True)**2,
        )

    def test_rz2volnorm_sqrt(self):
        self.assertAlmostEqual(
            e.rz2volnorm(scalar_R, scalar_Z, scalar_t),
            e.rz2volnorm(scalar_R, scalar_Z, scalar_t, sqrt=True)**2,
        )
        self.assertAlmostEqual(
            et.rz2volnorm(scalar_R, scalar_Z, scalar_t),
            et.rz2volnorm(scalar_R, scalar_Z, scalar_t, sqrt=True)**2,
        )

    def test_rz2rmid_sqrt_rho_False(self):
        self.assertAlmostEqual(
            e.rz2rmid(scalar_R, scalar_Z, scalar_t),
            e.rz2rmid(scalar_R, scalar_Z, scalar_t, sqrt=True)**2,
        )
        self.assertAlmostEqual(
            et.rz2rmid(scalar_R, scalar_Z, scalar_t),
            et.rz2rmid(scalar_R, scalar_Z, scalar_t, sqrt=True)**2,
        )

    def test_rz2rmid_sqrt_rho_True(self):
        self.assertAlmostEqual(
            e.rz2rmid(scalar_R, scalar_Z, scalar_t, rho=True),
            e.rz2rmid(scalar_R, scalar_Z, scalar_t, rho=True, sqrt=True)**2,
        )
        self.assertAlmostEqual(
            et.rz2rmid(scalar_R, scalar_Z, scalar_t, rho=True),
            et.rz2rmid(scalar_R, scalar_Z, scalar_t, rho=True, sqrt=True)**2,
        )

    def test_rz2roa_sqrt(self):
        self.assertAlmostEqual(
            e.rz2roa(scalar_R, scalar_Z, scalar_t),
            e.rz2roa(scalar_R, scalar_Z, scalar_t, sqrt=True)**2,
        )
        self.assertAlmostEqual(
            et.rz2roa(scalar_R, scalar_Z, scalar_t),
            et.rz2roa(scalar_R, scalar_Z, scalar_t, sqrt=True)**2,
        )

    # psinorm -> *
    def test_psinorm2phinorm_sqrt(self):
        self.assertAlmostEqual(
            e.psinorm2phinorm(scalar_psinorm, scalar_t),
            e.psinorm2phinorm(scalar_psinorm, scalar_t, sqrt=True)**2,
        )
        self.assertAlmostEqual(
            et.psinorm2phinorm(scalar_psinorm, scalar_t),
            et.psinorm2phinorm(scalar_psinorm, scalar_t, sqrt=True)**2,
        )

    def test_psinorm2volnorm_sqrt(self):
        self.assertAlmostEqual(
            e.psinorm2volnorm(scalar_psinorm, scalar_t),
            e.psinorm2volnorm(scalar_psinorm, scalar_t, sqrt=True)**2,
        )
        self.assertAlmostEqual(
            et.psinorm2volnorm(scalar_psinorm, scalar_t),
            et.psinorm2volnorm(scalar_psinorm, scalar_t, sqrt=True)**2,
        )

    def test_psinorm2rmid_sqrt_rho_False(self):
        self.assertAlmostEqual(
            e.psinorm2rmid(scalar_psinorm, scalar_t),
            e.psinorm2rmid(scalar_psinorm, scalar_t, sqrt=True)**2,
        )
        self.assertAlmostEqual(
            et.psinorm2rmid(scalar_psinorm, scalar_t),
            et.psinorm2rmid(scalar_psinorm, scalar_t, sqrt=True)**2,
        )

    def test_psinorm2rmid_sqrt_rho_True(self):
        self.assertAlmostEqual(
            e.psinorm2rmid(scalar_psinorm, scalar_t, rho=True),
            e.psinorm2rmid(scalar_psinorm, scalar_t, rho=True, sqrt=True)**2,
        )
        self.assertAlmostEqual(
            et.psinorm2rmid(scalar_psinorm, scalar_t, rho=True),
            et.psinorm2rmid(scalar_psinorm, scalar_t, rho=True, sqrt=True)**2,
        )

    def test_psinorm2roa_sqrt(self):
        self.assertAlmostEqual(
            e.psinorm2roa(scalar_psinorm, scalar_t),
            e.psinorm2roa(scalar_psinorm, scalar_t, sqrt=True)**2,
        )
        self.assertAlmostEqual(
            et.psinorm2roa(scalar_psinorm, scalar_t),
            et.psinorm2roa(scalar_psinorm, scalar_t, sqrt=True)**2,
        )

    # phinorm -> *
    def test_phinorm2psinorm_sqrt(self):
        self.assertAlmostEqual(
            e.phinorm2psinorm(scalar_psinorm, scalar_t),
            e.phinorm2psinorm(scalar_psinorm, scalar_t, sqrt=True)**2,
        )
        self.assertAlmostEqual(
            et.phinorm2psinorm(scalar_psinorm, scalar_t),
            et.phinorm2psinorm(scalar_psinorm, scalar_t, sqrt=True)**2,
        )

    def test_phinorm2volnorm_sqrt(self):
        self.assertAlmostEqual(
            e.phinorm2volnorm(scalar_psinorm, scalar_t),
            e.phinorm2volnorm(scalar_psinorm, scalar_t, sqrt=True)**2,
        )
        self.assertAlmostEqual(
            et.phinorm2volnorm(scalar_psinorm, scalar_t),
            et.phinorm2volnorm(scalar_psinorm, scalar_t, sqrt=True)**2,
        )

    def test_phinorm2rmid_sqrt_rho_False(self):
        self.assertAlmostEqual(
            e.phinorm2rmid(scalar_psinorm, scalar_t),
            e.phinorm2rmid(scalar_psinorm, scalar_t, sqrt=True)**2,
        )
        self.assertAlmostEqual(
            et.phinorm2rmid(scalar_psinorm, scalar_t),
            et.phinorm2rmid(scalar_psinorm, scalar_t, sqrt=True)**2,
        )

    def test_phinorm2rmid_sqrt_rho_True(self):
        self.assertAlmostEqual(
            e.phinorm2rmid(scalar_psinorm, scalar_t, rho=True),
            e.phinorm2rmid(scalar_psinorm, scalar_t, rho=True, sqrt=True)**2,
        )
        self.assertAlmostEqual(
            et.phinorm2rmid(scalar_psinorm, scalar_t, rho=True),
            et.phinorm2rmid(scalar_psinorm, scalar_t, rho=True, sqrt=True)**2,
        )

    def test_phinorm2roa_sqrt(self):
        self.assertAlmostEqual(
            e.phinorm2roa(scalar_psinorm, scalar_t),
            e.phinorm2roa(scalar_psinorm, scalar_t, sqrt=True)**2,
        )
        self.assertAlmostEqual(
            et.phinorm2roa(scalar_psinorm, scalar_t),
            et.phinorm2roa(scalar_psinorm, scalar_t, sqrt=True)**2,
        )

    # volnorm -> *
    def test_volnorm2psinorm_sqrt(self):
        self.assertAlmostEqual(
            e.volnorm2psinorm(scalar_psinorm, scalar_t),
            e.volnorm2psinorm(scalar_psinorm, scalar_t, sqrt=True)**2,
        )
        self.assertAlmostEqual(
            et.volnorm2psinorm(scalar_psinorm, scalar_t),
            et.volnorm2psinorm(scalar_psinorm, scalar_t, sqrt=True)**2,
        )

    def test_volnorm2phinorm_sqrt(self):
        self.assertAlmostEqual(
            e.volnorm2phinorm(scalar_psinorm, scalar_t),
            e.volnorm2phinorm(scalar_psinorm, scalar_t, sqrt=True)**2,
        )
        self.assertAlmostEqual(
            et.volnorm2phinorm(scalar_psinorm, scalar_t),
            et.volnorm2phinorm(scalar_psinorm, scalar_t, sqrt=True)**2,
        )

    def test_volnorm2rmid_sqrt_rho_False(self):
        self.assertAlmostEqual(
            e.volnorm2rmid(scalar_psinorm, scalar_t),
            e.volnorm2rmid(scalar_psinorm, scalar_t, sqrt=True)**2,
        )
        self.assertAlmostEqual(
            et.volnorm2rmid(scalar_psinorm, scalar_t),
            et.volnorm2rmid(scalar_psinorm, scalar_t, sqrt=True)**2,
        )

    def test_volnorm2rmid_sqrt_rho_True(self):
        self.assertAlmostEqual(
            e.volnorm2rmid(scalar_psinorm, scalar_t, rho=True),
            e.volnorm2rmid(scalar_psinorm, scalar_t, rho=True, sqrt=True)**2,
        )
        self.assertAlmostEqual(
            et.volnorm2rmid(scalar_psinorm, scalar_t, rho=True),
            et.volnorm2rmid(scalar_psinorm, scalar_t, rho=True, sqrt=True)**2,
        )

    def test_volnorm2roa_sqrt(self):
        self.assertAlmostEqual(
            e.volnorm2roa(scalar_psinorm, scalar_t),
            e.volnorm2roa(scalar_psinorm, scalar_t, sqrt=True)**2,
        )
        self.assertAlmostEqual(
            et.volnorm2roa(scalar_psinorm, scalar_t),
            et.volnorm2roa(scalar_psinorm, scalar_t, sqrt=True)**2,
        )

    # Rmid -> *
    def test_rmid2psinorm_sqrt(self):
        self.assertAlmostEqual(
            e.rmid2psinorm(scalar_Rmid, scalar_t),
            e.rmid2psinorm(scalar_Rmid, scalar_t, sqrt=True)**2,
        )
        self.assertAlmostEqual(
            et.rmid2psinorm(scalar_Rmid, scalar_t),
            et.rmid2psinorm(scalar_Rmid, scalar_t, sqrt=True)**2,
        )

    def test_rmid2phinorm_sqrt(self):
        self.assertAlmostEqual(
            e.rmid2phinorm(scalar_Rmid, scalar_t),
            e.rmid2phinorm(scalar_Rmid, scalar_t, sqrt=True)**2,
        )
        self.assertAlmostEqual(
            et.rmid2phinorm(scalar_Rmid, scalar_t),
            et.rmid2phinorm(scalar_Rmid, scalar_t, sqrt=True)**2,
        )

    def test_rmid2volnorm_sqrt(self):
        self.assertAlmostEqual(
            e.rmid2volnorm(scalar_Rmid, scalar_t),
            e.rmid2volnorm(scalar_Rmid, scalar_t, sqrt=True)**2,
        )
        self.assertAlmostEqual(
            et.rmid2volnorm(scalar_Rmid, scalar_t),
            et.rmid2volnorm(scalar_Rmid, scalar_t, sqrt=True)**2,
        )

    def test_rmid2roa_sqrt(self):
        self.assertAlmostEqual(
            e.rmid2roa(scalar_Rmid, scalar_t),
            e.rmid2roa(scalar_Rmid, scalar_t, sqrt=True)**2,
        )
        self.assertAlmostEqual(
            et.rmid2roa(scalar_Rmid, scalar_t),
            et.rmid2roa(scalar_Rmid, scalar_t, sqrt=True)**2,
        )

    # roa -> *
    def test_roa2psinorm_sqrt(self):
        self.assertAlmostEqual(
            e.roa2psinorm(scalar_psinorm, scalar_t),
            e.roa2psinorm(scalar_psinorm, scalar_t, sqrt=True)**2,
        )
        self.assertAlmostEqual(
            et.roa2psinorm(scalar_psinorm, scalar_t),
            et.roa2psinorm(scalar_psinorm, scalar_t, sqrt=True)**2,
        )

    def test_roa2phinorm_sqrt(self):
        self.assertAlmostEqual(
            e.roa2phinorm(scalar_psinorm, scalar_t),
            e.roa2phinorm(scalar_psinorm, scalar_t, sqrt=True)**2,
        )
        self.assertAlmostEqual(
            et.roa2phinorm(scalar_psinorm, scalar_t),
            et.roa2phinorm(scalar_psinorm, scalar_t, sqrt=True)**2,
        )

    def test_roa2volnorm_sqrt(self):
        self.assertAlmostEqual(
            e.roa2volnorm(scalar_psinorm, scalar_t),
            e.roa2volnorm(scalar_psinorm, scalar_t, sqrt=True)**2,
        )
        self.assertAlmostEqual(
            et.roa2volnorm(scalar_psinorm, scalar_t),
            et.roa2volnorm(scalar_psinorm, scalar_t, sqrt=True)**2,
        )

    # TODO: Does not support sqrt keyword (probably doesn't need to, but should
    # think about it...)
    # def test_roa2rmid_sqrt_rho_False(self):
    #     self.assertAlmostEqual(
    #         e.roa2rmid(scalar_psinorm, scalar_t),
    #         e.roa2rmid(scalar_psinorm, scalar_t, sqrt=True)**2,
    #     )
    #     self.assertAlmostEqual(
    #         et.roa2rmid(scalar_psinorm, scalar_t),
    #         et.roa2rmid(scalar_psinorm, scalar_t, sqrt=True)**2,
    #     )

class PsinormConsistencyTestCase(unittest.TestCase):
    def test_psinorm_to_phinorm(self):
        self.assertAlmostEqual(
            e.phinorm2psinorm(e.psinorm2phinorm(scalar_psinorm, scalar_t), scalar_t),
            scalar_psinorm,
            places=places
        )
        self.assertAlmostEqual(
            et.phinorm2psinorm(et.psinorm2phinorm(scalar_psinorm, scalar_t), scalar_t),
            scalar_psinorm,
            places=places
        )

    def test_psinorm_to_volnorm(self):
        self.assertAlmostEqual(
            e.volnorm2psinorm(e.psinorm2volnorm(scalar_psinorm, scalar_t), scalar_t),
            scalar_psinorm,
            places=places
        )
        self.assertAlmostEqual(
            et.volnorm2psinorm(et.psinorm2volnorm(scalar_psinorm, scalar_t), scalar_t),
            scalar_psinorm,
            places=places
        )

    def test_psinorm_to_rmid_rho_False(self):
        self.assertAlmostEqual(
            e.rmid2psinorm(e.psinorm2rmid(scalar_psinorm, scalar_t), scalar_t),
            scalar_psinorm,
            places=places
        )
        self.assertAlmostEqual(
            et.rmid2psinorm(et.psinorm2rmid(scalar_psinorm, scalar_t), scalar_t),
            scalar_psinorm,
            places=places
        )

    def test_psinorm_to_rmid_rho_True(self):
        # Need to use roa2psinorm, since there isn't a mirror option.
        self.assertAlmostEqual(
            e.roa2psinorm(e.psinorm2rmid(scalar_psinorm, scalar_t, rho=True), scalar_t),
            scalar_psinorm,
            places=places
        )
        self.assertAlmostEqual(
            et.roa2psinorm(et.psinorm2rmid(scalar_psinorm, scalar_t, rho=True), scalar_t),
            scalar_psinorm,
            places=places
        )

    def test_psinorm_to_roa(self):
        self.assertAlmostEqual(
            e.roa2psinorm(e.psinorm2roa(scalar_psinorm, scalar_t), scalar_t),
            scalar_psinorm,
            places=places
        )
        self.assertAlmostEqual(
            et.roa2psinorm(et.psinorm2roa(scalar_psinorm, scalar_t), scalar_t),
            scalar_psinorm,
            places=places
        )

class PhinormConsistencyTestCase(unittest.TestCase):
    def test_phinorm_to_psinorm(self):
        self.assertAlmostEqual(
            e.psinorm2phinorm(e.phinorm2psinorm(scalar_psinorm, scalar_t), scalar_t),
            scalar_psinorm,
            places=places
        )
        self.assertAlmostEqual(
            et.psinorm2phinorm(et.phinorm2psinorm(scalar_psinorm, scalar_t), scalar_t),
            scalar_psinorm,
            places=places
        )

    def test_phinorm_to_volnorm(self):
        self.assertAlmostEqual(
            e.volnorm2phinorm(e.phinorm2volnorm(scalar_psinorm, scalar_t), scalar_t),
            scalar_psinorm,
            places=places
        )
        self.assertAlmostEqual(
            et.volnorm2phinorm(et.phinorm2volnorm(scalar_psinorm, scalar_t), scalar_t),
            scalar_psinorm,
            places=places
        )

    def test_phinorm_to_rmid_rho_False(self):
        self.assertAlmostEqual(
            e.rmid2phinorm(e.phinorm2rmid(scalar_psinorm, scalar_t), scalar_t),
            scalar_psinorm,
            places=places
        )
        self.assertAlmostEqual(
            et.rmid2phinorm(et.phinorm2rmid(scalar_psinorm, scalar_t), scalar_t),
            scalar_psinorm,
            places=places
        )

    def test_phinorm_to_rmid_rho_True(self):
        # Need to use roa2phinorm, since there isn't a mirror option.
        self.assertAlmostEqual(
            e.roa2phinorm(e.phinorm2rmid(scalar_psinorm, scalar_t, rho=True), scalar_t),
            scalar_psinorm,
            places=places
        )
        self.assertAlmostEqual(
            et.roa2phinorm(et.phinorm2rmid(scalar_psinorm, scalar_t, rho=True), scalar_t),
            scalar_psinorm,
            places=places
        )

    def test_phinorm_to_roa(self):
        self.assertAlmostEqual(
            e.roa2phinorm(e.phinorm2roa(scalar_psinorm, scalar_t), scalar_t),
            scalar_psinorm,
            places=places
        )
        self.assertAlmostEqual(
            et.roa2phinorm(et.phinorm2roa(scalar_psinorm, scalar_t), scalar_t),
            scalar_psinorm,
            places=places
        )

class VolnormConsistencyTestCase(unittest.TestCase):
    def test_volnorm_to_psinorm(self):
        self.assertAlmostEqual(
            e.psinorm2volnorm(e.volnorm2psinorm(scalar_psinorm, scalar_t), scalar_t),
            scalar_psinorm,
            places=places
        )
        self.assertAlmostEqual(
            et.psinorm2volnorm(et.volnorm2psinorm(scalar_psinorm, scalar_t), scalar_t),
            scalar_psinorm,
            places=places
        )

    def test_volnorm_to_phinorm(self):
        self.assertAlmostEqual(
            e.phinorm2volnorm(e.volnorm2phinorm(scalar_psinorm, scalar_t), scalar_t),
            scalar_psinorm,
            places=places
        )
        self.assertAlmostEqual(
            et.phinorm2volnorm(et.volnorm2phinorm(scalar_psinorm, scalar_t), scalar_t),
            scalar_psinorm,
            places=places
        )

    def test_volnorm_to_rmid_rho_False(self):
        self.assertAlmostEqual(
            e.rmid2volnorm(e.volnorm2rmid(scalar_psinorm, scalar_t), scalar_t),
            scalar_psinorm,
            places=places
        )
        self.assertAlmostEqual(
            et.rmid2volnorm(et.volnorm2rmid(scalar_psinorm, scalar_t), scalar_t),
            scalar_psinorm,
            places=places
        )

    def test_volnorm_to_rmid_rho_True(self):
        # Need to use roa2volnorm, since there isn't a mirror option.
        self.assertAlmostEqual(
            e.roa2volnorm(e.volnorm2rmid(scalar_psinorm, scalar_t, rho=True), scalar_t),
            scalar_psinorm,
            places=places
        )
        self.assertAlmostEqual(
            et.roa2volnorm(et.volnorm2rmid(scalar_psinorm, scalar_t, rho=True), scalar_t),
            scalar_psinorm,
            places=places
        )

    def test_volnorm_to_roa(self):
        self.assertAlmostEqual(
            e.roa2volnorm(e.volnorm2roa(scalar_psinorm, scalar_t), scalar_t),
            scalar_psinorm,
            places=places
        )
        self.assertAlmostEqual(
            et.roa2volnorm(et.volnorm2roa(scalar_psinorm, scalar_t), scalar_t),
            scalar_psinorm,
            places=places
        )

class RmidConsistencyTestCase(unittest.TestCase):
    def test_rmid_to_psinorm(self):
        self.assertAlmostEqual(
            e.psinorm2rmid(e.rmid2psinorm(scalar_Rmid, scalar_t), scalar_t),
            scalar_Rmid,
            places=places
        )
        self.assertAlmostEqual(
            et.psinorm2rmid(et.rmid2psinorm(scalar_Rmid, scalar_t), scalar_t),
            scalar_Rmid,
            places=places
        )

    def test_rmid_to_phinorm(self):
        self.assertAlmostEqual(
            e.phinorm2rmid(e.rmid2phinorm(scalar_Rmid, scalar_t), scalar_t),
            scalar_Rmid,
            places=places
        )
        self.assertAlmostEqual(
            et.phinorm2rmid(et.rmid2phinorm(scalar_Rmid, scalar_t), scalar_t),
            scalar_Rmid,
            places=places
        )

    def test_rmid_to_volnorm(self):
        self.assertAlmostEqual(
            e.volnorm2rmid(e.rmid2volnorm(scalar_Rmid, scalar_t), scalar_t),
            scalar_Rmid,
            places=places
        )
        self.assertAlmostEqual(
            et.volnorm2rmid(et.rmid2volnorm(scalar_Rmid, scalar_t), scalar_t),
            scalar_Rmid,
            places=places
        )

    def test_rmid_to_roa(self):
        self.assertAlmostEqual(
            e.roa2rmid(e.rmid2roa(scalar_Rmid, scalar_t), scalar_t),
            scalar_Rmid,
            places=places
        )
        self.assertAlmostEqual(
            et.roa2rmid(et.rmid2roa(scalar_Rmid, scalar_t), scalar_t),
            scalar_Rmid,
            places=places
        )

class RoaConsistencyTestCase(unittest.TestCase):
    def test_roa_to_psinorm(self):
        self.assertAlmostEqual(
            e.psinorm2roa(e.roa2psinorm(scalar_psinorm, scalar_t), scalar_t),
            scalar_psinorm,
            places=places
        )
        self.assertAlmostEqual(
            et.psinorm2roa(et.roa2psinorm(scalar_psinorm, scalar_t), scalar_t),
            scalar_psinorm,
            places=places
        )

    def test_roa_to_phinorm(self):
        self.assertAlmostEqual(
            e.phinorm2roa(e.roa2phinorm(scalar_psinorm, scalar_t), scalar_t),
            scalar_psinorm,
            places=places
        )
        self.assertAlmostEqual(
            et.phinorm2roa(et.roa2phinorm(scalar_psinorm, scalar_t), scalar_t),
            scalar_psinorm,
            places=places
        )

    def test_roa_to_volnorm(self):
        self.assertAlmostEqual(
            e.volnorm2roa(e.roa2volnorm(scalar_psinorm, scalar_t), scalar_t),
            scalar_psinorm,
            places=places
        )
        self.assertAlmostEqual(
            et.volnorm2roa(et.roa2volnorm(scalar_psinorm, scalar_t), scalar_t),
            scalar_psinorm,
            places=places
        )

    def test_roa_to_rmid_rho_False(self):
        self.assertAlmostEqual(
            e.rmid2roa(e.roa2rmid(scalar_psinorm, scalar_t), scalar_t),
            scalar_psinorm,
            places=places
        )
        self.assertAlmostEqual(
            et.rmid2roa(et.roa2rmid(scalar_psinorm, scalar_t), scalar_t),
            scalar_psinorm,
            places=places
        )

class SelfConsistencyTestCase(unittest.TestCase):
    """Test the internal self-consistency of the basic conversions.
    """
    def test_rz2psi(self):
        R, Z = numpy.meshgrid(e.getRGrid(), e.getZGrid())
        out = e.rz2psi(R, Z, e.getTimeBase())
        out_t = et.rz2psi(R, Z, e.getTimeBase())

        # Psi gets its sign reversed, so I use plus here.
        diff = numpy.sqrt(((out + e.getFluxGrid())**2).max()) / numpy.absolute(e.getFluxGrid()).max()
        self.assertLessEqual(diff, tol)

        diff = numpy.sqrt(((out_t + et.getFluxGrid())**2).max()) / numpy.absolute(et.getFluxGrid()).max()
        self.assertLessEqual(diff, tol)

    def test_rz2psinorm(self):
        R, Z = numpy.meshgrid(e.getRGrid(), e.getZGrid())
        out = e.rz2psinorm(R, Z, e.getTimeBase())
        out_t = et.rz2psinorm(R, Z, e.getTimeBase())

        psinorm_grid = (
            (-1 * e.getFluxGrid() - e.getFluxAxis()[:, numpy.newaxis, numpy.newaxis]) /
            (e.getFluxLCFS()[:, numpy.newaxis, numpy.newaxis] - e.getFluxAxis()[:, numpy.newaxis, numpy.newaxis])
        )
        psinorm_grid_t = (
            (-1 * et.getFluxGrid() - et.getFluxAxis()[:, numpy.newaxis, numpy.newaxis]) /
            (et.getFluxLCFS()[:, numpy.newaxis, numpy.newaxis] - et.getFluxAxis()[:, numpy.newaxis, numpy.newaxis])
        )

        diff = numpy.sqrt(((out - psinorm_grid)**2).max()) / numpy.absolute(psinorm_grid).max()
        self.assertLessEqual(diff, tol)

        diff = numpy.sqrt(((out_t - psinorm_grid_t)**2).max()) / numpy.absolute(psinorm_grid_t).max()
        self.assertLessEqual(diff, tol)

    def test_psinorm2volnorm(self):
        psinorm_grid = numpy.linspace(0, 1, e.getFluxVol().shape[1])
        psinorm_grid_t = numpy.linspace(0, 1, et.getFluxVol().shape[1])

        volnorm_grid = e.getFluxVol() / e.getFluxVol()[:, -1, numpy.newaxis]
        volnorm_grid_t = et.getFluxVol() / et.getFluxVol()[:, -1, numpy.newaxis]

        out = e.psinorm2volnorm(psinorm_grid, e.getTimeBase())
        out_t = et.psinorm2volnorm(psinorm_grid, et.getTimeBase())

        diff = numpy.sqrt(((out - volnorm_grid)**2).max()) / numpy.absolute(volnorm_grid).max()
        self.assertLessEqual(diff, tol)

        diff = numpy.sqrt(((out_t - volnorm_grid_t)**2).max()) / numpy.absolute(volnorm_grid_t).max()
        self.assertLessEqual(diff, tol)

    def test_psinorm2rmid(self):
        psinorm_grid = numpy.linspace(0, 1, e.getRmidPsi().shape[1])
        psinorm_grid_t = numpy.linspace(0, 1, et.getRmidPsi().shape[1])

        out = e.psinorm2rmid(psinorm_grid, e.getTimeBase())
        out_t = et.psinorm2rmid(psinorm_grid, et.getTimeBase())

        diff = numpy.sqrt(((out - e.getRmidPsi())**2).max()) / numpy.absolute(e.getRmidPsi()).max()
        self.assertLessEqual(diff, tol)

        self.assertLessEqual(diff, tol)

        diff = numpy.sqrt(((out_t - et.getRmidPsi())**2).max()) / numpy.absolute(et.getRmidPsi()).max()
        self.assertLessEqual(diff, tol)

    def test_roa2rmid(self):
        roa_grid = numpy.linspace(0, 1, len(e.getRGrid()))
        out = e.roa2rmid(roa_grid, e.getTimeBase())

        rmid_grid = (
            roa_grid[numpy.newaxis, :] * (e.getRmidOut()[:, numpy.newaxis] -
            e.getMagR()[:, numpy.newaxis]) + e.getMagR()[:, numpy.newaxis]
        )

        diff = numpy.sqrt(((out - rmid_grid)**2).max()) / numpy.absolute(rmid_grid).max()
        self.assertLessEqual(diff, tol)

        roa_grid_t = numpy.linspace(0, 1, len(et.getRGrid()))
        out_t = et.roa2rmid(roa_grid_t, et.getTimeBase())

        rmid_grid_t = (
            roa_grid_t[numpy.newaxis, :] * (et.getRmidOut()[:, numpy.newaxis] -
            et.getMagR()[:, numpy.newaxis]) + et.getMagR()[:, numpy.newaxis]
        )

        diff = numpy.sqrt(((out_t - rmid_grid_t)**2).max()) / numpy.absolute(rmid_grid_t).max()
        self.assertLessEqual(diff, tol)

    def test_psinorm2phinorm(self):
        pass
        # TODO: Write a good test for this.


if __name__ == '__main__':
    unittest.main()
