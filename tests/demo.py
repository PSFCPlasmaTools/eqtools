# This script demonstrates some basic functionality of eqtools, giving basic
# inputs and the expected outputs. For a more detailed demo, refer to the online
# documentation at eqtools.readthedocs.org. For a more detailed set of tests,
# run the files test.py and unittests.py in this directory.

import eqtools

# Load the sample data (for local C-Mod users the data are taken from the
# MDSplus tree, for other users the pickled data are used):
try:
    shot = 1120914027
    # Run tests with both of these to be sure that tspline does everything right:
    e = eqtools.CModEFITTree(shot)
except:
    import warnings
    warnings.warn(
        "Could not access MDSplus data. Defaulting to pickled data. You may want "
        "to modify unittests.py to use your own local data system to ensure "
        "consistency for your use case.",
        RuntimeWarning
    )
    import cPickle as pkl
    with open('test_data.pkl', 'rb') as f:
        shot, e, et = pkl.load(f)

# Convert psinorm = 0.5 to Rmid at t=[0.25, 0.5, 0.75, 1.0]:
Rmid = e.psinorm2rmid(0.5, [0.25, 0.5, 0.75, 1.0])
# Rmid should now be: [0.81060089, 0.81021389, 0.81252954, 0.81248401]