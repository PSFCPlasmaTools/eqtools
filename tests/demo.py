# This script demonstrates some basic functionality of eqtools, giving basic
# inputs and the expected outputs. For a more detailed demo, refer to the online
# documentation at eqtools.readthedocs.org. For a more detailed set of tests,
# run the files test.py and unittests.py in this directory.

import eqtools
import cPickle as pkl

# Load the sample data:
with open('test_data.pkl', 'rb') as f:
    shot, e, et = pkl.load(f)

# Convert psinorm = 0.5 to Rmid at t=[0.25, 0.5, 0.75, 1.0]:
Rmid = e.psinorm2rmid(0.5, [0.25, 0.5, 0.75, 1.0])
# Rmid should now be: [0.81060089, 0.81021389, 0.81252954, 0.81248401]