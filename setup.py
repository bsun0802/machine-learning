# Making sure modules are ready to go.

import scipy
import numpy
import matplotlib
import pandas
import statsmodels
import sklearn
import idx2numpy

if __name__ == '__main__':
    print('scipy: %s' % scipy.__version__)
    print('numpy: %s' % numpy.__version__)
    print('matplotlib: %s' % matplotlib.__version__)
    print('pandas: %s' % pandas.__version__)
    print('statsmodels: %s' % statsmodels.__version__)
    print('sklearn: %s' % sklearn.__version__)
    print('idx2numpy(used in MNIST): %s' % idx2numpy.__version__)
    print("All packages are working as expected!")
