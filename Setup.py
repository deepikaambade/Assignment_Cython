
from setuptools import setup
from Cython.Build import cythonize
setup(extension_modules= cythonize("data_generationpy.pyx")) 
setup(extension_modules= cythonize("data_splitting.pyx")) 
setup(extension_modules= cythonize("data.pyx")) 
setup(extension_modules= cythonize("datasplitting.pyx")) 
setup(extension_modules= cythonize("model_eval.pyx")) 
setup(extension_modules= cythonize("model_evaluation.pyx")) 
setup(extension_modules= cythonize("model_training.pyx")) 
setup(extension_modules= cythonize("model_trainingpy.pyx")) 
