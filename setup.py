from setuptools import setup

setup(name='permuted_brunner_munzel',
      version='0.1',
      description='Python implementation of the permuted brunner munzel',
      url='https://github.com/MatthewCorney',
      author='Matthew Corney',
      author_email='matthew_corney@yahoo.com',
      license='MIT',
      packages=['permuted_brunnermunzel'],
      install_requires=[
          'numpy', 'scipy',
      ],
      test_suite='pytest.collector',
      tests_require=['pytest'],
      zip_safe=False)
