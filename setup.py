from setuptools import setup, find_packages

setup(name='permuted_brunnermunzel',
      version='0.1',
      description='Python implementation of the permuted brunner munzel',
      url='https://github.com/MatthewCorney',
      author='Matthew Corney',
      author_email='matthew_corney@yahoo.co.uk',
      license='MIT',
      package_dir={'':"src"},
      packages=find_packages("src"),
      install_requires=[
          'numpy', 'scipy',
      ],
      setup_requires=['pytest-runner'],
      tests_require=['pytest'],
      zip_safe=False)
