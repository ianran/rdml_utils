from setuptools import setup

setup(name='rdml_utils',
      version='0.0.1',
      description='Package for Robot Decision Making Lab Utility Functions and Classes',
      url='http://tb.d',
      author='Seth McCammon',
      author_email='mccammos@oregonstate.edu',
      license='None',
      packages=['rdml_utils'],
      install_requires=['numpy>=1.14.0', 'matplotlib>=2.1.2', 'haversine>=0.4.5', 'scipy>=1.0.0', 'deepdish>=0.3.6', 'shapely>=1.6.4.post2', 'opencv-python-headless', 'netCDF4', 'oyaml'],
      zip_safe=False)
