from setuptools import setup

setup(name='apc',
      version='0.1.0',
      description='Age-Period-Cohort Analysis',
      url='http://github.com/JonasHarnau/apc',
      author='Jonas Harnau',
      author_email='j.harnau@economics.oxon.org',
      license='GPLv3',
      packages=['apc'],
      install_requires=['matplotlib','numpy', 'pandas', 'scipy', 'seaborn', 'statsmodels'],
      include_package_data=True,
      classifiers=['Development Status :: 3 - Alpha'],
      long_description=open('README.rst').read(),
      zip_safe=False)