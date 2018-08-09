"""apc: Age-Period-Cohort and extended Chain-Ladder Analysis."""

import setuptools

with open("README.md", "r", encoding='utf8') as fh:
    long_description = fh.read()

setuptools.setup(
    name='apc',
    version='1.0.1',
    description='Age-Period-Cohort and extended Chain-Ladder Analysis',
    url='http://github.com/JonasHarnau/apc',
    author='Jonas Harnau',
    author_email='j.harnau@outlook.com',
    license='GPLv3',
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
    install_requires=[
        'matplotlib', 'numpy', 'pandas', 'quad_form_ratio', 'seaborn',
        'scipy', 'statsmodels'
    ],
    include_package_data=True,
    classifiers=['Development Status :: 3 - Alpha'],
    long_description=long_description,
    long_description_content_type="text/markdown",
    zip_safe=False)
