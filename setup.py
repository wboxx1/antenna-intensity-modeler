from distutils.core import setup

setup(
    name='antenna-intensity-modeler',
    version='0.1.0',
    packages=['antennas',],
    license='Creative Commons Attribution-Noncommercial-Share Alike license',
    long_description=open('README.txt').read(),
    install_requires=[
        numpy >= 1.15.2,
        pandas >= 0.23.4,
        scipy >= 1.1.0,
	matplotlib >= 3.0.0
    ],
)