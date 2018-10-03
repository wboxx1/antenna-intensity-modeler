from distutils.core import setup

setup(
    name='antenna-intensity-modeler',
    version='0.1.0',
    packages=['antennas',],
    license='GNU General Public License V3',
    long_description=open('README.md').read(),
    install_requires=[
        numpy >= 1.15.2,
        pandas >= 0.23.4,
        scipy >= 1.1.0,
	    matplotlib >= 3.0.0
    ],
)