from setuptools import setup, find_packages

setup(name='lesionmatching',
      packages=find_packages(),
      install_requires=['numpy>=1.21', 'torch>=1.13', 'SimpleITK', 'scipy', 'skimage',
                        'gryds', 'elastix-py', 'monai>=1.1', 'matplotlib', 'seaborn', 'pandas'])
