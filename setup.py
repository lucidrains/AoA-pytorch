from setuptools import setup, find_packages

setup(
  name = 'aoa_pytorch',
  packages = find_packages(exclude=['examples']),
  version = '0.0.2',
  license='MIT',
  description = 'Attention on Attention - Pytorch',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  url = 'https://github.com/lucidrains/SAoA-pytorch',
  keywords = [
    'artificial intelligence',
    'attention mechanism',
    'visual question answering'
  ],
  install_requires=[
    'torch>=1.6',
    'einops>=0.3'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
