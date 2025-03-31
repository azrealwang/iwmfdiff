from setuptools import setup, find_packages

setup(
    name='iwmfdiff',
    version='0.1.0',
    packages=find_packages(exclude=['scripts']),
    install_requires=[
        'numpy<2',
        'torch==2.1.2',         # Base version, CUDA variant fetched via extra index
        'torchvision==0.16.2',  # Base version, CUDA variant fetched via extra index
        'PyYAML==6.0.1',
        'tqdm==4.66.2',
        'facenet-pytorch==2.5.3',
    ],
    package_data={
        'iwmfdiff': ['configs/*.yml'],
    },
    author='Hanrui Wang',
    description='A package for image processing with diffusion models',
    url='https://github.com/azrealwang/iwmfdiff',
    python_requires='>=3.9',
    # Specify the PyTorch index for CUDA wheels
    extra_requires={
        'cuda': [
            'torch==2.1.2+cu118',
            'torchvision==0.16.2+cu118',
        ]
    },
    dependency_links=[
        'https://download.pytorch.org/whl/torch_stable.html'
    ]
)