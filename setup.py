from setuptools import setup, find_packages

setup(
    name='disentangled_retriever',
    version='0.0.2',
    packages=find_packages("src"),
    package_dir={'': 'src'},
    description='Disentangled Modeling of Domain and Relevance for Adaptable Dense Retrieval',
    url='https://github.com/jingtaozhan/disentangled-retriever',
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    author='Jingtao Zhan',
    author_email='jingtaozhan@gmail.com',
    install_requires=[
        "pytrec-eval",
        "ujson",
        "adapter-transformers",
        # 'torch >= 1.10.1', # install manually
        # 'faiss-gpu == 1.7.2', # install manually
    ],
)