from setuptools import setup

setup(
    name="libkge-semantic",
    version="0.1.0-alpha",
    description="An extension of the libkge knowledge graph embedding library that focuses on the inclusion of "
                "semantics into the embeddings",
    url="https://github.com/sfschouten/semantic-kge",
    author="Stefan F. Schouten",
    author_email="sfschouten@gmail.com",
    packages=["sem_kge"],
    install_requires=[
        "torch>=1.3.1",
        "libkge>=0.1",
        "mdmm==0.1.3",
        "pyro-ppl>=1.5.2",
    ],
    python_requires='>=3.7',  
)
