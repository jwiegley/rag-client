from setuptools import setup

setup(
    name="rag-client",
    version="0.1.0",
    description="RAG client command-line utility",
    # Include both Python files as modules
    py_modules=["main", "rag", "api"],
    # Create the rag-client executable
    entry_points={
        'console_scripts': [
            'rag-client=main:main',
        ],
    },
    # Add any dependencies your project needs
    install_requires=[
        # List dependencies here
    ],
)
