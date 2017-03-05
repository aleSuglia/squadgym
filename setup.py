from setuptools import setup

setup(
    name="squadgym",
    version="0.0.1",
    description="SQuAD-Gym environment",
    long_description="Environment that can be used to evaluate reasoning capabilities of artificial agents",
    url="https://www.github.com/aleSuglia/squadgym",
    author="Alessandro Suglia",
    license="Apache (v2)",
    install_requires=["gym", "nltk", "numpy"],
    classifiers=[
        "Development Status :: 1 - Planning",

        "Intended Audience :: ML practitioners and researchers",
        "Topic :: Artificial Intelligence :: Machine Learning",

        "License :: OSI Approved :: Apache (v2)",

        "Programming Language :: Python :: 3.6",
    ]
)
