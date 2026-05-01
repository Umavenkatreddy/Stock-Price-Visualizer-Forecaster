from setuptools import setup, find_packages

setup(
    name="stock-price-visualizer-forecaster",
    version="1.0.0",
    description="A Dash web application for visualizing and forecasting stock prices using ML",
    author="Uma Venkat",
    url="https://github.com/Umavenkatreddy/Stock-Price-Visualizer-Forecaster",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "dash>=2.17.1",
        "plotly>=5.22.0",
        "pandas>=2.2.2",
        "yfinance>=0.2.40",
        "scikit-learn>=1.5.0",
        "numpy>=1.26.4",
        "gunicorn>=22.0.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
    entry_points={
        "console_scripts": [
            "stock-app=Stock.app:server",
        ],
    },
)
