from setuptools import setup, find_packages


setup(
    name='DIABETES PREDICTION',  
    version='0.1.0', 
    author='M SHIVA RAMA KRISHNA REDDY',
    author_email='msrkreddy111@gmail.com',
    description='The objective of this project is to develop a web-based Diabetes Prediction application that uses machine learning algorithms to predict the likelihood of an individual having diabetes based on theirhealth parameters',
    long_description=open('README.md').read(), 
    long_description_content_type='text/markdown',
    url='https://github.com/msrkreddy0928/Diabetes-Prediction-ML-Project/tree/master',
    packages=find_packages(),  
    install_requires=[  
        'pandas',
        'numpy',
        'seaborn',
        'matplotlib',
        'scikit-learn',
        'flask',
        'imbalanced-learn',
        'joblib',
        'scipy',
        'xgboost'
      
    ],
    classifiers=[
        'Programming Language :: Python :: 3.10.16',
        'Operating System :: OS Independent',
    ],
    python_requires='3.10.16',  
)
