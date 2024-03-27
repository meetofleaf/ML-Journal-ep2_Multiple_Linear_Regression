# ML Journal - Machine Learning Fundamentals' Series
In this series of repositories, we'll explore various models, documenting the code and thought process behind each one.  The goal is to create a journal-like experience for both myself and anyone following along. By sharing the journey, we can:

- Break down complex concepts: We'll approach each model step-by-step, making the learning process manageable.
- Learn from mistakes: Documenting the process allows us to identify and learn from any errors along the way.
- Build a foundation: Each repository will build upon the knowledge from the previous one, creating a solid foundation in machine learning basics.
- We believe this approach can be particularly helpful for beginners struggling to find a starting point in the vast world of machine learning.


# Ep-2 - Multiple Linear Regression
This repository is the second addition to the 'ML Journal' series aimed at revisiting fundamental machine learning models. This specific repository focuses on Multiple Linear Regression, a widely used technique for modeling linear relationships between multiple features and a dependent/target variable.

Compared to the previous linear regression model, we have multiple independent variables and sometimes it is necessary to select the best variables to get the most optimal and accurate model. This process of selection is called feature selection. There are variety of feature selection methods but for the purpose of multiple regression, stepwise selection is the most common method. Our ML library, sklearn, uses backward elimination method which is a type of stepwise feature selection method and it does so automatically but if you want to learn more about stepwise feature selection visit [Dataaspirant](https://dataaspirant.com/stepwise-regression/).

Covering all feature selection methods is out of scope of this repository. So, to learn more about feature selection from a broader perspective, visit [neptune.ai](https://neptune.ai/blog/feature-selection-methods).


## Data
This repository includes the dataset (50_Startups.csv) suitable for mutliple linear regression.
The dataset contains 5 variables and 50 instances, uploaded by [Farhan](https://www.kaggle.com/farhanmd29).

Dataset link: https://www.kaggle.com/datasets/farhanmd29/50-startups

Independent Variables: 4
- _R&D Spend_
- _Administration_
- _Marketing Spend_
- _State_

Dependent Variables: 1
- _Profit_


## Code
The Python code file (multiple_linear_regression.py) demonstrates how to implement multiple linear regression using a popular machine learning library scikit-learn. You will find guiding comments in the code specifying purpose of each block of code in 1-3 lines.


## Requirements
Following is a list of the programs and libraries, with the versions, used in the code:

- Python==3.12.1
  - pandas==2.2.0
  - numpy==1.26.3
  - scikit-learn==1.4.1


## Code Specific Reference Links:
- [Feature selection](https://neptune.ai/blog/feature-selection-methods)
- [Stepwise feature selection](https://dataaspirant.com/stepwise-regression/)
- [One hot encoding/dummy variables](https://www.shiksha.com/online-courses/articles/handling-categorical-variables-with-one-hot-encoding/)
- [Dummy variable trap](https://www.learndatasci.com/glossary/dummy-variable-trap/)


## Getting Started
- Clone this repository.
- Ensure you have all the required programs and libraries installed or install using the requirements file.
- Simply run the Python script either from your OS' command prompt or from your choice of IDE.
- Follow the comments and code execution to understand the process.
- I encourage you to experiment with the code, modify the data, and play around with the model!
- Lastly, feel free to share any suggestions, corrections, ideas or questions you might have.

Also feel free to reach out to me for any discussion or collaboration. I'd be glad to productively interact with you.

This is just the first step in our machine learning journey. Stay tuned for future repositories exploring other fundamental models!


## References & Guidance Links:
- Python: https://www.python.org/
  - Scikit-learn: https://scikit-learn.org/stable/install.html
  - Pandas: https://pandas.pydata.org/pandas-docs/stable/getting_started/install.html
  - NumPy: https://numpy.org/install/
  - Matplotlib: https://matplotlib.org/stable/users/installing/index.html
- Pip (Python Package Manager): https://pip.pypa.io/en/stable/user_guide/
- Git: https://git-scm.com/
