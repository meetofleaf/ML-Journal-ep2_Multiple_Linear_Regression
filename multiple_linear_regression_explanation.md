# What is Multiple Linear Regression?
Today we're venturing into the world of multiple linear regression, a powerful tool for uncovering relationships between several independent variables and one dependent variable. In this case, we're aiming to predict a company's Profit based on factors like R&D Spend, Administration, and Marketing Spend.

You have data for 50 companies, including their spending on research and development (R&D), administration, and marketing, along with their profits.  Multiple linear regression helps you find a model that best explains how these spending factors influence the profit.

## Key Players
- **Dependent Variable (Y):** The variable you're trying to predict (Profit).
- **Independent Variables (X):** The variables you think influence the dependent variable (R&D Spend, Administration, Marketing Spend).
- **Coefficients (β):** These represent the influence of each independent variable on the dependent variable.
- **A positive coefficient** indicates that as the independent variable increases, the dependent variable tends to increase as well (e.g., more R&D spend might lead to higher profits).
- **A negative coefficient** suggests the opposite (e.g., higher administrative costs might decrease profits).
- **Intercept (β₀):** The point where the regression line (in a 2D plot with one independent variable) would cross the Y-axis. It represents the predicted profit when all independent variables are zero (not very realistic in our case, but useful for interpretation).


## The Equation Behind the Magic
This is where things get a bit more mathematical, but don't worry, we'll break it down:

#### **`Y = β₀ + β₁X₁ + β₂X₂ + β₃X₃ + ... + ε`**

- `Y` = Predicted value of the dependent variable (Profit)
- `β₀` = Intercept
- `β₁` to β₃ = Coefficients for each independent variable (R&D Spend, Administration, Marketing Spend)
- `X₁` to X₃ = Values of the independent variables
- `ε` = Error term (the difference between the actual and predicted profit)


## Feature Selection
As the model is trained based on multiple variables, it is important to select the correct ones for optimal accuracy and performance of the model. This process is called feature selection or extraction.

Including unnecessary variables/features can:
- Increase training time
- Reduce model interpretability
- Lead to overfitting (the model performs well on the training data but poorly on unseen data)

Multiple linear regression functions in most of the Python libraries perform feature selection automatically. There are multiple methods and categories of methods of feature extraction. The library that we used, scikt-learn, automatically uses Backward Elimination method. Let's understand feature selection via Backward Elimination, which is a type of Stepwise feature selection.

#### Backward Elimination in Action:
- Start Big: Begin with the full set of features in your multiple linear regression model.
- Evaluate and Remove: Fit the model and identify the feature with the least significant coefficient (think of it as the feature contributing the least to explaining the dependent variable).
- Out it Goes: Remove the least significant feature from the model.
- Repeat: Refit the model with the remaining features and iterate steps 2 and 3.
- Stop When Necessary: Keep eliminating features until a stopping criterion is met. This could be:
- A predetermined number of features remaining.
- A significant drop in model performance when removing another feature.
- When the coefficients of the remaining features become statistically significant.

#### Benefits of Backward Elimination:
- Reduced Training Time: By eliminating irrelevant features, the model trains faster.
- Improved Model Performance: Focusing on the most informative features can lead to better generalization and reduced overfitting.
- Enhanced Interpretability: With fewer features, it's easier to understand the relationships between the independent variables and the dependent variable.

#### Things to Consider:
- Backward elimination is a stepwise selection method, and the order of feature removal can influence the final set. Running the process multiple times might yield slightly different results.
- There are other feature selection techniques, such as forward selection and stepwise selection, that might be better suited for specific datasets.


## Finding the Best Fit Model
Multiple linear regression algorithms find the coefficients (β) and intercept (β₀) that minimize the overall error (ε) between the actual and predicted profits.


#### Important to Remember
- Multiple linear regression assumes linear relationships between the independent variables and the dependent variable. Visualizing the data can help assess this assumption.
- It's crucial to consider factors like outliers and multicollinearity (when independent variables are highly correlated) that can affect the model's accuracy.
- There will always be some error, and we use statistical tests to evaluate the model's goodness-of-fit.

## Further Exploration
This is just a glimpse into multiple linear regression. As you delve deeper, you'll explore:
- More Feature Selection Techniques: Choosing the most relevant independent variables.
- Model Selection Techniques: Selecting the best model complexity for optimal performance.
- Regularization: Techniques to reduce overfitting and improve model generalizability.
- Bonus: There are advanced regression techniques like decision trees and neural networks that can handle more complex relationships, but that's a topic for another day.

For now, you're equipped with the fundamentals of multiple linear regression. If you want to dig deeper and learn more about multiple linear regression, go through the resources given at the end and learn more in-depth concepts.

Remember, practice makes one perfect, so grab your data and start exploring!

## Additional Resources

- [Wikipedia - Multiple Linear Regression](https://en.wikipedia.org/wiki/Linear_regression)
  - The Wikipedia page on multiple linear regression provides a comprehensive overview of the mathematical details and statistical properties.

- [StatTrek - Multiple Linear Regression](https://stattrek.com/tutorials/regression-tutorial)
  - This webpage offers a clear explanation of multiple linear regression with step-by-step examples.

- [Harvard University - Introduction to Multiple Regression](https://www.youtube.com/watch?v=q1RD5ECsSB0)
  - This Harvard resource provides an introduction to multiple linear regression using the R programming language.

- [Coursera - Regression Models](https://www.coursera.org/learn/regression-models)
  - This course covers multiple regression along with other regression techniques in depth.

- [Khan Academy - Multiple Regression](https://www.khanacademy.org/math/statistics-probability/describing-relationships-quantitative-data/multiple-regression/v/multiple-regression)
  - Khan Academy offers a comprehensive tutorial on multiple regression, explaining the concepts step by step.

- [YouTube - Multiple Regression Analysis in Excel](https://www.youtube.com/watch?v=KLoABW1a03s)
  - This video tutorial demonstrates how to perform multiple regression analysis using Excel, which can be helpful for practical understanding.

- [Statistics Solutions - Multiple Regression Analysis](https://www.statisticssolutions.com/multiple-regression-analysis/)
  - This resource provides a detailed explanation of multiple regression analysis, including assumptions, interpretation of results, and practical examples.

- [Towards Data Science - Understanding Multiple Regression](https://towardsdatascience.com/understanding-multiple-regression-249b16bde83e)
  - This article on Towards Data Science delves into the intuition behind multiple regression and explores various aspects of the technique.

- [Penn State University - Multiple Regression with Many Predictor Variables](https://online.stat.psu.edu/stat501/lesson/9)
  - This resource from Penn State University offers a comprehensive guide to multiple regression, covering topics such as model building and interpretation.

- [DataCamp - Multiple and Logistic Regression Course](https://www.datacamp.com/courses/multiple-and-logistic-regression)
  - DataCamp offers a course specifically focused on multiple regression, providing hands-on experience with real-world datasets.

- [MIT OpenCourseWare - Introduction to Statistical Methods in Economics](https://ocw.mit.edu/courses/economics/14-32-econometric-methods-spring-2019/)
  - This course covers multiple regression and its applications in economics, providing lectures, assignments, and readings for a comprehensive understanding.

- [UC Business Analytics R Programming Guide - Multiple Regression](https://uc-r.github.io/multiple_regression)
  - This guide offers a detailed explanation of multiple regression using R programming language, including code examples and interpretation of results.

- [ResearchGate - Multiple Regression Analysis in Medical Research](https://www.researchgate.net/publication/237492569_Multiple_Regression_Analysis_in_Medical_Research)
  - This paper provides insights into the application of multiple regression analysis in medical research, highlighting its importance and challenges.