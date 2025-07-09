import numpy as np
import statsmodels as sm
# Sample data
X = np.array([1, 2, 3, 4, 5, 6, 7, 8])
y = np.array([2, 4, 5, 4, 5, 6, 5, 6])
# Add constant for intercept term  
X = sm.add_constant(X)
# Fit the model
model = sm.OLS(y, X).fit()
# View the summary
print(model.summary())
# Predict
y_pred = model.predict(X)
print("Predictions:", y_pred)
