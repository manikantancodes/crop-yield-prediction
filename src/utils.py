import matplotlib.pyplot as plt

def plot_residuals(y_test, y_pred):
    """Plot residuals for model evaluation."""
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals)
    plt.hlines(y=0, xmin=min(y_pred), xmax=max(y_pred), color='r')
    plt.xlabel('Predicted values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Predicted values')
    plt.savefig('results/residuals_plot.png')
    plt.show()

def plot_actual_vs_predicted(y_test, y_pred):
    """Plot actual vs predicted values."""
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='r', linestyle='--')
    plt.xlabel('Actual values')
    plt.ylabel('Predicted values')
    plt.title('Actual vs Predicted values')
    plt.savefig('results/actual_vs_predicted.png')
    plt.show()
