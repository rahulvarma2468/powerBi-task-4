# powerBi-task-4

# **Company name** : CODETECH IT SOLUTIONS
# **Name** : Mudunuru Rahul varma
# **Intern ID** :CT04DN302
# **Domain** : Power BI
# **Duration** : 4 weeks
# **Mentor** :Neela Santosh
# Description
Using **Python** or **R scripts within Power BI** enables analysts, data scientists, and business users to perform **advanced data analysis**, **machine learning**, and **custom data visualizations** far beyond the capabilities of Power BI’s built-in tools. This integration combines the best of both worlds: Power BI's powerful data connectivity and visualization framework with the analytical depth of Python and R programming languages.

---

## What Are Python and R Scripts in Power BI?

Power BI supports scripting using both Python and R at different stages of data processing and visualization. These languages can be used in:

1. **Power Query Editor** for data transformation and cleaning.
2. **Report canvas** as Python or R visual elements.
3. **Data modeling layer** to add calculated columns or prediction outputs.

This enables users to run custom scripts directly in Power BI to analyze data using statistical or machine learning models, conduct time series forecasting, generate advanced charts (e.g., violin plots, heatmaps), or manipulate data with more control and precision.

---

## Setting Up Python or R in Power BI

To use scripting languages in Power BI, you must first install the appropriate engine on your local machine:

* For **Python**: Install Python (preferably via [Anaconda](https://www.anaconda.com/)) and common libraries such as `pandas`, `matplotlib`, `seaborn`, and `scikit-learn`.
* For **R**: Install the R language and packages such as `ggplot2`, `dplyr`, `forecast`, and `caret`.

Next, in Power BI:

* Navigate to **File > Options and Settings > Options > Python scripting** or **R scripting**.
* Set the home directory for the language engine.

Once configured, you can begin inserting scripts into Power BI for either transformation or visualization.

---

## Use Case 1: Advanced Data Cleaning and Transformation

Power BI’s query editor is powerful but sometimes lacks the flexibility needed for complex data operations. Python or R can bridge that gap. For example, you might need to remove outliers, impute missing values, or normalize numeric fields.

**Python Example (in Power Query):**

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler

df = dataset.copy()
df = df[df['sales'] < df['sales'].quantile(0.95)]  # Remove outliers
scaler = StandardScaler()
df[['sales', 'profit']] = scaler.fit_transform(df[['sales', 'profit']])
```

This script removes the top 5% of outliers in the `sales` column and normalizes selected fields.

---

## Use Case 2: Custom Visualizations

Some advanced visualizations like violin plots, pair plots, or multi-dimensional histograms are not available natively in Power BI. Using Python or R visuals, you can render these custom charts.

**R Visualization Example:**

```r
library(ggplot2)

ggplot(dataset, aes(x = Category, y = Sales, fill = Category)) +
  geom_bar(stat = "identity") +
  theme_minimal()
```

This produces a clean bar chart using `ggplot2`, which can be fully customized in ways that go beyond Power BI’s native visual engine.

**Python Visualization Example:**

```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.boxplot(x='Region', y='Sales', data=dataset)
plt.title("Sales Distribution by Region")
plt.show()
```

You can use Python’s `matplotlib` and `seaborn` to create highly detailed plots with just a few lines of code.

---

## Use Case 3: Machine Learning and Forecasting

You can also use Python or R to perform **predictive modeling** right inside Power BI.

**Example: Linear Regression with Python**

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(dataset[['ad_spend']], dataset['sales'])
dataset['predicted_sales'] = model.predict(dataset[['ad_spend']])
```

This adds a column to the dataset containing predicted sales values based on advertising spend.

---

## Limitations and Considerations

* Scripts run **locally** and won’t work in Power BI Service unless a **personal gateway** is configured.
* Large datasets may hit performance or memory limits.
* Scripting is best suited for **moderate-sized datasets** and **exploratory analysis**.

---

## Conclusion

Incorporating Python or R scripts into Power BI extends its analytical power significantly. Whether you're a data scientist needing complex statistical modeling or a business analyst looking to create unique visualizations, scripting enables you to turn Power BI into a comprehensive, end-to-end analytics platform. With a little setup and the right code, your dashboards can become smarter, more dynamic, and visually compelling—bridging the gap between raw data and actionable insight.

# OUTPUT
