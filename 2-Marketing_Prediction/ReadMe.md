```python
import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt

os.chdir("/Users/jacobrichards/Desktop/DS_DA_Projects/2-Marketing_Prediction")

data = pd.read_csv("data.csv", na_values=["", "NA"])

import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.figsize'] = [8, 6]  # Default figure size
plt.rcParams['figure.dpi'] = 100  # Controls display resolution
```

#### data preview


```python
print(data.head())
```

       id  age      dist       income gender marital_status  target
    0   1   73  4.371654    90-99,999      M              S       1
    1   2   89  1.582733  100-149,999      M            NaN       1
    2   3   85  1.223810    10-19,999      F              S       1
    3   4   76  2.962427    90-99,999      M              M       1
    4   5   76  2.594408    10-19,999      M              S       1


#### Income: continuous, distance: integer, gender/marital_status/target: binary.


```python
import pandas as pd
import numpy as np

# Replace missing or empty values in columns 4, 5, 6 with "unknown"
data.iloc[:, [3, 4, 5]] = data.iloc[:, [3, 4, 5]].applymap(lambda x: "unknown" if pd.isna(x) or x == "" else x)

income_mapping = {
    "unknown": 55000,
    "Under $10k": 5000,
    "10-19,999": 15000,
    "20-29,999": 25000,
    "30-39,999": 35000,
    "40-49,999": 45000,
    "50-59,999": 55000,
    "60-69,999": 65000,
    "70-79,999": 75000,
    "80-89,999": 85000,
    "90-99,999": 95000,
    "100-149,999": 125000,
    "150 - 174,999": 162500,
    "175 - 199,999": 187500,
    "200 - 249,999": 225000,
    "250k+": 250000
}
data["income"] = data["income"].map(income_mapping).astype(int)

# Replace gender with 1 for male, 0 for female, and 0 for unknown
gender_mapping = {"M": 1, "F": 0, "unknown": 0}
data["gender"] = data["gender"].map(gender_mapping).astype(int)

# Replace marital_status with 1 for married, 0 for single, and 1 for unknown
marital_status_mapping = {"M": 1, "S": 0, "unknown": 1}
data["marital_status"] = data["marital_status"].map(marital_status_mapping).astype(int)

# Convert target to categorical values (0 or 1)
data["target"] = data["target"].astype(int)

# Convert distance (dist) column to integer
data["dist"] = data["dist"].astype(int)
```

#### Examine strength of continuous predictors within data set by plotting observed probability of response corresponding to predictor values.   


```python
import seaborn as sns
from scipy.stats import pointbiserialr

def plot_ratio_positive_negative_with_corr(df, continuous_var, categorical_var='target', positive_value=1, negative_value=0):
    grouped = df.groupby([continuous_var, categorical_var]).size().reset_index(name='count')
    pivot = grouped.pivot(index=continuous_var, columns=categorical_var, values='count').fillna(0)
    
    positive_counts = pivot[positive_value] if positive_value in pivot.columns else 0
    total_counts = positive_counts + pivot[negative_value] if negative_value in pivot.columns else positive_counts
    ratio = positive_counts / (total_counts + 1e-9)
    
    plot_df = pd.DataFrame({continuous_var: ratio.index, 'ratio_positive_negative': ratio.values})
    return plot_df, *pointbiserialr(df[continuous_var], df[categorical_var])

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))

dist_plot_df, dist_corr, dist_p = plot_ratio_positive_negative_with_corr(data, 'dist', 'target')
sns.scatterplot(x='dist', y='ratio_positive_negative', data=dist_plot_df, color='blue', s=50, label='Data Points', ax=ax1)

dist_plot_df_lt10 = dist_plot_df[dist_plot_df['dist'] <= 10]
dist_plot_df_gt10 = dist_plot_df[dist_plot_df['dist'] > 10]

sns.regplot(x='dist', y='ratio_positive_negative', data=dist_plot_df_lt10, scatter=False,
            lowess=True, color='red', line_kws={'lw': 2}, label='Lowess Smoother (≤10)', ax=ax1)
sns.regplot(x='dist', y='ratio_positive_negative', data=dist_plot_df_gt10, scatter=False,
            lowess=True, color='green', line_kws={'lw': 2}, label='Lowess Smoother (>10)', ax=ax1)

ax1.text(0.05, 0.95, f'Point-Biserial Corr: {dist_corr:.4f}\nP-Value: {dist_p:.4e}', transform=ax1.transAxes, fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))
ax1.set_title('Ratio of Positive Outcomes and Correlation by Distance')
ax1.set_xlabel('Distance')
ax1.set_ylabel('Ratio (Positive / Total)')
ax1.set_ylim(0, 0.50)
ax1.legend()
ax1.grid()

income_plot_df, income_corr, income_p = plot_ratio_positive_negative_with_corr(data, 'income', 'target')
sns.scatterplot(x='income', y='ratio_positive_negative', data=income_plot_df, color='blue', s=50, label='Data Points', ax=ax2)

income_plot_df_lt65k = income_plot_df[income_plot_df['income'] <= 65000]
income_plot_df_gt65k = income_plot_df[income_plot_df['income'] > 65000]

sns.regplot(x='income', y='ratio_positive_negative', data=income_plot_df_lt65k, scatter=False,
            lowess=True, color='red', line_kws={'lw': 2}, label='Lowess Smoother (≤65k)', ax=ax2)
sns.regplot(x='income', y='ratio_positive_negative', data=income_plot_df_gt65k, scatter=False,
            lowess=True, color='green', line_kws={'lw': 2}, label='Lowess Smoother (>65k)', ax=ax2)

ax2.text(0.05, 0.95, f'Point-Biserial Corr: {income_corr:.4f}\nP-Value: {income_p:.4e}', transform=ax2.transAxes, fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))
ax2.set_title('Ratio of Positive Outcomes and Correlation by Income')
ax2.set_xlabel('Income')
ax2.set_ylabel('Ratio (Positive / Total)')
ax2.set_ylim(0, 0.50)
ax2.legend()
ax2.grid()

age_plot_df, age_corr, age_p = plot_ratio_positive_negative_with_corr(data, 'age', 'target')
sns.scatterplot(x='age', y='ratio_positive_negative', data=age_plot_df, color='blue', s=50, label='Data Points', ax=ax3)

age_plot_df_lt82 = age_plot_df[age_plot_df['age'] <= 82]
age_plot_df_gt82 = age_plot_df[age_plot_df['age'] > 82]

sns.regplot(x='age', y='ratio_positive_negative', data=age_plot_df_lt82, scatter=False,
            lowess=True, color='red', line_kws={'lw': 2}, label='Lowess Smoother (≤82)', ax=ax3)
sns.regplot(x='age', y='ratio_positive_negative', data=age_plot_df_gt82, scatter=False,
            lowess=True, color='green', line_kws={'lw': 2}, label='Lowess Smoother (>82)', ax=ax3)

ax3.text(0.05, 0.95, f'Point-Biserial Corr: {age_corr:.4f}\nP-Value: {age_p:.4e}', transform=ax3.transAxes, fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))
ax3.set_title('Ratio of Positive Outcomes and Correlation by Age')
ax3.set_xlabel('Age')
ax3.set_ylabel('Ratio (Positive / Total)')
ax3.set_ylim(0, 0.50)
ax3.legend()
ax3.grid()

plt.tight_layout()
plt.show()
```


    
![png](Main_files/Main_6_0.png)
    


These predictors are relatively week individually within the aggregate of the data. They also have mixed effects.

In model testing, "knotting" only the variable age yielded the best results. 


```python
data['age_lt80'] = np.where(data['age'] < 80, data['age'], 80)
data['age_ge80'] = np.where(data['age'] >= 80, data['age'] - 79, 0)
print(data.head())
```

       id  age  dist  income  gender  marital_status  target  age_lt80  age_ge80
    0   1   73     4   95000       1               0       1        73         0
    1   2   89     1  125000       1               1       1        80        10
    2   3   85     1   15000       0               0       1        80         6
    3   4   76     2   95000       1               1       1        76         0
    4   5   76     2   15000       1               0       1        76         0


#### Examine strength of categorical predictors while checking for interactions.


```python
predictors = pd.DataFrame({
    'Single': [data[(data['gender'] == 0) & (data['marital_status'] == 0)]['target'].mean(),
               data[(data['gender'] == 1) & (data['marital_status'] == 0)]['target'].mean()],
    'Married': [data[(data['gender'] == 0) & (data['marital_status'] == 1)]['target'].mean(),
                data[(data['gender'] == 1) & (data['marital_status'] == 1)]['target'].mean()],
    'Overall': [data[data['gender'] == 0]['target'].mean(),
                data[data['gender'] == 1]['target'].mean()]
}, index=['Female', 'Male'])

print("Proportion of Gender that Responded") 
print(predictors)
```

    Proportion of Gender that Responded
              Single   Married  Overall
    Female  0.224093  0.219920  0.22151
    Male    0.214156  0.304501  0.27927


Gender effects relationship between marital_status and response. 

##### Evaluate Logistic Regression Model with every possible combination of interaction terms and select combination with best performance. Model performance is measured by the percentage of total responses captured among the top 40% of customers, ranked by predicted response probability visualized in a lift chart. 


```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.metrics import roc_curve, auc, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from itertools import combinations

base_features = ['age_lt80', 'age_ge80', 'dist', 'income', 'gender', 'marital_status']
interaction_terms = [
    ('age_lt80', 'dist'), ('age_lt80', 'income'), ('age_lt80', 'gender'), ('age_lt80', 'marital_status'),
    ('age_ge80', 'dist'), ('age_ge80', 'income'), ('age_ge80', 'gender'), ('age_ge80', 'marital_status'),
    ('dist', 'income'), ('dist', 'gender'), ('dist', 'marital_status'),
    ('income', 'gender'), ('income', 'marital_status'), ('gender', 'marital_status')
]

results = []

for r in range(len(interaction_terms) + 1):
    for terms in combinations(interaction_terms, r):
        X = data[base_features].copy()
        
        for t1, t2 in terms:
            X[f'{t1}_{t2}'] = X[t1] * X[t2]
            
        y = data['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = sm.Logit(y_train, sm.add_constant(X_train)).fit(disp=0)
        y_pred_proba_test = model.predict(sm.add_constant(X_test))
        
        test_data = pd.DataFrame({'target': y_test, 'predicted_probs': y_pred_proba_test})
        test_data_sorted = test_data.sort_values('predicted_probs', ascending=False)
        
        total_pos = test_data_sorted['target'].sum()
        n_rows = len(test_data_sorted)
        cutoff_index = int(0.4 * n_rows)
        lift_at_40 = test_data_sorted.iloc[:cutoff_index]['target'].sum() / total_pos * 100
        
        results.append({
            'num_interactions': len(terms),
            'interactions': terms,
            'lift_at_40': lift_at_40
        })

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('lift_at_40', ascending=False)

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

print("\nTop 10 interaction combinations by lift at 40%:")
display(results_df.head(10))
```

    
    Top 10 interaction combinations by lift at 40%:



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>num_interactions</th>
      <th>interactions</th>
      <th>lift_at_40</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>10034</th>
      <td>8</td>
      <td>((age_lt80, dist), (age_lt80, income), (age_lt80, gender), (age_lt80, marital_status), (age_ge80, income), (dist, income), (dist, marital_status), (gender, marital_status))</td>
      <td>62.672811</td>
    </tr>
    <tr>
      <th>13078</th>
      <td>9</td>
      <td>((age_lt80, dist), (age_lt80, income), (age_lt80, gender), (age_lt80, marital_status), (age_ge80, income), (age_ge80, marital_status), (dist, income), (dist, marital_status), (gender, marital_status))</td>
      <td>62.672811</td>
    </tr>
    <tr>
      <th>15077</th>
      <td>10</td>
      <td>((age_lt80, dist), (age_lt80, income), (age_lt80, gender), (age_lt80, marital_status), (age_ge80, income), (age_ge80, marital_status), (dist, income), (dist, gender), (income, gender), (income, marital_status))</td>
      <td>62.672811</td>
    </tr>
    <tr>
      <th>6175</th>
      <td>6</td>
      <td>((age_lt80, marital_status), (age_ge80, income), (age_ge80, marital_status), (dist, income), (dist, gender), (dist, marital_status))</td>
      <td>62.211982</td>
    </tr>
    <tr>
      <th>3131</th>
      <td>5</td>
      <td>((age_lt80, marital_status), (age_ge80, income), (dist, income), (dist, gender), (dist, marital_status))</td>
      <td>62.211982</td>
    </tr>
    <tr>
      <th>7895</th>
      <td>7</td>
      <td>((age_lt80, dist), (age_lt80, marital_status), (age_ge80, income), (age_ge80, marital_status), (dist, income), (dist, marital_status), (income, gender))</td>
      <td>62.211982</td>
    </tr>
    <tr>
      <th>1228</th>
      <td>4</td>
      <td>((age_lt80, marital_status), (age_ge80, marital_status), (dist, income), (income, gender))</td>
      <td>62.211982</td>
    </tr>
    <tr>
      <th>12731</th>
      <td>8</td>
      <td>((age_lt80, gender), (age_ge80, income), (age_ge80, marital_status), (dist, income), (dist, gender), (dist, marital_status), (income, gender), (income, marital_status))</td>
      <td>62.211982</td>
    </tr>
    <tr>
      <th>1242</th>
      <td>4</td>
      <td>((age_lt80, marital_status), (dist, income), (dist, gender), (income, gender))</td>
      <td>62.211982</td>
    </tr>
    <tr>
      <th>10602</th>
      <td>8</td>
      <td>((age_lt80, dist), (age_lt80, income), (age_lt80, marital_status), (age_ge80, marital_status), (dist, income), (dist, gender), (dist, marital_status), (income, marital_status))</td>
      <td>62.211982</td>
    </tr>
  </tbody>
</table>
</div>


#### Logistic Regression Model evaluated with best combination of interaction terms.


```python
best_terms = results_df.iloc[0]['interactions']

X = data[base_features].copy()
for t1, t2 in best_terms:
    X[f'{t1}_{t2}'] = X[t1] * X[t2]

y = data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = sm.Logit(y_train, sm.add_constant(X_train)).fit(disp=0)
y_pred_proba_test = model.predict(sm.add_constant(X_test))

test_data = pd.DataFrame({'target': y_test, 'predicted_probs': y_pred_proba_test})
for col in X_test.columns:
    test_data[col] = X_test[col]
test_data_sorted = test_data.sort_values('predicted_probs', ascending=False)

total_pos = test_data_sorted['target'].sum()
n_rows = len(test_data_sorted)
deciles = np.linspace(0, n_rows, 11, dtype=int)
lift_curve = [test_data_sorted.iloc[:i]['target'].sum() / total_pos * 100 for i in deciles]
baseline = np.linspace(0, 100, 11)

print(f'Yield at 40%: {lift_curve[4]:.1f}%')

plt.figure(figsize=(10, 6), dpi=300)
plt.plot(baseline, lift_curve, marker='o', label='Lift Curve', linewidth=2)
plt.plot(baseline, baseline, linestyle='--', marker='o', label='Baseline', linewidth=2)
plt.xlabel('Percentage of Data', fontsize=12)
plt.ylabel('Percentage of Positive Cases', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6), dpi=300)
plt.plot(range(11), np.array(lift_curve) / baseline, marker='o', linewidth=2)
plt.axhline(y=1, linestyle='--', color='r', linewidth=2)
plt.xlabel('Observations', fontsize=12)
plt.ylabel('Lift Curve / Baseline', fontsize=12)
plt.title('Advantage Plot', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

from sklearn.metrics import roc_curve, auc
fpr, tpr, _ = roc_curve(y_test, y_pred_proba_test)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 6), dpi=300)
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})', linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("\nTop 10 observations by predicted probability:")
display(test_data_sorted[['target', 'predicted_probs'] + base_features].head(10))

print("\nBottom 10 observations by predicted probability:")
display(test_data_sorted[['target', 'predicted_probs'] + base_features].tail(10))

```

    Yield at 40%: 62.7%



    
![png](Main_files/Main_16_1.png)
    



    
![png](Main_files/Main_16_2.png)
    



    
![png](Main_files/Main_16_3.png)
    


    
    Top 10 observations by predicted probability:



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>target</th>
      <th>predicted_probs</th>
      <th>age_lt80</th>
      <th>age_ge80</th>
      <th>dist</th>
      <th>income</th>
      <th>gender</th>
      <th>marital_status</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1405</th>
      <td>0</td>
      <td>0.593696</td>
      <td>80</td>
      <td>1</td>
      <td>2</td>
      <td>5000</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2586</th>
      <td>1</td>
      <td>0.574583</td>
      <td>80</td>
      <td>2</td>
      <td>0</td>
      <td>15000</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>184</th>
      <td>0</td>
      <td>0.572051</td>
      <td>80</td>
      <td>6</td>
      <td>1</td>
      <td>5000</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3331</th>
      <td>1</td>
      <td>0.565112</td>
      <td>80</td>
      <td>2</td>
      <td>3</td>
      <td>15000</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>279</th>
      <td>1</td>
      <td>0.563361</td>
      <td>80</td>
      <td>1</td>
      <td>5</td>
      <td>15000</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2682</th>
      <td>0</td>
      <td>0.548136</td>
      <td>80</td>
      <td>5</td>
      <td>4</td>
      <td>15000</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>862</th>
      <td>0</td>
      <td>0.545801</td>
      <td>80</td>
      <td>11</td>
      <td>2</td>
      <td>5000</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>893</th>
      <td>0</td>
      <td>0.543115</td>
      <td>80</td>
      <td>2</td>
      <td>3</td>
      <td>25000</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>63</th>
      <td>0</td>
      <td>0.536778</td>
      <td>80</td>
      <td>2</td>
      <td>0</td>
      <td>35000</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3379</th>
      <td>0</td>
      <td>0.535007</td>
      <td>80</td>
      <td>3</td>
      <td>11</td>
      <td>15000</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>


    
    Bottom 10 observations by predicted probability:



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>target</th>
      <th>predicted_probs</th>
      <th>age_lt80</th>
      <th>age_ge80</th>
      <th>dist</th>
      <th>income</th>
      <th>gender</th>
      <th>marital_status</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1476</th>
      <td>0</td>
      <td>0.034667</td>
      <td>65</td>
      <td>0</td>
      <td>18</td>
      <td>125000</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>990</th>
      <td>0</td>
      <td>0.033738</td>
      <td>66</td>
      <td>0</td>
      <td>18</td>
      <td>225000</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1532</th>
      <td>0</td>
      <td>0.032145</td>
      <td>65</td>
      <td>0</td>
      <td>4</td>
      <td>250000</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>538</th>
      <td>0</td>
      <td>0.030922</td>
      <td>68</td>
      <td>0</td>
      <td>7</td>
      <td>250000</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1926</th>
      <td>1</td>
      <td>0.028749</td>
      <td>79</td>
      <td>0</td>
      <td>22</td>
      <td>225000</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1916</th>
      <td>0</td>
      <td>0.028738</td>
      <td>70</td>
      <td>0</td>
      <td>17</td>
      <td>187500</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3594</th>
      <td>0</td>
      <td>0.024361</td>
      <td>80</td>
      <td>14</td>
      <td>18</td>
      <td>225000</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>270</th>
      <td>0</td>
      <td>0.021723</td>
      <td>80</td>
      <td>3</td>
      <td>25</td>
      <td>225000</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3052</th>
      <td>0</td>
      <td>0.021652</td>
      <td>67</td>
      <td>0</td>
      <td>18</td>
      <td>187500</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2476</th>
      <td>0</td>
      <td>0.016573</td>
      <td>65</td>
      <td>0</td>
      <td>12</td>
      <td>250000</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



```python
import seaborn as sns
import matplotlib.pyplot as plt

def plot_full_pairplot_with_corr_binary_target(df, continuous_vars, target_var='target', positive_value=1, negative_value=0):
    pairplot_data = df[continuous_vars + [target_var]].copy()
    pairplot_data[target_var] = pairplot_data[target_var].astype('category')
    correlation_matrix = pairplot_data[continuous_vars].corr()
    g = sns.pairplot(pairplot_data, hue=target_var, palette={positive_value: 'green', negative_value: 'red'}, diag_kind='kde', corner=False)
    for i, row_var in enumerate(continuous_vars):
        for j, col_var in enumerate(continuous_vars):
            if i != j:
                g.axes[i, j].annotate(f"Corr: {correlation_matrix.loc[row_var, col_var]:.2f}", xy=(0.5, 0.1), xycoords="axes fraction", ha="center", fontsize=9, color="blue")
    plt.suptitle(f"Full Pair Plot for Predictors Colored by '{target_var}'", y=1.02, fontsize=16)
    plt.show()

plot_full_pairplot_with_corr_binary_target(data, ['age_lt80', 'age_ge80', 'dist', 'income', 'gender', 'marital_status'], 'target')
```


    
![png](Main_files/Main_17_0.png)
    


#### As the data lacks strongly predictive patterns between predictors and response, the logistic regression model’s moderate confidence at best in response is actually an appropriate representation of the true relationship available to us in the given data.
