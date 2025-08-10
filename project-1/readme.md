# **purpose**:adding important libraries required in the project
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
```


# **purpose**: set a default size of plots , fetch the data used in project and 
# read the CSV files.
```
plt.rcParams['figure.figsize'] = (9, 5)
female_paths = '/content/sample_data/nhanes_adult_female_bmx_2020.csv'
male_paths = '/content/sample_data/nhanes_adult_male_bmx_2020.csv'
columns = [
    'weight_kg',           
    'height_cm',           
    'upper_arm_length_cm', 
    'upper_leg_length_cm', 
    'arm_circumference_cm', 
    'hip_circumference_cm', 
    'waist_circumference_cm'
]

female_df = pd.read_csv(female_paths, comment='#', header=0, skiprows=[1], names=columns)
male_df = pd.read_csv(male_paths, comment='#', header=0, skiprows=[1], names=columns)
```
# **purpose**: convert to NumPy matrices as requested


```
female = female_df.to_numpy()
male   = male_df.to_numpy()
```

# **interpretation**:
# - We have loaded two datasets as pandas DataFrames and NumPy arrays.
# - The column names are the seven variables described in the brief.
---

# **Purpose**: draw two histograms for ploting female weights (First)and male weights(Second) make sure they have identical x-axis limits.


```
female_weights = female_df['weight_kg'].values
male_weights   = male_df['weight_kg'].values
all_min = min(female_weights.min(), male_weights.min())
all_max = max(female_weights.max(), male_weights.max())
pad = (all_max - all_min) * 0.05
xlim = (all_min - pad, all_max + pad)

fig, axes = plt.subplots(2, 1, sharex=True)
axes[0].hist(female_weights, bins=20)
axes[0].set_title('Female weights (kg)')

axes[1].hist(male_weights, bins=20)
axes[1].set_title('Male weights (kg)')
for ax in axes:
    ax.set_xlim(xlim)
    ax.set_ylabel('Count')
axes[-1].set_xlabel('Weight (kg)')
plt.tight_layout()
plt.show()
```

# **interpretation**:
# - Plots show the distribution of weights for females and males.
# - Using identical x-axis makes the graphs to compare easily



---


#** Purpose**: show side-by-side box-and-whisker plots for female and male weights to compare medians,


```
plt.figure(figsize=(7,5))
plt.boxplot([female_weights, male_weights], labels=['Female','Male'])
plt.ylabel('Weight (kg)')
plt.title('Comparison of female and male weights')
plt.grid(axis='y', linestyle=':', alpha=0.5)
plt.show()
```
# **interpretation:**
# - By inspect median lines to see which group has higher central tendency we find that males have the higher central tendency
# - we can see that IQR of both males and females is approximately the same
# - both males and females have more spread on the higher-weighing side 
# - females have less high-end outliers than males



---
# Purpose: compute mean, median, std, variance, min, max, quartiles, skewness and kurtosis and compare the weights of both males and females



```
def summarise_vector(v):
    return {
        'count': int(np.size(v)),
        'mean': float(np.mean(v)),
        'median': float(np.median(v)),
        'std': float(np.std(v, ddof=0)),
        'var': float(np.var(v, ddof=0)),
        'min': float(np.min(v)),
        'q1': float(np.percentile(v,25)),
        'q3': float(np.percentile(v,75)),
        'max': float(np.max(v)),
        'skewness': float(stats.skew(v)),
        'kurtosis': float(stats.kurtosis(v))
    }

female_stats = summarise_vector(female_weights)
male_stats = summarise_vector(male_weights)

summary_df = pd.DataFrame([female_stats, male_stats], index=['Female','Male']).T
print(summary_df.round(3))

```
#**interpretation**:
# - male median weight is higher than female median weight
# - male mean weight is higher than female mean weight 
# - By comparing means and median we found that both males and females are right-skewed with a tail extending towards heavier weights.
# - males have a slightly wider IQR, meaning more variability in the middle 50% of their weights.
# - Both groups have longer upper whiskers than lower whiskers, showing greater spread in higher weights before outliers.



---



# **Purpose**: compute BMI for females and append it as a new column named 'BMI' to the female DataFrame and update Numpy matrix(female) to include BMI as 8th column.


```
female_df = female_df.copy()  # operate on a copy to be safe
female_df['BMI'] = female_df['weight_kg'] / ( (female_df['height_cm'] / 100.0) ** 2 )
female = female_df.to_numpy()

print('Added BMI column — now female DataFrame has columns:', female_df.columns.tolist())
print(female_df[['weight_kg','height_cm','BMI']].head())
```
# **Interpretation**:
# - BMI is computed as weight(kg) divided by (height(m))^2.
# - We added the BMI column: the NumPy matrix female now has 8 columns.



---



# **Purpose**: standardise each numeric column in the female DataFrame (subtract mean, divide by standard deviation) and produce a new NumPy matrix zfemale containing z-scores.


```
numeric_cols = female_df.columns.tolist()  # includes BMI now
zfemale_df = (female_df - female_df.mean()) / female_df.std(ddof=0)
zfemale = zfemale_df.to_numpy()

print('zfemale shape:', zfemale.shape)
print('\nPreview (first 5 rows):')
print(zfemale_df.head())

```


# **Interpretation**:
# - Each column now has mean approximately 0 and standard deviation 1.
# - Standardisation is necessary if we want variables on the same scale for methods sensitive to scale.



---

# **Purpose**: draw a scatterplot matrix for standardized height, weight, waist circ, hip circ, BMI and compute Pearson's and Spearman's correlation coefficients for all pairs.

```
vars_of_interest = ['height_cm','weight_kg','waist_circumference_cm','hip_circumference_cm','BMI']

sns.pairplot(zfemale_df[vars_of_interest], diag_kind='hist', plot_kws={'s': 15, 'alpha': 0.6})
plt.suptitle('Pairplot of standardized female variables', y=1.02)
plt.show()

pearson_corr = zfemale_df[vars_of_interest].corr(method='pearson')
spearman_corr = zfemale_df[vars_of_interest].corr(method='spearman')

print('Pearson correlation matrix:\n', pearson_corr.round(3))
print('\nSpearman correlation matrix:\n', spearman_corr.round(3))
```

# **Interpretation**:
# - Pearsons measures linear relationships; Spearman's measures monotonic relationships.
# - Values near ±1 indicate strong relationships.
#   relationships differ from linear ones.



---



# **Purpose**: add two columns (waist_to_height_ratio and waist_to_hip_ratio) to both DataFrames and to their NumPy matrix equivalents.


```
for df in (female_df, male_df):
    df['waist_to_height_ratio'] = df['waist_circumference_cm'] / df['height_cm']
    df['waist_to_hip_ratio'] = df['waist_circumference_cm'] / df['hip_circumference_cm']

female = female_df.to_numpy()
male = male_df.to_numpy()

print('Female columns now:', female_df.columns.tolist())
print('Male columns now:', male_df.columns.tolist())
```

# **Interpretation**:
# - Both ratios are unitless and useful adiposity indicators.



---



# **Purpose**: visually compare distributions of the two ratios between females and males using four boxes


```
female_whr = female_df['waist_to_height_ratio']
male_whr   = male_df['waist_to_height_ratio']
female_whip = female_df['waist_to_hip_ratio']
male_whip   = male_df['waist_to_hip_ratio']

plt.figure(figsize=(9,6))
plt.boxplot([female_whr, male_whr, female_whip, male_whip], labels=['Female W/H', 'Male W/H', 'Female W/Hip', 'Male W/Hip'])
plt.ylabel('Ratio (unitless)')
plt.title('Comparison of waist-to-height and waist-to-hip ratios by sex')
plt.grid(axis='y', linestyle=':', alpha=0.5)
plt.show()
```

# **Interpretation**:
# - Females have higher median in waist-to-height ratio than males
# -  Males have higher median in waist-to-hip ration than females



---



# **Purpose**: provide a concise list of pros/cons for the three adiposity indicators.

# **BMI (Body Mass Index)**:
#   Advantages:
- Simple to compute from weight and height.
- Widely used and allows broad population-level comparisons.
# Disadvantages:
- Does not distinguish fat mass from lean mass .
- Does not describe fat distribution (visceral vs subcutaneous).

#**Waist-to-Height Ratio (WHtR)**:
#Advantages:
- Accounts for central adiposity relative to height; better correlate of cardiometabolic risk than BMI in many studies.
- Unitless and easy to communicate .
#Disadvantages:
- Does not provide body composition details; measurement technique matters (consistency required).

#**Waist-to-Hip Ratio (WHR)**:
#Advantages:
- Captures body fat distribution (apple vs pear shapes); linked to cardiovascular risk.
#Disadvantages:
- Requires two consistent measurements; can be influenced by hip anatomy, not just adiposity.
- Slightly more complex for non-expert self-measurement.
---
# **Purpose**: identify the 5 smallest and 5 largest BMI values and print the standardized body measurements for those persons (zfemale_df rows). 

```
inds_sorted = female_df['BMI'].argsort().to_numpy()
lowest5_inds = inds_sorted[:5]
highest5_inds = inds_sorted[-5:][::-1]  # highest to lowest order

print('Indices of 5 lowest BMI (female dataset):', lowest5_inds)
print('Indices of 5 highest BMI (female dataset):', highest5_inds)

print('\nStandardised measurements for 5 lowest BMI females:')
print(zfemale_df.iloc[lowest5_inds][numeric_cols].round(3))

print('\nStandardised measurements for 5 highest BMI females:')
print(zfemale_df.iloc[highest5_inds][numeric_cols].round(3))
```
# **Interpretation**:
# - This highlights how their other measures compare relative to the female population.
# - For the lowest BMI group other body measures (height, waist, hip) are also low (z<0)
# - For the highest BMI group , elevated waist z-scores indicate central adiposity.



---
