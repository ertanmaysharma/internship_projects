import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

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

female = female_df.to_numpy()
male   = male_df.to_numpy()

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


plt.figure(figsize=(7,5))
plt.boxplot([female_weights, male_weights], labels=['Female','Male'])
plt.ylabel('Weight (kg)')
plt.title('Comparison of female and male weights')
plt.grid(axis='y', linestyle=':', alpha=0.5)
plt.show()

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

female_df = female_df.copy()  # operate on a copy to be safe
female_df['BMI'] = female_df['weight_kg'] / ( (female_df['height_cm'] / 100.0) ** 2 )
female = female_df.to_numpy()

print('Added BMI column — now female DataFrame has columns:', female_df.columns.tolist())
print(female_df[['weight_kg','height_cm','BMI']].head())

numeric_cols = female_df.columns.tolist()  
zfemale_df = (female_df - female_df.mean()) / female_df.std(ddof=0)
zfemale = zfemale_df.to_numpy()

print('zfemale shape:', zfemale.shape)
print('\nPreview (first 5 rows):')
print(zfemale_df.head())

vars_of_interest = ['height_cm','weight_kg','waist_circumference_cm','hip_circumference_cm','BMI']

sns.pairplot(zfemale_df[vars_of_interest], diag_kind='hist', plot_kws={'s': 15, 'alpha': 0.6})
plt.suptitle('Pairplot of standardized female variables', y=1.02)
plt.show()

pearson_corr = zfemale_df[vars_of_interest].corr(method='pearson')
spearman_corr = zfemale_df[vars_of_interest].corr(method='spearman')

print('Pearson correlation matrix:\n', pearson_corr.round(3))
print('\nSpearman correlation matrix:\n', spearman_corr.round(3))

for df in (female_df, male_df):
    df['waist_to_height_ratio'] = df['waist_circumference_cm'] / df['height_cm']
    df['waist_to_hip_ratio'] = df['waist_circumference_cm'] / df['hip_circumference_cm']

female = female_df.to_numpy()
male = male_df.to_numpy()

print('Female columns now:', female_df.columns.tolist())
print('Male columns now:', male_df.columns.tolist())

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

inds_sorted = female_df['BMI'].argsort().to_numpy()
lowest5_inds = inds_sorted[:5]
highest5_inds = inds_sorted[-5:][::-1]  # highest to lowest order

print('Indices of 5 lowest BMI (female dataset):', lowest5_inds)
print('Indices of 5 highest BMI (female dataset):', highest5_inds)

print('\nStandardised measurements for 5 lowest BMI females:')
print(zfemale_df.iloc[lowest5_inds][numeric_cols].round(3))

print('\nStandardised measurements for 5 highest BMI females:')
print(zfemale_df.iloc[highest5_inds][numeric_cols].round(3))


