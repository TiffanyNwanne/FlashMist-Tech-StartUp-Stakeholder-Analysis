# Methodology

## Introduction

These is a framework of all the methods applied to perform this analysis

## Data Extraction

The company’s database is hosted by PostgreSQL and the data for the analysis was extracted using this query

```sql

SELECT
    user_id,
    ab_group,
    converted,
    retention_days,
    retained,
    clickstream_path,
    review_text,
    sentiment,
    pipeline_status
FROM flashmist_data;

```
[![](https://github.com/TiffanyNwanne/FlashMist-Tech-StartUp-Stakeholder-Analysis/blob/main/Images/flashmist%20pgsql%20database.png)](https://github.com/TiffanyNwanne/FlashMist-Tech-StartUp-Stakeholder-Analysis/blob/main/Images/flashmist%20pgsql%20database.png)

The results were then exported as a CSV file.

## 1. **A/B Testing for Product Feature**

### Data Exploration

In order to determine which onboarding UI version improves conversion, and whether differences are statistically significant, I tried to understand base conversion rate per group and check if group sizes were similar for clean A/B test.

```python

df[['ab_group', 'converted']].groupby('ab_group').agg(['mean', 'count'])

```

### Data Cleaning

I ensure only valid `A` or `B` groups exist and removed missing/null `converted` values

```python

df = df[df['ab_group'].isin(['A', 'B'])]
df = df.dropna(subset=['converted'])

```

### Data Analysis

**Q1: Which version leads to better conversion?**

```python

conversion_rates = df.groupby('ab_group')['converted'].mean()

```

**Q2: Are the observed differences statistically significant?**

```python

from scipy.stats import ttest_ind

group_a = df[df['ab_group'] == 'A']['converted']
group_b = df[df['ab_group'] == 'B']['converted']

t_stat, p_value = ttest_ind(group_a, group_b)

```

- **Insight:** If `p < 0.05`, difference is statistically significant.

**Q3: What user segments respond best?**

```python

df.groupby(['ab_group', 'retained'])['converted'].mean().unstack()

```

## 2. **User Retention Analysis**

---

### Data Exploration

I tried to identify usage patterns and behaviors that influence retention.

```python
df['retention_days'].describe()
df['retained'].value_counts(normalize=True)

```

- Understand retention distribution
- Explore correlation with `converted`, `clickstream_path`

### Data Cleaning

I removed outlier retention values (e.g., > 30 if capped at first month) and standardized clickstream paths for parsing

```python
df = df[df['retention_days'] <= 30]

```

### Data Analysis

**Q1: Key drop-off points?**

```python

df['retention_days'].plot.hist(bins=30)

```

- Histogram of days shows drop-off trends

**Q2: Product usage vs. retention**

```python
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
click_matrix = vectorizer.fit_transform(df['clickstream_path'])

# Compare average frequency of each page between retained vs. non-retained

```

---

## 3. **Sentiment Analysis on Product Reviews**

---

### 🔍 Data Exploration

```python
df['sentiment'].value_counts()
df['review_text'].sample(5)

```

### Data Cleaning

I normalized the review text and remove empty or irrelevant reviews

```python

df = df[df['review_text'].notnull()]

```

### Data Analysis

**Q1: Prevailing sentiments?**

```python

df['sentiment'].value_counts(normalize=True)

```

## 4. **Clickstream Analysis**

---

### Data Exploration

I tried to understand how users navigate the app and how it affects retention/churn.

```python
df['clickstream_path'].sample(5)

```

### Data Cleaning

I had to split clickstream strings and remove anomalies or repeated entries.

```python
df['clickstream_list'] = df['clickstream_path'].str.split(' > ')

```

### Data Analysis

**Q1: Navigation through key workflows**

```python
# Extract path frequency or key transitions
from collections import Counter
transitions = []

for path in df['clickstream_list']:
    transitions += zip(path, path[1:])

Counter(transitions).most_common(10)

```

**Q3: Clickstream vs. churn**

```python

# Churn if not retained
df['churned'] = 1 - df['retained']

# Analyze avg. steps or last page before churn

```

---
