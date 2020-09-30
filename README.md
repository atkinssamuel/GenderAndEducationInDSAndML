# Salary Data Exploration
## Objectives
1. Perform exploratory data analysis to analyze the survey dataset and to summarize its main
characteristics. Present 3 graphical figures that represent different trends in the data. For your
explanatory data analysis, you can consider Country, Age, Education, Professional Experience, and
Salary.
2. Estimate the difference between average salary (Q10) of males vs. females. 
    - Compute and report descriptive statistics for each group (remove missing data, if
necessary).
    - If suitable, perform a two-sample t-test with 0.05 threshold. Explain your rationale.
    - Bootstrap your data for comparing the mean of salary (Q10) for the two groups.
Note that the number of instances you sample from each group should be relative to its
size. Use 1000 replications. Plot two bootstrapped distributions (for males and females)
and the distribution of the difference in means.
    - If suitable, perform a two-sample t-test with 0.05 threshold on the bootstrapped
data. Explain your rationale.
    - Comment on your findings.
3. Select “highest level of formal education” (Q4) from the dataset and repeat steps a to e, this
time use analysis of variance (ANOVA) instead of t test for hypothesis testing to compare the means
of salary for three groups (Bachelor’s degree, Doctoral degree, and Master’s degree). 