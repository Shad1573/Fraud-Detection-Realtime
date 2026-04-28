import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the data
# We use ../ because the script is in /src and the data is in /data
df = pd.read_csv('data/creditcard.csv') 

print("--- Dataset Info ---")
print(df.info())

print("\n--- First 5 Rows ---")
print(df.head())

# 2. Check for Imbalance (The "Fraud" problem)
fraud_count = df['Class'].value_counts()
print("\n--- Transaction Counts ---")
print(fraud_count)

# 3. Quick Visualization
plt.figure(figsize=(8, 5))
sns.barplot(x=fraud_count.index, y=fraud_count.values, palette='viridis')
plt.title('Transaction Distribution (0: Normal, 1: Fraud)')
plt.ylabel('Number of Transactions')
plt.xlabel('Class')
plt.show()