import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import SpectralBiclustering

ausact_bic_array = np.random.rand(10, 10)
model = SpectralBiclustering(n_clusters=3, random_state=0)  
model.fit(ausact_bic_array)    
bcn = model.rows_
print(bcn)                            
bcn = [
    np.array([0, 1, 2]),
    np.array([3, 4]),     
    np.array([5, 6, 7, 8])  
]
cl12 = np.full(ausact_bic_array.shape[0], np.nan)
for k, bicluster in enumerate(bcn):
    cl12[bicluster] = k + 1 
print(cl12)  
cl12 = np.array([1, 2, 3, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, np.nan, 3, 4, np.nan])
cl12_series = pd.Series(cl12)
frequency_table = cl12_series.value_counts(dropna=False)
print(frequency_table) 
cl12 = np.array([1, 2, 3, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, np.nan, 3, 4, np.nan])
cl12_series = pd.Series(cl12)

cl12_3 = (cl12_series.notna() & (cl12_series == 3))
cl12_3 = cl12_3.replace({False: "Not Segment 3", True: "Segment 3"})
print(cl12_3)

spendpppd = np.random.rand(100) * 100  
cl12_3 = np.random.choice(["Not Segment 3", "Segment 3"], size=100) 
df = pd.DataFrame({'spendpppd': spendpppd, 'cl12_3': cl12_3})

plt.figure(figsize=(8, 6))
sns.boxplot(x='cl12_3', y='spendpppd', data=df, notch=True, width=0.6)

plt.yscale('log')

plt.ylabel('AUD per person per day')

plt.show()

ausActivDesc = pd.DataFrame({
    'book1': np.random.randint(0, 10, size=100),
    'book2': np.random.randint(0, 10, size=100),
    'book3': np.random.randint(0, 10, size=100),
    'book4': np.random.randint(0, 10, size=100),
    'cl12.3': np.random.choice(["Not Segment 3", "Segment 3"], size=100)
})

book_columns = [col for col in ausActivDesc.columns if col.startswith("book")]

prop_df = ausActivDesc[book_columns].apply(lambda x: x / x.sum(), axis=0)


plt.figure(figsize=(8, 6))


sns.barplot(data=prop_df, orient='h')
plt.xlabel('Percent')
plt.xlim(-2, 102) 
plt.ylabel('Books')

plt.show()

ausActivDesc = pd.DataFrame({
    'info1': np.random.randint(0, 10, size=100),
    'info2': np.random.randint(0, 10, size=100),
    'info3': np.random.randint(0, 10, size=100),
    'info4': np.random.randint(0, 10, size=100),
    'cl12.3': np.random.choice(["Not Segment 3", "Segment 3"], size=100)
})
info_columns = [col for col in ausActivDesc.columns if col.startswith("info")]

prop_df = ausActivDesc[info_columns].apply(lambda x: x / x.sum(), axis=0)

plt.figure(figsize=(8, 6))

sns.barplot(data=prop_df, orient='h')

plt.xlabel('Percent')
plt.xlim(-2, 102) 
plt.ylabel('Info Categories')


plt.show()

ausActivDesc = pd.DataFrame({
    'TV.channel': np.random.choice(['Channel 1', 'Channel 2', 'Channel 3'], size=100),
    'cl12.3': np.random.choice(["Not Segment 3", "Segment 3"], size=100)
})


contingency_table = pd.crosstab(ausActivDesc['cl12.3'], ausActivDesc['TV.channel'])


plt.figure(figsize=(8, 6))
mosaic(contingency_table, colorizer=lambda x: "lightblue" if x >= 0 else "lightcoral")

plt.xlabel('cl12.3 (Group Segmentation)')
plt.ylabel('TV Channel')
plt.title('Mosaic Plot of TV Channel vs Group Segmentation')


plt.show() 