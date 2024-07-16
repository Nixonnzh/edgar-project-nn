import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

yahoo_filename = R'C:\edgar\test_folder_csv\yahoo_csv.csv'
ten_k_filename = R'C:\edgar\test_folder_csv\test_csv.csv'
yahoo_df = pd.read_csv(yahoo_filename)
ten_k_df = pd.read_csv(ten_k_filename)

df = pd.merge(yahoo_df, ten_k_df, left_on = ['Symbol', 'date'], right_on = ['ticker', 'filing_date'], how = 'inner')
df['positive_prop'] = df['positive'] / df['word_count']
df['negative_prop'] = df['negative'] / df['word_count']
df['uncertainty_prop'] = df['uncertainty'] / df['word_count']
df['litigious_prop'] = df['litigious'] / df['word_count']
df['constraining_prop'] = df['constraining'] / df['word_count']
df['modal_prop'] = df['modal'] / df['word_count']
df.drop(columns = ['positive', 'negative', 'uncertainty', 'litigious', 'constraining', 'modal', 'Symbol', 'date', 'ticker', 'filing_date', 'word_count', 'high', 'low', 'price', 'volume'], inplace = True)

df_train, df_test = train_test_split(df, test_size = 0.2, random_state = 0)
features = ['positive_prop', 'negative_prop', 'uncertainty_prop', 'litigious_prop', 'constraining_prop', 'modal_prop']

x_train = df_train[features]
y_train = df_train['1dailyreturn']

x_test = df_test[features]
y_test = df_test['1dailyreturn']

model = LinearRegression()
model.fit(x_train, y_train)

corr = df.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values,
            vmin = -1,
            vmax = 1,
            cmap = 'PiYG',
            center = 0,
            annot = True)

with open(R'C:\edgar\test_folder_csv\linear_correlation.csv', 'w', newline = '') as output_file:
    corr.to_csv(output_file)

y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)

print(f'r2 score is: {r2}')

sns.relplot(x = y_test, y=y_pred, color = '#67B587')

x = np.linspace(-0.125, 0.075)
y = x
plt.ylim(-0.015, 0.015)
plt.plot(x, y, color = '#000000', linestyle = '--', linewidth = 1)
plt.show()
