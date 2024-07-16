import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
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

corr = df.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values,
            vmin = -1,
            vmax = 1,
            cmap = 'PiYG',
            center = 0,
            annot = True)
with open(R'C:\edgar\test_folder_csv\tree_correlation.csv', 'w', newline = '') as output_file:
    corr.to_csv(output_file)

def hyper_tuning(x_train, y_train, x_test, y_test, depth_list, sample_list):
    data = []
    for depth in depth_list:
        for sample in sample_list:
            inner_dict = {}
            inner_dict['depth'] = depth
            dt = DecisionTreeRegressor(max_depth = depth, min_samples_leaf = sample, random_state = 0)
            dt.fit(x_train, y_train)
            y_pred = dt.predict(x_test)
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = mse**0.5
            mae = mean_absolute_error(y_test,y_pred)
            inner_dict['sample'] = sample
            inner_dict['mae'] = mae
            inner_dict['mse'] = mse
            inner_dict['rmse'] = rmse
            inner_dict['r2'] = r2
            data.append(inner_dict)
            data.append(inner_dict)
            
    summary_df = pd.DataFrame(data)
    summary_df.sort_values('rmse', inplace = True)
    output = summary_df.head(1)
    return output

depth_list = [3,4,5,6,7,8,9,10]
sample_list = [0.3, 0.2, 0.15, 0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02]
hyper_parameters = hyper_tuning(x_train, y_train, x_test, y_test, depth_list, sample_list)

dtr = DecisionTreeRegressor(min_samples_leaf = hyper_parameters['sample'].item(),
                            max_depth = hyper_parameters['depth'].item(),
                            random_state = 0)
dtr.fit(x_train, y_train)
y_pred = dtr.predict(x_test)

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse**0.5
print(f'r2 score is: {r2}')
print(f'rmse score is {rmse}')

plt.figure(figsize = (10, 5))
plot_tree(dtr, feature_names = features, max_depth = 3, filled = True)
plt.show()
