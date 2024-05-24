import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import re
import missingno as mso
from scipy import stats
from scipy.stats import ttest_ind
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd

# Đọc dữ liệu từ file CSV
data = pd.read_csv("seattle-weather.csv")
data.head()
data.shape

import warnings
warnings.filterwarnings('ignore') 

# Đếm số lượng từng loại thời tiết
sns.countplot(x="weather", data=data, palette='hls')
countrain = len(data[data.weather == 'rain'])
countsun = len(data[data.weather == 'sun'])
countdrizzle = len(data[data.weather == 'drizzle'])
countsnow = len(data[data.weather == 'snow'])
countfog = len(data[data.weather == 'fog'])



# Tính phần trăm từng loại thời tiết
print('percent of rain:{:.2f}%'.format((countrain / len(data.weather)) * 100))
print('percent of sun:{:.2f}%'.format((countsun / len(data.weather)) * 100))
print('percent of drizzle:{:.2f}%'.format((countdrizzle / len(data.weather)) * 100))
print('percent of snow:{:.2f}%'.format((countsnow / len(data.weather)) * 100))
print('percent of fog:{:.2f}%'.format((countfog / len(data.weather)) * 100))

# Mô tả thống kê các cột cụ thể
data[['precipitation', 'temp_max', 'temp_min', 'wind']].describe()

# Biểu đồ phân phối các biến số
sns.set(style='darkgrid')
fig, axs = plt.subplots(2, 2, figsize=(10, 8))
sns.histplot(data=data, x='precipitation', kde=True, ax=axs[0, 0], color='green')
sns.histplot(data=data, x='temp_max', kde=True, ax=axs[0, 1], color='red')
sns.histplot(data=data, x='temp_min', kde=True, ax=axs[1, 0], color='blue')
sns.histplot(data=data, x='wind', kde=True, ax=axs[1, 1], color='orange')

# Biểu đồ violin cho các biến số
sns.set(style='darkgrid')
fig, axs = plt.subplots(2, 2, figsize=(10, 8))
sns.violinplot(data=data, x='precipitation', ax=axs[0, 0], color='green')
sns.violinplot(data=data, x='temp_max', ax=axs[0, 1], color='red')
sns.violinplot(data=data, x='temp_min', ax=axs[1, 0], color='blue')
sns.violinplot(data=data, x='wind', ax=axs[1, 1], color='orange')

# Biểu đồ hộp (boxplot) cho các biến số so với weather
plt.figure(figsize=(12, 6))
sns.boxplot(x='weather', y='precipitation', data=data, palette='YlOrBr')
plt.figure(figsize=(12, 6))
sns.boxplot(x='weather', y='temp_max', data=data, palette='inferno')
plt.figure(figsize=(12, 6))
sns.boxplot(x='wind', y='weather', data=data, palette='YlOrBr')
plt.figure(figsize=(12, 6))
sns.boxplot(x='temp_min', y='weather', data=data, palette='YlOrBr')



# Loại bỏ cột 'date' nếu có
data_numeric = data.select_dtypes(include=['number'])

# Tính toán ma trận tương quan và vẽ heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(data_numeric.corr(), annot=True, cmap='coolwarm')



# Biểu đồ scatter cho các biến số
data.plot("precipitation", 'temp_max', style='o')
print('pearsons correlation: ', data['precipitation'].corr(data['temp_max']))
print('T test and P value: ', stats.ttest_ind(data['precipitation'], data['temp_max']))

data.plot("wind", 'temp_max', style='o')
print('pearsons correlation: ', data['wind'].corr(data['temp_max']))
print('T test and P value: ', stats.ttest_ind(data['wind'], data['temp_max']))

data.plot('temp_max', 'temp_min', style='o')

# Kiểm tra giá trị thiếu
data.isna().sum()
plt.figure(figsize=(12, 6))
axz = plt.subplot(1, 2, 2)
mso.bar(data.drop(['date'], axis=1), ax=axz, fontsize=12)

# Loại bỏ cột 'date' vì không cần thiết
data = data.drop(['date'], axis=1)

# Loại bỏ các giá trị ngoại lai (outliers)
# Chọn các cột dạng số
numeric_columns = data.select_dtypes(include=['number'])

# Tính toán phân vị
Q1 = numeric_columns.quantile(0.25)
Q3 = numeric_columns.quantile(0.75)
IQR = Q3 - Q1

# Loại bỏ các dòng nằm ngoài khoảng ngoại lệ
data = data[~((numeric_columns < (Q1 - 1.5 * IQR)) | (numeric_columns > (Q3 + 1.5 * IQR))).any(axis=1)]


# Chuẩn hóa dữ liệu bằng căn bậc hai (sqrt)
import numpy as np
data.precipitation = np.sqrt(data.precipitation)
data.wind = np.sqrt(data.wind)

# Biểu đồ phân phối sau khi chuẩn hóa
sns.set(style='darkgrid')
fig, axs = plt.subplots(2, 2, figsize=(10, 8))
sns.histplot(data=data, x="precipitation", kde=True, ax=axs[0, 0], color='green')
sns.histplot(data=data, x="temp_max", kde=True, ax=axs[0, 1], color='red')
sns.histplot(data=data, x="temp_min", kde=True, ax=axs[1, 0], color='blue')
sns.histplot(data=data, x="wind", kde=True, ax=axs[1, 1], color='orange')

# Mã hóa biến phân loại 'weather'
lc = LabelEncoder()
data['weather'] = lc.fit_transform(data['weather'])
data.head()

# Tách dữ liệu thành đầu vào (X) và đầu ra (y)
x = ((data.loc[:, data.columns != 'weather']).astype(int)).values[:, 0:]
y = data['weather'].values
data.weather.unique()

# Chia tập dữ liệu thành tập huấn luyện và tập kiểm tra
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=2)

# Huấn luyện và đánh giá mô hình KNN
knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
print('KNN accuracy:{:.2f}%'.format(knn.score(x_test, y_test) * 100))

# Huấn luyện và đánh giá mô hình SVM
svm = SVC()
svm.fit(x_train, y_train)
print('SVM accuracy:{:.2f}%'.format(svm.score(x_test, y_test) * 100))

# Huấn luyện và đánh giá mô hình Gradient Boosting
gbc = GradientBoostingClassifier(subsample=0.5, n_estimators=450, max_depth=5, max_leaf_nodes=25)
gbc.fit(x_train, y_train)
print('GBC accuracy:{:.2f}%'.format(gbc.score(x_test, y_test) * 100))

# Huấn luyện và đánh giá mô hình XGBoost
xgb = XGBClassifier()
xgb.fit(x_train, y_train)
print('XGB accuracy:{:.2f}%'.format(xgb.score(x_test, y_test) * 100))



# Dự đoán loại thời tiết cho một mẫu đầu vào mới
input = [[2, 8.9, 4, 2.123]]
ot = xgb.predict(input)
print('the weather is:')
if ot == 0:
    print('Drizzle')
elif ot == 1:
    print('Fog')
elif ot == 2:
    print('Rain')
elif ot == 3:
    print('Snow')
else:
    print('Sun')

# Lưu mô hình đã huấn luyện vào file
import pickle
file = 'rainfortecast.pkl'
pickle.dump(xgb, open(file, 'wb'))
