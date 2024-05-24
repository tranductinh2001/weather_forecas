import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import missingno as mso
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from xgboost import XGBRegressor
import warnings
import pickle

# Bỏ qua các cảnh báo
warnings.filterwarnings('ignore')

# Đọc dữ liệu từ file CSV
data = pd.read_csv("weatherHistory.csv")

# Hiển thị các thông tin cơ bản về dữ liệu
print(data.head())
print(data.shape)

# Mô tả thống kê các cột cụ thể
print(data[['Precip Type', 'Temperature (C)', 'Apparent Temperature (C)', 'Humidity', 'Wind Speed (km/h)', 'Wind Bearing (degrees)', 'Visibility (km)', 'Pressure (millibars)']].describe())

# Kiểm tra giá trị thiếu
print(data.isna().sum())
# Xóa các dòng có giá trị thiếu trong cột 'Precip Type'
data.dropna(subset=['Precip Type'], inplace=True)
print(data.isna().sum())
plt.figure(figsize=(12, 6))
axz = plt.subplot(1, 2, 2)
mso.bar(data.drop(['Formatted Date'], axis=1), ax=axz, fontsize=12)

# Loại bỏ cột 'Formatted Date' vì không cần thiết
data = data.drop(['Formatted Date'], axis=1)

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
data['Wind Speed (km/h)'] = np.sqrt(data['Wind Speed (km/h)'])

# Khởi tạo các LabelEncoder
lc_precip = LabelEncoder()
lc_daily_summary = LabelEncoder()
lc_summary = LabelEncoder()

# Mã hóa các cột
data['Precip Type'] = lc_precip.fit_transform(data['Precip Type'].fillna('none'))
data['Daily Summary'] = lc_daily_summary.fit_transform(data['Daily Summary'])
data['Summary'] = lc_summary.fit_transform(data['Summary'])

# Tạo X bằng cách loại bỏ các cột không cần thiết
X = data.drop('Temperature (C)', axis=1)

# Tạo y bằng cách chọn các cột mục tiêu
y = data['Temperature (C)']

# Hiển thị biến y
print(y)

# Chia tập dữ liệu thành tập huấn luyện và tập kiểm tra cho dự đoán nhiệt độ
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=42)

# Huấn luyện và đánh giá mô hình XGBoost cho dự đoán nhiệt độ
xgb_model = XGBRegressor()
xgb_model.fit(X_train, y_train)
print('XGBRegressor Type accuracy:{:.2f}%'.format(xgb_model.score(X_test, y_test) * 100))

# Sử dụng xác thực chéo với 5 lần gập (5-fold cross-validation)
scores = cross_val_score(xgb_model, X, y, cv=5)

# In kết quả của từng lần gập và trung bình
print("Cross-validation scores: ", scores)
print("Mean cross-validation score: {:.2f}%".format(scores.mean() * 100))

# Dự đoán nhiệt độ cho một mẫu đầu vào mới
input_data = np.array([[3, 3, 3, 3, 3, 3, 3, 3, 3, 3]])  # Cần cập nhật các giá trị này phù hợp với dữ liệu thực tế

# Dự đoán với mô hình XGBoost
temperature_prediction = xgb_model.predict(input_data)

# In kết quả dự đoán nhiệt độ
print('Predicted temperature:', temperature_prediction[0])

# Lưu mô hình đã huấn luyện và các LabelEncoder vào file
with open('Temperature.pkl', 'wb') as f:
    pickle.dump((xgb_model, lc_precip, lc_daily_summary, lc_summary), f)
