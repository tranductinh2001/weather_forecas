from flask import Flask, request, render_template
import pickle
import csv
from sklearn.preprocessing import LabelEncoder

# Load the models and label encoders
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('Temperature.pkl', 'rb') as f:
    model_Temperature, lc_precip, lc_daily_summary, lc_summary = pickle.load(f)

app = Flask(__name__)

@app.route('/temperature')
def temperature():
    # Khởi tạo các tập hợp để lưu các giá trị không trùng lặp
    unique_summaries = set()
    unique_precip_types = set()
    unique_daily_summaries = set()

    # Đọc dữ liệu từ file CSV
    with open('weatherHistory.csv', 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Bỏ qua header nếu có
        for row in reader:
            unique_summaries.add(row[1])
            unique_precip_types.add(row[2])
            unique_daily_summaries.add(row[11])

    # Chuyển các tập hợp thành danh sách
    summaries = list(unique_summaries)
    precip_types = list(unique_precip_types)
    daily_summaries = list(unique_daily_summaries)

    return render_template('temperature.html', summaries=summaries, precip_types=precip_types, daily_summaries=daily_summaries)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictTemperature', methods=['POST'])
def predict_temperature():
    # Lấy dữ liệu từ form
    ApparentTemperature = float(request.form['ApparentTemperature'])
    Humidity = float(request.form['Humidity'])
    WindSpeed = float(request.form['WindSpeed'])
    WindBearing = float(request.form['WindBearing'])
    Visibility = float(request.form['Visibility'])
    LoudCover = float(request.form['LoudCover'])
    Pressure = float(request.form['Pressure'])
    precip_type = request.form['PrecipType']
    daily_summary = request.form['DailySummary']
    summary = request.form['Summary']
    
    def custom_encode(encoder, value):
        # Nếu giá trị đã được mã hoá trước đó, trả về giá trị mã hoá
        if value in encoder.classes_:
            return encoder.transform([value])[0]
        # Nếu giá trị mới, trả về một giá trị đặc biệt khác, chẳng hạn -1
        else:
            return -1

    # Sử dụng hàm tùy chỉnh để mã hoá giá trị mới
    precip_type_encoded = custom_encode(lc_precip, precip_type)
    daily_summary_encoded = custom_encode(lc_daily_summary, daily_summary)
    summary_encoded = custom_encode(lc_summary, summary)

    print('Precip Type Encoded:', precip_type_encoded)
    print('Daily Summary Encoded:', daily_summary_encoded)
    print('Summary Encoded:', summary_encoded)



    # Dự đoán nhiệt độ
    Temperature = model_Temperature.predict([[summary_encoded, precip_type_encoded, ApparentTemperature, Humidity, WindSpeed, WindBearing, Visibility, LoudCover, Pressure, daily_summary_encoded]])[0]
    
    # Lấy lại các giá trị không trùng lặp để hiển thị lại trong template
    unique_summaries = set()
    unique_precip_types = set()
    unique_daily_summaries = set()

    with open('weatherHistory.csv', 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Bỏ qua header nếu có
        for row in reader:
            unique_summaries.add(row[1])
            unique_precip_types.add(row[2])
            unique_daily_summaries.add(row[11])

    summaries = list(unique_summaries)
    precip_types = list(unique_precip_types)
    daily_summaries = list(unique_daily_summaries)
    
    return render_template('temperature.html', result=Temperature, summaries=summaries, precip_types=precip_types, daily_summaries=daily_summaries)

@app.route('/predictWeather', methods=['POST'])
def predict_weather():
    precipitation = float(request.form['precipitation'])
    temp_max = float(request.form['temp_max'])
    temp_min = float(request.form['temp_min'])
    wind = float(request.form['wind'])

    prediction = model.predict([[precipitation, temp_max, temp_min, wind]])[0]

    if prediction == 0:
        weather_type = 'Mưa phùn'
    elif prediction == 1:
        weather_type = 'Sương mù'
    elif prediction == 2:
        weather_type = 'Mưa nhiều'
    elif prediction == 3:
        weather_type = 'Có tuyết rơi'
    else:
        weather_type = 'Nắng'

    return render_template('index.html', result=weather_type)

if __name__ == '__main__':
    app.run(debug=True)
