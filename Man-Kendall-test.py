import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import pymannkendall as mk

# Đường dẫn tệp đầu vào
file_path = r'E:\CAO HOC_HAI_DUONG\Thesis-Luanvan\Model\Data\Testing_data\WLraw data\ChauDoc.xlsx' # Đường dẫn đến tệp Excel

# Đường dẫn tệp kết quả
output_file_path = r'E:\CAO HOC_HAI_DUONG\Thesis-Luanvan\Model\Data\Testing_data\Results\ChauDoc_max.xlsx'

# Đọc dữ liệu từ tệp Excel
df = pd.read_excel(file_path, sheet_name='Ave')  # Đọc trang tính 'Sheet1'

# Giả sử dữ liệu nằm trong cột A (hoặc một cột cụ thể nào đó)
data = df['Data'].tolist()  # Chuyển đổi cột thành danh sách (giá trị trong [] là tên của tiêu đề)
time = df['Time']
# Chuyển đổi danh sách thành pandas Series
data_series = pd.Series(data)

# 0. Thống kê dữ liệu
mean_value = round(data_series.mean(),2)
max_value = round(data_series.max(),2)
min_value = round(data_series.min(),2)
std_value = round(data_series.std(),2)

print("Sumamry Statistics:") 
print(f"Mean: {mean_value}")
print(f"Max: {max_value}")
print(f"Min: {min_value}")
print(f"Standard Deviation: {std_value}")

# 1. Thực hiện kiểm định Mann-Kendall
mk_test = mk.original_test(data_series)
# Làm tròn giá trị của Mann-Kendall Tau và p-value
tau_value = round(mk_test.Tau, 2)  # Làm tròn Tau đến 2 chữ số thập phân
p_value = round(mk_test.p, 4)  # Làm tròn p-value đến 3 chữ số thập phân

print("Mann-Kendall Test:") 
print(f"Tau: {tau_value}") #Đo lương mối quan hệ giữa thứ hạng các cặp giá trị. Tau>0 xu hướnng tăng và ngược lại
print(f"p-value: {p_value}") # Mức ý nghĩa thống kê
print(f"Trend: {mk_test.trend}") #Xu hướng chuỗi dữ liệu
print(f"Hypothesis Test Result: {mk_test.h}") #True: giả thuyết có ý nghĩa, False: giả thuyết không có ý nghĩa

# 2. Tính toán Sen's Slope
n = len(data)
slopes = []

for i in range(n - 1):
    for j in range(i + 1, n):
        slope = (data_series[j] - data_series[i]) / (j - i)
        slopes.append(slope)
print(slopes)
# Sen's Slope là median của tất cả các slope
sens_slope = round(np.median(slopes),0)
print(sens_slope)
print("\nSen's Slope:")
print(f"Slope: {sens_slope}")

#############################
#############################
######### Xuất kết quả#######
# Tạo DataFrame cho thống kê tóm tắt
data_from_source = pd.read_excel(file_path, sheet_name='Ave')  # Đọc lại dữ liệu từ nguồn

summary_stats = pd.DataFrame({
    'Statistic': ['Mean', 'Max', 'Min', 'Standard Deviation'],
    'Value': [mean_value, max_value, min_value, std_value]
})

# Tạo DataFrame cho kết quả kiểm định Mann-Kendall và Sen's Slope
mankendall_slope_results = pd.DataFrame({
    'Test': ['Mann-Kendall Tau', 'Mann-Kendall p-value', 'Mann-Kendall Trend', 'Mann-Kendall Hypothesis Test Result', 'Sen\'s Slope'],
    'Value': [mk_test.Tau, mk_test.p, mk_test.trend, mk_test.h, sens_slope]
})

# Ghi dữ liệu vào cùng một sheet, bắt đầu từ cột E
with pd.ExcelWriter(output_file_path, engine='openpyxl') as writer:

    # Ghi dữ liệu gốc từ tệp Excel vào ô bắt đầu từ cột E
    data_from_source.to_excel(writer, sheet_name='Results', startrow=0, index=False)

    # Ghi dữ liệu thống kê tóm tắt bắt đầu từ cột E (cột số 5)
    summary_stats.to_excel(writer, sheet_name='Results', startrow=0, startcol=4, index=False)
    
    # Ghi dữ liệu kiểm định Mann-Kendall và Sen's Slope bắt đầu từ cột E, cách 3 hàng sau thống kê tóm tắt
    mankendall_slope_results.to_excel(writer, sheet_name='Results', startrow=len(summary_stats) + 3, startcol=4, index=False)

print(f"Results have been saved to {output_file_path}")


# Vẽ biểu đồ
plt.figure(figsize=(10, 6))

# Vẽ chuỗi dữ liệu gốc
plt.plot(time, data_series, label='Dữ liệu', marker='o')

# Vẽ đường xu hướng độ dốc nếu
if mk_test.trend != 'no trend':
    slope_line = [data_series[0] + sens_slope * (i - 0) for i in range(len(time))]
    plt.plot(time, slope_line, color='red', linestyle='--', label="Độ dốc Sen's")

# Thêm tiêu đề và nhãn cho biểu đồ
plt.title('Biểu đồ xu thế tại trạm Châu Đốc')
plt.xticks(np.arange(int(time.min()), int(time.max()) + 1, 2))
plt.xlabel('Năm')
plt.ylabel('Mực nước (cm)')
plt.grid()

# Hiển thị các thông số thống kê trên biểu đồ
plt.text(0.8, 0.75, f"Mann-Kendall Tau: {tau_value:.2f}\np-value: {p_value:.4f}\nTrend: {mk_test.trend}\nSen's Slope: {sens_slope}",
         transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
         bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))  # Thêm nền cho văn bản

# Hiển thị chú thích (legend)
plt.legend(loc='best')

# Hiển thị biểu đồ
plt.tight_layout()
plt.show()
