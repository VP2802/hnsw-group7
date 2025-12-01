import matplotlib.pyplot as plt
import os

# Biểu đồ chủ đề (top 10)
topics = ['Thế giới', 'Thể thao', 'Thời sự', 'Giải trí', 'Giáo dục', 
          'Sức khỏe', 'Tin quốc tế', 'Kinh doanh', 'Bóng đá', 'Khoa học']
counts = [685, 548, 517, 414, 363, 360, 260, 255, 227, 198]

plt.figure(figsize=(12, 6))
bars = plt.barh(topics[::-1], counts[::-1], color='skyblue')
plt.xlabel('Số lượng bài báo')
plt.title('Top 10 chủ đề bài báo')
for i, (bar, count) in enumerate(zip(bars, counts[::-1])):
    plt.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2, 
             f'{count} bài', va='center')
plt.tight_layout()
plt.savefig('topic_distribution.png', dpi=300)
plt.show()

# Biểu đồ ngôn ngữ
languages = ['Tiếng Việt', 'Tiếng Anh']
sizes = [3488, 971]
colors = ['#ff9999', '#66b3ff']

plt.figure(figsize=(8, 6))
plt.pie(sizes, labels=languages, colors=colors, autopct='%1.1f%%', startangle=90)
plt.title('Phân bố bài báo theo ngôn ngữ')
plt.axis('equal')
plt.savefig('language_distribution.png', dpi=300)
plt.show()

# Biểu đồ nguồn báo (top 10)
sources = ['Thanh Niên', 'Dân Trí', 'VnExpress', 'VietnamNet', 'Tuổi Trẻ',
           'VTV', 'ZingNews', 'Daily Mail', 'The Guardian', 'New Scientist']
source_counts = [800, 778, 729, 420, 324, 250, 187, 137, 100, 100]

plt.figure(figsize=(12, 6))
bars = plt.barh(sources[::-1], source_counts[::-1], color='lightgreen')
plt.xlabel('Số lượng bài báo')
plt.title('Top 10 nguồn báo')
for i, (bar, count) in enumerate(zip(bars, source_counts[::-1])):
    plt.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2, 
             f'{count} bài', va='center')
plt.tight_layout()
plt.savefig('source_distribution.png', dpi=300)
plt.show()