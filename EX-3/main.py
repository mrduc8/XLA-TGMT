import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from matplotlib.colors import ListedColormap

# Tải dữ liệu Iris và chỉ lấy hai đặc trưng là chiều dài và rộng của đài hoa
iris = datasets.load_iris()
X = iris.data[:, :2]  # Lấy hai đặc trưng đầu tiên
y = iris.target

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Khởi tạo các mô hình
models = {
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=3),
    'Support Vector Machine': SVC(kernel='linear', C=1),
    'Decision Tree': DecisionTreeClassifier(max_depth=5)
}

# Thiết lập đồ thị
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
h = .02  # Bước của lưới

# Định màu sắc cho đồ thị
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# Vẽ các biểu đồ
for i, (name, clf) in enumerate(models.items()):
    # Huấn luyện mô hình
    clf.fit(X_train, y_train)

    # Tạo lưới điểm để biểu diễn biên phân tách
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Dự đoán trên lưới điểm
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Vẽ vùng quyết định
    axes[i].contourf(xx, yy, Z, cmap=cmap_light)

    # Vẽ các điểm huấn luyện
    scatter = axes[i].scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=20)
    axes[i].set_xlim(xx.min(), xx.max())
    axes[i].set_ylim(yy.min(), yy.max())
    axes[i].set_xticks(())
    axes[i].set_yticks(())
    axes[i].set_title(name)

# Thêm chú thích
fig.legend(*scatter.legend_elements(), title="Classes", loc="upper right")
plt.show()
