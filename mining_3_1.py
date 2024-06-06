import cv2
import numpy as np
from sklearn.decomposition import PCA
import os
from PIL import Image
import pickle
from tqdm import trange
from os.path import join
from skimage.feature import hog
from sklearn import model_selection

# 将cifar100数据集转为图片保存于datasets/cifar100
def my_mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
    return dict
src_dir = 'cifar-100-python' # the dir you uncompress the dataset
dst_dir = 'datasets/cifar100' # the dir you want the img_dataset to be
if __name__ == '__main__':
    meta = unpickle(join(src_dir, 'meta')) # KEYS: {'fine_label_names', 'coarse_label_names'}
    my_mkdirs(dst_dir)
    for data_set in ['train', 'test']:
        print('Unpickling {} dataset......'.format(data_set))
        data_dict = unpickle(join(src_dir, data_set)) # KEYS: {'filenames', 'batch_label', 'fine_labels', 'coarse_labels', 'data'}
        my_mkdirs(join(dst_dir, data_set))
        for fine_label_name in meta['fine_label_names']:
            my_mkdirs(join(dst_dir, data_set, fine_label_name))
        for i in trange(data_dict['data'].shape[0]):
            img = np.reshape(data_dict['data'][i], (3, 32, 32))
            i0 = Image.fromarray(img[0])
            i1 = Image.fromarray(img[1])
            i2 = Image.fromarray(img[2])
            img = Image.merge('RGB', (i0, i1, i2))
            img.save(join(dst_dir, data_set, meta['fine_label_names'][data_dict['fine_labels'][i]], data_dict['filenames'][i]))
    print('转图片完成')

# 定义LBP特征算法
def calc_lbp(image):
    height, width = image.shape
    dst = np.zeros((height-2, width-2), dtype=image.dtype)
    for i in range(1, height-1):
        for j in range(1, width-1):
            center = image[i][j]
            code = 0
            code |= (image[i-1][j-1] > center) << 7
            code |= (image[i-1][j] > center) << 6
            code |= (image[i-1][j+1] > center) << 5
            code |= (image[i][j+1] > center) << 4
            code |= (image[i+1][j+1] > center) << 3
            code |= (image[i+1][j] > center) << 2
            code |= (image[i+1][j-1] > center) << 1
            code |= (image[i][j-1] > center) << 0
            dst[i-1][j-1] = code
    return dst



def div(img, cell_x, cell_y, cell_w):
    cell = np.zeros(shape=(cell_x, cell_y, cell_w, cell_w))
    img_x = np.split(img, cell_x, axis=0)
    for i in range(cell_x):
        img_y = np.split(img_x[i], cell_y, axis=1)
        for j in range(cell_y):
            cell[i][j] = img_y[j]
    return cell


def get_bins(grad_cell, ang_cell):
    bins = np.zeros(shape=(grad_cell.shape[0], grad_cell.shape[1], 9))
    for i in range(grad_cell.shape[0]):
        for j in range(grad_cell.shape[1]):
            binn = np.zeros(9)
            grad_list = grad_cell[i, j].flatten()
            ang_list = ang_cell[i, j].flatten()
            left = np.int8(ang_list / 20.0)
            right = left + 1
            right[right >= 8] = 0
            left_rit = (ang_list - 20 * left) / 20.0
            right_rit = 1.0 - left_rit
            binn[left] += left_rit * grad_list
            binn[right] += right_rit * grad_list
            bins[i, j] = binn
    return bins


def hog(img, cell_x, cell_y, cell_w):
    img = np.matmul(img, np.array([0.299, 0.587, 0.114]))
    gx = cv2.Sobel(img, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=3)
    gy = cv2.Sobel(img, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy)
    ang = np.arctan2(gx, gy)
    ang[ang < 0] = np.pi + ang[ang < 0]
    ang *= (180.0 / np.pi)
    ang[ang >= 180] -= 180
    grad_cell = div(grad, cell_x, cell_y, cell_w)
    ang_cell = div(ang, cell_x, cell_y, cell_w)
    bins = get_bins(grad_cell, ang_cell)
    feature = []
    for i in range(cell_x - 1):
        for j in range(cell_y - 1):
            tmp = []
            tmp.append(bins[i, j])
            tmp.append(bins[i + 1, j])
            tmp.append(bins[i, j + 1])
            tmp.append(bins[i + 1, j + 1])
            tmp -= np.mean(tmp)
            feature.append(tmp.flatten())
            x = len(feature)
    return np.array(feature).flatten()



def calc_sift(image):
    # 创建SIFT对象
    sift = cv2.xfeatures2d.SIFT_create()
    # 检测图像中的关键点和描述符
    keypoints, descriptors = sift.detectAndCompute(image, None)
    # 显示检测到的关键点
    img_with_keypoints = cv2.drawKeypoints(image, keypoints, None)
    img_with_keypoints = (img_with_keypoints.reshape(img_with_keypoints.shape[0], img_with_keypoints.shape[1] * img_with_keypoints.shape[2]))
    return img_with_keypoints


base_dir = 'datasets/cifar100'
for path_1 in os.listdir(base_dir):
    for path_2 in os.listdir(os.path.join(base_dir, path_1)):
        for path_3 in os.listdir(os.path.join(base_dir, path_1, path_2)):
            image_path = os.path.join(base_dir, path_1, path_2, path_3)
            # 提取LBP特征
            image = cv2.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            lbp = calc_lbp(gray)
            pca = PCA(n_components=2)
            pca_lbp_feature = pca.fit_transform(lbp)
            print(pca_lbp_feature.shape)
            # 提取HOG特征
            image = Image.open(image_path)
            image = np.array(image)
            image = cv2.resize(image, (20, 20), interpolation=cv2.INTER_CUBIC)
            cell_w = 10
            cell_x = int(image.shape[0] / cell_w)
            cell_y = int(image.shape[1] / cell_w)
            feature = hog(image, cell_x, cell_y, cell_w)
            print(feature.shape)
            # 提取SIFT特征
            sift = calc_sift(image)
            pca_sift_feature = pca.fit_transform(sift)
            print(sift)

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

file = unpickle("cifar-100-python/meta")
label_names = file[b'fine_label_names']
file = unpickle("cifar-100-python/train")
images = file[b'data']
labels = file[b'filenames']

train_images = np.array(images)
train_labels = np.array(labels)
train_images = train_images.reshape(50000, 3072)
train_labels = train_labels.reshape(50000)

batch = unpickle(f'cifar-100-python/test')
test_images = batch[b'data'].reshape(10000, 3072)
test_labels = np.array(batch[b'fine_labels']).reshape(10000)

images = images.reshape(50000, 3, 32, 32).transpose(0, 2, 3, 1)
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(test_images, test_labels, test_size=0.3)

# 朴素贝叶斯
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
# 训练模型
model.fit(X_train, Y_train)
# 在测试集上评估模型的性能
preds = model.predict(X_test)
# 准确率
from sklearn.metrics import accuracy_score
acc1 = accuracy_score(preds, Y_test)
print(acc1)

# KNN分类
from sklearn.neighbors import KNeighborsClassifier
K = 3
model = KNeighborsClassifier(n_neighbors=K)
model.fit(X_train, Y_train)
preds = model.predict(X_test)
from sklearn.metrics import accuracy_score
acc2 = accuracy_score(preds, Y_test)
print(acc2)

# 逻辑回归
from sklearn import linear_model
model = linear_model.LogisticRegression()
model.fit(X_train, Y_train)
preds = model.predict(X_test)
from sklearn.metrics import accuracy_score
acc3 = accuracy_score(preds, Y_test)
print(acc3)

# 持久化模型
from joblib import dump, load
dump(acc1, acc2, acc3, 'mining_3_1.joblib')
