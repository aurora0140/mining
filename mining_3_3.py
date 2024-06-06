import pickle
from PIL import Image
import os
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

import streamlit as st

st.header('CIFAR-100数据集的分类任务', divider='rainbow')

st.write('2109120118 熊雨欢')

# 创建一个文件夹用于保存上传的文件
if not os.path.exists("uploads"):
    os.makedirs("uploads")
# # 页面标题和说明文字
st.write("上传文件")
# 选择文件并重命名
file_name = st.text_input("要上传的数据集")
uploaded_file = st.file_uploader("选择文件", type="zip")
# 保存文件
if uploaded_file is not None:
    if file_name.strip():
        file_path = os.path.join("uploads", file_name+".docx")
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"已保存文件: {file_path}")
    else:
        st.error("请输入学号+姓名来命名文件")

st.write("图片可视化")
base_dir = 'cifar-100-python'
with open(os.path.join(base_dir, 'meta'),'rb') as file:
    meta = pickle.load(file)

with open(os.path.join(base_dir,'train'),'rb') as file:
    train = pickle.load(file,encoding='bytes')
    train_data = train[b'data']
    train_labels = train[b'fine_labels']
    train_imgname = train[b'filenames']

with open(os.path.join(base_dir, 'test'),'rb') as file:
    test = pickle.load(file, encoding='bytes')
    test_data = test[b'data']
    test_labels = test[b'fine_labels']
    test_imgname = test[b'filenames']

train_img = train_data.reshape(train_data.shape[0], 3, 32, 32)
test_img = test_data.reshape(test_data.shape[0], 3, 32, 32)
label_names = meta['fine_label_names']


figure = plt.figure(figsize=(len(label_names), 10))
idxs = list(range(len(train_img)))
np.random.shuffle(idxs)
count = [0] * len(label_names)

for idx in idxs:
    label = train_labels[idx]
    if count[label] >= 10:
        continue
    if sum(count) > 10 * len(label_names):
        break
    img = Image.merge('RGB', (Image.fromarray(train_img[idx][0]), Image.fromarray(train_img[idx][1]), Image.fromarray(train_img[idx][2])))
    label_name = label_names[label]
    subplot_idx = count[label] * len(label_names) + label + 1
    print(label, subplot_idx)
    plt.subplot(10, len(label_names), subplot_idx)
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    if count[label] == 0:
        plt.title(label_name)
    count[label] += 1

if st.button('图片'):
    st.dataframe(plt)
st.pyplot(plt)

# 模型加载
from joblib import dump, load
model1 = load('mining_3_1.joblib')
model2 = load('mining_3_2.joblib')

# C:\Users\26323\PycharmProjects\untitled
# streamlit run mining_3_3.py