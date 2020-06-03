import numpy as np
import random
import matplotlib.pyplot as plt
import tensorflow as tf
from mpl_toolkits.axes_grid1 import ImageGrid
from keras.utils import to_categorical, np_utils
from sklearn import preprocessing
import scipy.io as sio
from numpy import random as nr

def load_data_subject(subjectIndex):
    log_dir = r'/yourDir/'
    s1  = sio.loadmat(log_dir + '/djc_1.mat')
    s2  = sio.loadmat(log_dir + '/jj_1.mat')
    s3  = sio.loadmat(log_dir + '/lqj_1.mat')
    s4  = sio.loadmat(log_dir + '/ly_1.mat')
    s5  = sio.loadmat(log_dir + '/mhw_1.mat')
    s6  = sio.loadmat(log_dir + '/phl_1.mat')
    s7  = sio.loadmat(log_dir + '/sxy_1.mat')
    s8  = sio.loadmat(log_dir + '/wk_1.mat')
    s9  = sio.loadmat(log_dir + '/wsf_1.mat')
    s10 = sio.loadmat(log_dir + '/ww_1.mat')
    s11 = sio.loadmat(log_dir + '/wyw_1.mat')
    s12 = sio.loadmat(log_dir + '/xyl_1.mat')
    s13 = sio.loadmat(log_dir + '/ys_1.mat')
    s14 = sio.loadmat(log_dir + '/zjy_1.mat')
    min_max_scaler = preprocessing.MinMaxScaler(feature_range = (-1, 1))
    s1 = min_max_scaler.fit_transform(s1['djc_1'])
    s2 = min_max_scaler.fit_transform(s2['jj_1'])
    s3 = min_max_scaler.fit_transform(s3['lqj_1'])
    s4 = min_max_scaler.fit_transform(s4['ly_1'])
    s5 = min_max_scaler.fit_transform(s5['mhw_1'])
    s6 = min_max_scaler.fit_transform(s6['phl_1'])
    s7 = min_max_scaler.fit_transform(s7['sxy_1'])
    s8 = min_max_scaler.fit_transform(s8['wk_1'])
    s9 = min_max_scaler.fit_transform(s9['wsf_1'])
    s10 = min_max_scaler.fit_transform(s10['ww_1'])
    s11 = min_max_scaler.fit_transform(s11['wyw_1'])
    s12 = min_max_scaler.fit_transform(s12['xyl_1'])
    s13 = min_max_scaler.fit_transform(s13['ys_1'])
    s14 = min_max_scaler.fit_transform(s14['zjy_1'])
    allData = np.concatenate((s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14), 0)
    target = allData[subjectIndex*3394:(subjectIndex+1)*3394]
    source = np.delete(allData, range(subjectIndex*3394, (subjectIndex+1)*3394), 0)
    
    label = sio.loadmat(log_dir + '/labels.mat')
    label = label['labels']
    label = np_utils.to_categorical(label,3)
    
    source_label = np.concatenate((label,label,label,label,label,label,label,label,label,label,label,label,label), 0)
    
    index = [i for i in range(len(target))]
    random.shuffle(index)
    target = target[index]
    target_label = label[index]
    
    index = [i for i in range(len(source))]
    random.shuffle(index)
    source = source[index]
    source_label = source_label[index]
    
    source = source[0:5000]
    source_label = source_label[0:5000]
    '''
    a = nr.randint(0,13) #randomly select a subject as source
    source = source[a*3394:(a+1)*3394]
    
    index = [i for i in range(len(source))]
    random.shuffle(index)
    source = source[index]
    source_label = label[index]
    
    index = [i for i in range(len(target))]
    random.shuffle(index)
    target = target[index]
    target_label = label[index]
    '''
    return source, source_label, target, target_label

def load_data_session(subjectName):
    log_dir = r'/yourDir/'
    
    s1_path = subjectName + '_1.mat'
    s2_path = subjectName + '_2.mat'
    s3_path = subjectName + '_3.mat'
    
    s1  = sio.loadmat(log_dir + s1_path)
    s2  = sio.loadmat(log_dir + s2_path)
    s3  = sio.loadmat(log_dir + s3_path)
    
    min_max_scaler = preprocessing.MinMaxScaler(feature_range = (-1, 1))
    
    s1 = min_max_scaler.fit_transform(s1[subjectName + '_1'])
    s2 = min_max_scaler.fit_transform(s2[subjectName + '_2'])
    s3 = min_max_scaler.fit_transform(s3[subjectName + '_3'])
    
    #source = np.concatenate((s1, s2), 0)
    source = s2
    target = s3

    label = sio.loadmat(log_dir + 'labels.mat')
    label = label['labels']
    label = np_utils.to_categorical(label,3)
    
    source_label = np.concatenate((label,label), 0)
    target_label = label

    index = [i for i in range(len(source))]
    random.shuffle(index)
    source = source[index]
    source_label = source_label[index]
    
    index = [i for i in range(len(target))]
    random.shuffle(index)
    target = target[index]
    target_label = target_label[index]
    
    return source, source_label, target, target_label


def shuffle_aligned_list(data):
    """Shuffle arrays in a list by shuffling each array identically."""
    num = data[0].shape[0]
    p = np.random.permutation(num)
    return [d[p] for d in data]


def batch_generator(data, batch_size, shuffle=True):
    """Generate batches of data.
    Given a list of array-like objects, generate batches of a given
    size by yielding a list of array-like objects corresponding to the
    same slice of each input.
    """
    if shuffle:
        data = shuffle_aligned_list(data)
    batch_count = 0
    while True:
        if batch_count * batch_size + batch_size >= len(data[0]):
            batch_count = 0

            if shuffle:
                data = shuffle_aligned_list(data)

        start = batch_count * batch_size
        end = start + batch_size
        batch_count += 1
        yield [d[start:end] for d in data]

def batch_generator_balance(data, batch_size, num_labels):
    images = data[0]
    labels = np.argmax(data[1], axis=1)
    n_per_label = batch_size/num_labels
    res = []
    for i in xrange(num_labels):
        a = images[labels == i]
        randIndex = np.random.choice(len(a), n_per_label)
        res.append(a[randIndex])
    image_classes = np.arange(num_labels)
    batch_images = np.concatenate(res, axis=0)
    batch_labels = np.repeat(image_classes, n_per_label)
    batch_labels = to_categorical(batch_labels, num_labels)
    return shuffle_aligned_list([batch_images, batch_labels])


def imshow_grid(images, shape=[2, 8]):
    """Plot images in a grid of a given shape."""
    fig = plt.figure(1)
    grid = ImageGrid(fig, 111, nrows_ncols=shape, axes_pad=0.05)

    size = shape[0] * shape[1]
    for i in range(size):
        grid[i].axis('off')
        grid[i].imshow(images[i])  # The AxesGrid object work as a list of axes.

    plt.show()


def plot_embedding(X, y, d, title=None):
    """Plot an embedding X with the class label y colored by the domain d."""
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure(figsize=(3,3))
    ax = plt.subplot(111)
    plt.xticks([]), plt.yticks([])
    for i in range(X.shape[0]):
        plt.scatter(X[i, 0], X[i, 1],
                    c = plt.cm.bwr(d[i] / 1.),
                    marker = 'o',
                    s = 12)
        savename = title + '.png'
        plt.savefig(savename, dpi=300)
        





