import numpy as np
import torch
import matplotlib.pyplot as plt
import tifffile

def CreateTensorDataset(data_path,data_type,image_size,scale):
    """
    从训练微观结构中随机生成张量数据集
    数据：数据路径
    数据类型：数据类型
    图像尺寸：图像大小
    比例：比例
    """
    if data_type in ['png', 'jpg', 'tif2D']:
        tensor_dataset = []
        for img in data_path:
            img = plt.imread(img) if type != 'tif2D' else tifffile.imread(img)
            if len(img.shape)>2:
                img = img[:,:,0]
            img = img[::scale,::scale]
            x_max, y_max= img.shape[:]
            phases = np.unique(img)
            data = np.empty([32 * 900, len(phases),image_size,image_size])
            for i in range(32 * 900):
                x = np.random.randint(1, x_max -image_size-1)
                y = np.random.randint(1, y_max -image_size-1)
                for cnt, phs in enumerate(phases):
                    img1 = np.zeros([image_size, image_size])
                    img1[img[x:x + image_size, y:y + image_size] == phs] = 1
                    data[i, cnt, :, :] = img1
            data = torch.FloatTensor(data)
            dataset = torch.utils.data.TensorDataset(data)
            tensor_dataset.append(dataset)

    elif data_type=='tif3D':
        tensor_dataset=[]
        img = np.array(tifffile.imread(data[0]))
        img = img[::scale,::scale,::scale]
        ## Create a data store and add random samples from the full image
        x_max, y_max, z_max = img.shape[:]
        print('training image shape: ', img.shape)
        vals = np.unique(img)
        for dim in range(3):
            data = np.empty([32 * 900, len(vals), image_size, image_size])
            print('dataset ', dim)
            for i in range(32*900):
                x = np.random.randint(0, x_max -image_size)
                y = np.random.randint(0, y_max - image_size)
                z = np.random.randint(0, z_max - image_size)
                # create one channel per phase for one hot encoding
                lay = np.random.randint(img.shape[dim]-1)
                for cnt,phs in enumerate(list(vals)):
                    img1 = np.zeros([image_size,image_size])
                    if dim==0:
                        img1[img[lay, y:y + image_size, z:z + image_size] == phs] = 1
                    elif dim==1:
                        img1[img[x:x + image_size,lay, z:z + image_size] == phs] = 1
                    else:
                        img1[img[x:x + image_size, y:y + image_size,lay] == phs] = 1
                    data_path[i, cnt, :, :] = img1[:,:]
                    # data[i, (cnt+1)%3, :, :] = img1[:,:]
            data = torch.FloatTensor(data)
            dataset = torch.utils.data.TensorDataset(data)
            tensor_dataset.append(dataset)

    elif data_type=='colour':
        ## Create a data store and add random samples from the full image
        tensor_dataset = []
        for img in data_path:
            img = plt.imread(img)
            img = img[::scale,::scale,:]
            ep_sz = 32 * 900
            data = np.empty([ep_sz, 3, image_size, image_size])
            x_max, y_max = img.shape[:2]
            for i in range(ep_sz):
                x = np.random.randint(0, x_max - image_size)
                y = np.random.randint(0, y_max - image_size)
                # create one channel per phase for one hot encoding
                data[i, 0, :, :] = img[x:x + image_size, y:y + image_size,0]
                data[i, 1, :, :] = img[x:x + image_size, y:y + image_size,1]
                data[i, 2, :, :] = img[x:x + image_size, y:y + image_size,2]
            print('converting')
            data = torch.FloatTensor(data)
            dataset = torch.utils.data.TensorDataset(data)
            tensor_dataset.append(dataset)

    elif data_type=='grayscale':
        tensor_dataset = []
        for img in data_path:
            img = plt.imread(img)
            if len(img.shape) > 2:
                img = img[:, :, 0]
            img = img/img.max()
            img = img[::image_size, ::image_size]
            x_max, y_max = img.shape[:]
            data = np.empty([32 * 900, 1,image_size, image_size])
            for i in range(32 * 900):
                x = np.random.randint(1, x_max - image_size - 1)
                y = np.random.randint(1, y_max - image_size - 1)
                subim = img[x:x + image_size, y:y + image_size]
                data[i, 0, :, :] = subim
            data = torch.FloatTensor(data)
            dataset = torch.utils.data.TensorDataset(data)
            tensor_dataset.append(dataset)
    return tensor_dataset