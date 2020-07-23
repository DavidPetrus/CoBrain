import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter
import cv2
from ann import ANN

def gabor_fn(sigma,theta,Lambda,psi,gamma):
    sigma_x = sigma/gamma
    sigma_y = float(sigma)

    # Bounding box
    nstds = 3
    xmax = max(abs(nstds*sigma_x*np.cos(theta)),abs(nstds*sigma_y*np.sin(theta)))
    xmax = np.ceil(max(1,xmax))
    ymax = max(abs(nstds*sigma_x*np.sin(theta)),abs(nstds*sigma_y*np.cos(theta)))
    ymax = np.ceil(max(1,ymax))
    xmin = -xmax; ymin = -ymax
    (x,y) = np.meshgrid(np.arange(xmin,xmax+1),np.arange(ymin,ymax+1 ))
    (y,x) = np.meshgrid(np.arange(ymin,ymax+1),np.arange(xmin,xmax+1 ))

    # Rotation
    x_theta=x*np.cos(theta)+y*np.sin(theta)
    y_theta=-x*np.sin(theta)+y*np.cos(theta)

    gb= np.exp(-.5*(x_theta**2/sigma_x**2+y_theta**2/sigma_y**2))*np.cos(2*np.pi/Lambda*x_theta+psi)
    return gb

gabor_filter = gabor_fn(sigma = 18, theta = 3*np.pi/4, Lambda = 30, psi = 0, gamma  = 0.9)
gabor_filter = np.ndarray.astype((gabor_filter+1.)*127.5,np.float32)

ann = ANN(5,8).cuda()
optimizer = torch.optim.SGD(ann.parameters(),lr=0.001)
losses = []

for train_idx in range(100000):
    contrast_left = np.random.uniform(0.,1.)
    contrast_right = np.random.uniform(0.,1.)
    if contrast_right > contrast_left:
        label = torch.ones([1,1]).cuda()
    else:
        label = torch.zeros([1,1]).cuda()

    filter_left = gabor_filter.copy()
    filter_right = gabor_filter.copy()
    filter_left = np.clip(filter_left,filter_left[0,0] - contrast_left*filter_left[0,0],filter_left[0,0] + contrast_left*filter_left[0,0])
    filter_right = np.clip(filter_right,filter_right[0,0] - contrast_right*filter_right[0,0],filter_right[0,0] + contrast_right*filter_right[0,0])

    #print(filter_left.max())
    #print(filter_left.min())
    #print(filter_left[0,0])

    stimulus = np.full([257,900],filter_left[0,0])

    # Specify filter locations using [left_upper_x,left_upper_y,right_lower_x,right_lower_y]
    left_loc = [300//2-filter_left.shape[0]//2,257//2-filter_left.shape[1]//2,
                300//2+filter_left.shape[0]//2,257//2+filter_left.shape[1]//2]
    right_loc = [600+300//2-filter_right.shape[0]//2,257//2-filter_right.shape[1]//2,
                600+300//2+filter_right.shape[0]//2,257//2+filter_right.shape[1]//2]

    #print(stimulus.shape)
    #print(right_loc)
    #print(stimulus[right_loc[1]:right_loc[3]+1,right_loc[0]:right_loc[2]+1].shape)
    stimulus[left_loc[1]:left_loc[3]+1,left_loc[0]:left_loc[2]+1] = filter_left.copy()
    stimulus[right_loc[1]:right_loc[3]+1,right_loc[0]:right_loc[2]+1] = filter_right.copy()

    stimulus[:,300] = 0.
    stimulus[:,600] = 0.
    stimulus = gaussian_filter(stimulus,sigma=2.)
    stimulus = cv2.resize(stimulus,dsize=(175,50))
    stimulus = np.reshape(stimulus,[1,1,175,50])

    pred = ann(torch.from_numpy(stimulus).cuda())
    print(pred)
    print(label)
    loss = F.binary_cross_entropy(pred, label)
    print("Iter {}, Loss: {}".format(train_idx,loss))
    losses.append(loss)
    if train_idx % 1000 == 0:
        np_losses = np.array(losses)
        np.save('np_loss.npy',np_losses)
    loss.backward()
    '''for key in ann.net.keys():
        print(key)
        print(ann.net[key].weight.grad)'''
    optimizer.step()
    optimizer.zero_grad()

'''
fig = plt.imshow(stimulus,cmap = 'gray',interpolation='nearest',vmin=0,vmax=255)
plt.axis('off')
plt.show()
'''
