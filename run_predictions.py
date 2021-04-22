import os
import numpy as np
import json
from PIL import Image

def compute_convolution(I, T, stride=None):
    '''
    This function takes an image <I> and a template <T> (both numpy arrays) 
    and returns a heatmap where each grid represents the output produced by 
    convolution at each location. You can add optional parameters (e.g. stride, 
    window_size, padding) to create additional functionality. 
    '''
    (n_rows,n_cols,n_channels) = np.shape(I)

    '''
    BEGIN YOUR CODE
    '''
    if stride is None:
        stride = 1

    heatmap = np.zeros((n_rows, n_cols))
    # Assume filter is odd
    i = len(T) // 2
    j = len(T[0]) // 2
    # Zero-pad the image
    temp = np.pad(I, ((i, i), (j, j), (0, 0)))

    # Compute correlation by hadamard product, then summation for each window
    for x in range(0, n_rows, stride):
        for y in range(0, n_cols, stride):
            heatmap[x, y] = np.sum(temp[x:x+len(T), y:y+len(T[0]), :] * T)

    # Scale by size of filter for stability
    heatmap /= len(T) * len(T[0])
    '''
    END YOUR CODE
    '''

    return heatmap 


def predict_boxes(heatmap):
    '''
    This function takes heatmap and returns the bounding boxes and associated
    confidence scores.
    '''

    output = []

    '''
    BEGIN YOUR CODE
    '''
    
    '''
    As an example, here's code that generates between 1 and 5 random boxes
    of fixed size and returns the results in the proper format.
    '''

    thresh = 2
    n_rows, n_cols = heatmap.shape
    detected = heatmap > thresh

    # First iteration?
    switch = False
    # Search each pixel
    for x in range(n_rows):
        for y in range(n_cols):
            # Skip if not above threshold
            if detected[x, y] == 0:
                continue
            # Skip if encountered
            if switch and x in range(bbox[0], bbox[2] + 1) and y in range(bbox[1], bbox[3] + 1):
                continue
            bbox = [x, y, x, y, 0]
            switch = True
            # Search radius
            for r in range(1, min(n_rows, n_cols) // 2):
                window = heatmap[x-r:x+r, y-r:y+r]
                mu = np.mean(window)
                if mu > thresh:
                    # Normalize with sigmoid
                    norm_mu = 1 / (1 + np.exp(-mu))
                    bbox = [x-r, y-r, x+r, y+r, np.maximum(0, norm_mu)]
                else:
                    break 
            if bbox[0] != bbox[2] and bbox[1] != bbox[3]:
                output.append(bbox)


    '''
    END YOUR CODE
    '''

    return output


def detect_red_light_mf(I):
    '''
    This function takes a numpy array <I> and returns a list <output>.
    The length of <output> is the number of bounding boxes predicted for <I>. 
    Each entry of <output> is a list <[row_TL,col_TL,row_BR,col_BR,score]>. 
    The first four entries are four integers specifying a bounding box 
    (the row and column index of the top left corner and the row and column 
    index of the bottom right corner).
    <score> is a confidence score ranging from 0 to 1. 

    Note that PIL loads images in RGB order, so:
    I[:,:,0] is the red channel
    I[:,:,1] is the green channel
    I[:,:,2] is the blue channel
    '''

    '''
    BEGIN YOUR CODE
    '''
    # Filter size
    template_height = 5
    template_width = 5

    # Accentuate red channel pixel
    n1 = np.ones((template_height, template_width, 1))
    n1[template_height//2, template_width//2] = 25
    # Subtract other channels
    n2 = -0.5 * np.ones((template_height, template_width, 2))
    n2[template_height//2, template_width//2] = 0
    T = np.concatenate((n1, n2), axis=-1)
    # z-score normalization
    I = (I - np.mean(I)) / np.std(I)

    heatmap = compute_convolution(I, T)
    output = predict_boxes(heatmap)

    '''
    END YOUR CODE
    '''

    for i in range(len(output)):
        assert len(output[i]) == 5
        assert (output[i][4] >= 0.0) and (output[i][4] <= 1.0)

    return output

# Note that you are not allowed to use test data for training.
# set the path to the downloaded data:
data_path = './data/RedLights2011_Medium/RedLights2011_Medium'

# load splits: 
split_path = './data/hw02_splits'
file_names_train = np.load(os.path.join(split_path,'file_names_train.npy'))
file_names_test = np.load(os.path.join(split_path,'file_names_test.npy'))

# set a path for saving predictions:
preds_path = './data/hw02_preds'
os.makedirs(preds_path, exist_ok=True) # create directory if needed

# Set this parameter to True when you're done with algorithm development:
done_tweaking = False

'''
Make predictions on the training set.
'''
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


preds_train = {}
for i in range(1, len(file_names_train)):
    print('{:.2f}%'.format(i/len(file_names_train)*100), end='\r')

    # read image using PIL:
    I = Image.open(os.path.join(data_path,file_names_train[i]))

    # convert to numpy array:
    I = np.asarray(I) 

    outputs = detect_red_light_mf(I)
    preds_train[file_names_train[i]] = outputs

    # Plot bounding boxes
    # plt.imshow(I)
    # for bbox in outputs:
    #     rx = bbox[3] - bbox[1]
    #     ry = bbox[2] - bbox[0]
    #     plt.gca().add_patch(Rectangle((bbox[1], bbox[0]), rx, ry, color='green'))
    # plt.show()
    

# save preds (overwrites any previous predictions!)
# with open(os.path.join(preds_path,'preds_train.json'),'w') as f:
#     json.dump(preds_train,f)

if done_tweaking:
    '''
    Make predictions on the test set. 
    '''
    preds_test = {}
    for i in range(len(file_names_test)):

        # read image using PIL:
        I = Image.open(os.path.join(data_path,file_names_test[i]))

        # convert to numpy array:
        I = np.asarray(I)

        preds_test[file_names_test[i]] = detect_red_light_mf(I)

    # save preds (overwrites any previous predictions!)
    with open(os.path.join(preds_path,'preds_test.json'),'w') as f:
        json.dump(preds_test,f)
