import pickle 
import numpy as np

def load_data(file):
    # Open 'pickle' file
    with open(file, 'rb') as f:
        d = pickle.load(f, encoding='latin1')
        """
        Data is a dictionary with four keys:
            'features' - is a 4D array with raw pixel data of the traffic sign images,
                         (number of examples, width, height, channels).
            'labels'   - is a 1D array containing the label id of the traffic sign image,
                         file label_names.csv contains id -> name mappings.
            'sizes'    - is a 2D array containing arrays (width, height),
                         representing the original width and height of the image.
            'coords'   - is a 2D array containing arrays (x1, y1, x2, y2),
                         representing coordinates of a bounding frame around the image.
        """
        # 4D numpy.ndarray type, for train = (34799, 32, 32, 3)
        input_data = d['features'].astype(np.float32)
        # 1D numpy.ndarray type, for train = (34799,)   
        target = d['labels']                       
        # 2D numpy.ndarray type, for train = (34799, 2) 
        sizes = d['sizes']                         
        # 2D numpy.ndarray type, for train = (34799, 4)
        coords = d['coords']                        
       
    return input_data, target, sizes, coords