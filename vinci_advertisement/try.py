import numpy as np
import pickle
if __name__ == '__main__':
    with open('keys.pkl','rb') as input:
        keys = pickle.load(input)
    for key in keys:
        print("================================***********************************")
        print(key)
