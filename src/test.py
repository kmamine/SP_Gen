from src.model import *
import pandas as pd
import numpy as np
import cv2
import os
import sys, getopt
import torch
from tqdm import tqdm

from time import time


def selct_chanels(output, mask): 
    chanel,cord = output.shape[-2:] 
    repeted_mask = mask.int().view(-1,chanel,1).repeat(1,1,cord)
    result = output*repeted_mask 
    return result[mask], mask.sum(1)

    
def predict_(img,out,path,i,temp):
    
    p = 'new_tests_myDS_DA_randn'
    name = path.split('.')[0]
    out  =out.squeeze().detach().cpu().numpy()  
    df = pd.DataFrame( np.squeeze(out),columns=['Y','X'])
    os.makedirs(f'./{p}/_predicted_path_{temp}/{name}/',exist_ok=True)
    df.to_csv(f'./{p}/_predicted_path_{temp}/{name}/{i:02d}.csv',index=False) 
          
    df['X'] = df['X'] * 320
    df['Y'] = df['Y'] * 240
    df = df [['X','Y']]
    out_p = df.values
    # print(name,i,np.int32([out_p]))
    img = cv2.polylines(img, np.int32([out_p]),isClosed=False, color=(0,0,255) ,thickness = 2)
    os.makedirs(f'./{p}/_predicted_images_{temp}/{name}/',exist_ok=True)
    cv2.imwrite(f'./{p}/_predicted_images_{temp}/{name}/{i:02d}.jpg',img)


def main(argv):

    try:
        opts, args = getopt.getopt(argv,"hi:w:",["i_image_folder=","weight_path="])
    except getopt.GetoptError:
        print('test.py -i <i_image_folder> -w <weight_path>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('test.py -i <i_image_folder> -w <weight_path>')
            sys.exit()
        elif opt in ("-i", "--i_image_folder"):
            i_image_folder = arg
        elif opt in ("-w", "--weight_path"):
            weight_path = arg

    
    # i_image_folder = 
    # weight_path = 
    # weight_path = '/media/Prisme-6TB/amine_new/ICME_Test_afterDA/weight.pt'
    # i_image_folder = '/media/Prisme-6TB/amine_new/ICME_Test_afterDA/weight.pt'
    
    
    temps = np.array(sorted(range(-100,101)))/10.
    temps = np.array(sorted(range(-100,101)))/10.

    temps = np.array([-9.8, -7.6, -5.3, -1.8 , -0.8, -0.3 , 0.0 , 0.4, 1.0 , 2.1 ,7.1 , 10.0])
    temps = [-9.8, -5.3]
    print(temps)
    model = ScanPathModel()
    model.cuda()
    model.load_state_dict(torch.load(weight_path), strict=False)

    for temp in temps:  
        t1 = time()
        print(f' testing the  temperture : {temp}.')
        for p in tqdm(sorted(list(os.listdir(i_image_folder)))):            
            if not (p.endswith(('png','jpg','jpeg'))):
                continue
            for i in range(15) :
                image_p = os.path.join(i_image_folder,p)
                image = cv2.imread(image_p)
                image = cv2.resize(image, (320, 240))
                image = image.astype(np.float32)
                img = torch.cuda.FloatTensor(image)
                img = img.permute(2,0,1) 
                out_path, gaussian_maps, mask_vector = model(img.unsqueeze(0),temp=temp)
                out_path, lnnnn = selct_chanels(out_path,mask_vector)
                predict_(image, out_path, p, i, temp)
        t2 =  time()
        print(f' temperture : {temp}  is done. \n time : {(t2-t1)} \n\n {(t2-t1) // 3600} hours {(t2-t1) // 60} minutes {(t2-t1) % 60} seconds')

if __name__ == "__main__":
   main(sys.argv[1:])
