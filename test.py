from src.model import *
import pandas as pd
import numpy as np
import cv2
import sys, getopt
import torch



def selct_chanels(output, mask)->tuple: 
    """
    Args:
        output (torch.tensor): the output of the model (Spatial corrdinates of the scanpath)
        mask (torch.tensor): the mask of the output (the mask of the scanpath)
    Returns:
        torch.tensor: The masked scanpath sequence
        torch.tensor: the number of the selected fixation points
    """
    chanel,cord = output.shape[-2:] 
    repeted_mask = mask.int().view(-1,chanel,1).repeat(1,1,cord)
    result = output*repeted_mask 
    return result[mask], mask.sum(1)

    
def predict_(img,sp)->None:
    """
    Draw the scanpath on the image.
    Args:
        img (np.array): the image to draw the scanpath on it
        sp (torch.tensor): the scanpath to draw
    Returns:
        None
    """
    # extract the scanpath coordinates to numpy array
    sp  =sp.squeeze().detach().cpu().numpy()
    # reverse the columns to be (X,Y) instead of (Y,X)
    sp = sp[:,[1,0]] 
    # multiply the coordinates by the image size
    # to get the real coordinates
    # the image size is 320x240
    sp[:,0] = sp[:,0]*320
    sp[:,1] = sp[:,1]*240
    # convert the coordinates to int
    sp = sp.astype(np.int32)
    # draw the scanpath on the image
    img = cv2.polylines(img, [sp],isClosed=False, color=(0,0,255) ,thickness = 2)
    # show the image
    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def load_image(image_p:str)->torch.tensor:
    """
    Load an image from a path and convert it to a tensor.
    Args:
        image_p (str): the path of the image
    Returns:
        torch.tensor: the image tensor
        np.array: the image in a numpy
    """
    image = cv2.imread(image_p)
    image = cv2.resize(image, (320, 240))
    image = image.astype(np.float32)
    img = torch.cuda.FloatTensor(image)
    img = img.permute(2,0,1) 
    return img.unsqueeze(0) , image


def main(argv) -> None:
    """
    The main function to test the model.
    Args:
        argv (list): the arguments of the script
    Returns:
        None
    """

    try:
        opts, args = getopt.getopt(argv,"hi:w:t:",["i_image_path=","weight_path=", "temperture="])
    except getopt.GetoptError:
        print('test.py -i <i_image_path> -w <weight_path> -t <temperture>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('test.py -i <i_image_path> -w <weight_path>')
            sys.exit()
        elif opt in ("-i", "--i_image_path"):
            i_image_path = arg
        elif opt in ("-w", "--weight_path"):
            weight_path = arg
            # src/weigths/weight.pt
        elif opt in ("-t", "--temperture"):
            temp = arg


    model = ScanPathModel(domain=False)
    model.cuda()
    model.load_state_dict(torch.load(weight_path), strict=False)

    img, image = load_image(i_image_path) 
    out_path, _ , mask_vector = model(img,temp=float(temp))
    out_path, _ = selct_chanels(out_path,mask_vector)
    predict_(image, out_path)

if __name__ == "__main__":
   main(sys.argv[1:])
