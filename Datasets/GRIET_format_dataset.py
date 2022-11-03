import os
import shutil
import xml.etree.ElementTree as ET
import tarfile, zipfile
import pickle
import numpy as np
from PIL import Image
import random
import cv2

def get_count(dir_path):
    count = 0
    # Iterate directory
    for path in os.listdir(dir_path):
        # check if current path is a file
        if os.path.isfile(os.path.join(dir_path, path)):
            count += 1
    return count

def get_unique_id(size : int, used : list):
    id = random.randrange(size) 
    while(used[id] != False):
        id = random.randrange(size)
    return id

def convert_img_into_grayscale(dir_path):
    # This funtion is to convert all the 3 and 4 channels images into 2D arrays and save them in the same place again

    for filename in os.listdir(dir_path):    
        image = Image.open(os.path.join(dir_path,filename))
        image_arr = np.array(image)
        if(len(image_arr.shape) == 2):
            continue
        else:
            image_arr = image_arr[:,:,0]
            image = Image.fromarray(image_arr)       
            image.save(filename)
        os.replace(filename, os.path.join(dir_path,filename))

'''
    # This part is to check the no of channels in each image in dataset/Griet_lines/lines 
    for filename in os.listdir(dir_path):
        print("filename : ",filename)   
        image = Image.open(os.path.join(dir_path,filename))
        image_arr = np.array(image)
        print(image_arr.shape)
 '''
 
def format_GRIET_line():
    """
    Formatting the dataset at the line level witht he split of (30 train, 10 valid and 9 for test)"""
    source_folder = "raw/GRIET"
    target_folder = "formatted/GRIET_lines"
    line_folder_path = os.path.join(target_folder, "lines")
    text_file_path = os.path.join(target_folder,"text_files")
    os.makedirs(target_folder, exist_ok = True)
    
    
    
    set_names = ["train", "valid", "test"]
    gt = {
        "train" : dict(),
        "valid" : dict(),
        "test" : dict()
    }
    charset = set()

    lines_path = os.path.join(source_folder,"lines")
    size = get_count(lines_path)
    convert_img_into_grayscale(lines_path)          #converting images into 2D/GrayScale to rectify the errors
    
    # creating directories for testing,training and validation folders.
    test_folder_path = os.path.join(target_folder,"test")
    train_folder_path = os.path.join(target_folder,"train")
    valid_folder_path = os.path.join(target_folder, "valid")
    os.makedirs(test_folder_path, exist_ok = True)
    os.makedirs(train_folder_path, exist_ok = True)
    os.makedirs(valid_folder_path, exist_ok = True)
    
    
    #loading the miges in lines folder and placing it into 

    used = [False]*size
    set_name = "train"
    counter = 0
    current_folder = train_folder_path
    done_count = 0
    
    # with open("sample_dataset_output.txt","w+") as o:       #remove this and back indent 90 to 139
    while(done_count < size):
        if(set_name == "train" and done_count > 0.8*size and done_count <= 0.9*size):
            set_name = "valid"
            current_folder = valid_folder_path
            counter = 0
        elif(set_name == "valid" and done_count > 0.9*size):
            set_name = "test"
            current_folder = test_folder_path
            counter = 0
        
        #get an random img 
        id = get_unique_id(size, used)
        
        used[id] = True
        img_path = os.path.join(source_folder,"lines",str(id)+".png")
        
        text_file = os.path.join(source_folder,"text_files", str(id)+".txt")
        with open(text_file) as f:
            label = f.readline()
            label = str(label)
            label = label.replace('\ufeff','')
            label = label.replace('\u200c','')
            label = label.replace("\n","")
            label_words = label.split(" ")
            new_label = ""
            for i in label_words:
                i = str(i)
                i = i.lstrip()
                i = i.rstrip()
                new_label += i+" "
            label = new_label[:-1]
            label = label.lstrip()
            label = label.rstrip()
            # o.write("###"+str(label)+"###\n\n")                #even remove this
            
        img_name = "{}_{}.png".format(set_name, counter)
        gt[set_name][img_name] = {
            "text" : label
        }
        charset = charset.union(set(label))
        new_path = os.path.join(current_folder, img_name)
        
        # this line does copies the imgs in lines folder in GRIET_lines folder  
        shutil.copy(img_path, new_path)  
        
        #this line deletes the contents of the lines folder in GRIET_lines folder.
        # os.replace(img_path, new_path)
        
        done_count += 1
        counter+=1
        
    with open(os.path.join(target_folder, "labels.pkl"), "wb") as f:
        pickle.dump({
            "ground_truth": gt,
            "charset": sorted(list(charset)),
        }, f)        
                                            
if __name__ == "__main__":
    format_GRIET_line()