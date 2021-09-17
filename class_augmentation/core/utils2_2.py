from core.config import cfg
from tqdm import tqdm

import os
import glob
import numpy as np
import json
import pandas as pd
import tensorflow as tf
import random
import PIL
from PIL import Image 

############################################################################################################################################################################
### car.json 생성 관련 함수                                                                                                                                               ###
############################################################################################################################################################################
def write_class_names(limit=-1):

    ##classes폴더에 train's json정보와 valid's json정보가 담긴 dataframe이 있는지 검사하고 없다면, 생성해줍니다.
    try :
        dataframe_train = pd.read_csv(cfg.NEXTLAB.TRAIN_DATAFRAME_PATHS, encoding='euc-kr')
        dataframe_valid = pd.read_csv(cfg.NEXTLAB.VALID_DATAFRAME_PATHS, encoding='euc-kr')
        
    except :
        paths_valid = get_paths(cfg.NEXTLAB.BACKUP_VALID_LABELS_PATH, 'json') 
        dataframe_valid = get_jsons(cfg.NEXTLAB.BACKUP_VALID_IMAGES_PATH, paths_valid)
        
    #dataframe 합치기
    dataframe_All = pd.concat([dataframe_train, dataframe_valid], axis=0, ignore_index=True)
    dataframe_All.to_csv(cfg.NEXTLAB.ALL_DATAFRAME_PATHS, mode='w',encoding='euc-kr', index=False)

    #'brand','model','year'를 조합해 클래스를 생성합니다.
    dataframe_All['class_name'] = dataframe_All['brand']+'/'+dataframe_All['model']+'/'+dataframe_All['year']

    # ##이미지의 수가 너무 적은 클래스를 Drop하기 위해서 클래스당 이미지의 수를 count합니다.
    
    counts = dataframe_All['class_name'].value_counts().to_frame().reset_index()
    counts.columns = ['class_name', 'count']
    dataframe_All = pd.merge(dataframe_All, counts, on= 'class_name')

    # 분류에 사용할 부분만 오려내기
    df = dataframe_All.drop(['brand', 'model', 'year', 'color'], axis=1)
       
    # ##dataframe_All을 class_name기준으로 병합합니다.
    
    df = dataframe_All.groupby(['class_name','count'])['image_path'].apply(','.join).reset_index()

    #'image_path'를 list형태로 바꿔서 dataframe에 저장합니다. 

    df.image_path = df.image_path.str.split(',') 

    #이미지의 수가 너무 적은 클래스를 Drop합니다.
    if limit >= 0 :
        drop_index = df[df['count'] < limit].index
        df = df.drop(drop_index).reset_index(drop=True)
        
    df = df.reset_index().rename(columns={"index": "label", "count": "num_per_class"})
    df.to_excel(cfg.NEXTLAB.MODIFIED_DATAFRAME_PATHS, encoding='euc-kr', index=False)

    ##JSON형태로 저장합니다.
    class_names = df.set_index('class_name').T.to_dict()
    with open(cfg.NEXTLAB.CLASSES_JSON, 'w', encoding='utf-8') as make_file:
        json.dump(class_names, make_file, indent="\t", ensure_ascii=False) 


############################################################################################################################################################################
### 기타함수                                                                                                                                                              ###
############################################################################################################################################################################
def get_length(file_name):
        tmp = file_name.split('-')
        tmp = tmp[1].split('.')
        return int(tmp[0])
        
def get_paths(path=str, target=str):
    if target == 'json' :
        __DIR = '/**/**/**.json'
    elif target == 'tfrecord' :
        __DIR = '/*.tfrecord'
    ##assert
    paths = path + __DIR
    paths = glob.glob(path + __DIR)
    return paths

def get_jsons(image_path=str, jsons_paths=list):
    json_dataframe = pd.DataFrame()
    for path in tqdm(jsons_paths):
        with open(path, "r", encoding='UTF8') as json_file:
            json_data = json.load(json_file)
            json_data = json_data['car']
            data = json_data['attributes']
            data['image_path'] = image_path + '/' + json_data['imagePath']
            json_dataframe = json_dataframe.append(data, ignore_index=True)
    return json_dataframe


def augmentaion(df, seq, n_aug, limit_flag=True):  
    for df_index in range(len(df)):
        image_paths = df['image_path'].iloc[df_index]
        image_paths_valid = df['image_path_valid'].iloc[df_index]
        n_image = len(image_paths)
        image_paths_toAgu = image_paths

        
        if (n_image == n_aug) :
            image_paths_toAgu = image_paths
        ## 클래스당 이미지 보유 수가 증강할 이미지의 수보다 작은 경우  
        elif (n_image < n_aug) :
            ## 증강할 이미지 수만큼 원본 이미지 경로를 랜덤하게 뽑아 추가해줍니다.
            while (len(image_paths_toAgu) < n_aug) :
                image_paths_toAgu+=[random.choice(image_paths)]     
        ## 클래스당 이미지 보유 수가 증강할 이미지의 수보다 클 경우 
        elif (n_image > n_aug): 
            ##이미지 최대 수를 n_aug의 수만큼 제한할거면.
            if limit_flag == True :
                ##이미지수를 증강할 이미지수로 변경   
                n_image=n_aug  
            ## 증강할 이미지 수만큼 원본 이미지 경로를 랜덤하게 뽑아 줄여줍니다.
                while (len(image_paths_toAgu) > n_aug) : 
                    image_paths_toAgu.pop(random.randint(0,len(image_paths_toAgu)-1))
                    image_paths_toAgu = set(image_paths_toAgu)
                    image_paths_toAgu = list(image_paths_toAgu)
        
        ##backup-dataset에서 원본 이미지를 열어서 dataset폴더에 저장합니다. (증강 안함)
        agued_img_paths=[]
        nonagued_img_paths = []
        for path in image_paths_toAgu[:n_image] :
            image = np.array(PIL.Image.open(path))
            image = Image.fromarray(image)
            nonagued_img_path = path.replace('/backup/','/')
            image.save(nonagued_img_path)
            nonagued_img_paths.append(nonagued_img_path)

        for path in image_paths_valid :
            image = np.array(PIL.Image.open(path))
            image = Image.fromarray(image)
            image.save(path.replace('/backup/','/'))

        ##가지고 있는 이미지 수가 증강할 이미지 수를 넘지 않는 경우에만 이미지를 증강합니다.
        if n_image !=  n_aug : 
            image_list=[]
            for path in tqdm(image_paths_toAgu[n_image:]) :
                image =  np.array(PIL.Image.open(path))
                image_list.append(image)

            #이미지를 증강합니다.
            images_aug = seq(images=image_list)

            #증강한 이미지를 저장합니다 경로명은 dataset에서 dataset_aug으로 변경하고 확장자 앞에 _aug+숫자 를 붙여줍니다.
            for i, image in enumerate(images_aug) :
                image = Image.fromarray(image)
                agued_img_path = image_paths_toAgu[i].replace('/backup/','/').replace('.jpg',f'_aug{i}.jpg')              
                image.save(agued_img_path) ##./data/dataset_aug/train/image/**/**/**_aug_{i}.jpg    
                agued_img_paths.append(agued_img_path)
        #증강한 이미지에 대응하는 경로명을 json파일에 수정 및 저장합니다.
        df['image_path'].iloc[df_index] = nonagued_img_paths + agued_img_paths
        print(f"[{df_index}] class::[{df['class_name'].iloc[df_index]}] is done!")
        
    class_names = df.set_index('class_name').T.to_dict()
    with open(cfg.NEXTLAB.CLASSES_JSON, 'w', encoding='utf-8') as make_file:
        json.dump(class_names, make_file, indent="\t", ensure_ascii=False)   

def generate_train_newpath(path, class_name):
    path = path.split('/')
    newpath = path[0]+'/'+path[1]+'/'+path[2]+'/'+'train'+'/'+class_name     
    if not os.path.exists(newpath):
        os.makedirs(newpath)                
    newpath = newpath + '/'+path[7]
    return newpath

def generate_valid_newpath(path, class_name):
    path = path.split('/')
    newpath = path[0]+'/'+path[1]+'/'+path[2]+'/'+'valid'+'/'+class_name     
    if not os.path.exists(newpath):
        os.makedirs(newpath)                
    newpath = newpath + '/'+path[7]
    return newpath

def augmentaionv2(df, seq, n_aug, valid_size, limit_flag=True):  
    #train dir, valid dir 생성
    train_dir = './data/dataset/train'
    valid_dir = './data/dataset/valid'
    try:
        if not os.path.exists(train_dir):
            os.makedirs(train_dir)
    except OSError:
        print ('Error: Creating directory. ' +  train_dir)

    try:        
        if not os.path.exists(valid_dir):
            os.makedirs(valid_dir)
    except OSError:
        print ('Error: Creating directory. ' +  valid_dir)


    
    for df_index in range(len(df)):
        class_name = df['class_name'].iloc[df_index].replace('/','_')
        image_paths = df['image_path'].iloc[df_index]
        
        #현재 가지고 있는 원본 image 나눠갖기
        n_image = len(image_paths)
         
                                
        train_n_image = int((1-valid_size)*n_image)
        train_n_aug = int((1-valid_size)*n_aug)


        train_image_paths= image_paths[:train_n_image]
        train_image_paths_toAgu = train_image_paths


        #나줘가진 train image에 대해
        if (train_n_image== train_n_aug) :
            train_image_paths_toAgu = train_image_paths
        ## 클래스당 이미지 보유 수가 증강할 이미지의 수보다 작은 경우  
        elif (train_n_image < train_n_aug) :
            ## 증강할 이미지 수만큼 원본 이미지 경로를 랜덤하게 뽑아 추가해줍니다.
            while (len(train_image_paths_toAgu) < train_n_aug) :
                train_image_paths_toAgu+=[random.choice(train_image_paths)]     
        ## 클래스당 이미지 보유 수가 증강할 이미지의 수보다 클 경우 
        elif (train_n_image > train_n_aug): 
            ##이미지 최대 수를 n_aug의 수만큼 제한할거면.
            if limit_flag == True :
                ##이미지수를 증강할 이미지수로 변경   
                train_n_image = train_n_aug  
            ## 증강할 이미지 수만큼 원본 이미지 경로를 랜덤하게 뽑아 줄여줍니다.
                while (len(train_image_paths_toAgu) > train_n_aug) : 
                    train_image_paths_toAgu.pop(random.randint(0,len(train_image_paths_toAgu)-1))
                    train_image_paths_toAgu = set(train_image_paths_toAgu)
                    train_image_paths_toAgu = list(train_image_paths_toAgu)


        valid_n_image = n_image - train_n_image
        valid_n_aug = n_aug - train_n_aug

        
        valid_image_paths= image_paths[train_n_image:]
        valid_image_paths_toAgu = valid_image_paths
        
        #나줘가진 valid image에 대해
        if (valid_n_image == valid_n_aug) :
            valid_image_paths_toAgu = valid_image_paths
        ## 클래스당 이미지 보유 수가 증강할 이미지의 수보다 작은 경우  
        elif (valid_n_image < valid_n_aug) :
            ## 증강할 이미지 수만큼 원본 이미지 경로를 랜덤하게 뽑아 추가해줍니다.
            while (len(valid_image_paths_toAgu) < valid_n_aug) :
                valid_image_paths_toAgu+=[random.choice(valid_image_paths)]     
        ## 클래스당 이미지 보유 수가 증강할 이미지의 수보다 클 경우 
        elif (valid_n_image > valid_n_aug): 
            ##이미지 최대 수를 n_aug의 수만큼 제한할거면.
            if limit_flag == True :
                ##이미지수를 증강할 이미지수로 변경   
                valid_n_image = valid_n_aug
            ## 증강할 이미지 수만큼 원본 이미지 경로를 랜덤하게 뽑아 줄여줍니다.
                while (len(valid_image_paths_toAgu) > valid_n_aug) : 
                    valid_image_paths_toAgu.pop(random.randint(0,len(valid_image_paths_toAgu)-1))
                    valid_image_paths_toAgu = set(valid_image_paths_toAgu)
                    valid_image_paths_toAgu = list(valid_image_paths_toAgu)




        ##원본 이미지를 dataset에서 열어서 dataset폴더에 저장합니다. (증강 안함)
        
        train_nonagued_img_paths = []
        valid_nonagued_img_paths = []

        #train_set에 원본주기
        for path in train_image_paths_toAgu[:train_n_image] :
            image = np.array(PIL.Image.open(path))
            image = Image.fromarray(image)
            train_nonagued_img_path = path.replace('/backup/','/')
            train_nonagued_img_path = generate_train_newpath(train_nonagued_img_path, class_name)
            image.save(train_nonagued_img_path)
            train_nonagued_img_paths.append(train_nonagued_img_path)

        
        
        #valid_set에 원본주기
        for path in valid_image_paths_toAgu[:valid_n_image] :
            image = np.array(PIL.Image.open(path))
            image = Image.fromarray(image)
            valid_nonagued_img_path = path.replace('/backup/','/')
            valid_nonagued_img_path = generate_valid_newpath(valid_nonagued_img_path, class_name)
            image.save(valid_nonagued_img_path)
            valid_nonagued_img_paths.append(valid_nonagued_img_path)

        ##가지고 있는 이미지 수가 증강할 이미지 수를 넘지 않는 경우에만 이미지를 증강합니다.

        train_agued_img_paths=[]
        valid_agued_img_paths=[]



            #train 증강
        if train_n_image !=  train_n_aug : 
            train_image_list=[]
            for path in tqdm(train_image_paths_toAgu[train_n_image:]) :
                image =  np.array(PIL.Image.open(path))
                train_image_list.append(image)

            #이미지를 증강합니다.
            train_images_aug = seq(images=train_image_list)

            #증강한 이미지를 저장합니다 경로명은 dataset에서 dataset_aug으로 변경하고 확장자 앞에 _aug+숫자 를 붙여줍니다.
                #증강한 이미지 나눠갖기
            for i, image in enumerate(train_images_aug) :
                image = Image.fromarray(image)
                train_agued_img_path = train_image_paths_toAgu[i].replace('/backup/','/').replace('.jpg',f'_aug{i}.jpg') 
                train_agued_img_path = generate_train_newpath(train_agued_img_path, class_name)              
                image.save(train_agued_img_path) ##./data/dataset_aug/train/image/**/**/**_aug_{i}.jpg    
                train_agued_img_paths.append(train_agued_img_path)

               #valid 증강
        if valid_n_image !=  valid_n_aug : 
            valid_image_list=[]
            for path in tqdm(valid_image_paths_toAgu[valid_n_image:]) :
                image =  np.array(PIL.Image.open(path))
                valid_image_list.append(image)

            #이미지를 증강합니다.
            valid_images_aug = seq(images=valid_image_list)

            #증강한 이미지를 저장합니다 경로명은 dataset에서 dataset_aug으로 변경하고 확장자 앞에 _aug+숫자 를 붙여줍니다.
                #증강한 이미지 나눠갖기
            for i, image in enumerate(valid_images_aug) :
                image = Image.fromarray(image)
                valid_agued_img_path = valid_image_paths_toAgu[i].replace('/backup/','/').replace('.jpg',f'_aug{i}.jpg') 
                valid_agued_img_path = generate_valid_newpath(valid_agued_img_path, class_name)              
                image.save(valid_agued_img_path) ##./data/dataset_aug/train/image/**/**/**_aug_{i}.jpg    
                valid_agued_img_paths.append(valid_agued_img_path)

            
        #증강한 이미지에 대응하는 json파일을 수정 및 저장합니다. ######train과 valid가 합쳐진 json ,, 분리하고 싶다..

        # df['image_path'].iloc[df_index] = train_nonagued_img_paths + train_agued_img_paths + valid_nonagued_img_paths + valid_agued_img_paths
        print(f"[{df_index}] class::[{df['class_name'].iloc[df_index]}] is done!")
        
    # class_names = df.set_index('class_name').T.to_dict()
    # with open(cfg.NEXTLAB.DATASET_CLASSES_JSON, 'w', encoding='utf-8') as make_file:
    #     json.dump(class_names, make_file, indent="\t", ensure_ascii=False)   


