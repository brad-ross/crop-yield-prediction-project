import numpy as np
import gdal
from joblib import Parallel, delayed
import datetime
import bucket_util as bu

################
# Data range
# MODIS: 2003-2016, 14 years
# MODIS_landcover: 2003-2013, 12 years
# MODIS_temperature: 2003_2015, 13 years

# Intersection: 2003-2013, 11 years
################

def divide_image(img,first,step,num):
    image_list=[]
    for i in range(0,num-1):
        image_list.append(img[:, :, first:first+step])
        first+=step
    image_list.append(img[:, :, first:])
    return image_list

def extend_mask(img,num):
    for i in range(0,num):
        img = np.concatenate((img, img[:,:,-2:-1]),axis=2)
    return img

def merge_image(MODIS_img_list,MODIS_temperature_img_list):
    MODIS_list=[]
    for i in range(0,len(MODIS_img_list)):
        img_shape=MODIS_img_list[i].shape
        img_temperature_shape=MODIS_temperature_img_list[i].shape
        img_shape_new=(img_shape[0],img_shape[1],img_shape[2]+img_temperature_shape[2])
        merge=np.empty(img_shape_new)
        for j in range(0,img_shape[2]/7):
            img=MODIS_img_list[i][:,:,(j*7):(j*7+7)]
            temperature=MODIS_temperature_img_list[i][:,:,(j*2):(j*2+2)]
            merge[:,:,(j*9):(j*9+9)]=np.concatenate((img,temperature),axis=2)
        MODIS_list.append(merge)
    return MODIS_list

def mask_image(MODIS_list,MODIS_mask_img_list):
    MODIS_list_masked = []
    for i in range(0, len(MODIS_list)):
        mask = np.tile(MODIS_mask_img_list[i],(1,1,MODIS_list[i].shape[2]))
        masked_img = MODIS_list[i]*mask
        MODIS_list_masked.append(masked_img)
    return MODIS_list_masked

def quality_dector(image_temp):
        filter_0=image_temp>0
        filter_5000=image_temp<5000
        filter=filter_0*filter_5000
        return float(np.count_nonzero(filter))/image_temp.size

def preprocess_save_data_parallel(fileInfo):
    prefix, datatype, file = fileInfo # Unpack the tuple
    
    data_yield = np.genfromtxt('yield_final.csv', delimiter=',', dtype=float)
    if file.endswith(".tif"):
        MODIS_path = bu.getFullPath(fileInfo)
        MODIS_temperature_path = bu.getFullPath(bu.replaceDatatype(fileInfo, 'temperature'))
        MODIS_mask_path = bu.getFullPath(bu.replaceDatatype(fileInfo, 'mask'))
        
        # get geo location
        raw = file.replace('_',' ').replace('.',' ').split()
        loc1 = int(raw[0])
        loc2 = int(raw[1])

        # read image
        try:
            MODIS_img = np.transpose(np.array(gdal.Open(MODIS_path).ReadAsArray(), dtype='uint16'),axes=(1,2,0))
        except ValueError as msg:
            print msg

        # read temperature
        MODIS_temperature_img = np.transpose(np.array(gdal.Open(MODIS_temperature_path).ReadAsArray(), dtype='uint16'),axes=(1,2,0))
        # shift
        MODIS_temperature_img = MODIS_temperature_img-12000
        # scale
        MODIS_temperature_img = MODIS_temperature_img*1.25
        # clean
        MODIS_temperature_img[MODIS_temperature_img<0]=0
        MODIS_temperature_img[MODIS_temperature_img>5000]=5000

        # read mask
        MODIS_mask_img = np.transpose(np.array(gdal.Open(MODIS_mask_path).ReadAsArray(), dtype='uint16'),axes=(1,2,0))
        # Non-crop = 0, crop = 1
        MODIS_mask_img[MODIS_mask_img != 12] = 0
        MODIS_mask_img[MODIS_mask_img == 12] = 1

        # Divide image into years
        MODIS_img_list=divide_image(MODIS_img, 0, 46 * 7, 14)
        MODIS_temperature_img_list = divide_image(MODIS_temperature_img, 0, 46 * 2, 14)
        MODIS_mask_img = extend_mask(MODIS_mask_img, 3)
        MODIS_mask_img_list = divide_image(MODIS_mask_img, 0, 1, 14)

        # Merge image and temperature
        MODIS_list = merge_image(MODIS_img_list,MODIS_temperature_img_list)

        # Do the mask job
        MODIS_list_masked = mask_image(MODIS_list,MODIS_mask_img_list)

        # check if the result is in the list
        year_start = 2003
        bu.setBucketLocation('~/cs231n-satellite-images-clean')
        for i in range(0, 14):
            year = i+year_start
            key = np.array([year,loc1,loc2])
            if np.sum(np.all(data_yield[:,0:3] == key, axis=1))>0:
                ## 1 save original file
                filename = bu.getFullPath((prefix, 'output_full', str(year)+'_'+str(loc1)+'_'+str(loc2)+'.npy'))
                np.save(filename,MODIS_list_masked[i])
                print datetime.datetime.now()
                print filename,':written '
        bu.setBucketLocation('~/cs231n-satellite-images') # Return bucket path to original

if __name__ == "__main__":
    bu.setBucketLocation('~/cs231n-satellite-images')
    files = bu.walk('data', 'image_full')
    Parallel(n_jobs=12)(delayed(preprocess_save_data_parallel)(file) for file in files)
