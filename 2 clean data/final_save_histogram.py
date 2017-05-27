import numpy as np
import math
import os
import bucket_util as bu
import datetime

class fetch_data():
    def __init__(self):
        self.dir = os.path.expanduser('~/cs231n-satellite-images-clean') + '/data_output_full_'
        self.outputdir = os.path.expanduser('~/cs231n-satellite-images-hist') + '/data_output_full_'

        # load yield data
        self.data_yield = np.genfromtxt('yield_final_highquality.csv', delimiter=',')
        self.locations = np.genfromtxt('locations_final.csv', delimiter=',')
        
        # generate index for all data
        length = self.data_yield.shape[0]
        self.index_all = np.arange(length)

    def filter_timespan(self,image_temp,start_day,end_day,bands):
        start_index=int(math.floor(start_day/8))*bands
        end_index=int(math.floor(end_day/8))*bands
        if end_index>image_temp.shape[2]:
            image_temp = np.concatenate((image_temp, 
                np.zeros((image_temp.shape[0],image_temp.shape[1],end_index-image_temp.shape[2]))),axis=2)
        return image_temp[:,:,start_index:end_index]

    def calc_histogram(self,image_temp,bin_seq,bins,times,bands):
        hist=np.zeros([bins,times,bands])
        for i in range(image_temp.shape[2]):
            density, _ = np.histogram(image_temp[:, :, i], bin_seq, density=False)
            if density.sum()==0: # WE UNCOMMENTED THIS BC SOME IMAGES ARE ALL 0S!!!
                 continue
            hist[:, i / bands, i % bands] = density / float(density.sum())

        return hist

    def save_data(self):
        output_image = np.zeros([self.index_all.shape[0], 32, 32, 9])
        output_yield = np.zeros([self.index_all.shape[0]])
        output_year = np.zeros([self.index_all.shape[0]])
        output_locations = np.zeros([self.index_all.shape[0],2])
        output_index = np.zeros([self.index_all.shape[0],2])

        for i in self.index_all:
            if i % 2000 == 0:
                print "Saving snapshot!"
                np.savez(self.outputdir+'histogram_all_full_snapshot_%d.npz' % i,
                         output_image=output_image,output_yield=output_yield,
                         output_year=output_year,output_locations=output_locations,output_index=output_index)

            year = str(int(self.data_yield[i, 0]))
            loc1 = str(int(self.data_yield[i, 1]))
            loc2 = str(int(self.data_yield[i, 2]))

            key = np.array([int(loc1),int(loc2)])
            index = np.where(np.all(self.locations[:,0:2].astype('int') == key, axis=1))
            longitude = self.locations[index,2]
            latitude = self.locations[index,3]

            print datetime.datetime.now()
            filename = year + '_' + loc1 + '_' + loc2 + '.npy'
            print "Examining file: %s!" % filename
            try:
                image_temp = np.load(self.dir + filename) # ?
                image_temp = self.filter_timespan(image_temp, 49, 305, 9) # ?
                print datetime.datetime.now()
                print image_temp.shape

                bin_seq = np.linspace(1, 4999, 33)
                image_temp = self.calc_histogram(image_temp, bin_seq ,32, 32, 9) # ?
                image_temp[np.isnan(image_temp)] = 0
                if np.sum(image_temp) < 250:
                    print 'broken image', filename

                output_image[i, :] = image_temp
                output_yield[i] = self.data_yield[i, 3]
                output_year[i] = int(year)
                output_locations[i, 0] = longitude
                output_locations[i, 1] = latitude
                output_index[i,:] = np.array([int(loc1),int(loc2)])

                print i,np.sum(image_temp),year,loc1,loc2
            except IOError:
                print "File: %s not found!" % filename
                continue
            except ValueError:
                print "File: Value error in %s!" % filename
                continue
        np.savez(self.outputdir+'histogram_all_full.npz',
                 output_image=output_image,output_yield=output_yield,
                 output_year=output_year,output_locations=output_locations,output_index=output_index)
        print 'save done'

if __name__ == '__main__':
    data=fetch_data()
    data.save_data()