import cv2
import numpy as np
import rasterio as rio
from skimage.transform import resize
import os
from utils_KMEANS import KMEANS
from utils_postCD import post_CD
from utils_vote import VOTE
from utils_plot import PLOT
from utils_merge import merge_grids
from glob import glob
from rasterio.mask import mask
import json
from shapely.geometry import box
import geopandas as gpd
import warnings
import shutil
from utils_Region_Based import region_based
from utils_RF import randomforest
# ignore warnings
warnings.filterwarnings("ignore")


################################# functions #################################

def UINT8(Data):  # _____ gets a multi-band image and returns it as uint8 image
    shape = Data.shape
    for i in range(shape[2]):
        data = Data[:, :, i]
        data = data / data.max()
        data = 255 * data
        Data[:, :, i] = data.astype(np.uint8)
    return Data


def getFeatures(gdf):  # _____ reads coordinates of a json shapefile
    return [json.loads(gdf.to_json())['features'][0]['geometry']]


def get_index(path):  # _____ splits the input path string to objects of a list that index can be extracted from it
    # path = path.replace("/", " ")
    path = path.replace("_", " ")
    path = path.replace(".", " ")
    return path.split()


def ndvi(img):  # _____ creates ndvi image
    B4 = img[:, :, 2]
    B8 = img[:, :, 6]
    ndvi = (B8 - B4) / (B8 + B4)
    return ndvi


def index(img):  # _____ gets a multi-band image and creates greenness and brightness indexes from it
    B = 0.3510 * img[:, :, 0] + 0.3813 * img[:, :, 1] + 0.3437 * img[:, :, 2] + 0.7196 * img[:, :, 6] + 0.2396 * img[:, :, 8] + 0.1949 * img[:, :, 9]
    G = -0.3599 * img[:, :, 0] - 0.3533 * img[:, :, 1] - 0.4734 * img[:, :, 2] + 0.6633 * img[:, :, 6] + 0.0087 * img[:, :, 8] - 0.2856 * img[:, :, 9]
    return B, G


# ______________________________________________________________________________
class change_detection:
    def __init__(self, root, serial, city_name, kmeans_type="TCI_band", n_bands=3, show_plots=False):
        city_params = {
            "tehran"    : {'n_cluster': 3, 'ndvi_thre': 0.3   , 'cva_thre' : 1},
            "alborz"    : {'n_cluster': 7, 'ndvi_thre': 0.35  , 'cva_thre' : 2.5},
            'qom'       : {'n_cluster': 6, 'ndvi_thre': 0.2   , 'cva_thre' : 1},
            'mazandaran': {'n_cluster': 6, 'ndvi_thre': 0.3   , 'cva_thre' : 1},
            'khoozestan': {'n_cluster': 7, 'ndvi_thre': 0.35  , 'cva_thre' : 1},
            'fars'      : {'n_cluster': 7, 'ndvi_thre': 0.165 , 'cva_thre' : 1}
        }
        ndvi_thre = city_params[city_name]['ndvi_thre']
        n_cluster = city_params[city_name]['n_cluster']
        cva_thre  = city_params[city_name]['cva_thre']
        self.root = root  # _____ root path is the path where the code is
        self.serial = serial  # _____ serial number determining the grid
        self.kmeans_type = kmeans_type  # _____ defines which bands should be used for kmeans(tct_ndvi / tct / all_bands or else)
        self.n_cluster = n_cluster  # _____ shows the number of clusters
        self.n_bands = n_bands  # _____ shows the number of bands that should say a pixel is change in kmeans
        self.show_plots = show_plots  # _____ true ===> shows plots / false ===> doesnt show the plots
        self.ndvi_thre = ndvi_thre  # _____ threshold for classifying a pixel to vegetation or non vegetation
        self.cva_thre = cva_thre

    def grid_to_chips(self, date):  # _____ creates small chips from grids
        s2_bands = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12', 'TCI_R', 'TCI_G', 'TCI_B']
        # _____ creating path to dem images / input grids and folder to save splited chips
        dem_path = os.path.join(self.root, 'data', 'dem')
        grid_path = os.path.join(self.root, 'data', date, 'ai')
        split_path = os.path.join(self.root, 'data', date, 'split')

        dem_images = sorted(glob(dem_path + f'/dem_{self.serial}*'))
        input_tiles = sorted(glob(grid_path + f'/sentinel2_{self.serial}*'))

        if not os.path.isdir(split_path):
            os.mkdir(split_path)
        # _____  creating small input images from big image
        done = 0
        print()
        print('*' * 20, 'creating small images from big slices', '*' * 20)
        for dem in dem_images:
            Dem = rio.open(dem)
            shape = Dem.shape
            slice = np.zeros((13, shape[0], shape[1]))
            ind = get_index(dem)[-2]
            bound = Dem.bounds
            metadata = Dem.meta.copy()
            left = bound[0]
            bot = bound[1]
            right = bound[2]
            top = bound[3]
            bbox = box(left, bot, right, top)
            geo = gpd.GeoDataFrame({'geometry': bbox}, index=[0], crs="EPSG:4326")
            geo = geo.to_crs(crs=4326)
            coords = getFeatures(geo)

            for tile in input_tiles:

                band = get_index(tile)[-2]
                if band in ['R', 'G', 'B']:
                    band = get_index(tile)[-3] + '_' + get_index(tile)[-2]
                band_num = s2_bands.index(band)
                data = rio.open(tile)
                out_img, out_transform = mask(dataset=data, shapes=coords, crop=True)
                out_img = np.squeeze(out_img)
                out_img = cv2.resize(out_img, (shape[1], shape[0])).astype(int)
                slice[band_num, :, :] = out_img

            done = done + 1
            total = len(dem_images)
            Done = int(50 * done / total)
            print("\r[%s%s]" % ('=' * Done, ' ' * (50 - Done)), f' {done}/{total}', end='')

            metadata.update({'transform': out_transform, 'width': shape[1], 'count': 13,
                             'height': shape[0], 'dtype': rio.float32})
            image_path = os.path.join(split_path, f'sentinel2_{self.serial}_{ind}_ai.tiff')
            with rio.open(image_path, 'w', **metadata) as tile_dataset:
                slice = slice.astype(np.float32)
                tile_dataset.write(np.array(slice))
                tile_dataset.close()

    ################################# change detection code #################################

    # finding all images in directory
    def CM(self):  # _____ main function for change detection
        input_dir = os.path.join(self.root, 'data', 'before', 'split')  # _____ path to splited chips as input images
        output_dir = os.path.join(self.root, 'data', 'chips_results')  # _____ path to directory to save results
        image1_paths = []
        image2_paths = []
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        files = sorted(glob(input_dir + f'/sentinel2_{self.serial}*'))
        # _____ creating address for image date1 and date2
        for i in range(len(files)):
            address = files[i]
            image1_paths.append(address)
            ind = get_index(address)[-3]
            address = f'{self.root}/data/after/split/sentinel2_{self.serial}_{ind}_ai.tiff'
            image2_paths.append(address)

        for i in range(len(image1_paths)):
            image1_path = image1_paths[i]
            self.index = get_index(image1_path)[-3]
            print()
            print('*' * 50, f'creating change mask for image with index {self.index} in grid {self.serial}', '*' * 50)
            image2_path = image2_paths[i]
            img1 = rio.open(image1_path)
            image1 = img1.read()
            img2 = rio.open(image2_path)
            image2 = img2.read()
            # _____ reshaping image from (c , h , w) to (h , w , c)
            image1 = image1.transpose(1, 2, 0)
            image2 = image2.transpose(1, 2, 0)
            # _____ histogram matching

            # Compute the histograms of the two images
            hist1, bins1 = np.histogram(image1, bins=65536, range=[0, 65535])
            hist2, bins2 = np.histogram(image2, bins=65536, range=[0, 65535])
            # Perform histogram matching
            image1 = np.interp(image1, bins1[:-1], bins2[:-1])

            # ___________________________________________________________________________________________

            # _____ for removing nodata from final mask
            mask = np.any(image1 == 0, axis=2) | np.any(image2 == 0, axis=2)
            # _____ replacing pixels with pixel value of 0 to 1 , these pixels are no data
            image1[mask] = 1
            image2[mask] = 1
            # _____ using tci bands for visualisation
            img1_vis = image1[:, :, 10:13].astype(np.uint8)
            img2_vis = image2[:, :, 10:13].astype(np.uint8)

            # _____ creating float32 image from uint16 image
            image1 = image1.astype(np.float32)
            image2 = image2.astype(np.float32)

            # _____ resizing image to execute kmeans(needed to be coefficient of 5)
            x, y = image1.shape[0], image1.shape[1]
            new_sizex = np.asarray(image1.shape[0]) / 5
            new_sizex = new_sizex.astype(int) * 5
            new_sizey = np.asarray(image1.shape[1]) / 5
            new_sizey = new_sizey.astype(int) * 5
            new_size = [new_sizey, new_sizex]
            image1_new = resize(image1, (new_sizex, new_sizey, image1.shape[2])).astype(int)
            image2_new = resize(image2, (new_sizex, new_sizey, image1.shape[2])).astype(int)

            # _____ create ndvi images
            ndvi1_area = ndvi(image1)
            ndvi2_area = ndvi(image2)
            # _____ create difference NDVI image
            diff_ndvig = ndvi1_area - ndvi2_area
            # _____ create change map of diff ndvi
            diff_ndvi = np.where(abs(diff_ndvig) >= self.ndvi_thre, 1, 0)
            
            
            ########### k-means_PCA CD ###########

            n_cluster = self.n_cluster  # _____ indicates the percentage of sureness for change pixel
            n_band = self.n_bands  # _____ indicates how many bands should say there is a change in pixel
            # kmeans_type="TCT" :using brightness , kmeans_type="all_bands" :using 13 bands of Sentinel2 
            # kmeans_type="TCI_band" :using TCI bands of  Sentinel2 
            change_mask_kmeans = KMEANS(image1_new, image2_new, new_size, n_cluster, n_band, use_bands=self.kmeans_type)
            
            
            ########### Change Vector Analysis(CVA) CD ###########
            B1, G1 = index(image1)
            B2, G2 = index(image2)
            change_mask_cva = np.sqrt((B2 - B1) ** 2 + (G2 - G1) ** 2)
            ########### post_classification CD (Unet++) ###########
            change_mask_pc, seg1, seg2 = post_CD(self.root, image1_path, image2_path, serial=self.serial)
           
            
            ########### post_classification CD (randomforest) ###########
            change_mask_RF= randomforest(image1,image2)



            ########### vote ###########
            # _____ use weight for difference NDVI and post CD  for pixels with high ndvi: weighted_method="NDVI"
            # _____ use weight for  CVA : weighted_method="CVA"
            weighted_method = "NDVI"
            final_mask = VOTE(weighted_method, change_mask_kmeans=change_mask_kmeans,change_mask_RF=change_mask_RF,
                              change_mask_cva=change_mask_cva, change_mask_pc=change_mask_pc, diff_ndvi=diff_ndvi, y=y, x=x,
                              ndvi1_area=ndvi1_area, ndvi2_area=ndvi2_area, cva_thre = self.cva_thre)

            # _____applying region based to small chips for faster process
            final_mask = region_based(final_mask, seg1[0], seg2[0])

            # _____  Save final change map
            transform = img1.transform
            crs = img1.crs
            shape = img1.shape
            os.chdir(output_dir)
            new_dataset = rio.open(f"CM_{self.serial}_{self.index}.tif", 'w', driver='GTiff', height=shape[0],
                                   width=shape[1],
                                   count=1, dtype=str('uint8'),
                                   crs=crs,
                                   transform=transform)
            new_dataset.write(np.array([final_mask]).astype('uint8'))
            new_dataset.close()

            ########### plot ###########
            if self.show_plots:
                PLOT(weighted_method, img1_vis, img2_vis, change_mask_cva, change_mask_kmeans,change_mask_RF, change_mask_pc, final_mask,
                     diff_ndvi, diff_ndvig, y, x , cva_thre = self.cva_thre)

    def main(self):
        #self.grid_to_chips(date='before')  # _____ create chips for date one images
        #self.grid_to_chips(date='after')  # _____ create chips for date two images
        self.CM()  # _____ calling main function to compute and save change mask images
        final_mask, out_transform, metadata = merge_grids(self.root, self.serial)  # _____ this function will assign one change class to a region instead of pixel based CM

        seg_path = os.path.join(root, 'data/seg')  # _____ path to save segmentation results for from to image
        chips_result = os.path.join(root, 'data/chips_results')  # _____ path to save small chips
        result_path = os.path.join(self.root, 'data/results/ai')  # _____ path for saving final change mask grid

        metadata.update({'transform': out_transform, 'width': final_mask.shape[1],
                         'height': final_mask.shape[0], 'dtype': np.float32})
        image_path = os.path.join(result_path, f'CM_{self.serial}.tif')
        with rio.open(image_path, 'w', **metadata) as tile_dataset:
            tile_dataset.write(np.array([final_mask]).astype('float32'))
            tile_dataset.close()
        print('*' * 50, f'grid {self.serial} is done ;)', '*' * 50)

        # _____ removing junk files that are not needed
        shutil.rmtree(chips_result)
        shutil.rmtree(seg_path)


root = os.getcwd()
change_process = change_detection(root, serial='R59C27', city_name="alborz", show_plots=True)
change_process.main()
