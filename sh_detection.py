# Reference
# Oyibo, P., Meulah, B., Bengston, M., van Lieshout, L., Oyibo, W., Diehl, J. C., Vdovin, G.,
# & Agbana, T., (2022). A Two-stage Automated Diagnosis Framework for Urogenital Schistosomiasis 
# in Microscopy Images from Low-resource Settings. Journal of Medical Imaging, [Submitted Manuscript].

r"""sautomated detection of Schistosoma haematobium eggs.

The following command runs this script and saves the following result images
1. Segmentation mask image showing the segmented eggs pixels 
2. Original input image with the detected Schistosoma haematobium  
   eggs highlighted in elipses`

```
python sh_detection.py \
  --model models/deeplabv3_mobilenetv3_sh12k_tflite_model.tflite \
  --input images/img1.jpg \
  --output ${HOME}/result
```
"""

import argparse
from PIL import Image
import math
from time import perf_counter

import numpy as np
import cv2
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects
from sklearn.mixture import GaussianMixture
import tensorflow as tf

def computeComplexity(region,k):
    
    c = region.area/500 - k

    return c

def computeAIC(region, ellipse_img, n, c):

    loss =  (1- np.sum(np.logical_and(region.image, ellipse_img))/region.area) + 1e-6

    aic = c*np.log(loss) + 2*n

    return aic

def computeOverlap(region_img, ellipse_img, area_e):

    overlap = np.sum(np.logical_and(region_img, ellipse_img))/area_e 
    
    return overlap


def computeEccentricity(axes_len):
  if axes_len[0] > axes_len[1]:
    a = axes_len[0]/2 
    b = axes_len[1]/2 
  else:
    a = axes_len[1]/2 
    b = axes_len[0]/2 

  c = np.sqrt(a**2 - b**2)
  e = c/a
  return e


def fitEllipse(region, n_ellipse):
    if n_ellipse == 1:
        center_coord = (round((region.centroid[1]-region.bbox[1])), round(region.centroid[0]-region.bbox[0]))
        axes_len = (round(region.major_axis_length), round(region.minor_axis_length))
        angle = (180 * (-region.orientation) / np.pi)+90
        e = computeEccentricity(axes_len)
        area = math.pi*(axes_len[0]/2)*(axes_len[1]/2)
        ellipses = [{'center_coord': center_coord, 'axes_len':axes_len, 'angle': angle, 'eccentricity': e, 'area': area}]     

    else:   
        # Instantiate and fit the model using GMM-EM
    
        r,c = np.where(region.image>0)
        x = np.array([c,r]).T
        if n_ellipse > 10:
          n_init = 10
        else:
          n_init = 100

        gmm = GaussianMixture(n_components=n_ellipse, n_init=n_init).fit(x) 
    
        ellipses = []
        for n in range(n_ellipse):
            covariances = gmm.covariances_[n][:2, :2]
            v, w = np.linalg.eigh(covariances)
            u = w[0] / np.linalg.norm(w[0])
            angle = np.arctan2(u[0], u[1])
            angle = (180 * (-angle) / np.pi)+90  # convert to degrees
            center_coord = (round(gmm.means_[n,0]), round(gmm.means_[n,1]))

            v = 2*1.96* np.sqrt(v)
            axes_len = (round(v[0]), round(v[1]) )
            e = computeEccentricity(axes_len)
            area = math.pi*(axes_len[0]/2)*(axes_len[1]/2)
            ellipses.append({'center_coord': center_coord, 'axes_len':axes_len, 'angle': angle, 'eccentricity': e, 'area': area})
    
    return ellipses

def createEllipseImage(img, img_b, ellipses, offset=None, scale=1):
    
    for ellipse in ellipses:
        if offset is None:
            center_coord = tuple([scale*x for x in ellipse['center_coord']])   # x,y
        else:
            center_coord = (scale*(ellipse['center_coord'][0]+offset[0]) \
                            , scale*(ellipse['center_coord'][1]+offset[1]))
     
        axes_len = tuple([round(scale*x/2) for x in ellipse['axes_len']])       # first, second
        
        angle = ellipse['angle']  # clockwise, first axis, starts horizontal
        start_angle = 0
        end_angle = 360
        color = (255,255,255)
        color_patch = (0,255,0)
        thickness = -1  
        cv2.ellipse(img, center_coord, axes_len, angle, \
                            start_angle, end_angle, color, thickness)
        cv2.ellipse(img_b, center_coord, axes_len, angle, \
                            start_angle, end_angle, color_patch, 0)
    if not offset:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    return img, img_b

def computeCountLimit(area_r):
  n_lower = math.ceil(0.65*area_r/500)
  n_upper = math.ceil(1.15*area_r/500)
    
  return n_lower, n_upper

def segmentEggsTFL(input_img, interpreter):
    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_data = np.asarray(input_img)/255
    mean=[0.59873112, 0.59390334, 0.64128324]
    std=[0.02849652, 0.02736885, 0.02703068]
    input_data = ((input_data - mean)/ std).astype('float32')
    input_data = np.expand_dims(input_data, axis=0).transpose(0,3,1,2)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    output_predictions = (output_data[0]>0).squeeze().astype('uint8') *255
    segmented_img = Image.fromarray(output_predictions)
    return segmented_img

def refinedSegmentation(segmented_img):
  segmented_img = segmented_img.resize((338,254))
  result = []

  arr=np.asarray(segmented_img)
  arr = remove_small_objects(arr>0, 100)
  label_img = label(arr)

  regions = regionprops(label_img)
  eggCount = 0
  for idx in range(len(regions)):
        
        min_r, min_c, _,_ = regions[idx].bbox
        offset = (min_c, min_r)
        n_lower, n_upper = computeCountLimit(regions[idx].area)

        best = {'areaR': regions[idx].area, 'areaE': None, 'n': None, 'overlap': None, 'ellipses': None, 'offset': offset, 'aic':float('INF')}

        for n in range(n_lower, n_upper+1):

            ellipses = fitEllipse(regions[idx], n)
            k = np.sum([ ell['area'] > 750 for ell in  ellipses])
            c = computeComplexity(regions[idx], k)
            
            h, w = regions[idx].image.shape
            ellipse_img_b = np.zeros((h,w,3), np.uint8)
            n_rows,n_cols = np.where(regions[idx].image>0)
            ellipse_img_b[n_rows, n_cols,:] = 255
            ellipse_img = np.zeros_like(ellipse_img_b, dtype=np.uint8) # creates a black image
        
            ellipse_img, ellipse_img_b = createEllipseImage(ellipse_img, ellipse_img_b, ellipses)
         
            aic = computeAIC(regions[idx], ellipse_img, n, c)

            area_e = np.sum(ellipse_img>0)
            overlap = computeOverlap(regions[idx].image, ellipse_img, area_e)

            if aic < best['aic']:
                best['n'] = n
                best['areaE'] = area_e
                best['overlap'] = overlap
                best['ellipses'] = ellipses
                best['aic'] = aic
    
        eggCount += best['n']
        result.append(best)

  return eggCount, result 


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True,
                      help='Path of the segmentation model.')
    parser.add_argument('--input', required=True,
                      help='File path of the input image.')
    parser.add_argument('--output', default='/result',
                      help='Directory path to save the output images.')
  
    args = parser.parse_args()

    image = Image.open(args.input)
    input_image = image.resize((676,507))

    # Load the TFLite model and allocate tensors
    interpreter = tf.lite.Interpreter(args.model)
    interpreter.allocate_tensors()

    #Detect and count S. haematobium eggs
    t_start = perf_counter()
    segmented_image = segmentEggsTFL(input_image, interpreter)
    fov_count, fov_result = refinedSegmentation(segmented_image)
    t_stop = perf_counter()
    
    # Save result images
    filename = args.input.split('/')[-1].split('.')[0]
    print(filename)
    segmented_image.save(f"{args.output}/{filename}_seg.jpg")   
    ellipse_img_b = np.asarray(input_image)
    ellipse_img = np.zeros_like(ellipse_img_b, dtype=np.uint8)                                     
    for region in fov_result:
        ellipse_img, ellipse_img_b = createEllipseImage(ellipse_img, ellipse_img_b, region['ellipses'], offset=region['offset'], scale=2)                                           
    Image.fromarray(ellipse_img_b).save(f"{args.output}/{filename}_ell.jpg")

    #print result output
    print('Done. Results saved at', args.output)
    print(f"Egg count:       {fov_count}")
    print(f"Processing time: {t_stop-t_start:.2f}s")

if __name__ == '__main__':
    main()