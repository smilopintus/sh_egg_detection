# Reference
# Oyibo, P., Meulah, B., Bengston, M., van Lieshout, L., Oyibo, W., Diehl, J. C., Vdovin, G.,
# & Agbana, T., (2022). A Two-stage Automated Diagnosis Framework for Urogenital Schistosomiasis 
# in Microscopy Images from Low-resource Settings. Journal of Medical Imaging [Accepted].

r"""automated detection of Schistosoma haematobium eggs.

The command below runs this script and saves the following result images
1. Segmentation mask image showing the segmented eggs pixels 
2. Detection result image with the detected Schistosoma haematobium  
   eggs highlighted in elipses`

```
python sh_detection.py \
  --model models/deeplabv3_mobilenetv3_sh12k_tflite_model.tflite \
  --input images/img1.jpg \
  --output result
```
"""

import argparse
from PIL import Image
import math
from time import perf_counter
import numpy as np
import cv2
from skimage import measure, morphology
from sklearn import mixture
import tensorflow as tf

DATASET_MEAN = (0.59873112, 0.59390334, 0.64128324)
DATASET_STD = (0.02849652, 0.02736885, 0.02703068)
MEAN_EGG_AREA = 500
INPUT_IMAGE_SIZE = (676, 507)
SEGMENTED_IMAGE_SIZE = (338, 254)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0) 

def compute_eccentricity(axes_length):
  # estimates the eccentricity of an ellipse
  if axes_length[0] > axes_length[1]:
    a = axes_length[0] / 2 
    b = axes_length[1] / 2 
  else:
    a = axes_length[1] / 2 
    b = axes_length[0] / 2 

  return np.sqrt(a ** 2 - b ** 2) / a

def fit_ellipse(region, n_ellipse):
    # fits an optimal number of ellipse to a segmented reion
    if n_ellipse == 1:
        center_coord = (round((region.centroid[1] - region.bbox[1])),\
                         round(region.centroid[0] - region.bbox[0]))
        axes_length = (round(region.major_axis_length), round(region.minor_axis_length))
        angle = (180 * (-region.orientation) / np.pi) + 90
        eccentricity = compute_eccentricity(axes_length)
        area = math.pi * (axes_length[0] / 2) * (axes_length[1] / 2)
        ellipses = [{'center_coord': center_coord, 
                     'axes_length': axes_length, 
                     'angle': angle, 
                     'eccentricity': eccentricity, 
                     'area': area}]     
    else:   
        # cluster pixels into ellipses using Gaussian Mixture Model
        r, c = np.where(region.image > 0)
        x = np.array([c, r]).T
        gmm = mixture.GaussianMixture(n_components=n_ellipse).fit(x) 
        
        ellipses = []
        for n in range(n_ellipse):
            covariances = gmm.covariances_[n][:2, :2]
            v, w = np.linalg.eigh(covariances)
            u = w[0] / np.linalg.norm(w[0])
            angle = (180 * (-np.arctan2(u[0], u[1])) / np.pi) + 90  
            center_coord = (round(gmm.means_[n, 0]), round(gmm.means_[n, 1]))
            v = 3.92 * np.sqrt(v)
            axes_length = (round(v[0]), round(v[1]) )
            eccentricity = compute_eccentricity(axes_length)
            area = math.pi * (axes_length[0] / 2) * (axes_length[1] / 2)
            new_ellipse = {'center_coord': center_coord, 
                           'axes_length':axes_length, 
                           'angle': angle, 
                           'eccentricity': eccentricity, 
                           'area': area}
            ellipses.append(new_ellipse)

    return ellipses

def create_ellipse_image(img, img_b, ellipses, offset=None, scale=1):
    # generate an image with the detected eggs outlined by ellipses
    for ellipse in ellipses:      
        if offset is None:
            center_coord = tuple([scale*x for x in ellipse['center_coord']])   
        else:
            center_coord = (scale * (ellipse['center_coord'][0] + offset[0]) \
                            , scale * (ellipse['center_coord'][1] + offset[1]))
     
        axes_length = tuple([round(scale * x / 2) for x in ellipse['axes_length']])      
        angle = ellipse['angle']  
        start_angle = 0
        end_angle = 360
        thickness = -1  
        cv2.ellipse(img, center_coord, axes_length, angle, \
                            start_angle, end_angle, WHITE, thickness)
        cv2.ellipse(img_b, center_coord, axes_length, angle, \
                            start_angle, end_angle, GREEN, 0)
        
    if not offset:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 

    return img, img_b

def segment_eggs(input_img, interpreter):
    # segment eggs using a trained deep learning model
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_data = np.asarray(input_img) / 255
    input_data = ((input_data - DATASET_MEAN) / DATASET_STD).astype('float32')
    input_data = np.expand_dims(input_data, axis=0).transpose(0, 3, 1, 2)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    output_predictions = (output_data[0] > 0).squeeze().astype('uint8') * 255
    segmented_img = Image.fromarray(output_predictions)

    return segmented_img

def refined_segmentation(segmented_img):
  # refines the segmentation result of the deep learning model and counts the detected eggs
  segmented_img = segmented_img.resize((SEGMENTED_IMAGE_SIZE))
  result = []
  arr = morphology.remove_small_objects(np.asarray(segmented_img) > 0, 100)
  label_img = measure.label(arr)
  regions = measure.regionprops(label_img)
  eggCount = 0

  for idx in range(len(regions)):  
        #obtain the optimaal number of ellipses for a segmented region
        min_r, min_c, _, _ = regions[idx].bbox
        offset = (min_c, min_r)
        n_lower = math.ceil(0.65 * regions[idx].area / MEAN_EGG_AREA)
        n_upper = math.ceil(1.15 * regions[idx].area / MEAN_EGG_AREA)
        best = {'region_area': regions[idx].area, 
                'ellipse_area': None, 
                'n': None, 
                'overlap': None, 
                'ellipses': None, 
                'offset': offset, 
                'aic': float('INF')}
        
        for n in range(n_lower, n_upper+1):
            ellipses = fit_ellipse(regions[idx], n)
            k = np.sum([ ell['area'] > 1.5 * MEAN_EGG_AREA for ell in  ellipses]) 
            c = regions[idx].area / MEAN_EGG_AREA - k
            h, w = regions[idx].image.shape
            ellipse_img_b = np.zeros((h, w, 3), np.uint8)
            n_rows, n_cols = np.where(regions[idx].image>0)
            ellipse_img_b[n_rows, n_cols,:] = 255
            ellipse_img = np.zeros_like(ellipse_img_b, dtype=np.uint8) # creates a black image
            ellipse_img, ellipse_img_b = create_ellipse_image(ellipse_img, ellipse_img_b, ellipses)
            loss =  (1 - np.sum(np.logical_and(regions[idx].image, ellipse_img)) / regions[idx].area) + 1e-6
            aic = c * np.log(loss) + 2 * n
            area_e = np.sum(ellipse_img > 0)
            overlap = np.sum(np.logical_and(regions[idx].image, ellipse_img)) / area_e 

            if aic < best['aic']:
                best['n'] = n
                best['ellipse_area'] = area_e
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
    input_image = Image.open(args.input).resize(INPUT_IMAGE_SIZE)

    # Load the TFLite model and allocate tensors
    interpreter = tf.lite.Interpreter(args.model)
    interpreter.allocate_tensors()

    #Detect and count S. haematobium eggs
    t_start = perf_counter()
    segmented_image = segment_eggs(input_image, interpreter)
    fov_count, fov_result = refined_segmentation(segmented_image)
    t_stop = perf_counter()
    
    # Save result images
    filename = args.input.split('/')[-1].split('.')[0]
    segmented_image.save(f"{args.output}/{filename}_segmentation_result.jpg")   
    ellipse_img_b = np.asarray(input_image)
    ellipse_img = np.zeros_like(ellipse_img_b, dtype=np.uint8)    

    for region in fov_result:
        ellipse_img, ellipse_img_b = create_ellipse_image(ellipse_img, ellipse_img_b, region['ellipses'], \
                                                          offset=region['offset'], scale=2)                                           
    Image.fromarray(ellipse_img_b).save(f"{args.output}/{filename}_detection_result.jpg")

    #print result output
    print(f"Done. output images saved in ./{args.output}")
    print(f"Egg count:       {fov_count}")
    print(f"Processing time: {t_stop-t_start:.2f}s")

if __name__ == '__main__':
    main()