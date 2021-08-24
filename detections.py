from detectron2 import model_zoo
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.data import MetadataCatalog
import numpy as np
import os
import pandas as pd
import cv2
os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")
import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image
import csv 


# Return predicted bounding boxes and write the result if path_result != None
def predict_bounding_boxes(predictor, image):
    outputs = predictor(image)
    v = Visualizer(image[:, :, ::-1],
                scale=0.5, 
                metadata=MetadataCatalog.get("my_dataset_training").set(thing_classes=["vertèbre"]),
                instance_mode=ColorMode.IMAGE_BW  
                )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    img = out.get_image()[:, :, ::-1]
    return outputs['instances'].get_fields()['pred_boxes'].tensor.cpu().numpy(), out.get_image()[:, :, ::-1]

# Crop image accrording to the bounding boxes and create csv file
def generate_patch(image, bounding_boxes, h_padding, v_padding):
    patches = []
    i=0
    for x1,y1,x2,y2 in bounding_boxes:
        x1 = int(x1)-h_padding
        y1 = int(y1)-v_padding
        x2 = int(x2)+h_padding
        y2 = int(y2)+v_padding

        x1 = x1 if x1 > 0 else 0
        y1 = y1 if y1 > 0 else 0
        x2 = x2 if x2 < image.shape[1]  else image.shape[1]
        y2 = y2 if y2 < image.shape[0] else image.shape[0]

        cropped_image = image.copy()[y1:y2, x1:x2]
        i+=1
        patches.append(cropped_image)
    return patches

# Generate patches of all images
def object_detection(predictor, image):
    bounding_boxes, image_bbox = predict_bounding_boxes(predictor, image.copy())
    df = pd.DataFrame(bounding_boxes, columns=list('ABCD'))
    bounding_boxes = df.sort_values(['B', 'D'], ascending=[True, False]).values
    h_padding = 15
    v_padding = 4
    patches = generate_patch(image, bounding_boxes, h_padding, v_padding)
    return patches, bounding_boxes, image_bbox

# Return the predicted landmarks of the 17 vertebra
def predict_landmarks(patches, model):
  landmarks = []
  for p in patches:
    im = np.array(p)
    im = np.delete(im, [1, 2], axis=2)
    im = np.array(im) / 255.0
    im = np.expand_dims(im, axis=0)
    lmarks = model.predict(im)
    lmarks= lmarks[0]
    lmarks[0:8:2] = lmarks[0:8:2] * im.shape[2]
    lmarks[1:8:2] = lmarks[1:8:2] * im.shape[1]
    landmarks.append(lmarks)
  return landmarks

# Return the mapped landmarks 
def map_landmarks(landmarks, bounding_boxes):
  boxs = []
  H_padding = 15
  V_padding = 4
  mapped_landmarks = []

  for i, b in enumerate(bounding_boxes):
      p1 = [b[0], b[1]]
      keypoints = []
      for m in range(0,8,2):
        keypoints.append((int(landmarks[i][m]+p1[0]), int(landmarks[i][m+1]+p1[1])))
      mapped_landmarks.append([keypoints[0], keypoints[1], keypoints[2], keypoints[3]])
      # mapped_landmarks.append(keypoints)
  return mapped_landmarks

# Visualize image
def visualize(landmarks, landmarks2, patches, im):
  for i, l in enumerate(landmarks):
    p = patches[i]
    im = cv2.circle(im, l[0], 2, (255, 0, 0), 10)
    im = cv2.circle(im, l[1], 2, (255, 0, 0), 10)
    im = cv2.circle(im, l[2], 2, (0, 0, 255, 255), 10)
    im = cv2.circle(im, l[3], 2, (0, 0, 255), 10)

    keypoints = []
    for m in range(0,8,2):
        keypoints.append((int(landmarks2[i][m]), int(landmarks2[i][m+1])))
  return im

# Predict landmarks of a set of images and make a csv file and write predicted images
def landmark_detection(image, patches, bounding_boxes, model): 
  normalized_landmarks = []
  landmarks = predict_landmarks(patches, model)
  mapped_landmarks = map_landmarks(landmarks, bounding_boxes)
  im = visualize(mapped_landmarks, landmarks, patches, image.copy())

  return mapped_landmarks, im

# Calculate Cobb angles from the detected landmarks
def calculate_angles(landmarks, image, bounding_boxes, lower_MT, upper_MT):
    img = image.copy()
    vertebra_slopes = []
    for lm in landmarks:
        slope1 = (round(lm[1][1] - round(lm[0][1])))/(round(lm[1][0] - round(lm[0][0]))) 
        slope2 = (round(lm[3][1] - round(lm[2][1])))/(round(lm[3][0] - round(lm[2][0]))) 
        vertebra_slopes.append((slope1 + slope2)/2)


    cobb_angles= [0.0,0.0,0.0]
    if not isinstance(vertebra_slopes, np.ndarray):
        vertebra_slopes= np.array(vertebra_slopes)

    if (lower_MT is None) and (upper_MT is None) :
        max_slope = np.amax(vertebra_slopes)
        min_slope = np.amin(vertebra_slopes)

        lower_MT= np.argmax(vertebra_slopes)
        upper_MT = np.argmin(vertebra_slopes)
    else :
        max_slope = vertebra_slopes[lower_MT]
        min_slope = vertebra_slopes[upper_MT]

    if lower_MT < upper_MT:
        lower_MT, upper_MT = upper_MT, lower_MT


    try:
        upper_max_slope= np.amax(vertebra_slopes[0:upper_MT+1])
        upper_min_slope = np.amin(vertebra_slopes[0:upper_MT+1])
    except ValueError:
        upper_max_slope= vertebra_slopes[upper_MT]
        upper_min_slope = vertebra_slopes[upper_MT]


    try:
        lower_max_slope=np.amax(vertebra_slopes[lower_MT:len(vertebra_slopes)-1])
        lower_min_slope=np.amin(vertebra_slopes[lower_MT:len(vertebra_slopes)-1])
    except ValueError:
        lower_max_slope=vertebra_slopes[lower_MT]
        lower_min_slope=vertebra_slopes[lower_MT]

    cobb_angles[0]= abs(np.rad2deg(np.arctan(max_slope))- np.rad2deg(np.arctan(min_slope)))
    cobb_angles[1]= abs(np.rad2deg(np.arctan(upper_max_slope))-np.rad2deg(np.arctan(upper_min_slope)))
    cobb_angles[2]= abs(np.rad2deg(np.arctan(lower_max_slope)) - np.rad2deg(np.arctan(lower_min_slope)))
    
    
    overlay = img.copy()
    cv2.rectangle(overlay, (round(bounding_boxes[upper_MT][0]), round(bounding_boxes[upper_MT][1])), (round(bounding_boxes[upper_MT][2]), round(bounding_boxes[upper_MT][3])), (0, 255, 0), -1)
    cv2.rectangle(overlay, (round(bounding_boxes[lower_MT][0]), round(bounding_boxes[lower_MT][1])), (round(bounding_boxes[lower_MT][2]), round(bounding_boxes[lower_MT][3])), (255, 0, 0), -1)
    def get_line(P, Q):
        line = {}
        line["a"] = P[1] - Q[1]
        line["b"] = Q[0] - P[0]
        line["c"] = line["a"]*(P[0]) + line["b"]*(P[1])

        return line

    def get_x(line, y):
        return ((-1)*line["b"]*y + line["c"])/line["a"]

    def get_y(line, x):
        return ((-1)*line["a"]*x + line["c"])/line["b"]



    upper_MT_x1 = (landmarks[upper_MT][0][0] +landmarks[upper_MT][3][0])/2
    upper_MT_x2 = (landmarks[upper_MT][1][0] +landmarks[upper_MT][2][0])/2
    upper_MT_y1 = (landmarks[upper_MT][0][1] +landmarks[upper_MT][3][1])/2
    upper_MT_y2 = (landmarks[upper_MT][1][1] +landmarks[upper_MT][2][1])/2

    lower_MT_x1 = (landmarks[lower_MT][0][0] +landmarks[lower_MT][3][0])/2
    lower_MT_x2 = (landmarks[lower_MT][1][0] +landmarks[lower_MT][2][0])/2
    lower_MT_y1 = (landmarks[lower_MT][0][1] +landmarks[lower_MT][3][1])/2
    lower_MT_y2 = (landmarks[lower_MT][1][1] +landmarks[lower_MT][2][1])/2

    upper_line = get_line((upper_MT_x1, upper_MT_y1), (upper_MT_x2, upper_MT_y2))

    upper_P = (0, round(get_y(upper_line, 0)))
    upper_Q = (img.shape[1], round(get_y(upper_line, img.shape[1])))

    lower_line = get_line((lower_MT_x1, lower_MT_y1), (lower_MT_x2, lower_MT_y2))

    lower_P = (0, round(get_y(lower_line, 0)))
    lower_Q = (img.shape[1], round(get_y(lower_line, img.shape[1])))


    im = cv2.line(img, upper_P, upper_Q, (213,1,2), 5)
    im = cv2.line(img, lower_P, lower_Q, (213,1,2), 5)

    upper_slope = (round(upper_Q[1] - round(upper_P[1])))/(round(upper_Q[0] - round(upper_P[0]))) 
    lower_slope = (round(lower_Q[1] - round(lower_P[1])))/(round(lower_Q[0] - round(lower_P[0]))) 

    if upper_P[1] < upper_Q[1]:
        upper_perp_P = ((upper_MT_x2 + upper_Q[0])/2, (upper_MT_y2 + upper_Q[1])/2)
        y = lower_Q[1]
        upper_perp_Q = (upper_perp_P[0] - upper_slope * (y - upper_perp_P[1]), y)
        upper_perp_P = (round(upper_perp_P[0]), round(upper_perp_P[1]))
        upper_perp_Q = (round(upper_perp_Q[0]), round(upper_perp_Q[1]))

        lower_perp_P = ((lower_MT_x2 + lower_Q[0])/2, (lower_MT_y2 + lower_Q[1])/2)
        y = upper_Q[1]
        lower_perp_Q = (lower_perp_P[0] - lower_slope * (y - lower_perp_P[1]), y)
        lower_perp_P = (round(lower_perp_P[0]), round(lower_perp_P[1]))
        lower_perp_Q = (round(lower_perp_Q[0]), round(lower_perp_Q[1]))

        im = cv2.line(img, upper_perp_P, upper_perp_Q, (0,0,255), 5)
        im = cv2.line(img, lower_perp_P, lower_perp_Q, (0,0,255), 5)

    elif upper_P[1] > upper_Q[1]:
        upper_perp_P = ((upper_MT_x1 + upper_P[0])/2, (upper_MT_y1 + upper_P[1])/2)
        y = lower_P[1]
        upper_perp_Q = (upper_perp_P[0] - upper_slope * (y - upper_perp_P[1]), y)
        upper_perp_P = (round(upper_perp_P[0]), round(upper_perp_P[1]))
        upper_perp_Q = (round(upper_perp_Q[0]), round(upper_perp_Q[1]))

        lower_perp_P = ((lower_MT_x1 + lower_P[0])/2, (lower_MT_y1 + lower_P[1])/2)
        y = upper_P[1]
        lower_perp_Q = (lower_perp_P[0] - lower_slope * (y - lower_perp_P[1]), y)
        lower_perp_P = (round(lower_perp_P[0]), round(lower_perp_P[1]))
        lower_perp_Q = (round(lower_perp_Q[0]), round(lower_perp_Q[1]))

        im = cv2.line(img, lower_perp_P, lower_perp_Q, (0,0,255), 5)
        im = cv2.line(img, upper_perp_P, upper_perp_Q, (0,0,255), 5)

    img_new = cv2.addWeighted(overlay, 0.2, img, 0.8, 0)
    return cobb_angles, upper_MT, lower_MT, img_new