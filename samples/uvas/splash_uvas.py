"""
Mask R-CNN
Train on the toy Grape dataset and implement color splash effect.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 grape.py train --dataset=/path/to/grape/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 grape.py train --dataset=/path/to/grape/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 grape.py train --dataset=/path/to/grape/dataset --weights=imagenet

    # Apply color splash to an image
    python3 grape.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Apply color splash to video using the last weights you trained
    python3 grape.py splash --weights=last --video=<URL or path to file>
"""

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
from os import getenv
import pymssql
from IPython.core.display import display, HTML

from deep_sort_pytorch.deep_sort import build_tracker
from deep_sort_pytorch.utils.draw import draw_boxes
from deep_sort_pytorch.utils.parser import get_config


# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
import mrcnn.visualize
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
# SQL server
############################################################


def insert_list(conn, lista):
    cursor = conn.cursor()
    cursor.executemany("INSERT INTO deteccion VALUES (%d, %d, %d, %d, %d, %d, %s, %s, %s, %s)", lista)
    conn.commit()

def read(conn):
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM deteccion')
    row = cursor.fetchone()

    while row:
        print("hilera=%s, racimo=%s, area=%s, lng=%s, lat=%s, lng=%s" % (row[4],row[5],row[6],row[7],row[8],row[9]))
        row = cursor.fetchone()

############################################################
#  Configurations
############################################################


class GrapeConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "uvas"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 2  # Background + (grape + trunk)

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 75% confidence
    DETECTION_MIN_CONFIDENCE = 0.75


############################################################
#  Dataset
############################################################

class GrapeDataset(utils.Dataset):

    def load_grape(self, dataset_dir, subset):
        """Load a subset of the Grape dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("object", 1, "uvas")
        self.add_class("object", 2, "tronco")

        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        annotations = json.load(open(os.path.join(dataset_dir, "via_region_data.json")))
        annotations = list(annotations.values())  # don't need the dict keys

        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]

        # Add images
        for a in annotations:
            # Get the x, y coordinates of points of the polygons that make up
            # the outline of each object instance. There are stores in the
            # shape_attributes (see json format above)
            # for b in a['regions'].values():
            #    polygons = [{**b['shape_attributes'], **b['region_attributes']}]
            # print("string=", polygons)
            # for r in a['regions'].values():
            #    polygons = [r['shape_attributes']]
            #    # print("polygons=", polygons)
            #    multi_numbers = [r['region_attributes']]
            # print("multi_numbers=", multi_numbers)
            if type(a["regions"]) is dict:

                polygons = [r['shape_attributes'] for r in a['regions'].values()]
                objects = [s['region_attributes'] for s in a['regions'].values()]
            else:
                polygons = [r['shape_attributes'] for r in a['regions']]
                objects = [s['region_attributes'] for s in a['regions']]

            # print("multi_numbers=", multi_numbers)
            # num_ids = [n for n in multi_numbers['number'].values()]
            # for n in multi_numbers:
            num_ids = [int(n['object']) for n in objects]
            # print("num_ids=", num_ids)
            # print("num_ids_new=", num_ids_new)
            # categories = [s['region_attributes'] for s in a['regions'].values()]
            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "object",
                image_id=a['filename'], # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons,
                num_ids=num_ids)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a grape dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "uvas":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "uvas":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

def color_splash(image, mask_red, mask_blue):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.


    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255


    # Copy color pixels from the original color image where mask is set
    red = image.copy()
    red[:, :, 1] = 0
    red[:, :, 2] = 0

    blue = image.copy()
    blue[:, :, 0] = 0
    blue[:, :, 1] = 0

    if mask_red.shape[-1] > 0 or mask_blue.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask_red = (np.sum(mask_red, -1, keepdims=True) >= 1)
        splash_red = np.where(mask_red, red, image).astype(np.uint8)
        mask_blue = (np.sum(mask_blue, -1, keepdims=True) >= 1)
        splash = np.where(mask_blue, blue, splash_red).astype(np.uint8)
    else:
        splash = image
    return splash


def detect_and_color_splash(model, image_path=None, video_path=None, connection=None):
    assert image_path or video_path

    # Image or video?
    if video_path:
        import cv2
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = 1280    #int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = 720    #int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))

        db_output = []

        totalFrames = int(vcapture.get(cv2.CAP_PROP_FRAME_COUNT))
        start_frame  = 500
        end_frame = 600
        for frameCount in range(totalFrames):
            success, image = vcapture.read()
            if frameCount < start_frame:
                continue
            if frameCount > end_frame: break
            if success == False: continue
            image = cv2.resize(image, (width, height))
            print("frame: ", frameCount)
            # Read next image

            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]

                # Detect objects
                r = model.detect([image], verbose=0)

                # Color splash
                splash = color_splash(image, r['red_masks'], r['blue_masks'])

                detections_blue = r['blue_rois']
                detections_red = r["red_rois"]
                if(detections_blue is not [] and len(detections_blue)!=0):
                    bbox_xywh_blue = detections_blue[:, :4].copy()
                    bbox_xywh_blue[:, 0] = (detections_blue[:, 1] + detections_blue[:, 3])/2
                    bbox_xywh_blue[:, 1] = (detections_blue[:, 0] + detections_blue[:, 2])/2
                    bbox_xywh_blue[:, 2] = (detections_blue[:, 3] - detections_blue[:, 1])
                    bbox_xywh_blue[:, 3] = (detections_blue[:, 2] - detections_blue[:, 0])
                    cls_conf_blue = r['blue_scores']
                    for i in range(len(detections_blue)):
                        cv2.putText(splash, str(r['blue_scores'][i]), (int(bbox_xywh_blue[i][0]), int(bbox_xywh_blue[i][1])), cv2.FONT_HERSHEY_PLAIN, 1, [255, 255, 255], 1)
                    outputs_blue = deepsort_blue.update(bbox_xywh_blue, cls_conf_blue, splash)
                    if len(outputs_blue) > 0:
                        bbox_xyxy_blue = outputs_blue[:, :4]
                        identities_blue = outputs_blue[:, -1]
                        splash = draw_boxes(splash, bbox_xyxy_blue, identities_blue, class_label="Tronco")

                if(detections_red is not [] and len(detections_red)!=0):
                    bbox_xywh_red = detections_red[:, :4].copy()
                    bbox_xywh_red[:, 0] = (detections_red[:, 1] + detections_red[:, 3])/2
                    bbox_xywh_red[:, 1] = (detections_red[:, 0] + detections_red[:, 2])/2
                    bbox_xywh_red[:, 2] = (detections_red[:, 3] - detections_red[:, 1])
                    bbox_xywh_red[:, 3] = (detections_red[:, 2] - detections_red[:, 0])
                    cls_conf_red = r['red_scores']
                    for i in range(len(detections_red)):
                        cv2.putText(splash, str(r['red_scores'][i]), (int(bbox_xywh_red[i][0]), int(bbox_xywh_red[i][1])), cv2.FONT_HERSHEY_PLAIN, 1, [255, 255, 255], 1)
                    outputs_red = deepsort_red.update(bbox_xywh_red, cls_conf_red, splash)
                    if len(outputs_red) > 0:
                        bbox_xyxy_red = outputs_red[:, :4]
                    #insert_list    identities_red = outputs_red[:, -1]
                        identities_red = outputs_red[:, -1]
                        splash = draw_boxes(splash, bbox_xyxy_red, identities_red, class_label="Uva")

                        
                        # Get and send database info
                        for identity in identities_red:
                            area = deepsort_red.tracker.get_area_by_id(identity)
                            if area >= 0:
                                #print(str(deepsort.tracker.get_area_by_id(identity)))

                                db_output.append((1, 1, 1, 1, int(identity), int(area), "71.32579919", "35.09985769", "carmenere", datetime.datetime.now()))

                # Show rectangle with grape count
                overlay = splash.copy()
                alpha = 0.6
                label_red = "Conteo de racimos: {}".format(str(deepsort_red.get_total_confirmed()))
                label_blue = "Conteo de troncos: {}".format(str(deepsort_blue.get_total_confirmed()))
                t_size_red = cv2.getTextSize(label_red, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                t_size_blue = cv2.getTextSize(label_blue, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                cv2.rectangle(overlay, (0, 0), (0 + max(t_size_blue[0], t_size_red[0]) + 6, 0 + t_size_blue[1] + 48), [255, 255, 255], -1)
                cv2.putText(overlay, label_red, (0, 0 + t_size_red[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 1, [0, 0, 0], 2)
                cv2.putText(overlay, label_blue, (0, 0 + t_size_blue[1] + 44), cv2.FONT_HERSHEY_SIMPLEX, 1, [0, 0, 0], 2)
                splash = cv2.addWeighted(overlay, alpha, splash, 1 - alpha, 0)

                # RGB -> BGR to save image to video
                splash = splash[..., ::-1]
                # Add image to video writer
                vwriter.write(splash)
                cv2.imshow("img", splash)


                if (cv2.waitKey(1) & 0xFF) == ord('q'): # Hit `q` to exit
                    break

        vwriter.release()
        #insert_list(connection, db_output)
        print("Saved to ", file_name)


############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect grapes.')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    args = parser.parse_args()


    cfg = get_config()
    cfg.merge_from_file("./deep_sort_pytorch/configs/yolov3.yaml")
    cfg.merge_from_file("./deep_sort_pytorch/configs/deep_sort.yaml")
    deepsort_blue = build_tracker(cfg, use_cuda=1)
    deepsort_red = build_tracker(cfg, use_cuda=1)


    # Validate arguments
    assert args.image or args.video, "Provide --image or --video to apply color splash"

    print("Weights: ", args.weights)
    print("Logs: ", args.logs)

    # Configurations
    class InferenceConfig(GrapeConfig):
        # Set batch size to 1 since we'll be running inference on
        # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
        GPU_COUNT = 1
        # this number was originally 1
        IMAGES_PER_GPU = 1
    config = InferenceConfig()
    config.display()

    # Create model
    model = modellib.MaskRCNN(mode="inference", config=config, model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)



    # Load database

    server = "54.233.255.233"
    user = "SA"
    password = "digevo"

    connection = pymssql.connect(server, user, password, "AustralFalconUvas")

    # Train or evaluate
    print("Saved model to disk")
    detect_and_color_splash(model, image_path=args.image, video_path=args.video, connection=connection)

    #read(connection)
    connection.close()
