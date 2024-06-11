#!/usr/bin/env python
# coding:utf-8

import os
import sys
import time

import cv2
import numpy as np
import onnxruntime
from tqdm import tqdm


class HumanDetectionResult(object):
    def __init__(self):
        self.left_top_x: int = None
        self.left_top_y: int = None
        self.right_bottom_x: int = None
        self.right_bottom_y: int = None
        self.center_x: int = None
        self.center_y: int = None
        self.width: int = -1
        self.height: int = -1
        self.prob: float = -1.0
        self.objectness: float = -1.0
        self.class_prob: float = -1.0
        self.category_type: int = -1


class HumanDetector(object):
    def __init__(self, args):
        self.model_path = args.model_path
        self.input_blob_name = args.input_blob_name
        self.input_type = args.input_type
        self.image_list_file = args.image_list_file
        self.movie_path = args.movie_path
        self.detect_threshold = args.detect_threshold
        self.iou_threshold = args.iou_threshold
        self.min_image_size = args.min_image_size
        self.max_image_size = args.max_image_size
        self.scale_list = args.scale_list
        self.num_classes = args.num_classes
        self.category_target_dict = args.category_target_dict
        self.category_colors = args.category_colors

        self.image_list = []
        self.cap = None
        self.frame_index = 0
        self.frame_length = 0
        self.inv_scales = []
        self.model = None

    def set_image_list(self):
        target_ext = (".jpg", ".jpeg", ".png", ".bmp")
        with open(self.image_list_file, "r", encoding="utf-8") as f:
            for line in f.readlines():
                image_path = line.strip()
                ext = os.path.splitext(os.path.basename(image_path))[1]
                if not ext.lower().endswith(target_ext):
                    print("Image file type is not supported")
                    sys.exit(1)
                self.image_list.append(image_path)
        self.frame_length = len(self.image_list)

    def set_movie(self):
        target_ext = ".mp4"
        ext = os.path.splitext(os.path.basename(self.movie_path))[1]
        if not ext.lower().endswith(target_ext):
            print("Movie file type is not supported")
            sys.exit(1)
        self.cap = cv2.VideoCapture(self.movie_path)
        if not self.cap.isOpened():
            print("Movie file cannot be opened")
            sys.exit(1)
        self.frame_length = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def load_image(self):
        try:
            image_path = self.image_list[self.frame_index]
            image = cv2.imread(image_path)
            if image is None:
                raise cv2.error(f"Failed to read image: {image_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image
        except IndexError:
            raise IndexError(f"Frame index {self.frame_index} is out of range.")

    def load_movie(self):
        ret, image = self.cap.read()
        if not ret:
            raise cv2.error("Failed to read frame from the video.")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def get_image_data(self):
        try:
            if self.input_type == "image":
                image = self.load_image()
            elif self.input_type == "movie":
                image = self.load_movie()
            else:
                raise ValueError(f"Invalid input type: {self.input_type}")
            self.frame_index += 1
            return image
        except (IndexError, cv2.error) as e:
            print(f"Error: {str(e)}")
            return None

    def determine_model_input_shape(self, src_image_width, src_image_height):
        # Calculate the remainder when dividing the height by a specific multiple
        height_remainder = src_image_height % 32

        # Calculate the remainder when dividing the width by a specific multiple
        width_remainder = src_image_width % 32

        # If there is a remainder in the height direction
        if height_remainder != 0:
            # Calculate the quotient when dividing by the multiple
            height_quotient = src_image_height // 32

            # Calculate the nearest multiple to the height value
            dst_image_height = 32 * (height_quotient + 1)
        else:
            dst_image_height = src_image_height

        # If there is a remainder in the width direction
        if width_remainder != 0:
            # Calculate the quotient when dividing by the multiple
            width_quotient = src_image_width // 32

            # Calculate the nearest multiple to the width value
            dst_image_width = 32 * (width_quotient + 1)
        else:
            dst_image_width = src_image_width

        return dst_image_width, dst_image_height

    def preprocess_image(self, src_image):
        # Variable to store resized images
        resize_image_list = []

        for scale in self.scale_list:
            self.inv_scales.append(1 / scale)

            # Get the image size after applying the scale value
            scale_image_height = int(src_image.shape[0] * scale)
            scale_image_width = int(src_image.shape[1] * scale)

            # Calculate the image size that can be input to the model
            tmp_image_width, tmp_image_height = self.determine_model_input_shape(scale_image_width, scale_image_height)

            # Create empty data (black image) of the input image size
            resize_image = np.zeros([tmp_image_height, tmp_image_width, 3], np.float32)

            # Compare the resized image size with the original image size, select the image interpolation algorithm, and resize the image
            if (tmp_image_height * tmp_image_width) > (scale_image_width * scale_image_height):
                tmp_image = cv2.resize(
                    src_image,
                    dsize=(scale_image_width, scale_image_height),
                    interpolation=cv2.INTER_LINEAR,
                )
                # Paste the resized image onto the pre-prepared dst_image_data
                resize_image[0:scale_image_height, 0:scale_image_width, :] = tmp_image
            else:
                resize_image = src_image.astype(np.float32)

            # Add the resized image
            resize_image_list.append(resize_image)

        # Variable to store preprocessed images
        dst_image_list = []

        # Loop through the number of resized images
        for resize_image in resize_image_list:
            # Normalization
            normalize_image = resize_image / 255

            # Convert shape to channel-first
            # (height, width, channels) -> (channels, height, width)
            reshape_image = np.transpose(normalize_image, (2, 0, 1))

            # Add the batch size dimension
            # (channels, height, width) -> (batch_size, channels, height, width)
            dst_image = np.expand_dims(reshape_image, 0)

            # Add the preprocessed image
            dst_image_list.append(dst_image)

        return dst_image_list

    def sigmoid(self, x):
        # Consider overflow
        mask = x >= 0
        y = np.zeros_like(x)  # Create an array initialized with 0s having the same shape as x
        y[mask] = 1.0 / (1 + np.exp(-x[mask]))  # x >= 0, f(x) = 1/(1+e^(-x))
        y[~mask] = np.exp(x[~mask]) / (1 + np.exp(x[~mask]))  # x < 0, f(x) = e^x/(e^x+1)
        return y

    def decode_outputs(self, outputs):
        # List to store pairs of height and width of feature maps after network output
        # Example) For an input image of 480×640, the sizes of the feature maps would be:
        # hw_list = [(60, 80), (30, 40), (15, 20)]
        hw_list = []

        # Flatten the feature maps of network outputs
        reshaped_outputs = []
        for output in outputs:
            # Store pairs of height and width of feature maps after network output
            h, w = output.shape[-2:]
            hw_list.append((h, w))

            # Flatten the feature maps of network output
            # output => (batch_size, 4 + 1 + num_classes, h, w)
            # reshaped_output => (4 + 1 + num_classes, h*w)
            reshaped_output = output[0].reshape(output.shape[1], -1)
            reshaped_outputs.append(reshaped_output)

        # Concatenate the flattened feature maps of network outputs
        # Example) For an input image of 480×640 (4 dimensions: bbox=(tx, ty, tw, th), 1 dimension: object confidence)
        # reshaped_output1 => (4 + 1 + num_classes, 60*80) => (4 + 1 + num_classes, 4800)
        # reshaped_output2 => (4 + 1 + num_classes, 30*40) => (4 + 1 + num_classes, 1200)
        # reshaped_output3 => (4 + 1 + num_classes, 15*20) => (4 + 1 + num_classes, 300)
        # concatenated_outputs => reshaped_output1 + reshaped_output2 + reshaped_output3 => (4 + 1 + num_classes, 6300)
        concatenated_outputs = np.concatenate(reshaped_outputs, axis=1)

        # Transpose dimensions
        # Example) For an input image of 480×640
        # transposed_outputs => (4 + 1 + num_classes, 6300) => (6300, 4 + 1 + num_classes)
        transposed_outputs = concatenated_outputs.T

        # Variable to store decoded results
        decoded_data = np.zeros_like(transposed_outputs)

        # Scale confidence and class probabilities through the sigmoid function to the range [0, 1]
        decoded_data[:, 4:] = self.sigmoid(transposed_outputs[:, 4:])

        grids = []
        strides = []

        for (h, w), stride in zip(hw_list, [8, 16, 32]):
            # Create grid coordinates based on the height and width of the feature map
            # Example) For a feature map of 3×2 (h=2, w=3)
            #                                                        x, y
            # grid_x = [[0, 1, 2]   grid_y = [[0, 0, 0]    grid = [[0, 0],
            #           [0, 1, 2]]            [1, 1, 1]]           [1, 0],
            #                                                      [2, 0],
            #                                                      [0, 1],
            #                                                      [1, 1],
            #                                                      [2, 1]]
            grid_y, grid_x = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
            grid = np.stack((grid_x, grid_y), axis=2).reshape(-1, 2)
            grids.append(grid)

            # Create an array (h*w, 1) with all elements set to the stride value based on the grid coordinates
            # The stride represents how much the feature map is downsampled relative to the original input image
            # Example) For an input image of 480×640
            # Feature map 1: height: 480 /  8 = 60, width: 640 /  8 = 80, stride = 8
            # Feature map 2: height: 480 / 16 = 30, width: 640 / 16 = 40, stride = 16
            # Feature map 3: height: 480 / 32 = 15, width: 640 / 32 = 20, stride = 32
            strides.append(np.full((grid.shape[0], 1), stride))

        # Concatenate the grid coordinates corresponding to feature maps of different sizes
        # Example) For an input image of 480×640
        # grid1 => (4800, 2)
        # grid2 => (1200, 2)
        # grid3 => (300, 2)
        # grid  => grid1 + grid2 + grid3 => (6300, 2)
        grids = np.concatenate(grids, axis=0).astype(decoded_data.dtype)

        # Concatenate the downsampling factors (strides) corresponding to feature maps of different sizes
        # Example) For an input image of 480×640
        # stride1 => (4800, 1)
        # stride2 => (1200, 1)
        # stride3 => (300, 1)
        # strides => stride1 + stride2 + stride3 => (6300, 1)
        strides = np.concatenate(strides, axis=0).astype(decoded_data.dtype)

        # Calculate the center of the bounding box (cx, cy)
        # Convert to the image coordinate system by adding the output (tx, ty) to the top-left coordinates of the grid (gx, gy) and multiplying by strides
        decoded_data[:, :2] = (transposed_outputs[:, :2] + grids) * strides

        # Calculate the size of the bounding box (bw, bh)
        # Convert to the image coordinate system by calculating (e^tw, e^th) from the output (tw, th) and multiplying by strides
        decoded_data[:, 2:4] = np.exp(transposed_outputs[:, 2:4]) * strides

        return decoded_data

    def get_detect_results(self, decode_data_list):
        # ----------------------------------------------------------#
        # Variable to store all detection results (0, 12)
        # [cx, cy, width, height, ltx, lty, rbx, rby, objectness, class_prob, prob, class_pred]
        detections = np.zeros((0, 12))
        # ----------------------------------------------------------#

        for scale_index, decode_data in enumerate(decode_data_list):
            # ----------------------------------------------------------#
            # decode_data => [total number of grids in all feature maps, 4 + 1 + num_classes]
            # ----------------------------------------------------------#

            # Create an array to store detection bounding box information
            # [cx, cy, w, h, ltx, lty, rbx, rby]
            bbox_info = np.zeros((decode_data.shape[0], 8))
            bbox_info[:, 0] = decode_data[:, 0]  # cx
            bbox_info[:, 1] = decode_data[:, 1]  # cy
            bbox_info[:, 2] = decode_data[:, 2]  # width
            bbox_info[:, 3] = decode_data[:, 3]  # height
            bbox_info[:, 4] = bbox_info[:, 0] - bbox_info[:, 2] / 2  # ltx
            bbox_info[:, 5] = bbox_info[:, 1] - bbox_info[:, 3] / 2  # lty
            bbox_info[:, 6] = bbox_info[:, 0] + bbox_info[:, 2] / 2  # rbx
            bbox_info[:, 7] = bbox_info[:, 1] + bbox_info[:, 3] / 2  # rby

            # ----------------------------------------------------------#
            # Store the probability of the class with the highest predicted probability for each anchor
            # class_prob => [total number of grids in all feature maps, 1]
            # ----------------------------------------------------------#
            class_prob = np.max(decode_data[:, 5 : 5 + self.num_classes], axis=1, keepdims=True)

            # ----------------------------------------------------------#
            # Store the index of the class corresponding to class_prob
            # class_pred => [total number of grids in all feature maps, 1]
            # ----------------------------------------------------------#
            class_pred = np.argmax(decode_data[:, 5 : 5 + self.num_classes], axis=1, keepdims=True)
            class_pred = class_pred.astype(np.float32)

            # ----------------------------------------------------------#
            # Store the objectness confidence
            # objectness => [total number of grids in all feature maps, 1]
            # ----------------------------------------------------------#
            objectness = decode_data[:, 4, np.newaxis]

            # ----------------------------------------------------------#
            # Create a mask indicating whether the product of objectness confidence and class prediction probability is above the detection threshold (True: above threshold, False: below threshold)
            # prob      => [total number of grids in all feature maps, 1]
            # prob_mask => [total number of grids in all feature maps, ]
            # ----------------------------------------------------------#
            prob = objectness * class_prob
            prob_mask = prob[:, 0] >= self.detect_threshold

            # -------------------------------------------------------------------------#
            # Store 12 pieces of information in detect_data
            # detect_data => [total number of grids in all feature maps, 12]
            # [cx, cy, width, height, ltx, lty, rbx, rby, objectness, class_prob, prob, class_pred]
            # -------------------------------------------------------------------------#
            detect_data = np.concatenate((bbox_info, objectness, class_prob, prob, class_pred), axis=1)

            # Apply the mask
            detect_data = detect_data[prob_mask]

            # Create an array to store the reciprocal of the scale value
            inv_scale = np.full((detect_data.shape[0], 1), self.inv_scales[scale_index])

            # -------------------------------------------------------------------------#
            # Adjust coordinates to match the size of the original image
            # -------------------------------------------------------------------------#
            # Adjust cx, cy, width, height
            detect_data[:, 0:4] = detect_data[:, 0:4] * inv_scale
            # Calculate ltx, lty
            detect_data[:, 4:6] = detect_data[:, 0:2] - (detect_data[:, 2:4] / 2.0)
            # Calculate rbx, rby
            detect_data[:, 6:8] = detect_data[:, 0:2] + (detect_data[:, 2:4] / 2.0)

            # Add the detection result data
            detections = np.concatenate([detections, detect_data], axis=0)

        # NMS algorithm
        nms_out_index = self.non_maximum_supression(detections[:, 4:8], detections[:, 10])

        # If NMS indices exist
        if len(detections) != 0:
            # Filter using the indices obtained from NMS processing
            detections = detections[nms_out_index]

        # List to store detection results
        results = []

        # Store detection results
        for result in detections:
            # Create an instance of the class to store detection results
            human_detection_result = HumanDetectionResult()

            # Store each result
            human_detection_result.center_x = int(result[0])
            human_detection_result.center_y = int(result[1])
            human_detection_result.width = int(result[2])
            human_detection_result.height = int(result[3])
            human_detection_result.left_top_x = int(result[4])
            human_detection_result.left_top_y = int(result[5])
            human_detection_result.right_bottom_x = int(result[6])
            human_detection_result.right_bottom_y = int(result[7])
            human_detection_result.objectness = float(result[8])
            human_detection_result.class_prob = float(result[9])
            human_detection_result.prob = float(result[10])
            human_detection_result.category_type = int(result[11])

            results.append(human_detection_result)

        return results

    def non_maximum_supression(self, boxes, scores):
        # Get the indices of the detection bounding boxes when sorted in ascending order of scores
        idxs = scores.argsort()

        # Variable to store the detection bounding boxes that will be kept after NMS processing
        keep = []

        while idxs.size > 0:
            # Store the index of the detection bounding box with the highest score
            max_score_index = idxs[-1]

            # Store the coordinates of the detection bounding box with the highest score
            # max_score_box => (1, 4)
            max_score_box = boxes[max_score_index][None, :]

            # Append the index of the selected detection bounding box to keep
            keep.append(max_score_index)

            if idxs.size == 1:
                break

            # Remove the index of the selected detection bounding box
            idxs = idxs[:-1]

            # Store the coordinates of the remaining detection bounding boxes that are not selected
            other_boxes = boxes[idxs]

            # Calculate the IoU between the selected detection bounding box and the remaining detection bounding boxes
            ious = self.calc_box_iou(max_score_box, other_boxes)

            # Keep only the detection bounding boxes with IoU below the threshold in idxs
            idxs = idxs[ious < self.iou_threshold]

        return np.array(keep)

    def calc_box_iou(self, box1, box2):
        # Calculate the area of the detection bounding boxes from the coordinates
        area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
        area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

        # Calculate the top-left coordinates of the intersection region of the two detection bounding boxes
        intersection_lt = np.maximum(box1[:, :2], box2[:, :2])

        # Calculate the bottom-right coordinates of the intersection region of the two detection bounding boxes
        intersection_rb = np.minimum(box1[:, 2:], box2[:, 2:])

        # Calculate the width and height of the intersection region of the two detection bounding boxes
        intersection_wh = intersection_rb - intersection_lt

        # Convert negative values to 0
        intersection_wh = np.maximum(0, intersection_wh)

        # Calculate the area of the intersection region
        intersection_area = intersection_wh[:, 0] * intersection_wh[:, 1]

        # Calculate the total area
        union = area1 + area2 - intersection_area

        # Calculate IoU
        iou = intersection_area / union

        return iou

    def plot_one_box(self, pt1, pt2, src_image, category, prob, color):
        # Plots one bounding box on image img
        tl = round(0.001 * max(src_image.shape[0:2])) + 1  # line thickness
        cv2.rectangle(img=src_image, pt1=pt1, pt2=pt2, color=color, thickness=tl)

        tf = max(tl - 2, 1)  # font thickness
        text = category + " " + f"{prob:.2f}"
        t_size, _ = cv2.getTextSize(text=text, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=tl / 3, thickness=tf)
        cv2.rectangle(
            img=src_image,
            pt1=pt1,
            pt2=(pt1[0] + t_size[0], pt1[1] - t_size[1] - 3),
            color=color,
            thickness=-1,
        )
        cv2.putText(
            img=src_image,
            text=text,
            org=(pt1[0], pt1[1] - 2),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=tl / 3,
            color=[0, 0, 0],
            thickness=tf,
            lineType=cv2.LINE_AA,
        )

    def is_valid_image_size(self, image):
        height, width, channels = image.shape
        return (
            width >= self.min_image_size[1]
            and width <= self.max_image_size[1]
            and height >= self.min_image_size[0]
            and height <= self.max_image_size[0]
            and channels == 3
        )

    def detect_image(self, image):
        # Check input image size
        if not self.is_valid_image_size(image):
            raise ValueError("Input image size is not supported.")

        # Preprocess the input image
        input_image_list = self.preprocess_image(image)

        decode_data_list = []
        for input_image in input_image_list:
            # Predict using the model
            outputs = self.model.run(None, {self.input_blob_name: input_image})

            # Decode model outputs
            decode_data = self.decode_outputs(outputs)
            decode_data_list.append(decode_data)

        # Get detection results
        results = self.get_detect_results(decode_data_list)

        return results

    def run(self):
        if self.input_type == "image":
            self.set_image_list()

            if not os.path.exists("exp/detect/images"):
                os.makedirs("exp/detect/images")

        if self.input_type == "movie":
            self.set_movie()

            if not os.path.exists("exp/detect/movie"):
                os.makedirs("exp/detect/movie")

            file_name = os.path.splitext(os.path.basename(self.movie_path))[0]
            output_movie_path = os.path.join("exp/detect/movie", file_name + "_detect.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(output_movie_path, fourcc, fps, (width, height))

        # load model
        self.model = onnxruntime.InferenceSession(self.model_path)

        sum_process_time = 0
        pbar = tqdm(total=self.frame_length)

        while True:
            image = self.get_image_data()
            if image is None:
                break

            # start process time
            start_time = time.perf_counter()

            # detect image
            results = self.detect_image(image)

            # end process time
            end_time = time.perf_counter()

            # calculate prcess time (s)
            process_time = end_time - start_time

            if len(results) != 0:
                for result in results:
                    category = self.category_target_dict[result.category_type]
                    pt1 = (result.left_top_x, result.left_top_y)
                    pt2 = (result.right_bottom_x, result.right_bottom_y)

                    self.plot_one_box(
                        pt1, pt2, image, category, result.prob, self.category_colors[result.category_type]
                    )

            if self.input_type == "image":
                image_path = self.image_list[self.frame_index - 1]
                file_name = os.path.basename(image_path)
                output_image_path = os.path.join("exp/detect/images", file_name)
                cv2.imwrite(output_image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

            if self.input_type == "movie":
                # file_name = os.path.splitext(os.path.basename(self.movie_path))[0]
                # frame_index = self.frame_index - 1
                # digit = len(str(self.frame_length))
                # output_image_path = (
                #     "exp/detect/movie/" + file_name + "_frame_" + str(frame_index).zfill(digit) + ".jpg"
                # )
                # cv2.imwrite(output_image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                out.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

            sum_process_time += process_time

            # update progress bar
            pbar.update()

            # wait for ESC key to exit
            if cv2.waitKey(30) == 27:
                break

            # check final frame
            if self.frame_length == self.frame_index:
                break

        # close progress bar
        pbar.close()

        average_process_time = sum_process_time / self.frame_length
        fps = 1 / average_process_time

        print(f"Process time: {average_process_time:.4f}[s], fps: {fps:.4f}")

        # movie data
        if self.input_type == 1:
            self.cap.release()
            cv2.destroyAllWindows()
            out.release()
