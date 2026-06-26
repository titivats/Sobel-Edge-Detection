YOLO IMAGE CLASSIFICATION
=========================

1. Label images in Label Studio with one choice per image, such as:
   PASS
   NG

2. Export the Label Studio project as JSON and place the JSON file in:

   Export JSON from label-studio

3. To prepare the dataset and start training, double-click:

   Train_model.bat

4. Check that both classes exist:

   dataset\train\PASS
   dataset\train\NG
   dataset\val\PASS
   dataset\val\NG

Training output:

   runs\pass_ng_classifier\weights\best.pt

PREDICTION
==========

1. Put new images in:

   predict_images

2. Double-click:

   Predict_model.bat

3. Review:

   prediction_results\predictions.csv
   prediction_results\images

REAL-TIME UI
============

Double-click:

   Realtime_UI.bat

The UI watches Input_files\Picture, converts each new image to Sobel edges,
classifies it as PASS or NG, and displays the confidence.

This workflow is image classification. It does not use YOLO bounding-box
labels or data.yaml.
