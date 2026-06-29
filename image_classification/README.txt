YOLO IMAGE CLASSIFICATION
=========================

1. Label images in Label Studio with one choice per image, such as:
   PASS
   NG

2. Export the Label Studio project as JSON and place the JSON file in:

   Export JSON from label-studio

3. To prepare the dataset and start training, run:

   python image_classification\prepare_dataset.py
   python image_classification\train.py

   The dataset folder is generated from the newest JSON export and can be
   deleted safely when you want to clean old training input.

4. Check that both classes exist:

   dataset\train\PASS
   dataset\train\NG
   dataset\val\PASS
   dataset\val\NG

Training output:

   runs\pass_ng_classifier\weights\best.pt

Only best.pt is required by the realtime dashboard. Other training reports,
batch preview images, and last.pt can be deleted safely.

PREDICTION
==========

1. Put new images in:

   predict_images

   Create this folder when you need folder-based testing.

2. Run:

   python image_classification\predict.py

3. Review:

   prediction_results\predictions.csv
   prediction_results\images

REAL-TIME UI
============

Open the unified Aurotek Edge UI and choose Realtime Dashboard.

The UI watches Input_files\Picture, converts each new image to Sobel edges,
classifies it as PASS or NG, and displays the confidence.

This workflow is image classification. It does not use YOLO bounding-box
labels or data.yaml.
