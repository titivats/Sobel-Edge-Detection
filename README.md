# Sobel Image Classification

Production workflow for Sobel fine-tuning, Label Studio classification,
Ultralytics training, batch prediction, and realtime PASS/NG monitoring.

## Operator workflow

1. Run `Finetune.bat` to create Sobel images under `Output_Sobel`.
2. Label those images as `PASS` or `NG` in Label Studio.
3. Export JSON into:

   `image_classification\Export JSON from label-studio`

4. Run `image_classification\Train_model.bat`.
5. Run `image_classification\Predict_model.bat` for folder-based testing.
6. Run `Realtime_Dashboard.bat` for the realtime dashboard.

## Important paths

- Trained model:
  `image_classification\runs\pass_ng_classifier\weights\best.pt`
- Prediction input:
  `image_classification\predict_images`
- Prediction report:
  `image_classification\prediction_results`
- Realtime input:
  `Input_files\Picture`
- Product CSV:
  `Input_files\Product Info`

This project uses whole-image YOLO classification. It does not use
object-detection boxes, detection labels, or `data.yaml`.
