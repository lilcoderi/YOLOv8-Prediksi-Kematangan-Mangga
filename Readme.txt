dataset_split = data split awal (manual)
yolo_dataset = data yolo (manual)
run - mangga_yolov8 = manual

Training Model YOLOv8 di run/train/mangga_yolov8
    isinya: 
    1. results.png — grafik loss, precision, recall, mAP
    2. confusion_matrix.png — jika klasifikasi
    3. weights/best.pt — model terbaik
    4. weights/last.pt — model terakhir

Evaluasi mAP, Precision, Recall di test set
    (run/detect/val)
    isinya:
    1. metrics/precision: rata-rata precision per kelas
    2. metrics/recall: rata-rata recall
    3. metrics/mAP50: Mean Average Precision (IoU > 0.5)
    4. metrics/mAP50-95: mAP dari IoU 0.5 sampai 0.95



