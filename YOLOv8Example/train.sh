#!/bin/sh

.venv/bin/yolo detect mode=train model=yolov8s.pt data=data.yaml imgsz=640 epochs=5 batch=1 project=runs name=train