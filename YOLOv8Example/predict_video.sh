#!/bin/sh

.venv/bin/yolo detect mode=predict model=./runs/train/weights/best.pt source=video.mp4 project=runs name=predict
