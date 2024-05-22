#!/bin/sh

.venv/bin/yolo detect mode=predict model=./runs/train/weights/best.pt source=image.jpg project=runs name=predict