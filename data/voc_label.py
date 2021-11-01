import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
from shutil import copy, rmtree
import random
from pathlib import Path


def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x, y, w, h)


def voc2coco(data_set, image_id, train_val="train"):
    # Pascal VOC format
    in_file = open('./labels/{}/{}.xml'.format(data_set, image_id))
    print(image_id)
    # COCO format
    out_file = open(
        './coco/{}/labels/{}/{}.txt'.format(data_set, train_val, image_id), 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(
            xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " +
                       " ".join([str(a) for a in bb]) + '\n')

    copy('./images/{}/{}.jpg'.format(data_set, image_id),
         './coco/{}/images/{}'.format(data_set, train_val))


def process(data_set, ratio=0.1):
    images_train = os.path.join("coco", data_set, "images", "train")
    labels_train = os.path.join("coco", data_set, "labels", "train")
    images_val = os.path.join("coco", data_set, "images", "val")
    labels_val = os.path.join("coco", data_set, "labels", "val")

    rmtree(images_train, ignore_errors=True)
    rmtree(labels_train, ignore_errors=True)
    rmtree(images_val, ignore_errors=True)
    rmtree(labels_val, ignore_errors=True)

    Path(images_train).mkdir(parents=True, exist_ok=True)
    Path(labels_train).mkdir(parents=True, exist_ok=True)
    Path(images_val).mkdir(parents=True, exist_ok=True)
    Path(labels_val).mkdir(parents=True, exist_ok=True)

    image_ids = [os.path.splitext(os.path.basename(x))[0] for x in os.listdir(
        './labels/{}'.format(data_set)) if x.endswith(".xml")]
    random.shuffle(image_ids)
    num = int(ratio * len(image_ids))
    val_set = image_ids[:num]
    train_set = image_ids[num:]
    for image_id in train_set:
        voc2coco(data_set, image_id, train_val="train")
    for image_id in val_set:
        voc2coco(data_set, image_id, train_val="val")


if __name__ == "__main__":
    classes = []
    with open("predefined_classes.txt") as file:
        classes = file.readlines()
        classes = [line.rstrip() for line in classes]
    process('cmnd', 0.2)
