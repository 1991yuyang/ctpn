import json
import os
import numpy as np
"""
convert labelme polygon json label file to txt file
"""

labelme_json_file_dir = r"/home/yuyang/data/vin_code/labels"
txt_file_save_dir = r"/home/yuyang/data/vin_code/txt"


def cvt_one_json_file(json_file_name):
    txt_file_name = "gt_" + json_file_name.replace("json", "txt")
    json_content = json.load(open(os.path.join(labelme_json_file_dir, json_file_name)))
    shapes = json_content["shapes"]
    s = ""
    for item in shapes:
        points = ",".join([str(i) for i in np.array(item["points"], dtype=int).ravel().tolist()])
        s += points
        s += "\n"
    s = s.strip("\n")
    with open(os.path.join(txt_file_save_dir, txt_file_name), "w", encoding="utf-8") as file:
        file.write(s)


def cvt_func():
    for json_file_name in os.listdir(labelme_json_file_dir):
        cvt_one_json_file(json_file_name)


if __name__ == "__main__":
    cvt_func()