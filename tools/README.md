# Tools Documentation

## generate_record_file.py

This file could generate list and record files for specified dataset.

Sample command:

```
python ./tools/generate_record_file.py --dataset-name voc_07_test --dataset-path ./data/VOCdevkit
```



## im2rec.py

This file is copied [MXNet](https://github.com/apache/incubator-mxnet/blob/master/tools/im2rec.py) project.

You could use im2rec.py to create your own record dataset in two steps.

### Step 1. Make an Image List File

After you prepare the dataset,  you need to make an image list file. The format is:

```
integer_image_index \t extra_header_width \t per_object_label_witdth \t objects_label... \t path_to_image
```

1. `extra_header_width` represents width from itself to labels, 
here `extra_header_width` = len(extra_header_width, per_object_label_witdth) = 2.
2. `per_object_label_witdth` is the label width for one object annotation, for voc dataset, it is 6 includes (x_min, 
y_min, x_max, y_max, cls_id, difficult).

Here is an example file (from voc07 test set):
```
0	2	6	47.00	239.00	194.00	370.00	11.00	0.00	7.00	11.00	351.00	497.00	14.00	0.00	./VOC2007/JPEGImages/000001.jpg
1	2	6	138.00	199.00	206.00	300.00	18.00	0.00	./VOC2007/JPEGImages/000002.jpg
2	2	6	122.00	154.00	214.00	194.00	17.00	0.00	238.00	155.00	306.00	204.00	8.00	0.00	./VOC2007/JPEGImages/000003.jpg
```

### Step 2. Create the Record File

Sample command:

```
python ./tools/imrec.py list_file_path image_root_dir --no-shuffle --pack-label --num-thread 4
```

  