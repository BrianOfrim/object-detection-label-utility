# Object detection label utility

**This project is now deprecated, it's functionality has been added to https://github.com/BrianOfrim/boja**

A matplotlib based GUI utility for annotating images with labels and bounding boxes.

![LabelSample](https://raw.githubusercontent.com/BrianOfrim/object-detection-label-utility/master/docs/assets/labelImage_480.jpg)

## Getting started
### Installing dependencies
Install required pip modules using:
 ```
$ pip install -r requirements.txt
 ```
### Providing inputs
 Fill in the blanks in the data/ directory structure  
 .  
├── data  
│   ├── **<labels.txt>**  
│   ├── annotations  
│   ├── images  
│   │   ├── **<Input_Images>**  
│   └── manifests  

The **labels.txt** file should contain the labels you wish to apply to the bounding boxes of your image. Each label should be on it's own line.  
An example contents of a labels.txt file would be as follows:  
aluminum  
compost  
glass  
paper  
plastic  
trash  

The **Input_Images** represents where the input images should be placed (in the directory data/images/).

### Running the application
To run the labeling application navigate to the odlu folder and run the label.py file:
```
$ cd odlu
$ python label.py
```
The output annotation files will be saved in **data/annotations/** by default.

The output manifest files generated will have a filename **[UNIX_TIMESTAMP]-manifest.txt**
Each line in the manifest contains a comma separated list of the image filename, annotation file name. For example:  
...  
image-123.jpg,annotation-abc.xml  
image-456.jpg,annotation-def.xml  
...  

When a new manifest file is generated the contents for the previous manifest file is copied into the new file and the new image file, annotation file pair lines are appended to the end of the new file. 

## Options
There are various command line options which can be seen by running:
```
$ python label.py --help
```
### AWS S3 integration
The application can be configured to retrieve images from, and send annotation/manifest files to, an AWS s3 bucket that the user has access to:
```
$ python label.py --s3_bucket_name <bucket_name>
```

The directory location within the s3 bucket that the images are to be pulled from and the annotations/manifest files will be pushed to can be configured using  the following flags:  
  --s3_annotation_dir: Prefix of the s3 image annotation objects. (default: 'data/annotations/')  
  --s3_bucket_name: S3 bucket to retrieve images from and upload manifest to.  
  --s3_image_dir: Prefix of the s3 image objects. (default: 'data/images/')  
  --s3_manifest_dir: Prefix of the s3 image annotation manifest objects (default: 'data/manifests/')  

The default s3 directory locations mimic the default local directory locations. The project root is just replaced with the s3 bucket root.

