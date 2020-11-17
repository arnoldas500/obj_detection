# obj_detection
object detection for clouds on mesonet images
Run using docker, tensorboard and jupyter lab:
On hulk:
$docker run --runtime=nvidia --ipc=host -p 8885:8885 -p 6006:6006 -it -v /raid/:/data clouds_arnold:v1
-Next run notebook using port 8885 (follow above)
http://hulk.asrc.albany.edu:8885
-Next can start tensorboard using port 6006 from new terminal by entering the already running docker container using docker exec -it
In new docker terminal:
$tensorboard --logdir /tmp/retrain_logs --port=6006
View on:
http://hulk.asrc.albany.edu:6006


Arnold object detection:
Start docker container:
docker run --runtime=nvidia --ipc=host -p 8879:8879 -p 6006:6006 -it -v /raid/:/data clouds_arnold:v1

docker run --runtime=nvidia --ipc=host -p 8879:8879 -p 6005:6006 -it -v /raid/:/home/ -w /home akurbanovas/clouds_a

Note : /workspace has old models 

git clone https://github.com/tensorflow/models.git
cd models/research/object_detection/

#will break pip and python kernel
apt-get install protobuf-compiler python-pil python-lxml python-tk
#end breaking cod
pip install --user Cython
pip install --user contextlib2
pip install --user jupyter
pip install --user matplotlib

# From tensorflow/models/research/ to compile protobufs to python executables
protoc object_detection/protos/*.proto --python_out=.

#coco set up (common objects in context)
#large image data set designed for object detection


jupyter notebook --generate-config
emacs /root/.jupyter/jupyter_notebook_config.py 
Or
cat > /root/.jupyter/jupyter_notebook_config.py
c.NotebookApp.allow_origin = '*'
c.NotebookApp.ip = '0.0.0.0'
jupyter notebook --port=8885 --ip=0.0.0.0 --allow-root --no-browser .

Steps to run:
Create folders data images and training 
Data is going to contain train and test csv and .record files plus category .pbtxt - dictionary describing each category label
Images are going to contain test and train folders with all train and test images with the corresponding .xml files
Training will contain model config 
Using LabelIImg create boxes for objects in each image and corresponding .xml files (split these into the test and train dirs under images)
LabelImg is downloaded on Cearas with setup to run over X forwarding, make sure XORG is installed if using a mac to ssh into Cearas. 
Cd into labeImg and type labelImg 
Click 'Change default saved annotation folder' in Menu/File (use the same folder to save XML files that you are grabbing the images from)
Click 'Open Dir'
Change ‘default class’ to your category name
Click 'Create RectBox'
Click and release left mouse to select a region to annotate the rect box  (save after every image or turn on auto save)

Convert XML files to singular CSV files from these splits using xml_to_csv notebook (run only first main and check csv files after they are generated)
Use singular CSV files to generate TF records that are needed for training (first check to make sure CSV files are not empty)
Need to run twice (1 for train and another for test), update to tensorflow 2.0 with v1 compat mode is generate_tfrecord_v2.py for version 1 use generate_tfrecord.py
python3 generate_tfrecord_V2.py --csv_input=/home/arnold/clouds_detection/obj_detection/data/train_labels.csv --output_path=/home/arnold/clouds_detection/obj_detection/data/train.record --image_dir=/home/arnold/clouds_detection/obj_detection/images/train/

  python3 generate_tfrecord_V2.py --csv_input=/home/arnold/clouds_detection/obj_detection/data/test_labels.csv --output_path=/home/arnold/clouds_detection/obj_detection/data/test.record --image_dir=/home/arnold/clouds_detection/obj_detection/images/test/
NOTE: if error “TypeError: None has type NoneType, but expected one of: int, long” go into CSV file and look at all the classes and make sure they are the same as categories under class_text_to_int in generate_tf_records. Need to find and delete labels like ‘precip’ and add any new labels that are being used
After need to grab config ssd_...config and model tar 
$ tar -xzf ssd_mobilenet_v1_coco_11_06_2017.tar.gz
NOTE: in config change PATH_TO_BE_CONFIG, num_classes (3 currently), train_config batch size, checkpoit name/path (fine_tune_checkpoint: "ssd_mobilenet_v1_coco_11_06_2017/model.ckpt" we download this from models) and label_map_path: "training/object-detect.pbtxt"
last update input_path for data/ train and test .record 

object-detect.pbtxt needs to be maunally created: (id needs to start at 1 NOT zero, once created drop this file in /data)
item {
     id:1
     name: 'scattered_clouds'
}
item {
     id:2
     name: 'clear'
}
item {
     id:3
     name: 'overcast'
}
item {
     id:4
     name: 'few'
}

COPY data, images, downloaded model folder, training and .config to models/object_detection
$ cp -r data images training ssd_mobilenet_v1_coco_11_06_2017 /home/arnold/clouds_detection/modelsV2/research/object_detection/legacy
Finally can train:
From within /home/arnold/clouds_detection/modelsV2/research/object_detection/legacy:
UPDATE : code is now in models research object detection legacy 
$ python3 train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/ssd_mobilenet_v1_pets.config

if have tf2.0 wont work need pip install tensorflow-gpu==1.15

want to get avg loss below 1 (if above 2 model iss trash)
If error between training make sure you delete everything except model config andd obj.pbtxt file before rerunning (so delete checkpoints graphs etc)
If stuck on global_step/sec: 0

Load up tensorboard
$ tensorboard --logdir=training/


## NOTES:
 ML Cloud Categorization
* Initialize new container
```
$ `docker run —runtime=nvidia -it -p 8886:8886 -v “/raid/images/“:”/data/“ tensorflow_v`
$`nvidia-smi`
$`git clone https://github.com/tensorflow/hub.git`
$`pip install tensorflow_hub`
* Configure & launch jupyter
$`jupyter notebook --generate-config`—>create config file
$`jupyter notebook password` —> create password
$`vim /root/.jupyter/jupyter_notebook_config.py`
	* add to file:
```
c.NotebookApp.allow_origin = '*'
c.NotebookApp.ip = '0.0.0.0'	
```
$`cd /data`or directory from which to launch
$`jupyter notebook --port=8886 --ip=0.0.0.0 --allow-root --no-browser .`
	* open browser and launch [Jupyter Notebook](http://hulk.asrc.albany.edu:8886/tree?)

* exit container but keep running
$`control + p + q to `
* re-enter saved container
$`docker ps`-> see running containers and get container name or ID
$`docker exec -it silly_thompson /bin/bash`-> run container
```
```
* paths:
	* images
		* outside container: /raid/images & /raid/images/images 
		* inside container: /data/images
	* scripts 
		* inside container: /workspace/hub/examples/image_retraining
		* outside container: ??
* retrain script:
	* path: /workspace/hub/examples/image_retraining
	* $`python retrain.py —how_many_training_steps 2000 —image_dir /data/images`
* label/test single image:
	* path: /workspace/hub/examples/image_retraining
	* $`python label_image.py --graph=/tmp/output_graph.pb --labels=/tmp/output_labels.txt --input_layer=Placeholder --output_layer=final_result --image='/data/20190101T230029_VOOR.jpg'`
```
