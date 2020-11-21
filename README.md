# Logo classifier V3.0

### Function

Recognize logo from a picture 

### Support

1. Support 1 folder of picture at a time.

2. The logo has to be in the middle of the picture and takes most of the area.

3. Only local command line and local file address input is supported.

4. Only following brands are included: 'BMW', 'Ford', 'Honda', 'Toyota', 'VW'

### Requirements

Packsages:

Pytorch
Pillow
numpy 
pandas 
matplotlib
natsort


Files:

train, val, unknown, tlmodelv3, class_names.txt, logoclassifierV3.0.py


### Usage

Try `python logoclassifierV3.0.py filename`

For example, `python logoclassifierV3.0.py ./unkown`

