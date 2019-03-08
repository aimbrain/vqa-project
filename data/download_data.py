#    Copyright 2018 AimBrain Ltd.

#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at

#        http://www.apache.org/licenses/LICENSE-2.0

#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os

# download input questions (training, validation and test sets)
os.system(
    'wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip -P zip/')
os.system(
    'wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip -P zip/')
os.system(
    'wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Test_mscoco.zip -P zip/')

# download annotations (training and validation sets)
os.system(
    'wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip -P zip/')
os.system(
    'wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip -P zip/')

# download pre-trained glove embeddings
os.system('wget http://nlp.stanford.edu/data/glove.6B.zip -P zip/')

# download rcnn extracted features (may take a while, both very large files)
os.system(
    'wget https://imagecaption.blob.core.windows.net/imagecaption/trainval_36.zip -P zip/')
os.system(
    'wget https://imagecaption.blob.core.windows.net/imagecaption/test2015_36.zip -P zip/')

# extract them
os.system('unzip zip/v2_Questions_Train_mscoco.zip -d raw/')
os.system('unzip zip/v2_Questions_Val_mscoco.zip -d raw/')
os.system('unzip zip/v2_Questions_Test_mscoco.zip -d raw/')
os.system('unzip zip/v2_Annotations_Train_mscoco.zip -d raw/')
os.system('unzip zip/v2_Annotations_Val_mscoco.zip -d raw/')
os.system('unzip zip/glove.6B.zip -d ./')
os.system('unzip zip/trainval_36.zip -d raw/')
os.system('unzip zip/test2015_36.zip -d raw/')