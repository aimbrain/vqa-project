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

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import argparse
import base64
import numpy as np
import csv
import sys
import h5py
import pandas as pd
import zarr
from tqdm import tqdm


csv.field_size_limit(sys.maxsize)


def features_to_zarr(phase):
    FIELDNAMES = ['image_id', 'image_w', 'image_h',
                  'num_boxes', 'boxes', 'features']

    if phase == 'trainval':
        infiles = [
            'raw/trainval_36/trainval_resnet101_faster_rcnn_genome_36.tsv',
        ]
    elif phase == 'test':
        infiles = [
            'raw/test2015_36/test2015_resnet101_faster_rcnn_genome_36.tsv',
        ]
    else:
        raise SystemExit('Unrecognised phase')

    # Read the tsv and load data in a dictionary
    in_data = {}
    for infile in infiles:
        print(infile)
        with open(infile, "r") as tsv_in_file:
            reader = csv.DictReader(
                tsv_in_file, delimiter='\t', fieldnames=FIELDNAMES)
            for item in reader:
                item['image_id'] = str(item['image_id'])
                item['image_h'] = int(item['image_h'])
                item['image_w'] = int(item['image_w'])
                item['num_boxes'] = int(item['num_boxes'])
                for field in ['boxes', 'features']:
                    encoded_str = base64.decodestring(
                        item[field].encode('utf-8'))
                    item[field] = np.frombuffer(encoded_str,
                                                dtype=np.float32).reshape((item['num_boxes'], -1))
                in_data[item['image_id']] = item

    # convert dict to pandas dataframe
    train = pd.DataFrame.from_dict(in_data)
    train = train.transpose()

    # create image sizes csv
    print('Writing image sizes csv...')
    d = train.to_dict()
    dw = d['image_w']
    dh = d['image_h']
    d = [dw, dh]
    dwh = {}
    for k in dw.keys():
        dwh[k] = np.array([d0[k] for d0 in d])
    image_sizes = pd.DataFrame(dwh)
    image_sizes.to_csv(phase + '_image_size.csv')

    # select bounding box coordinates and fill hdf5
    h = h5py.File(phase + 'box.hdf5', mode='w')
    t = train['boxes']
    d = t.to_dict()
    print('Creating bounding box file...')
    for k, v in tqdm(d.items()):
        h.create_dataset(str(k), data=v)
    if h:
        h.close()

    # convert to zarr
    print('Writing zarr file...')
    i_feat = h5py.File(phase + 'box.hdf5', 'r', libver='latest')
    dest = zarr.open_group(phase + '_boxes.zarr', mode='w')
    zarr.copy_all(i_feat, dest)
    i_feat.close()
    dest.close()
    os.remove(phase + 'box.hdf5')

    # select features and fill hdf5
    h = h5py.File(phase + '.hdf5', mode='w')
    t = train['features']
    d = t.to_dict()
    print('Creating image features file...')
    for k, v in tqdm(d.items()):
        h.create_dataset(str(k), data=v)
    if h:
        h.close()

    # convert to zarr
    print('Writing zarr file...')
    i_feat = h5py.File(phase + '.hdf5', 'r', libver='latest')
    dest = zarr.open_group(phase + '.zarr', mode='w')
    zarr.copy_all(i_feat, dest)
    i_feat.close()
    dest.close()
    os.remove(phase + '.hdf5')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                        description='Preprocessing for VQA v2 image data')
    parser.add_argument('-d', '--data', nargs='+', help='trainval, and/or test, list of data phases to be processed', required=True)
    args, unparsed = parser.parse_known_args()
    if len(unparsed) != 0:
        raise SystemExit('Unknown argument: {}'.format(unparsed))

    phase_list = args.data

    for phase in phase_list:
        # First download and extract

        if not os.path.exists(phase + '.zarr'):
            print('Converting features tsv to zarr file...')
            features_to_zarr(phase)

    print('Done')
