# ckwg +29
# Copyright 2019 by Kitware, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#    * Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
#    * Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
#    * Neither name of Kitware, Inc. nor the names of any contributors may be used
#    to endorse or promote products derived from this software without specific
#    prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
from __future__ import print_function
from __future__ import division

import numpy as np
import torch
import pickle
import os
import signal
import sys
import time
import yaml
import mmcv

from collections import namedtuple
from PIL import Image
from distutils.util import strtobool
from shutil import copyfile

from kwiver.vital.algo import DetectedObjectSetOutput, TrainDetector
from kwiver.vital.types import (
    BoundingBoxD, CategoryHierarchy, DetectedObject, DetectedObjectSet,
)

from learn.algorithms.CutLER.pretrain_algo import PretrainAlgo


_Option = namedtuple('_Option', ['attr', 'config', 'default', 'parse', 'help'])


class CutLERTrainer( TrainDetector ):
    """
    Implementation of TrainDetector class
    """

    _options = [
        _Option('_gpu_count', 'gpu_count', -1, int, ''),
        _Option('_launcher', 'launcher', 'pytorch', str, ''), # "none, pytorch, slurm, or mpi" 
        
        _Option('_config_file', 'config_file', '', str, ''),

        _Option('_output_directory', 'output_directory', '', str, '')
    ]

    def __init__( self ):
        TrainDetector.__init__( self )

        for opt in self._options:
            setattr(self, opt.attr, opt.default)

        #self.config = yaml.safe_load(self._config_file)
        #print('config:', self.config)

    def get_configuration( self ):
        # Inherit from the base class
        cfg = super( TrainDetector, self ).get_configuration()

        for opt in self._options:
            cfg.set_value(opt.config, str(getattr(self, opt.attr)))
        return cfg

    def set_configuration( self, cfg_in ):
        cfg = self.get_configuration()
        cfg.merge_config( cfg_in )

        for opt in self._options:
            setattr(self, opt.attr, opt.parse(cfg.get_value(opt.config)))

        """
        # ============================
        self.toolset = 
        register_new_losses()

        mmdet_config_path = os.path.join(get_original_cwd(), './learn/algorithms/MMDET/configs/'+self.config["mmdet_model_config_file"])
        coco_data_path = os.path.join(top_data_dir, f"annotations/{self.toolset['stage']}_maskcut_annotations.json")

        self.mmdet_config = self.set_config(mmdet_config_path, coco_data_path, train_image_dir)
        self.mmdet_config.work_dir = self.work_dir

        print(f'Config:\n{self.mmdet_config.pretty_text}')

        self.original_chkpt_file = str(os.path.join(self.toolset['protocol_config']['domain_network_selector']['params']["pretrained_network_dir"],
            self.config["model_checkpoint_file"]))

        seed = self.config["seed"]
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        self.mmdet_config.seed = seed

        replace(self.mmdet_config, 500)
        """
        
        
    def check_configuration( self, cfg ):
        return True
        if not cfg.has_value( "config_file" ) or len( cfg.get_value( "config_file") ) == 0:
            print( "A config file must be specified!" )
            return False

    def load_network( self ):
        """
        ##### Train an MMDetection model rather than detectron
        meta = dict()
        env_info_dict = collect_env()
        env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
        dash_line = '-' * 60 + '\n'
        print('Environment info:\n' + dash_line + env_info + '\n' + dash_line)
        meta['env_info'] = env_info
        meta['config'] = self.mmdet_config.pretty_text
        meta['seed'] = self.mmdet_config.seed

        print("building dataset")
        datasets = [build_dataset(self.mmdet_config.data.train)]

        print("building detector")
        model = build_detector(
            self.mmdet_config.model, train_cfg=self.mmdet_config.get('train_cfg'), test_cfg=self.mmdet_config.get('test_cfg')
        )

        checkpoint_file = self.original_chkpt_file
        chkpt = load_checkpoint(model, checkpoint_file)

        print("training model")

        model.train()
        train_detector(model, datasets, self.mmdet_config, distributed=False, validate=False, meta=meta)


        print("finished training")
        self.model = model
        save_checkpoint(model, os.path.join(self.work_dir, "pretrain.pth"))

        gc.collect()  # Make sure all object have ben deallocated if not used
        torch.cuda.empty_cache()
        return
        """

    def add_data_from_disk( self, categories, train_files, train_dets, test_files, test_dets ):
        print('add_data_from_disk')

        if len( train_files ) != len( train_dets ):
            print( "Error: train file and groundtruth count mismatch" )
            return
        
        cats = []
        for cat in categories.all_class_names():
            cat_id = categories.get_class_id(cat)
            cats.append({'name': cat, 'id': int(cat_id)})

        for split in [train_files, test_files]:
            is_train = ( split == train_files )
            num_images = len(split)
            
            images = []
            annotations = []
            annotation_id = 0

            for index in range(num_images):
                filename = split[index]
                img = mmcv.image.imread(filename)
                height, width = img.shape[:2]
                targets = train_dets[index] if is_train else test_dets[index]

                image_dct = {'file_name': filename, # dataset.root + '/' + dataset.image_fnames[index],
                    'height': height,
                    'width': width,
                    'id': int(index)
                }

                image_anno_ctr = 0

                for target in targets:
                    bbox = [  target.bounding_box.min_x(),
                              target.bounding_box.min_y(),
                              target.bounding_box.max_x(),
                              target.bounding_box.max_y() ] # tlbr
                    
                    # skip bboxes with 0 width or height
                    if (bbox[2] - bbox[0]) <= 0 or (bbox[3] - bbox[1]) <= 0:
                        continue

                    class_lbl = target.type.get_most_likely_class()
                    if categories is not None:
                        class_id = categories.get_class_id( class_lbl )
                    else:
                        if class_lbl not in self._categories:
                            self._categories.append( class_lbl )
                        class_id = self._categories.index( class_lbl )

                    annotation_dct = {'bbox': [bbox[0], bbox[1], (bbox[2]-bbox[0]), (bbox[3]-bbox[1])],  # coco annotations in file need x, y, width, height
                        'image_id': int(index),
                        'area': (bbox[2]-bbox[0]) * (bbox[3]-bbox[1]),
                        'category_id': class_id,
                        'id': annotation_id,
                        'iscrowd': 0
                    }
                    annotations.append(annotation_dct)
                    annotation_id += 1
                    image_anno_ctr += 1

                images.append(image_dct)

            coco_format_json = dict(
                images=images,
                annotations=annotations,
                categories=cats)

            fn = 'train_data_coco.json' if is_train else 'test_data_coco.json'
            self.work_dir = ''
            output_file = os.path.join(self.work_dir, fn)
            mmcv.dump(coco_format_json, output_file)
            
            print(f"Transformed the dataset into COCO style: {output_file} "
                  f"Num Images {len(images)} and Num Annotations: {len(annotations)}")
        
    def update_model( self ):
        self.trainer.train(self._output_dir)
        print( "\nModel training complete!\n" )

    def interupt_handler( self ):
        self.proc.send_signal( signal.SIGINT )
        timeout = 0
        while self.proc.poll() is None:
            time.sleep( 0.1 )
            timeout += 0.1
            if timeout > 5:
                self.proc.kill()
                break
        sys.exit( 0 )



def __vital_algorithm_register__():
    from kwiver.vital.algo import algorithm_factory

    # Register Algorithm
    implementation_name = "cutler_supervised_mmdet"

    if algorithm_factory.has_algorithm_impl_name(
      CutLERTrainer.static_type_name(), implementation_name ):
        return

    algorithm_factory.add_algorithm( implementation_name,
      "PyTorch CutLER supervised mmdet training routine", CutLERTrainer )

    algorithm_factory.mark_algorithm_as_loaded( implementation_name )
