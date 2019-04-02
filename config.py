'''
Copyright (c) College of Mechatronics and Control Engineering, Shenzhen University.
All rights reserved.

Description :
Some config

Author：Team Li
'''
from enum import Enum, unique

######### net building config ##########
## anchor config, total 6 layers, each layer can produce different size of anchors##
normal_anchor_range = [0.05 ,0.7] #two interval value indicate the range of anchor in normal layer.
special_anchor_range = [0.02, 0.03] ##two value indicate the size of anchor in feat_layer_1


img_size = (418, 418) #default img height and width

##supported backbone name
supported_backbone_name = ['vgg_16', 'mobilenet_v2']

## extracted features name, each model can extract six feature layer##
extract_feat_name = {'vgg_16':['backbone/vgg_16/conv4/conv4_3','backbone/vgg_16/conv5/conv5_3',
                              'backbone/vgg_16/block7/conv7','backbone/vgg_16/block8/conv3x3',
                              'backbone/vgg_16/block9/conv3x3','backbone/vgg_16/block10/conv3x3'],
                     'mobilenet_v2':['layer_7','layer_14','layer_17',
                                     'layer_19','layer_21', 'layer_23']}

## only for input 418x418x3
feat_size_all_layers = {'mobilenet_v2':{'layer_1':(53, 53), 'layer_2':(27, 27),'layer_3':(14, 14),
                                  'layer_4':(7, 7),'layer_5':(4, 4),'layer_6':(2, 2)},
                        'vgg_16': {'layer_1': (52, 52), 'layer_2': (26, 26), 'layer_3': (13, 13),
                                         'layer_4': (7, 7), 'layer_5': (4, 4), 'layer_6': (2, 2)}
                        }
######### net building config ##########


########################################################
############ config_dict when building net #############
########################################################
### method used to process backbone endpoints ###
class process_backbone_method(Enum):
    NONE = 0
    CFE = 1
    RESIZE = 2
    MSF = 3

## train range ##
class train_range(Enum):
    REFINE = 0 ##only train refine net
    ALL = 1 ##train all net

# config mergeFeatures #
@unique
class merge_method(Enum):
    CONCAT = 0  ## tf.concat([up_feat, de_feat], axis=-1)
    ADD = 1     ## up_feat + de_feat


@unique
class deconv_method(Enum):
    LEARN_HALF = 0  ## learn half and reuse half
    LEARN_ALL = 1     ## learn half
########################################################


## loss building config ###
# config gt_refine_loss #
@unique
class refine_method(Enum):
    NEAREST_NEIGHBOR = 0
    JACCARD_BIGGER = 1
    JACCARD_TOPK = 2

refine_pos_jac_val_all_layers = [0.2, 0.3, 0.4, 0.4, 0.3, 0.3]
det_pos_jac_val_all_layers = [0.5, 0.6, 0.7, 0.7, 0.6, 0.6]

####### dataset config #############
total_obj_n = 11 ##include background

