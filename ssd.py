import os 
import numpy as np
import tensorflow as tf
import cv2 as cv
import json 
from pprint import pprint
from sys import getsizeof
import time
import math
from sys import path
#tf.enable_eager_execution()


class Preprocessing(object):
    def __init__(self, IDs,IDs_val, img_prefix,img_prefix_val,):

        self.hflip_prob=0.5
        self.vflip_prob=0.5
        self.cut_prob=0.5
        self.do_crop = True
        self.crop_area_range = [0.75, 1.0]
        self.aspect_ratio_range = [3./4. , 4./3.]
        self.batch_size_P = BTSZ

        self.shuffle_limit = len(IDs) 
        self.shuffle_limit_val = len(IDs_val) 

        self.id_index = 0
        self.id_index_val = 0
        self.prefix = img_prefix
        self.prefix_val = img_prefix_val
        self.Prep_ID_List = IDs
        self.Prep_ID_List_val = IDs_val
        self.anan = anchor((320,320,3), feature_shapes,anchor_sizes, anchor_ratios, anchor_steps, dtype=np.float32)



    def horizontal_flip(self, img, y):
        if np.random.random() < self.hflip_prob:
            img = img[:, ::-1]
            y[:, [0, 2]] = 1 - y[:, [2, 0]]
        return img, y
    
    def vertical_flip(self, img, y):
        if np.random.random() < self.vflip_prob:
            img = img[::-1]
            y[:, [1, 3]] = 1 - y[:, [3, 1]]
        return img, y
    
    def random_sized_crop(self, img, targets,targets_labels):
        img_w = img.shape[1]
        img_h = img.shape[0]
        img_area = img_w * img_h
        random_scale = np.random.random()
        random_scale *= (self.crop_area_range[1] -
                         self.crop_area_range[0])
        random_scale += self.crop_area_range[0]
        target_area = random_scale * img_area
        random_ratio = np.random.random()
        random_ratio *= (self.aspect_ratio_range[1] -
                         self.aspect_ratio_range[0])
        random_ratio += self.aspect_ratio_range[0]
        w = np.round(np.sqrt(target_area * random_ratio))     
        h = np.round(np.sqrt(target_area / random_ratio))
        if np.random.random() < 0.5:
            w, h = h, w
        w = min(w, img_w)
        w_rel = w / img_w
        w = int(w)
        h = min(h, img_h)
        h_rel = h / img_h
        h = int(h)
        x = np.random.random() * (img_w - w)
        x_rel = x / img_w
        x = int(x)
        y = np.random.random() * (img_h - h)
        y_rel = y / img_h
        y = int(y)
        img = img[y:y+h, x:x+w]
        new_targets = []
        new_targets_labels = []
        for i,box in enumerate(targets):
            cx = 0.5 * (box[0] + box[2])
            cy = 0.5 * (box[1] + box[3])
            if (x_rel < cx < x_rel + w_rel and
                y_rel < cy < y_rel + h_rel):
                xmin = (box[0] - x_rel) / w_rel
                ymin = (box[1] - y_rel) / h_rel
                xmax = (box[2] - x_rel) / w_rel
                ymax = (box[3] - y_rel) / h_rel
                xmin = max(0, xmin)
                ymin = max(0, ymin)
                xmax = min(1, xmax)
                ymax = min(1, ymax)
                box[:4] = [xmin, ymin, xmax, ymax]
                new_targets.append(box)
                new_targets_labels.append(targets_labels[i])
        new_targets = np.asarray(new_targets).reshape(-1, targets.shape[1])
        new_targets_labels = np.asarray(new_targets_labels).reshape(-1)
        return img, new_targets,new_targets_labels
    
    def generate(self, y_,l_, train_flag=True):
         
        
        while True:
            img_list = []
            
            ground_list = []

            if train_flag:
                self.id_index = 0
                np.random.shuffle(self.Prep_ID_List)
                pl = self.Prep_ID_List
                pre = self.prefix
            else:
                self.id_index_val = 0
                np.random.shuffle(self.Prep_ID_List_val)
                pl = self.Prep_ID_List_val
                pre = self.prefix_val
            for iii in pl:
                temp_image = None
                temp_image = cv.imread(pre + iii, cv.IMREAD_COLOR)
                if(type(temp_image) == type(None)):
                    continue
                temp_image = cv.cvtColor(temp_image, cv.COLOR_BGR2RGB)
                img = temp_image.copy()
                y = np.array(y_[iii]).copy()
                l = np.array(l_[iii]).copy()
                if self.hflip_prob > np.random.random():
                    img, y = self.horizontal_flip(img, y)
                if self.vflip_prob > np.random.random():
                    img, y = self.vertical_flip(img, y)
                if self.cut_prob > np.random.random():
                    img, y,l = self.random_sized_crop(img, y,l)
                e_lb,e_lc,e_s,e_msk = encode_layers(labels_t=l,bboxes_t=y,num_class_t=11,anchors_t=self.anan,thresholdvalue=0.5)
                if(np.sum(e_msk)==0):
                    continue
                e_ground = np.concatenate([e_lb,e_lc,e_s,e_msk],axis=-1)

                
                img = cv.resize(img,(320,320)).astype(np.float32)
                

                img_list.append(img)
                
                ground_list.append(e_ground)


        
                if(len(img_list)==self.batch_size_P):
                    i_l = np.array(img_list)
                    img_list = []
                    
                    g_l = np.array(ground_list)
                    ground_list = []
                    
                    yield  i_l, g_l


def DPWConv2D_Layer_s2(input_x, num_filters, multiplier,  block_id ):

    x = tf.keras.layers.Conv2D(filters = num_filters * multiplier, kernel_size=(1, 1), strides=(1,1), 
                                use_bias = False, name = 'DPWConv2D_s2_%d_conv_expand' % block_id)(input_x)
    x = tf.keras.layers.BatchNormalization(epsilon=1e-3, momentum=0.999,name='DPWConv2D_s2_%d_BTN_expand' %block_id)(x)
    x = tf.keras.layers.Activation(activation = tf.nn.relu6, name='DPWConv2D_s2_%d_relu6_expand'%block_id)(x)

    
    x = tf.keras.layers.DepthwiseConv2D(kernel_size = (3,3), strides = (1,1), padding = 'same', depth_multiplier = 1,
                                        use_bias = False, name = 'DPWConv2D_s2_%d_conv_depth'%block_id)(x)
    x = tf.keras.layers.BatchNormalization(epsilon=1e-3, momentum=0.999,name='DPWConv2D_s2_%d_BTN_depth' %block_id)(x)
    x = tf.keras.layers.Activation(activation = tf.nn.relu6 ,name='DPWConv2D_s2_%d_relu6_depth'%block_id)(x)


    x = tf.keras.layers.Conv2D(filters = num_filters , kernel_size=(1, 1), strides=(1,1), 
                                use_bias = False, name = 'DPWConv2D_s2_%d_conv_project' % block_id)(x)
    x = tf.keras.layers.BatchNormalization(epsilon=1e-3, momentum=0.999,name='DPWConv2D_s2_%d_BTN_project' %block_id)(x)
    x = tf.keras.layers.Activation(activation = tf.nn.relu6 ,name='DPWConv2D_s2_%d_relu6_project'%block_id)(x)

    return x

def DPWConv2D_Layer_s1(input_x, num_filters, multiplier,  block_id ):

    x = tf.keras.layers.Conv2D(filters =  num_filters * multiplier, kernel_size=(1, 1), strides=(1,1), 
                                use_bias = False, name = 'DPWConv2D_s1_%d_conv_expand' % block_id)(input_x)
    x = tf.keras.layers.BatchNormalization(epsilon=1e-3, momentum=0.999,name='DPWConv2D_s1_%d_BTN_expand' %block_id)(x)
    x = tf.keras.layers.Activation(activation = tf.nn.relu6, name='DPWConv2D_s1_%d_relu6_expand'%block_id)(x)

    
    x = tf.keras.layers.DepthwiseConv2D(kernel_size = (3,3), strides = (1,1), padding = 'same', depth_multiplier = 1,
                                        use_bias = False, name = 'DPWConv2D_s1_%d_conv_depth'%block_id)(x)
    x = tf.keras.layers.BatchNormalization(epsilon=1e-3, momentum=0.999,name='DPWConv2D_s1_%d_BTN_depth' %block_id)(x)
    x = tf.keras.layers.Activation(activation = tf.nn.relu6 ,name='DPWConv2D_s1_%d_relu6_depth'%block_id)(x)


    x = tf.keras.layers.Conv2D(filters =  num_filters , kernel_size=(1, 1), strides=(1,1), 
                                use_bias = False, name = 'DPWConv2D_s1_%d_conv_project' % block_id)(x)
    x = tf.keras.layers.BatchNormalization(epsilon=1e-3, momentum=0.999,name='DPWConv2D_s1_%d_BTN_project' %block_id)(x)
    x = tf.keras.layers.Activation(activation = tf.nn.relu6 ,name='DPWConv2D_s1_%d_relu6_project'%block_id)(x)
    x = tf.keras.layers.add([x,input_x])

    return x

def DPWConv2D_layer_Conv2D(input_x, num_filters,block_id):

    x = tf.keras.layers.Conv2D(filters = num_filters // 2, kernel_size=(1, 1), strides=(1,1), 
                                use_bias = False, name = 'L_Conv2D_%d_conv_expand' % block_id)(input_x)
    x = tf.keras.layers.BatchNormalization(epsilon=1e-3, momentum=0.999,name='L_Conv2D_%d_BTN_expand' %block_id)(x)
    x = tf.keras.layers.Activation(activation = tf.nn.relu6, name='L_Conv2D_%d_relu6_expand'%block_id)(x)

    
    x = tf.keras.layers.DepthwiseConv2D(kernel_size = (3,3), strides = (2,2), padding = 'same', depth_multiplier = 2,
                                        use_bias = False, name = 'L_Conv2D_%d_conv_depth'%block_id)(x)
    x = tf.keras.layers.BatchNormalization(epsilon=1e-3, momentum=0.999,name='L_Conv2D_%d_BTN_depth' %block_id)(x)
    x = tf.keras.layers.Activation(activation = tf.nn.relu6 ,name='L_Conv2D_%d_relu6_depth'%block_id)(x)


    x = tf.keras.layers.Conv2D(filters = num_filters , kernel_size=(1, 1), strides=(1,1), 
                                use_bias = False, name = 'L_Conv2D_%d_conv_project' % block_id)(x)
    mid_x = x
    x = tf.keras.layers.BatchNormalization(epsilon=1e-3, momentum=0.999,name='L_Conv2D_%d_BTN_project' %block_id)(x)
    x = tf.keras.layers.Activation(activation = tf.nn.relu6 ,name='L_Conv2D_%d_relu6_project'%block_id)(x)
    return x, mid_x

def DPWConv2D_layer_Conv2D_nonpad(input_x, num_filters,block_id):

    x = tf.keras.layers.Conv2D(filters = num_filters // 2, kernel_size=(1, 1), strides=(1,1), 
                                use_bias = False, name = 'L_Conv2D_%d_conv_expand' % block_id)(input_x)
    x = tf.keras.layers.BatchNormalization(epsilon=1e-3, momentum=0.999,name='L_Conv2D_%d_BTN_expand' %block_id)(x)
    x = tf.keras.layers.Activation(activation = tf.nn.relu6, name='L_Conv2D_%d_relu6_expand'%block_id)(x)

    
    x = tf.keras.layers.DepthwiseConv2D(kernel_size = (3,3), strides = (2,2), padding = 'valid', depth_multiplier = 2,
                                        use_bias = False, name = 'L_Conv2D_%d_conv_depth'%block_id)(x)
    x = tf.keras.layers.BatchNormalization(epsilon=1e-3, momentum=0.999,name='L_Conv2D_%d_BTN_depth' %block_id)(x)
    x = tf.keras.layers.Activation(activation = tf.nn.relu6 ,name='L_Conv2D_%d_relu6_depth'%block_id)(x)


    x = tf.keras.layers.Conv2D(filters = num_filters , kernel_size=(1, 1), strides=(1,1), 
                                use_bias = False, name = 'L_Conv2D_%d_conv_project' % block_id)(x)
    mid_x = x
    x = tf.keras.layers.BatchNormalization(epsilon=1e-3, momentum=0.999,name='L_Conv2D_%d_BTN_project' %block_id)(x)
    x = tf.keras.layers.Activation(activation = tf.nn.relu6 ,name='L_Conv2D_%d_relu6_project'%block_id)(x)
    return x, mid_x

def Prediction_Conv(input_x,NUM_ANCHOR,NUM_CLASSES,feature_shape ):
    box_classes = tf.keras.layers.Conv2D(filters = NUM_ANCHOR * NUM_CLASSES, kernel_size=(3,3), 
                                                    strides=(1,1), padding = 'same', use_bias = True)(input_x)
    #box_classes = tf.keras.layers.Reshape((feature_shape[0],feature_shape[1],NUM_ANCHOR,NUM_CLASSES))(box_classes)
    box_classes = tf.keras.layers.Flatten()(box_classes)
    box_regress = tf.keras.layers.Conv2D(filters = NUM_ANCHOR * 4, kernel_size=(3,3), strides = (1,1), padding = 'same',
                                                    use_bias = True)(input_x)
    box_regress = tf.keras.layers.Flatten()(box_regress)
    #box_regress = tf.keras.layers.Reshape((feature_shape[0],feature_shape[1],NUM_ANCHOR,4))(box_regress)
    return box_classes, box_regress 
def SSD(image_shape):
    alpha = 1.0
    defaultshape = (224,224,3)
    Dummy_Input = tf.keras.Input(image_shape)
    
    mobilenetv2 = tf.keras.applications.MobileNetV2(input_shape = defaultshape, include_top = False, weights = 'imagenet')
    Feature = tf.keras.Model(inputs = mobilenetv2.input, outputs = mobilenetv2.get_layer('block_5_add').output)
    
    x = Feature(Dummy_Input)

    x = DPWConv2D_Layer_s2(x, 64, 6,  0 )
    x = DPWConv2D_Layer_s1(x, 64, 6,  0 )

    x = DPWConv2D_Layer_s2(x, 64, 6,  1 )
    x = DPWConv2D_Layer_s1(x, 64, 6,  1 )

    x = DPWConv2D_Layer_s2(x, 96, 6,  2 )
    x = DPWConv2D_Layer_s1(x, 96, 6,  2 )

    
    x = DPWConv2D_Layer_s2(x, 160, 6,  3 )
    x = DPWConv2D_Layer_s1(x, 160, 6,  3 )
    
    x = DPWConv2D_Layer_s2(x, 320, 6,  4 )
    
    x = DPWConv2D_Layer_s1(x, 320, 6,  4 )
    
    x = tf.keras.layers.Conv2D(filters =  1280, kernel_size=(1, 1), strides=(1,1), 
                                use_bias = False, name = 'MNF_conv' )(x)
    x = tf.keras.layers.BatchNormalization(epsilon=1e-3, momentum=0.999,name='MNF_BTN' )(x)
    x = tf.keras.layers.Activation(activation = tf.nn.relu6 ,name='MNF_relu6')(x)
    
    fm0_c , fm0_r = Prediction_Conv(x,6,11,(32,32))
    fm0_c = tf.keras.layers.Flatten()(fm0_c) 
    fm0_r = tf.keras.layers.Flatten()(fm0_r)

    x, fm1 = DPWConv2D_layer_Conv2D(x,1024,0)

    fm1_c , fm1_r = Prediction_Conv(fm1,6,11,(16,16))
    fm1_c = tf.keras.layers.Flatten()(fm1_c) 
    fm1_r = tf.keras.layers.Flatten()(fm1_r)

    x, fm2 = DPWConv2D_layer_Conv2D(x,512,1)
    
    fm2_c , fm2_r = Prediction_Conv(fm2,6,11,(8,8))
    fm2_c = tf.keras.layers.Flatten()(fm2_c) 
    fm2_r = tf.keras.layers.Flatten()(fm2_r) 

    x, fm3 = DPWConv2D_layer_Conv2D_nonpad(x,256,2)
    
    fm3_c , fm3_r = Prediction_Conv(fm3,6,11,(4,4))
    fm3_c = tf.keras.layers.Flatten()(fm3_c) 
    fm3_r = tf.keras.layers.Flatten()(fm3_r) 

    x, fm4 = DPWConv2D_layer_Conv2D(x,256,3)
    
    fm4_c , fm4_r = Prediction_Conv(fm4,6,11,(2,2))
    fm4_c = tf.keras.layers.Flatten()(fm4_c) 
    fm4_r = tf.keras.layers.Flatten()(fm4_r) 

    x, fm5 = DPWConv2D_layer_Conv2D(x,256,4)
    
    fm5_c , fm5_r = Prediction_Conv(fm5,4,11,(1,1)) 
    fm5_c = tf.keras.layers.Flatten()(fm5_c) 
    fm5_r = tf.keras.layers.Flatten()(fm5_r) 
    
    fm_c = tf.keras.layers.concatenate([fm0_c,fm1_c,fm2_c,fm3_c,fm4_c,fm5_c], axis = 1,name='concatenate_flatten_c')
    fm_r = tf.keras.layers.concatenate([fm0_r,fm1_r,fm2_r,fm3_r,fm4_r,fm5_r], axis = 1,name='concatenate_flatten_r')
    
    num_boxes = fm_c.get_shape().as_list()[1] // 11
    
    fm_c = tf.keras.layers.Reshape((num_boxes,11),name='reshape_c')(fm_c)
    fm_r = tf.keras.layers.Reshape((num_boxes,4),name='reshape_r')(fm_r)
    fm_c_p = tf.keras.layers.Activation(activation = tf.nn.softmax ,name='softmax')(fm_c)
 
    
    preds = tf.keras.layers.concatenate([fm_c,fm_r,fm_c_p],axis = 2,name='predictions')
    
    base_model = tf.keras.Model(inputs = Dummy_Input, outputs = preds)

    return base_model




##################################################################################################
#           anchor and encoding
##################################################################################################

def anchor(img_shape, featuremaps_shape,in_anchor_size, in_anchor_ratio, steps, dtype=np.float32):

    anchors_list = []
    for i, feature_shape in enumerate(featuremaps_shape):
        
        y, x = np.mgrid[0:feature_shape[0], 0:feature_shape[1]]

        y = (y.astype(dtype) + 0.5) * steps[i] / img_shape[0]
        x = (x.astype(dtype) + 0.5) * steps[i] / img_shape[1]
        y = np.expand_dims(y, axis=-1)
        x = np.expand_dims(x, axis=-1)

        num_anchors = len(in_anchor_size[i]) + len(in_anchor_ratio[i])
        h = np.zeros((num_anchors, ), dtype=dtype)
        w = np.zeros((num_anchors, ), dtype=dtype)
        h[0] = in_anchor_size[i][0] / img_shape[0]
        w[0] = in_anchor_size[i][0] / img_shape[1]
        di = 1
        if len(in_anchor_size[i]) > 1:
            h[1] = math.sqrt(in_anchor_size[i][0] * in_anchor_size[i][1]) / img_shape[0]
            w[1] = math.sqrt(in_anchor_size[i][0] * in_anchor_size[i][1]) / img_shape[1]
            di += 1
        for j, r in enumerate(in_anchor_ratio[i]):
            h[j+di] = in_anchor_size[i][0] / img_shape[0] / math.sqrt(r)
            w[j+di] = in_anchor_size[i][0] / img_shape[1] * math.sqrt(r)
        anchors_list.append((y,x,w,h))

    return anchors_list


def encode_layers(labels_t, bboxes_t,num_class_t, anchors_t,thresholdvalue):
    target_labels = []
    target_localizations = []
    target_scores = []
    target_masks = []
    for al in anchors_t:
        tlabel, tlocal, tscore, tmask = encode_label_layer(labels_=labels_t, bboxes_ =  bboxes_t,num_classes_ = num_class_t,anchors_layer = al, prior_scaling=[0.1, 0.1, 0.2, 0.2],thresholdvalue_l = thresholdvalue, dtype = np.float32)

        target_labels.append(tlabel)
        target_localizations.append(tlocal)
        target_scores.append(tscore)
        target_masks.append(tmask)
        
    target_labels = np.concatenate([target_labels[0],target_labels[1],target_labels[2],target_labels[3],target_labels[4],target_labels[5]])
    target_localizations = np.concatenate([target_localizations[0],target_localizations[1],target_localizations[2],target_localizations[3],target_localizations[4],target_localizations[5]])
    target_scores = np.concatenate([target_scores[0],target_scores[1],target_scores[2],target_scores[3],target_scores[4],target_scores[5]])
    target_masks = np.concatenate([target_masks[0],target_masks[1],target_masks[2],target_masks[3],target_masks[4],target_masks[5]])
    
    target_labels = np.reshape(target_labels,(target_labels.shape[0],1))

    target_localizations = np.reshape(target_localizations,(target_localizations.shape[0]//4,4))
    target_scores = np.reshape(target_scores,(target_scores.shape[0],1))
    target_masks = np.reshape(target_masks,(target_masks.shape[0],1)).astype(np.float32)

    return target_labels, target_localizations, target_scores, target_masks


def encode_label_layer(labels_, bboxes_,num_classes_, anchors_layer, prior_scaling,thresholdvalue_l,dtype = np.float32):


    yref, xref, href, wref = anchors_layer

    ymin = yref - href / 2.
    xmin = xref - wref / 2.
    ymax = yref + href / 2.
    xmax = xref + wref / 2.
    vol_anchors = (xmax - xmin) * (ymax - ymin)
 
    
    shape = (yref.shape[0], yref.shape[1], href.size)
    feat_labels = np.zeros(shape, dtype=np.int64)
    feat_scores = np.zeros(shape, dtype=dtype)
 
    feat_ymin = np.zeros(shape, dtype=dtype)
    feat_xmin = np.zeros(shape, dtype=dtype)
    feat_ymax = np.ones(shape, dtype=dtype)
    feat_xmax = np.ones(shape, dtype=dtype)
 
    def jaccard_with_anchors(bbox):
    
        int_ymin = np.maximum(ymin, bbox[1])
        int_xmin = np.maximum(xmin, bbox[0])
        int_ymax = np.minimum(ymax, bbox[3])
        int_xmax = np.minimum(xmax, bbox[2])
        h = np.maximum(int_ymax - int_ymin, 0.)
        w = np.maximum(int_xmax - int_xmin, 0.)
        
        inter_vol = h * w
        union_vol = vol_anchors - inter_vol \
            + (bbox[3] - bbox[1]) * (bbox[2] - bbox[0])
        jaccard = np.divide(inter_vol, union_vol)
        
        return jaccard
 
  
 

    def body(i, feat_labels, feat_scores,
             feat_ymin, feat_xmin, feat_ymax, feat_xmax):

        label = labels_[i]
        bbox = bboxes_[i]
        
        jaccard = jaccard_with_anchors(bbox)
     
        mask = np.greater(jaccard, feat_scores)
        mask_ = np.greater(jaccard,thresholdvalue_l)
        mask = np.logical_and(mask,mask_)
    
 
        
        imask = mask.astype(np.int32)
        fmask = mask.astype(np.float32)
      
        feat_labels = imask * label + (1 - imask) * feat_labels

        feat_scores = np.where(mask, jaccard, feat_scores)
 
        feat_ymin = fmask * bbox[1] + (1 - fmask) * feat_ymin
        feat_xmin = fmask * bbox[0] + (1 - fmask) * feat_xmin
        feat_ymax = fmask * bbox[3] + (1 - fmask) * feat_ymax
        feat_xmax = fmask * bbox[2] + (1 - fmask) * feat_xmax
 
        return [i+1, feat_labels, feat_scores,
                feat_ymin, feat_xmin, feat_ymax, feat_xmax]

    i = 0
    
    while(i < labels_.shape[0]):
       

        [i, feat_labels, feat_scores,
        feat_ymin, feat_xmin,
         feat_ymax, feat_xmax]          = body(i, feat_labels, feat_scores,
                                                feat_ymin, feat_xmin,
                                                feat_ymax, feat_xmax)
                                           
    feat_cy = (feat_ymax + feat_ymin) / 2.
    feat_cx = (feat_xmax + feat_xmin) / 2.
    feat_h = feat_ymax - feat_ymin
    feat_w = feat_xmax - feat_xmin

    feat_cy = (feat_cy - yref) / href / prior_scaling[0]

    feat_cx = (feat_cx - xref) / wref / prior_scaling[1]
    feat_h = np.log(feat_h / href) / prior_scaling[2]
    feat_w = np.log(feat_w / wref) / prior_scaling[3]

    feat_localizations = np.stack([feat_cx, feat_cy, feat_w, feat_h], axis=-1)
 
    feat_labels = feat_labels.flatten()
    feat_localizations = feat_localizations.flatten()
    feat_scores = feat_scores.flatten()
    feat_mask = feat_scores > thresholdvalue_l

    return feat_labels, feat_localizations, feat_scores, feat_mask




#################################################################################################
#           loss
#################################################################################################
class BOXES_LOSS():
    def __init__(self):
        self.match_threshold=0.1,
        self.negative_ratio=3.
        self.alpha=1.
        self.label_smoothing=0.
        self.batch_size = BTSZ

    def l1_smooth_loss(self,y_true_l,y_pred_l):
        abs_loss = tf.abs(y_true_l - y_pred_l)
        sq_loss = 0.5 * (y_true_l - y_pred_l) ** 2
        l1_loss = tf.where(tf.less(abs_loss,1.0),sq_loss,abs_loss - 0.5)
        return tf.reduce_sum(l1_loss,axis = -1)


    def compute_loss(self, y_true, y_pred):

        batch_size_loss = tf.shape(y_true)[0]

        num_boxex_loss = tf.to_float(tf.shape(y_true)[1])

        logits_l = tf.nn.softmax(y_pred[:,:,:11])

        loc_loss = self.l1_smooth_loss(y_true[:,:,1:5],y_pred[:,:,11:15])

        num_pos = tf.reduce_sum(y_true[:,:,-1],axis = -1)

        y_true_int = tf.to_int32(y_true[:,:,0])
 
        num_pos_ex = tf.expand_dims(num_pos,axis = -1)

        pos_conf_loss = tf.losses.sparse_softmax_cross_entropy(labels = y_true_int,logits=y_pred[:,:,:11],weights = y_true[:,:,-1])# / num_pos_ex)

        pos_loc_loss = tf.reduce_sum(loc_loss*y_true[:,:,-1], axis = 1)

        #get negative
        num_neg = tf.minimum(self.negative_ratio * num_pos, num_boxex_loss - num_pos)
        num_neg = tf.maximum(num_neg , num_boxex_loss // 7)
        num_neg = tf.to_int32(num_neg)
        max_conf = tf.reduce_max(logits_l[:,:,1:11] ,axis = 2)
        n_max_conf = max_conf * (1. - y_true[:,:,-1])
        num_neg = tf.reduce_min(num_neg)
        
        value_l, indice_l = tf.nn.top_k(n_max_conf, k = num_neg)
       
        value_l = tf.expand_dims(value_l[:,-1], axis = -1)
        n_mask = tf.greater(n_max_conf, value_l)

        n_mask = tf.to_float(n_mask)
        #n_mask = n_mask * y_pred[:,:,-1]
        num_n = tf.reduce_sum(n_mask, axis = -1)
        num_n_ex = tf.expand_dims(num_n,axis = -1) + 1.
         
        neg_conf_loss = tf.losses.sparse_softmax_cross_entropy(labels = y_true_int,logits=y_pred[:,:,:11],weights = n_mask) #/ num_n_ex)

        total_loss = (neg_conf_loss / tf.reduce_sum(n_mask) + pos_conf_loss / tf.reduce_sum(num_pos)) #/ ( + )

        total_loss = total_loss + self.alpha * tf.reduce_sum(pos_loc_loss / num_pos)
 
        return total_loss / self.batch_size

#################################################################################################
#           decode   
#################################################################################################
class Boxes_decode(object):
    def __init__(self):
        self.score_theshold = 0.85
        self.top_k = 50
        self.nms_thshd = 0.6
        self.boxes = tf.placeholder(dtype='float32', shape=(None,4))
        self.scores = tf.placeholder(dtype='float32', shape =(None,))
        self.nms = tf.image.non_max_suppression(self.boxes, self.scores,self.top_k,iou_threshold=self.nms_thshd)
        self.sess = tf.Session()
        self.b_num_class = 11
        self.color_boxes = [ (255,0,0),
                            (0,255,0),
                            (0,0,255),
                            (255,255,0),
                            (255,0,255),
                            (0,255,255),
                            (255,255,255),
                            (255,128,255),
                            (128,255,255),
                            (255,255,128)]
        self.anchor_dc = reform_anchor(ananan)#anchor((512,512,3), feature_shapes,anchor_sizes, anchor_ratios, anchor_steps, dtype=np.float32)
        self.id_box = np.array(range(8188)).astype(np.int32)

    def decode_id(self, idx_):
        i = 0
        x = 0
        y = 0

        if(idx_<6144):

            temp = np.floor(idx_ / 6)
            m = idx_ % 6
            y = np.floor(temp / 32)
            x = temp % 32

        elif(idx_<7680):
            i = 1
            m = idx_ % 6
            temp = np.floor((idx_ - 6144) / 6)
            y = np.floor(temp / 16)
            x = temp % 16
        elif(idx_<8064):
            i = 2
            m = idx_ % 6
            temp = np.floor((idx_ - 7680) / 6)
            y = np.floor(temp / 8)
            x = temp % 8
        elif(idx_<8160):
            i = 3
            m = idx_ % 6
            temp = np.floor((idx_ - 8064) / 6)
            y = np.floor(temp / 4)
            x = temp % 4
        elif(idx_<8184):
            i = 4
            m = idx_ % 6
            temp = np.floor((idx_ - 8160) / 6)
            y = np.floor(temp / 2)
            x = temp % 2
        else:
            i = 5
            temp = np.floor((idx_ - 8184) / 4)
            m = (idx_ - 8184) % 4
            y = np.floor(temp )
            x = np.floor(temp )
        return i, int(x), int(y), m
        
    def decode_boxes(self,prd_rlt, sc_imgs):
        #prd_boxes = prd_rlt[:,:,11:15]
        prd_score = prd_rlt[:,:,15:]
        

        for i in range(len(prd_rlt)):
            temp_box = prd_rlt[i,:,11:15]
            sc_img = sc_imgs[i]
            sc_img = cv.cvtColor(sc_img, cv.COLOR_RGB2BGR)
            for c in range(1,self.b_num_class):
                sc_c_prd = prd_rlt[i,:,15+c]
                sc_mask = sc_c_prd > self.score_theshold
                if(len(sc_mask)>0):
                    score_sc = sc_c_prd[sc_mask]
                    boxes_sc = self.decode_pre_process(temp_box)
                    boxes_sc = boxes_sc[sc_mask]
                    
                    feed_dict = {self.boxes:boxes_sc, self.scores:score_sc}
                    idx = self.sess.run(self.nms, feed_dict=feed_dict)
                   
                    for index in idx:
                        boxxx = boxes_sc[index]
                        xxmin = (boxxx[1] * sc_img.shape[1]).astype(np.int32)
                        xxmax = (boxxx[3] * sc_img.shape[1]).astype(np.int32)
                        yymin = (boxxx[0] * sc_img.shape[0]).astype(np.int32)
                        yymax = (boxxx[2] * sc_img.shape[0]).astype(np.int32)
                        cv.rectangle(sc_img,(xxmin,yymin),(xxmax,yymax),self.color_boxes[c-1],2)
                        cv.putText(sc_img,'T:%s/S:%s'%(c,score_sc[index]), (xxmin,yymin),cv.FONT_HERSHEY_SIMPLEX, 1,self.color_boxes[c-1],1)
            file_id = int(np.random.random() * 1000)
            cv.imwrite('test%s.jpg'%file_id,sc_img)
    def decode_pre_process(self, in_box ):

        yy = in_box[:,1] * ps[0] * self.anchor_dc[:,3] + self.anchor_dc[:,1]
        xx = in_box[:,0] * ps[1] * self.anchor_dc[:,2] + self.anchor_dc[:,0]
        hh = np.exp(in_box[:,3] * ps[3]) * self.anchor_dc[:,3]
        ww = np.exp(in_box[:,2] * ps[2]) * self.anchor_dc[:,2] 
 
        xxmin = np.expand_dims(xx - ww/2. ,axis = -1)
        xxmax = np.expand_dims(xx + ww/2. ,axis = -1)
        yymin = np.expand_dims(yy - hh/2. ,axis = -1)
        yymax = np.expand_dims(yy + hh/2. ,axis = -1)
 
        out_box = np.concatenate([yymin,xxmin,yymax,xxmax],axis = -1)

        return out_box


    def decode_boxes_t(self,prd_rlt, sc_img):
        prd_boxes = prd_rlt[1]
        prd_score = prd_rlt[2]
        prd_lb = prd_rlt[0]

        color_box = self.color_boxes[0]
        sc_img = cv.cvtColor(sc_img, cv.COLOR_RGB2BGR)

        sc_c_prd = prd_score[:,0]
        print(prd_score.shape)
        
        print(prd_score)
        
        sc_mask = prd_score[:,0] > 0.5#self.score_theshold
        print(sc_mask.shape)
        if(len(sc_mask)>0):
            score_sc = sc_c_prd[sc_mask]
            boxes_sc = self.decode_pre_process(prd_boxes)
            boxes_sc = boxes_sc[sc_mask]
            print(boxes_sc.shape)
            id_sc = self.id_box[sc_mask]
            feed_dict = {self.boxes:boxes_sc, self.scores:score_sc}
            idx = self.sess.run(self.nms, feed_dict=feed_dict)
            
            for index in idx:

                boxxx = boxes_sc[index]
                print(boxxx)
                

                xxmin = (boxxx[1] * sc_img.shape[1]).astype(np.int32)
                xxmax = (boxxx[3] * sc_img.shape[1]).astype(np.int32)
                yymin = (boxxx[0] * sc_img.shape[0]).astype(np.int32)
                yymax = (boxxx[2] * sc_img.shape[0]).astype(np.int32)
                
                
                cv.rectangle(sc_img,(xxmin,yymin),(xxmax,yymax),color_box,1)
                
        file_id = int(np.random.random() * 1000)
        cv.imwrite('test%s.jpg'%file_id,sc_img)

def reform_anchor( pr_anan):

    num_layer = len(pr_anan)
    out_anan = []
    for i in range(num_layer):
        ry, rx, rw, rh = pr_anan[i]
        for j in range(ry.shape[0]):
            for k in range(ry.shape[1]):
                for m in range(len(rw)):
                    out_anan.append([rx[j][k],ry[j][k],rw[m],rh[m]])
    out_anan = np.array(out_anan)
    return out_anan


#################################################################################################
#           main
#################################################################################################

anchor_sizes = [(48., 96.), (96., 140.8), (140.8, 198.4),
                 (198.4, 243.2), (243.2, 288.), (288., 332.8)]

anchor_ratios = [[1.5, 1./1.5, 1.3, 1./1.3], [1.5, 1./1.5, 1.3, 1./1.3], [1.5, 1./1.5, 1.3, 1./1.3], [1.5, 1./1.5, 1.3, 1./1.3],
                     [1.5, 1./1.5, 1.3, 1./1.3], [1.5, 1./1.5]]
feature_shapes = [(40,40) , (20,20), (10,10), (4,4), (2,2), (1,1)]
anchor_steps=[ 8, 16, 32, 80 , 160, 320]
ps = [0.1, 0.1, 0.2, 0.2]
ananan = anchor((320,320,3), feature_shapes,anchor_sizes, anchor_ratios, anchor_steps, dtype=np.float32)

BTSZ = 16


if __name__=='__main__':
 

    input_image_shape = (1280,720,3)

    EPOCH = 5

    SSD512 = SSD((320,320,3))


    SSD512.load_weights('./checkpoints/weights300.02-1.52.hdf5', by_name=True)

    DDBox = Boxes_decode()

    for L in SSD512.layers:

        if L.name == 'model':
            L.trainable = False


    base_lr = 3e-4

    def schedule(epoch, decay=0.9):
        return base_lr * decay**(epoch)

    callbacks = [tf.keras.callbacks.ModelCheckpoint('./checkpoints/weights300.{epoch:02d}-{val_loss:.2f}.hdf5',
                                             verbose=1,
                                             save_weights_only=True),
                                             tf.keras.callbacks.LearningRateScheduler(schedule)]


    optim = tf.keras.optimizers.Adam(lr=base_lr)
    SSD512.compile(optimizer=optim,
              loss=BOXES_LOSS().compute_loss)



    gen = Preprocessing(training_IDs,training_IDs_val, data_root,data_root_val)

###############################################################
#           Training
###############################################################

    #print(gen.shuffle_limit//16)
    print('Start training...')
    history = SSD512.fit_generator(gen.generate( location,label,train_flag=True),
                           steps_per_epoch=gen.shuffle_limit//BTSZ,
                           #steps_per_epoch=3,
                           epochs=EPOCH, verbose=1,
                           callbacks=callbacks,
                           validation_data=gen.generate(location_val,label_val,train_flag=False),
                           validation_steps=gen.shuffle_limit_val//BTSZ
                           #validation_steps=3
                           )

