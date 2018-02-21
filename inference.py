#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 16:48:51 2017

@author: chang
"""
import tensorflow as tf
import os
import numpy as np
import time
import cv2

with tf.gfile.FastGFile('./exp_1/frozen_flower.pb', 'rb') as f:   
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(
                graph_def, 
                input_map=None, 
                return_elements=None, 
                name='', 
                op_dict=None, 
                producer_op_list=None
                )

#for op in tf.get_default_graph().get_operations():
#    print(op.name)
#op = tf.get_default_graph().get_operations()
#num_op = len(op)

#op[0].name
#op[num_op-1].name

#node_num = len(graph_def.node)
#node_first = graph_def.node[0]
#node_last  = graph_def.node[node_num-1]
#print(node_first)
#print(node_last)

class_name = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips' ]

      
x = tf.get_default_graph().get_tensor_by_name('input:0')
y = tf.get_default_graph().get_tensor_by_name('InceptionV3/Predictions/Reshape_1:0')

capture = cv2.VideoCapture(0)
print('image width %d' % capture.get(3))
print('image height %d' % capture.get(4))

capture.set(3, 640)
capture.set(4, 480)

sess = tf.Session()
while(1):
  ret, frame = capture.read()

  #if(ret == True):
    
  cropped = frame
  #if(True):
  cropped = cv2.resize(cropped,(299,299))
  image = np.reshape(cropped, (1, 299, 299, 3))
  image = image / 255.
  image = image - 0.5
  image = image * 2
    
  preview = frame
   
  probability = sess.run(y, feed_dict={x: image})
  p = probability[0]
  p_ind = np.argmax(p)
  prediction = class_name[p_ind]    

  font = cv2.FONT_HERSHEY_SIMPLEX
  cv2.putText(preview, prediction, (10, 100), font, 2, (255, 0, 0), 3, cv2.LINE_AA)

  cv2.imshow('frame1', frame)        
    #if(frame.size>0):
      
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

capture.release()
cv2.destroyAllWindows()


