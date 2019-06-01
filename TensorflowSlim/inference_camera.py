#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import time
import cv2

tf.app.flags.DEFINE_string(
    'pb_file', '',
    'frozen file.')

tf.app.flags.DEFINE_string(
    'label_path', '',
    '')

tf.app.flags.DEFINE_integer(
    'image_size', 224,
    'network input image size')

FLAGS = tf.app.flags.FLAGS

with tf.gfile.FastGFile(FLAGS.pb_file, 'rb') as f:
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

class_name = []
f = open(FLAGS.label_path, 'r')
lines = f.readlines()
for line in lines:
  line = line[:-1] # remove end line char
  class_name.append(line)
f.close()

x = tf.get_default_graph().get_tensor_by_name('input:0')
y = tf.get_default_graph().get_tensor_by_name('MobilenetV2/Predictions/Reshape_1:0') # mobilenet_1.0_224

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
  cropped = cv2.resize(cropped,(FLAGS.image_size, FLAGS.image_size))
  image = np.reshape(cropped, (1, FLAGS.image_size, FLAGS.image_size, 3))
  image = image / 255.
  image = image - 0.5
  image = image * 2

  # swap BGR -> RGB
  temp = image[:, :, 2].copy()
  image[:, :, 2] = image[:, :, 0]
  image[:, :, 0] = temp
    
  preview = frame

  start_time = time.time()
  probability = sess.run(y, feed_dict={x: image})
  print("--- %f fps ---" % (1.0 / float(time.time() - start_time)))
  p = probability[0]
  p_ind = np.argmax(p)
  prediction = class_name[p_ind]    

  font = cv2.FONT_HERSHEY_SIMPLEX
  cv2.putText(preview, prediction, (10, 100), font, 1, (255, 0, 0), 3, cv2.LINE_AA)

  cv2.imshow('frame1', frame)        
    #if(frame.size>0):
      
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

capture.release()
cv2.destroyAllWindows()


