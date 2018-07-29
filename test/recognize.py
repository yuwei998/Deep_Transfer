# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 11:02:11 2017

@author: Alpha
"""

import datetime
import tensorflow as tf

def inference(image_path):
# Read in the image_data
    image_data = tf.gfile.FastGFile(image_path, 'rb').read()
    
    # Loads label file, strips off carriage return
    print('done')
    label_lines = [line.rstrip() for line in tf.gfile.GFile("../output_labels_new.txt")]

    #name_dict = {'car': '汽车',
    #             'fishingboat': '渔船',
    #             'armoredcar': '装甲车',
    #             'coach': '客车',
    #             'cargoship': '货船',
    #             'gunship': '武装直升机',
    #             'fighterplane': '战斗机',
    #             'transportaircraft': '运输机',
    #             'multibarrelrocketlauncher': '火箭炮车',
    #             'civilavitation': '民用航空飞机',
    #             'aircraftcarrier': '航空母舰',
    #             'truck': '货车',
    #             'mainbattletank': '主战坦克',
    #             'frigate': '护卫舰',
    #             'millitaryvehicle': '军用越野车'}

    time1 = datetime.datetime.now()
    # Unpersists graph from file
    # with tf.gfile.FastGFile("output_graph_new.pb", 'rb') as f:
    #     graph_def = tf.GraphDef()
    #     graph_def.ParseFromString(f.read())
    #     _ = tf.import_graph_def(graph_def, name='')

    
    init_ops = tf.global_variables_initializer()
    
    with tf.Session() as sess:
        sess.run(init_ops)
        # Feed the image_data as input to the graph and get first prediction
        time2 = datetime.datetime.now()
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

        predictions = sess.run(softmax_tensor, \
                 {'DecodeJpeg/contents:0': image_data})

        # Sort to show labels of first prediction in order of confidence
        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
        print(top_k)
        for node_id in top_k:
            human_string = label_lines[node_id]
    #        human_string = name_dict[label_lines[node_id]]
            score = predictions[0][node_id]
            print('%s (score = %.5f)' % (human_string, score))
    #        print('%s (概率 = %.5f)' % (human_string, score))
    time3 = datetime.datetime.now()

    loadtime = time2 - time1
    procetime = time3- time2
    print('time for loading network: %s' % loadtime)
    print('time for recognize: %s' % procetime)
    return label_lines[top_k[0]]


