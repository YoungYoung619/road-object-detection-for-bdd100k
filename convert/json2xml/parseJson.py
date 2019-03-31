#-*-coding:utf-8-*-
#parse json，input json filename,output info needed by voc

import json

## bdd100k obj types ##
## if you don't wanne some types ,just delete those types in this list.
categorys = ['bus','traffic light',
             'traffic sign','person','bike',
             'truck','motor','car','train','rider']

def parseJson(jsonFile):
    objs = []
    f = open(jsonFile)
    info = json.load(f)
    objects = info['frames'][0]['objects']
    for i in objects:
        obj = []
        if(i['category'] in categorys):
            obj.append(int(i['box2d']['x1']))
            obj.append(int(i['box2d']['y1']))
            obj.append(int(i['box2d']['x2']))
            obj.append(int(i['box2d']['y2']))
            obj.append(i['category'])
            objs.append(obj.copy())
            #print("objs",objs)
    return objs
