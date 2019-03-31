#-*-coding:utf-8-*-
import os
from convert.json2xml import pascal_voc_io
from convert.json2xml import parseJson

if __name__ == '__main__':
    ## bdd json file dir##
    dirName = "g:/Data/BDD100K/bdd/labels/100k/train"

    ## where you wanne save the xml file ##
    savePath = "../Annotations"

    i = 1
    for dirpath,dirnames,filenames in os.walk(dirName):
        for filepath in filenames:
            fileName = os.path.join(dirpath,filepath)
            print("processing: ",i)
            i = i + 1
            xmlFileName = filepath[:-5]
            #print("xml: ",xmlFileName)
            objs = parseJson.parseJson(str(fileName))
            if len(objs):
                tmp = pascal_voc_io.PascalVocWriter(savePath=savePath,filename=xmlFileName,
                                                imgSize=(720,1280,3),databaseSrc="BDD100K")
                for obj in objs:
                    tmp.addBndBox(obj[0],obj[1],obj[2],obj[3],obj[4])
                tmp.save()
            else:
                print(fileName)
