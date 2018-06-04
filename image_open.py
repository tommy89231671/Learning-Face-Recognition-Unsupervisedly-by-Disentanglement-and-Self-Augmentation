#from PIL import Image
#import numpy as np
#im = Image.open( "0001.jpg" )
from PIL import Image,ImageDraw
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

import torchvision.transforms as transforms
import torchvision.models as models

tree = ET.ElementTree(file='./img/Tara_gtOrig.xml')

root = tree.getroot()
frame_num=int(root.attrib['end_frame'])-int(root.attrib['start_frame'])+1

a=[0]*frame_num
for i in range(frame_num):
    a[i]=[]

for child in root:
    for grandson in child:
        #print(int(grandson.attrib["frame_no"])-1)
        #print((int(grandson.attrib["x"]),int(grandson.attrib["y"]),int(grandson.attrib["width"]),int(grandson.attrib['height'])))
        a[int(grandson.attrib["frame_no"])-1].append((int(grandson.attrib["x"]),int(grandson.attrib["y"]),int(grandson.attrib["width"]),int(grandson.attrib['height'])))

#n=input("Enter no:")
#tmp=n.zfill(4)+'.jpg'
#
#im=Image.open(tmp)
#draw = ImageDraw.Draw(im)
#i=0
#for j in a[int(n)]:
#    #draw.line([(j[0],j[1]),(j[0]+j[2],j[1]),(j[0]+j[2],j[1]+j[3]),(j[0],j[1]+j[3]),(j[0],j[1])], fill = (255,0,0), width = 5)
#    #box=(j[0],j[1]+j[3],j[0]+j[2],j[1])
#    #plt.subplot(len(a[int(n)]),1,i+1)
#    i=i+1
#    box=(j[0],j[1],j[0]+j[2],j[1]+j[3])
#    nim=im.crop(box)
#    
#    #nim.thumbnail( (j[2],j[3]) )
#    #nim2 = nim.resize( (0,0), Image.BILINEAR )
#    print(nim)
#    #plt.axis('off')
#    #plt.imshow(nim2)
#    #print(nim2)
#    fname=str(n)+'_'+str(i)+'.jpg'
#    nim.save(fname)
#    #plt.savefig(fname)
#    #plt.imshow(nim)
#    #plt.show()
##    cr.save('1.jpg')
##    plt.imshow(cr)
##    tmp=str(j)+'.jpg'
##    plt.savefig(tmp)
##
##    
#    
'''all'''
for frameno in range(5):
    i=0
    tmp=str(frameno+1).zfill(4)
    img='./img/'+tmp+".jpg"
    im=Image.open(img)   
    for face in a[frameno]:
        i=i+1
        box=(face[0],face[1],face[0]+face[2],face[1]+face[3])
        nim=im.crop(box)
        fname='./crop/'+str(frameno+1)+'_'+str(i)+'.jpg'
        nim.save(fname)
#    tmp=str(i+1).zfill(4)
#    img=tmp+".jpg"
#    im=Image.open(img)       
#    draw = ImageDraw.Draw(im)
#    print(img)
#    for j in a[i]:
#        draw.line([(j[0],j[1]),(j[0]+j[2],j[1]),(j[0]+j[2],j[1]+j[3]),(j[0],j[1]+j[3]),(j[0],j[1])], fill = (255,0,0), width = 5)
#    fname=img[0:4]+'new.jpg'
#    plt.imshow(im)
#    plt.savefig(fname)
#    