import os
import cv2
import face_recognition as fr
from PIL import Image, ImageDraw
import numpy


videos_src_path = "/home/eternalding/ML/videos/"
videos_save_path = "/home/eternalding/ML/video_crop/"

videos = os.listdir(videos_src_path)
videos = filter(lambda x: x.endswith('avi'), videos)

batch_size=100

ignore=0

known_face=numpy.zeros(shape=(200,128))
zeros_128=numpy.zeros(128)
pics=numpy.zeros(128)

total_face=0

for each_video in videos:
    ignore=0  
    total_face=0
    print (each_video)

    # get the name of each video, and make the directory to save frames
    each_video_name, _ = each_video.split('.')
    if not os.path.isdir(videos_save_path + '/' + each_video_name):
      os.mkdir(videos_save_path + '/' + each_video_name)               

    each_video_save_full_path = os.path.join(videos_save_path, each_video_name) + '/'

    # get the full path of each video, which will open the video tp extract frames
    each_video_full_path = os.path.join(videos_src_path, each_video)

    cap  = cv2.VideoCapture(each_video_full_path)
    frame_count = 1
    success = True
    while(success):
        people=0
        success, frame = cap.read()
        print ("Read a new frame: ", success," now at frame: ",frame_count)
        if(success):
          cv2.imwrite(each_video_save_full_path + each_video_name + "_%d.jpg" % frame_count, frame,)
          frame_count = frame_count + 1
    
    #now we have video frames, begin to crop faces and classify them
    if not os.path.isdir(videos_save_path + '/' + each_video_name+'_face_crop'):
      os.mkdir(videos_save_path + '/' + each_video_name+'_face_crop') 
    
    dir_1 =videos_save_path + each_video_name+'_face_crop'
    i=1
    known_people=0
    ID=0
    while(1):
      
      path=each_video_save_full_path + each_video_name + "_%d.jpg" % i
      #image not exist then break
      if(os.path.isfile(path)==0):
        break
        
      #fr load image  
      image = fr.load_image_file(path)
      
      face_loactions = fr.face_locations(image)
  
      
      j=len(face_loactions)   
      print("now cropping number ",i," th picture. ", "Found ",j," faces in this frame")
      
      
      for face_location in face_loactions:    
        total_face=total_face+1  
        print("now cropping ",j,"face in this frame")
        # Print the location of each face in this image
        top, right, bottom, left = face_location       

        # You can access the actual face itself like this:
        
        face_image = image[top:bottom, left:right]
        
        
        #now seperate different people to different directory
        #face_landmark_list: number of faces in one frame
        #image:fr.load of frame
        #pil_image:face location of image 
        #pil_image_2:face_image->Image type
        
        
        #target = cropped face now
        
        #knownface[]
           
        print(len(fr.face_encodings(face_image)),"elements")   
        #can not be encoded
        if(len(fr.face_encodings(face_image))==0):
          
          if not os.path.isdir(dir_1 + '/' + each_video_name + '_'+'ignore'):
            os.mkdir(dir_1 + '/' + each_video_name + '_'+'ignore')
          save_dir = os.path.join(dir_1 + '/' + each_video_name + '_'+'ignore','frame_'+str(i)+'_crop_'+str(ignore)+'.jpeg')
          pil_image_2 = Image.fromarray(face_image)
          pil_image_2.save(save_dir,'jpeg') 
          ignore=ignore+1
          j=j-1
          print("cannot be encoded ,ignore,ignore rate=",ignore/total_face)
          
        else:
          print(fr.face_encodings(face_image)[0].shape,"fr.face_encodings(face_image)[0].shape")
          
          target_face_encoding = fr.face_encodings(face_image)[0]   
        
          if((known_face[0]==zeros_128).any()):#no known faces
            known_face[known_people]=target_face_encoding;
            known_people=known_people+1
            ID=0
          else:
            results = fr.compare_faces(known_face, target_face_encoding)
            for index in range(0,known_people): #searching in known people
              if(results[index]==1):
                print("ID is ",index)
                ID=index
                break
            if(not True in results):#unknown people
              print("Unknown person, ID=",known_people)
              ID=known_people
              known_face[known_people]=target_face_encoding;
              known_people=known_people+1
            
          #save
          if not os.path.isdir(dir_1 + '/' + each_video_name + '_'+str(ID)):
            os.mkdir(dir_1 + '/' + each_video_name + '_'+str(ID))   
          
          save_dir = os.path.join(dir_1 + '/' + each_video_name + '_'+str(ID),'frame_'+str(i)+'_crop_'+str(int(pics[ID]))+'.jpeg')
          
          pics[ID]=pics[ID]+1
          pil_image_2 = Image.fromarray(face_image)
          pil_image_2.save(save_dir,'jpeg')
          j=j-1  
      i=i+1;
  print("ignore rate:",ignore/total_face)     
        
cap.release()

