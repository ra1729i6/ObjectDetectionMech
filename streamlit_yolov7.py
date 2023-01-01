import singleinference_yolov7
from singleinference_yolov7 import SingleInference_YOLOV7
import os
import streamlit as st
import logging
import requests
from PIL import Image
from io import BytesIO
import numpy as np
import cv2
import numpy as np
from skimage.feature import greycomatrix,greycoprops
import pickle
import pandas as pd
import base64
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"jpg"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
class Streamlit_YOLOV7(SingleInference_YOLOV7):
    '''
    streamlit app that uses yolov7
    '''
    def __init__(self,):
        self.logging_main=logging
        self.logging_main.basicConfig(level=self.logging_main.DEBUG)

    def new_yolo_model(self,img_size,path_yolov7_weights,path_img_i,device_i='cpu'):
        '''
        SimpleInference_YOLOV7
        created by Steven Smiley 2022/11/24
        INPUTS:
        VARIABLES                    TYPE    DESCRIPTION
        1. img_size,                    #int#   #this is the yolov7 model size, should be square so 640 for a square 640x640 model etc.
        2. path_yolov7_weights,         #str#   #this is the path to your yolov7 weights 
        3. path_img_i,                  #str#   #path to a single .jpg image for inference (NOT REQUIRED, can load cv2matrix with self.load_cv2mat())
        OUTPUT:
        VARIABLES                    TYPE    DESCRIPTION
        1. predicted_bboxes_PascalVOC   #list#  #list of values for detections containing the following (name,x0,y0,x1,y1,score)
        CREDIT
        Please see https://github.com/WongKinYiu/yolov7.git for Yolov7 resources (i.e. utils/models)
        @article{wang2022yolov7,
            title={{YOLOv7}: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors},
            author={Wang, Chien-Yao and Bochkovskiy, Alexey and Liao, Hong-Yuan Mark},
            journal={arXiv preprint arXiv:2207.02696},
            year={2022}
            }
        
        '''
        super().__init__(img_size,path_yolov7_weights,path_img_i,device_i=device_i)
    def main(self):
        st.title('Defect Detection')
        st.subheader(""" Upload an image and run YoloV7 on it for object detection.\n""")
        st.markdown(
            """
        <style>
        .reportview-container .markdown-text-container {
            font-family: monospace;
        }
        .sidebar .sidebar-content {
            background-image: linear-gradient(#2e7bcf,#2e7bcf);
            color: black;
        }
        .Widget>label {
            color: green;
            font-family: monospace;
        }
        [class^="st-b"]  {
            color: green;
            font-family: monospace;
        }
        .st-bb {
            background-color: black;
        }
        .st-at {
            background-color: green;
        }
        footer {
            font-family: monospace;
        }
        .reportview-container .main footer, .reportview-container .main footer a {
            color: black;
        }
        header .decoration {
            background-image: None);
        }
        </style>
        """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <style>
            .reportview-container {
                #background: url("misc/galaxy-11098__340.jpg")
                add_bg_from_local('galaxy-11098__340.jpg')
            }
        .sidebar .sidebar-content {
                #background: url("https://raw.githubusercontent.com/stevensmiley1989/STREAMLIT_YOLOV7/main/misc/IMG_0512_reduce.JPG")
                add_bg_from_local('milky-way-2695569__340.jpg')
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        text_i_list=[]
        for i,name_i in enumerate(self.names):
            #text_i_list.append(f'id={i} \t \t name={name_i}\n')
            text_i_list.append(f'{i}: {name_i}\n')
        st.selectbox('Classes',tuple(text_i_list))
        self.conf_selection=st.selectbox('Confidence Threshold',tuple([0.1,0.25,0.5,0.75,0.95]))
        
        self.response=requests.get(self.path_img_i)

        self.img_screen=Image.open(BytesIO(self.response.content))

        st.image(self.img_screen, caption=self.capt, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
        st.markdown('YoloV7 on streamlit.  Demo of object detection with YoloV7 with a web application.')
        self.im0=np.array(self.img_screen.convert('RGB'))
        self.load_image_st()
        predictions = st.button('Predict on the image?')
        if predictions:
            self.predict()
            predictions=False

    def load_image_st(self):
        uploaded_img=st.file_uploader(label='Upload an image')
        if type(uploaded_img) != type(None):
            self.img_data=uploaded_img.getvalue()
            st.image(self.img_data)
            self.im0=Image.open(BytesIO(self.img_data))#.convert('RGB')
            self.im0=np.array(self.im0)

            return self.im0
        elif type(self.im0) !=type(None):
            return self.im0
        else:
            return None
    
    def predict(self):
        self.conf_thres=self.conf_selection
        st.write('Loading image')
        self.load_cv2mat(self.im0)
        st.write('Making inference')
        self.inference()

        self.img_screen=Image.fromarray(self.image).convert('RGB')
        
        self.capt='DETECTED:'
        if len(self.predicted_bboxes_PascalVOC)>0:
            for item in self.predicted_bboxes_PascalVOC:
                name=str(item[0])
                conf=str(round(100*item[-1],2))
                self.capt=self.capt+ ' name='+name+' confidence='+conf+'%, '
        st.image(self.img_screen, caption=self.capt, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
        self.image=None
    

if __name__=='__main__':
    app=Streamlit_YOLOV7()
    # GLCM Technique
#     img_gray = cv2.cvtColor(st.file_uploader(label='upload image here!'), cv2.COLOR_BGR2GRAY);
#     glcmMatrix=(greycomatrix(img_gray, [1], [0], levels=256))
#     proList = ['contrast', 'dissimilarity', 'homogeneity', 'ASM', 'energy'];
#     for j in range(0, len(proList)):
#         properties[j]=(greycoprops(glcmMatrix, prop=proList[j]))
#     features = np.array([properties[0],properties[1],properties[2],properties[3],properties[4]]);
#     filename = 'gclm_model.sav'; 
#     neigh1 = pickle.load(open(filename, 'rb'));
#     testt1=neigh1.predict(features);
#     if testt1==1:
#        st.write("crease") ;
#     elif testt1== 2:
#         st.write("crescent_gap");
#     elif testt1 == 3:
#         st.write("inclusion");
#     elif testt1 == 4 :
#        st.write("oil_spot");
#     elif testt1 == 5:
#         st.write("punching_hole");
#     elif testt1 == 6:
#         st.write("rolled_pit");
#     elif testt1 == 7:
#         st.write("silk_spot");
#     elif testt1 == 8:
#         st.write("waist folding");
#     elif testt1 == 9:
#         st.write("water_spot");
#     else:
#         st.write("welding_line");
    
    #INPUTS for YOLOV7
    img_size=1056
    path_yolov7_weights="weights/best.pt"
    path_img_i="https://storage.googleapis.com/kagglesdsdata/datasets/711184/1240214/3/img_01_424826300_00950.jpg?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=databundle-worker-v2%40kaggle-161607.iam.gserviceaccount.com%2F20230101%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20230101T155850Z&X-Goog-Expires=345600&X-Goog-SignedHeaders=host&X-Goog-Signature=6da32fb717306045b6e73491f133928858d36f594bdad2e2ebc3c213400527411a56af75210c4ea727abef72e533ad20d12299456598b241dd84beb5211f86d8cb57a1a7fa1ef6ccea262ca44ecf3dd31476c6de514d248fa0b4b4a9414e95a9e44b117a03a4a962e3aebbefaaddbe93a4c8d557b84fe112a709b222fa9bfd243dadc19840f94338401b40324e6ec2e1bc0d4056c7e5b7cfa5e43241b62e74c87036d3ba81f21ee04b737b6f816ba02437ec6a773401020afb48e660e301324deaaac2d3191b8747042c965bea7adb352d26ec3aba14d269930eeb1a6dec998b9ea1aa42819c6229e200cb3fe12a83026cdc9a4ca87933070ee92bf9568d185e"
#     with open(path_img_i, "rb") as image_file:
#         encoded_string = base64.b64encode(image_file.read())
#     st.markdown(
#     f"""
#     <style>
#     .stApp {{
#         background-image: url(data:image/{"jpg"};base64,{encoded_string.decode()});
#         background-size: cover
#     }} 
    
    #INPUTS for webapp
    app.capt="Initial Image"
    app.new_yolo_model(path_img_i,path_yolov7_weights,path_img_i)
    app.conf_thres=0.65
    app.load_model() #Load the yolov7 model
    
    app.main()
