        
# import argparse
# import io
# from PIL import Image
# import datetime

# from matplotlib.pylab import seed
# import torch
# import cv2
# import numpy as np 
# import tensorflow as tf 
# from re import DEBUG, sub
# from flask import Flask, render_template, request, redirect, send_file, url_for, Response
# from werkzeug.utils import secure_filename, send_from_directory
# import os
# import subprocess
# from subprocess import Popen
# import re
# import requests
# import shutil
# import time
# import glob


# from ultralytics import YOLO


# app = Flask(__name__)


# @app.route("/")
# def hello_world():
#     return render_template('index.html')

    
# @app.route("/", methods=["GET", "POST"])
# def predict_img():
#     if request.method == "POST":
#         if 'file' in request.files:
#             f = request.files['file']
#             basepath = os.path.dirname(__file__)
#             filepath = os.path.join(basepath,'uploads',f.filename)
#             print("upload folder is ", filepath)
#             f.save(filepath)
#             global imgpath
#             predict_img.imgpath = f.filename
#             print("printing predict_img :::::: ", predict_img)
                                               
#             file_extension = f.filename.rsplit('.', 1)[1].lower() 
            
#             if file_extension == 'jpg':
#                 img = cv2.imread(filepath)

#                 # Perform the detection
#                 model = YOLO(r'C:\Users\vijen\Desktop\yolov8_env\Object-Detection-Web-Application-with-Flask-and-YOLOv9-main\best.pt')
#                 detections =  model(img, save=True) 
#                 return display(f.filename)
            
#             elif file_extension == 'mp4': 
#                 video_path = filepath  # replace with your video path
#                 cap = cv2.VideoCapture(video_path)

#                 # get video dimensions
#                 frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#                 frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            
#                 # Define the codec and create VideoWriter object
#                 fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#                 out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (frame_width, frame_height))
                
#                 # initialize the YOLOv8 model here
#                 model = YOLO(r'C:\Users\vijen\Desktop\yolov8_env\Object-Detection-Web-Application-with-Flask-and-YOLOv9-main\best.pt')
                
#                 while cap.isOpened():
#                     ret, frame = cap.read()
#                     if not ret:
#                         break                                                      

#                     # do YOLOv8 detection on the frame here 
#                     #model = YOLO(r'C:\Users\vijen\Desktop\yolov8_env\Object-Detection-Web-Application-with-Flask-and-YOLOv9-main\best.pt')
#                     results = model(frame, save=True)  #working
#                     print(results)
#                     cv2.waitKey(1)

#                     # res_plotted = results[0].plot()
#                     # cv2.imshow("result", res_plotted)
                    
#                     # write the frame to the output video
#                     # out.write(res_plotted)

#                     if cv2.waitKey(1) == ord('q'):
#                         break

#                 return video_feed()            


            
#     folder_path = 'runs/detect'
#     subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]    
#     latest_subfolder = max(subfolders, key=lambda x: os.path.getctime(os.path.join(folder_path, x)))    
#     image_path = folder_path+'/'+latest_subfolder+'/'+f.filename 
#     return render_template('index.html', image_path=image_path)
#     #return "done"



# # #The display function is used to serve the image or video from the folder_path directory.
# @app.route('/<path:filename>')
# def display(filename):
#     folder_path = 'runs/detect'
#     subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]    
#     latest_subfolder = max(subfolders, key=lambda x: os.path.getctime(os.path.join(folder_path, x)))    
#     directory = folder_path+'/'+latest_subfolder    
#     print("printing directory: ",directory) 
#     files = os.listdir(directory)
#     latest_file = files[0]
    
#     print(latest_file)

#     filename = os.path.join(folder_path, latest_subfolder, latest_file)

#     file_extension = filename.rsplit('.', 1)[1].lower()

#     environ = request.environ
#     if file_extension == 'jpg':      
#         return send_from_directory(directory,latest_file,environ) #shows the result in seperate tab

#     else:
#         return "Invalid file format"
        
        
        

# def get_frame():
#     folder_path = os.getcwd()
#     mp4_files = 'output.mp4'
#     video = cv2.VideoCapture(mp4_files)  # detected video path
#     while True:
#         success, image = video.read()
#         if not success:
#             break
#         ret, jpeg = cv2.imencode('.jpg', image) 
      
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')   
#         time.sleep(0.1)  #control the frame rate to display one frame every 100 milliseconds: 


# # function to display the detected objects video on html page
# @app.route("/video_feed")
# def video_feed():
#     print("function called")

#     return Response(get_frame(),
#                     mimetype='multipart/x-mixed-replace; boundary=frame')
        
 
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Flask app exposing yolov9 models")
#     parser.add_argument("--port", default=5000, type=int, help="port number")
#     args = parser.parse_args()
#     model = YOLO(r'C:\Users\vijen\Desktop\yolov8_env\Object-Detection-Web-Application-with-Flask-and-YOLOv9-main\best.pt')
#     app.run(host="0.0.0.0", port=args.port) 





 #Function to start webcam

#     app.route("/webcam feed")
#     def webcam seed():
#         source O
#         cap cv2.VideoCapture
#         def generate():
#             img Image.open(io.BytesIO(frame))
# model(img, save=True) results
# print(results) cv2.waitKey(1)

# while True:
#     success, frame cap.read()
# if not success:
#     break
#     ret, buffer cv2.imencode('.jpg', frame)
#     frame buffer.tobytes()
# print (type (frame))
# model YOLO('yolovic.pt')
# res_plotted results[0].plot()
# cv2.imshow("result", res_plotted)
# if cv2.waitKey(1) ord(''):
#     break
# read image as BGR img BGR cv2.cvtColor(res_plotted, cv2.COLOR_RGB2BGR)


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Flask app exposing yolov9 models")
#     parser.add_argument("--port", default=5000, type=int, help="port number")
#     args = parser.parse_args()
#     model = YOLO(r'C:\Users\vijen\Desktop\yolov8_env\Object-Detection-Web-Application-with-Flask-and-YOLOv9-main\best.pt')
#     app.run(host="0.0.0.0", port=args.port) 










# ... (keep all your imports and initial setup)

# import argparse
# import io
# from PIL import Image
# import datetime

# from matplotlib.pylab import seed
# import torch
# import cv2
# import numpy as np
# import tensorflow as tf
# from re import DEBUG, sub
# from flask import Flask, json, render_template, request, redirect, send_file, url_for, Response
# from werkzeug.utils import secure_filename, send_from_directory
# import os
# import subprocess
# from subprocess import Popen
# import re
# import requests
# import shutil
# import time
# import glob


# from ultralytics import YOLO



# app = Flask(__name__)

# # Load model and insect data ONCE at startup
# model = YOLO(r'C:\Users\vijen\Desktop\yolov8_env\Object-Detection-Web-Application-with-Flask-and-YOLOv9-main\best.pt')
# with open('insect_data.json') as f:
#     insect_db = json.load(f)

# @app.route("/")
# def hello_world():
#     return render_template('index.html')

# @app.route("/", methods=["GET", "POST"])
# def predict_img():
#     if request.method == "POST":
#         if 'file' in request.files:
#             f = request.files['file']
#             basepath = os.path.dirname(__file__)
#             filepath = os.path.join(basepath, 'uploads', f.filename)
#             f.save(filepath)
            
#             file_extension = f.filename.rsplit('.', 1)[1].lower()
            
#             if file_extension == 'jpg':
#                 # Image Processing with Metadata
#                 results = model(filepath, save=True)
                
#                 # Get insect info from JSON
#                 detected_insects = []
#                 for box in results[0].boxes:
#                     class_id = int(box.cls[0].item())
#                     confidence = round(float(box.conf[0].item()), 2)
#                     insect_info = insect_db.get(str(class_id), {})
#                     insect_info['confidence'] = f"{confidence:.2f}%"
#                     detected_insects.append(insect_info)
                
#                 # Get result image path
#                 folder_path = 'runs/detect'
#                 subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]    
#                 latest_subfolder = max(subfolders, key=lambda x: os.path.getctime(os.path.join(folder_path, x)))    
#                 image_path = f'/runs/detect/{latest_subfolder}/{f.filename}'

#                 return render_template('results.html', 
#                                     image_path=image_path,
#                                     insects=detected_insects)



#             # ================================================
#             # KEEP EXISTING VIDEO HANDLING CODE BELOW (UNCHANGED)
#             # ================================================
#             elif file_extension == 'mp4': 
#                 video_path = filepath
#                 cap = cv2.VideoCapture(video_path)
#                 frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#                 frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#                 fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#                 out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (frame_width, frame_height))
                
#                 while cap.isOpened():
#                     ret, frame = cap.read()
#                     if not ret:
#                         break
#                     results = model(frame, save=True)
#                     if cv2.waitKey(1) == ord('q'):
#                         break

#                 return video_feed()

#     return render_template('index.html')

# @app.route('/runs/<path:filename>')
# def serve_runs(filename):
#     return send_from_directory('runs', filename)

# # ================================================
# # KEEP EXISTING VIDEO FEED CODE BELOW (UNCHANGED)
# # ================================================
# def get_frame():
#     video = cv2.VideoCapture('output.mp4')
#     while True:
#         success, image = video.read()
#         if not success:
#             break
#         ret, jpeg = cv2.imencode('.jpg', image)      
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')   
#         time.sleep(0.1)

# @app.route("/video_feed")
# def video_feed():
#     return Response(get_frame(),
#                     mimetype='multipart/x-mixed-replace; boundary=frame')

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000)















import argparse
import os
import time
import json
import cv2
from flask import Flask, render_template, request, send_from_directory, Response
from ultralytics import YOLO


app = Flask(__name__)

@app.route("/")
def hello_world():
    return render_template('index.html')


@app.route("/", methods=["GET", "POST"])
def predict_img():
    if request.method == "POST":
        if 'file' in request.files:
            f = request.files['file']
            basepath = os.path.dirname(__file__)
            filepath = os.path.join(basepath, 'uploads', f.filename)
            f.save(filepath)
            file_extension = f.filename.rsplit('.', 1)[1].lower()

            if file_extension == 'jpg':
                img = cv2.imread(filepath)

                # Load YOLO model
                model = YOLO(r'C:\Users\vijen\Desktop\yolov8_env\Object-Detection-Web-Application-with-Flask-and-YOLOv9-main\best.pt')
                results = model(img, save=True)

                # Extract class name
                boxes = results[0].boxes
                class_id = int(boxes.cls[0].item())
                class_name = results[0].names[class_id]

                # # Load insect data from JSON
                # json_path = os.path.join(basepath, 'insect_data.json')
                # with open(json_path, 'r') as f_json:
                #     insect_data = json.load(f_json)

                # info = insect_data.get(class_name, {
                #     "problems caused": "No information found.",
                #     "solutions": "No information found."
                # })
                # Load insect data
                json_path = os.path.join(basepath, 'insect_data.json')
                with open(json_path, 'r') as f_json:
                    insect_data = json.load(f_json)

# Make sure class_name is the full insect name
                print("Detected Insect Class Name:", class_name)

# Get info based on detected name
                info = insect_data.get(class_name.lower(), {
                    "benefits": "No information found.",
                    "problems caused": "No information found.",
                    "solutions": "No information found."
})

         


                # Get latest detected image
                folder_path = 'runs/detect'
                subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
                latest_subfolder = max(subfolders, key=lambda x: os.path.getctime(os.path.join(folder_path, x)))
                image_path = folder_path + '/' + latest_subfolder + '/' + f.filename

                return render_template("results.html",
                                       image_path=image_path,
                                       class_name=class_name,
                                       problems=info["problems caused"],
                                       solutions=info["solutions"],
                                       info=info)

            elif file_extension == 'mp4':
                video_path = filepath
                cap = cv2.VideoCapture(video_path)
                frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (frame_width, frame_height))
                model = YOLO(r'C:\Users\vijen\Desktop\yolov8_env\Object-Detection-Web-Application-with-Flask-and-YOLOv9-main\best.pt')

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    results = model(frame, save=True)
                    cv2.waitKey(1)

                return video_feed()

    return render_template('index.html')


@app.route('/<path:filename>')
def display(filename):
    folder_path = 'runs/detect'
    subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
    latest_subfolder = max(subfolders, key=lambda x: os.path.getctime(os.path.join(folder_path, x)))
    directory = folder_path + '/' + latest_subfolder
    files = os.listdir(directory)
    latest_file = files[0]
    file_extension = filename.rsplit('.', 1)[1].lower()

    if file_extension in ['jpg','mp4']:
        return send_from_directory(directory, latest_file, environ=request.environ)
    else:
        return "Invalid file format"


def get_frame():
    mp4_files = 'output.mp4'
    video = cv2.VideoCapture(mp4_files)
    while True:
        success, image = video.read()
        if not success:
            break
        ret, jpeg = cv2.imencode('.jpg', image)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
        time.sleep(0.1)


@app.route("/video_feed")
def video_feed():
    return Response(get_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask app exposing YOLOv9 models")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()
    app.run(host="0.0.0.0", port=args.port)
