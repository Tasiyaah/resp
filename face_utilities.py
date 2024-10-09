import cv2
import numpy as np
import dlib
from imutils import face_utils
import imutils
from collections import OrderedDict


class Face_utilities():
    '''
    This class contains all needed functions to work with faces in a frame
    '''
    
    def __init__(self, face_width = 200):
        self.detector = None
        
        self.predictor = None 
        self.age_net = None
        self.gender_net = None
        
        self.MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
        
        
        self.desiredLeftEye=(0.35, 0.35)
        self.desiredFaceWidth = face_width
        self.desiredFaceHeight = None
        
        if self.desiredFaceHeight is None:
            self.desiredFaceHeight = self.desiredFaceWidth
        #For dlib’s 68-point facial landmark detector:
        self.FACIAL_LANDMARKS_68_IDXS = OrderedDict([
            ("mouth", (48, 68)),
            ("right_eyebrow", (17, 22)),
            ("left_eyebrow", (22, 27)),
            ("right_eye", (36, 42)),
            ("left_eye", (42, 48)),
            ("nose", (27, 36)),
            ("jaw", (0, 17))
        ])

        #For dlib’s 5-point facial landmark detector:
        self.FACIAL_LANDMARKS_5_IDXS = OrderedDict([
            ("right_eye", (2, 3)),
            ("left_eye", (0, 1)),
            ("nose", (4))   
        ])
        
        #last params
        self.last_age = None
        self.last_gender = None
        self.last_rects = None
        self.last_shape = None
        self.last_aligned_shape = None
        
        #FACIAL_LANDMARKS_IDXS = FACIAL_LANDMARKS_68_IDXS
    
    def face_alignment(self, frame, shape):
       
        if (len(shape)==68):
			# extract the left and right eye (x, y)-coordinates
            (lStart, lEnd) = self.FACIAL_LANDMARKS_68_IDXS["left_eye"]
            (rStart, rEnd) = self.FACIAL_LANDMARKS_68_IDXS["right_eye"]
        else:
            (lStart, lEnd) = self.FACIAL_LANDMARKS_5_IDXS["left_eye"]
            (rStart, rEnd) = self.FACIAL_LANDMARKS_5_IDXS["right_eye"]
        
        leftEyePts = shape[lStart:lEnd]
        rightEyePts = shape[rStart:rEnd]
        # compute the center of mass for each eye
        leftEyeCenter = leftEyePts.mean(axis=0).astype("int")
        rightEyeCenter = rightEyePts.mean(axis=0).astype("int")
        
        # compute the angle between the eye centroids
        dY = rightEyeCenter[1] - leftEyeCenter[1]
        dX = rightEyeCenter[0] - leftEyeCenter[0]
        angle = np.degrees(np.arctan2(dY, dX)) - 180
        
        desiredRightEyeX = 1.0 - self.desiredLeftEye[0]
       
        dist = np.sqrt((dX ** 2) + (dY ** 2))
        desiredDist = (desiredRightEyeX - self.desiredLeftEye[0])
        desiredDist *= self.desiredFaceWidth
        scale = desiredDist / dist
        
        # compute center (x, y)-coordinates (i.e., the median point)
        # between the two eyes in the input image
        eyesCenter = (int((leftEyeCenter[0] + rightEyeCenter[0]) // 2),
            int((leftEyeCenter[1] + rightEyeCenter[1]) // 2))
        
        # grab the rotation matrix for rotating and scaling the face
        M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)
        
        # update the translation component of the matrix
        tX = self.desiredFaceWidth * 0.5
        tY = self.desiredFaceHeight * self.desiredLeftEye[1]
        M[0, 2] += (tX - eyesCenter[0])
        M[1, 2] += (tY - eyesCenter[1])
        
        # apply the affine transformation
        (w, h) = (self.desiredFaceWidth, self.desiredFaceHeight)
        aligned_face = cv2.warpAffine(frame, M, (w, h),
            flags=cv2.INTER_CUBIC)
            
        #print("1: aligned_shape_1 = {}".format(aligned_shape))
        #print(aligned_shape.shape)
        
        if(len(shape)==68):
            shape = np.reshape(shape,(68,1,2))
        
                    
        else:
            shape = np.reshape(shape,(5,1,2))
          
                  
        aligned_shape = cv2.transform(shape, M)
        aligned_shape = np.squeeze(aligned_shape)        
      
        return aligned_face,aligned_shape
    
    def face_detection(self, frame):
        
        if self.detector is None:
            self.detector = dlib.get_frontal_face_detector()
        
        if frame is None:
            return
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #get all faces in the frame
        rects = self.detector(gray, 0)
        # to get the coords from a rect, use: (x, y, w, h) = face_utils.rect_to_bb(rects[0])
        
        return rects
    
    
    def get_landmarks(self, frame, type):
     
        if self.predictor is None:
            print("[INFO] load " + type + " facial landmarks model ...")
            self.predictor = dlib.shape_predictor("shape_predictor_" + type + "_face_landmarks.dat")
            print("[INFO] Load model - DONE!")
        
        if frame is None:
            return None, None
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = self.face_detection(frame)
        
        if len(rects)<0 or len(rects)==0:
            return None, None
            
        shape = self.predictor(gray, rects[0])
        shape = face_utils.shape_to_np(shape)
        
       
        return shape, rects
    
    def ROI_extraction(self, face, shape):
       
        if (len(shape)==68):
            ROI1 = face[shape[29][1]:shape[33][1], #right cheek
                    shape[54][0]:shape[12][0]]
                    
            ROI2 =  face[shape[29][1]:shape[33][1], #left cheek
                    shape[4][0]:shape[48][0]]
            
            ROI3 = face[shape[29][1]:shape[33][1], #nose
                        shape[31][0]:shape[35][0]]
        else:
            ROI1 = face[int((shape[4][1] + shape[2][1])/2):shape[4][1], #right cheek
                    shape[2][0]:shape[3][0]]
                    
            ROI2 =  face[int((shape[4][1] + shape[2][1])/2):shape[4][1], #left cheek
                    shape[1][0]:shape[0][0]]
            ROI3 = face[int((shape[4][1] + shape[2][1]) / 2):shape[4][1], #nose
                   shape[1][0]:shape[0][0]]
                   
                
        return ROI1, ROI2, ROI3      
   
    def facial_landmarks_remap(self,shape):
      
        
        remapped_shape = shape.copy()
        # left eye brow
        remapped_shape[17] = shape[26]
        remapped_shape[18] = shape[25]
        remapped_shape[19] = shape[24]
        remapped_shape[20] = shape[23]
        remapped_shape[21] = shape[22]
        # right eye brow
        remapped_shape[22] = shape[21]
        remapped_shape[23] = shape[20]
        remapped_shape[24] = shape[19]
        remapped_shape[25] = shape[18]
        remapped_shape[26] = shape[17]
        # neatening 
        remapped_shape[27] = shape[0]
        
        remapped_shape = cv2.convexHull(shape)
        #to use remapped_shape
        #mask = np.zeros((face_frame.shape[0], face_frame.shape[1])) #create a black rectangle mask with w, h of the face
        #cv2.fillConvexPoly(mask, remapped_shape[0:27], 1) #fill convex to the mask with remapped_shape
        
        return remapped_shape       
    
    def no_age_gender_face_process(self, frame, type):
     
        if(type=="5"):
            shape, rects = self.get_landmarks(frame, "5")
            
            if shape is None:
                return None
        else:    
            shape, rects = self.get_landmarks(frame, "68")
            if shape is None:
                return None
        
        (x, y, w, h) = face_utils.rect_to_bb(rects[0])
        
        face = frame[y:y+h,x:x+w]
        aligned_face,aligned_shape = self.face_alignment(frame, shape)
        
                
        return rects, face, shape, aligned_face, aligned_shape
        
    def face_full_process(self, frame, type, face_detect_on, others_on):
        #assign from last params
        rects = self.last_rects
        shape = self.last_shape
        aligned_shape = self.last_aligned_shape
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if face_detect_on:
            if(type=="5"):
                shape, rects = self.get_landmarks(frame, "5")
                #mask = None
                
                if shape is None:
                    return None
            else:    
                shape, rects = self.get_landmarks(frame, "68")
               
                if shape is None:
                    return None
        
        (x, y, w, h) = face_utils.rect_to_bb(rects[0])
        
        face = frame[y:y+h,x:x+w]        
        
        
        aligned_face, aligned_face = self.face_alignment(frame, shape)
        
        self.last_rects = rects
        self.last_shape = shape
        self.last_aligned_shape = aligned_shape
        
        #return rects, face,  shape, aligned_face, mask
        return rects, face,  shape, aligned_face, aligned_shape
