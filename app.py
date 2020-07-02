from flask import Flask, render_template, request, send_from_directory
import cv2, imutils
import sys
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

def resize(img, max_dim=640):
    h, w = img.shape[:2]
    if h > w:
        hnew = max_dim
        wnew = (max_dim*w)//h
    else:
        wnew = max_dim
        hnew = (max_dim*h)//w

    img = cv2.resize(img, (wnew, hnew))
    return img

def getGrid(img):
    grayImageBlur = cv2.blur(img,(3,3))
    edgedImage = cv2.Canny(grayImageBlur, 100, 300, 3)
    
    allContours = cv2.findContours(edgedImage.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    allContours = imutils.grab_contours(allContours)
    allContours = sorted(allContours, key=cv2.contourArea, reverse=True)[:1]
    
    perimeter = cv2.arcLength(allContours[0], True) 
    ROIdimensions = cv2.approxPolyDP(allContours[0], 0.02*perimeter, True)
    
    try:
        ROIdimensions = ROIdimensions.reshape(4,2)
    except:
        return img
    rect = np.zeros((4,2), dtype="float32")
    
    s = np.sum(ROIdimensions, axis=1)
    rect[0] = ROIdimensions[np.argmin(s)]
    rect[2] = ROIdimensions[np.argmax(s)]

    diff = np.diff(ROIdimensions, axis=1)
    rect[1] = ROIdimensions[np.argmin(diff)]
    rect[3] = ROIdimensions[np.argmax(diff)]

    (tl, tr, br, bl) = rect

    widthA = np.sqrt((tl[0] - tr[0])**2 + (tl[1] - tr[1])**2 )
    widthB = np.sqrt((bl[0] - br[0])**2 + (bl[1] - br[1])**2 )
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt((tl[0] - bl[0])**2 + (tl[1] - bl[1])**2 )
    heightB = np.sqrt((tr[0] - br[0])**2 + (tr[1] - br[1])**2 )
    maxHeight = max(int(heightA), int(heightB))
    
    dst = np.array([
        [0,0],
        [maxWidth-1, 0],
        [maxWidth-1, maxHeight-1],
        [0, maxHeight-1]], dtype="float32")
    transformMatrix = cv2.getPerspectiveTransform(rect, dst)
    
    return cv2.warpPerspective(img, transformMatrix, (maxWidth, maxHeight)) 

def getNums(cropped):
    region_1d = [(i*70+i+i//3, (i+1)*70+i+i//3) for i in range(9)]
    region_2d = [(x, y) for x in region_1d for y in region_1d]

    nums = []

    for dim in region_2d:
        p, q = dim
        x1, x2 = p
        y1, y2 = q
        num_i = cropped[x1:x2, y1:y2]
        nums.append(num_i)
        
    return nums

def recognizeDigits(nums, crop_size=7, mean_th=240):
    sudoku = []
    for num_i in nums:
        ret, temp = cv2.threshold(num_i, 160, 255, cv2.THRESH_BINARY)
        num_cr = temp[crop_size:-crop_size, crop_size:-crop_size]
        mn = num_cr.mean()
        if mn < mean_th:
            sudoku.append(model.predict_classes(cv2.resize(num_cr, (28, 28)).reshape(1, 28, 28, 1))[0])
        else:
            sudoku.append('.')
    return sudoku       

def smart_solve(grids):
    grids = grids.copy()
    for _ in range((grids == 0).sum((1, 2)).max()):
        preds = np.array(solver.predict(to_categorical(grids)))  # get predictions
        probs = preds.max(2).T  # get highest probability for each 81 digit to predict
        values = preds.argmax(2).T + 1  # get corresponding values
        zeros = (grids == 0).reshape((grids.shape[0], 81))  # get blank positions

        for grid, prob, value, zero in zip(grids, probs, values, zeros):
            if any(zero):  # don't try to fill already completed grid
                where = np.where(zero)[0]  # focus on blanks only
                confidence_position = where[prob[zero].argmax()]  # best score FOR A ZERO VALUE (confident blank)
                confidence_value = value[confidence_position]  # get corresponding value
                grid.flat[confidence_position] = confidence_value  # fill digit inplace
    return grids[0]


app = Flask(__name__)

model = load_model('neuralnetworks/lens-digi.h5')
solver = load_model('neuralnetworks/sudoku.h5')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/confirm', methods=['GET', 'POST'])
def confirm():
    f = request.files['file']  
    f.save('static/'+f.filename)
    
    img = cv2.imread('static/'+f.filename, 0)
    img = utils.resize(img, 640)
    cropped = cv2.resize(utils.getGrid(img), (640, 640))
    nums = utils.getNums(cropped)
    nums = utils.recognizeDigits(nums)
    quiz = []
    for i in range(9):
        quiz.append(nums[i*9:(i+1)*9])

    return render_template("confirm.html", quiz=quiz, filename=f.filename) 
    
@app.route('/solve', methods=['GET', 'POST'])
def solve():
    sudoku = list(request.form.values())
    quiz = []
    for i in range(9):
        quiz.append(list(map(int, [0 if x=='.' else x for x in sudoku[i*9:(i+1)*9]])))
    quiz = np.array([quiz])
    solution = utils.smart_solve(quiz)        
    return render_template('solve.html', solution=solution)

if __name__ == '__main__':
    app.run()