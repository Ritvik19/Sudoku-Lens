from flask import Flask, render_template, request, send_from_directory
import utils, cv2
import numpy as np

app = Flask(__name__)

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
    app.run(debug=True)