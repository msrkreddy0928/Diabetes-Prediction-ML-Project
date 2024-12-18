from flask import Flask, render_template

newapp = Flask(__name__)

@newapp.route("/")
def home():
    return 'welcome to flask'

@newapp.route('/h')
def hello():
    return "hello shiva"

if __name__ == '__main__':
    newapp.run(debug=True)
    

    