# import os
from flask import Flask,flash, render_template,url_for,request,redirect,send_file
# from wtforms import Form
# from wtforms import StringField,SubmitField
# from wtforms.validators import DataRequired, Length, Email
# from werkzeug.utils import secure_filename

# from flask_mysqldb import MySQL




# app=Flask(__name__)

# app.secret_key = os.urandom(24)


# app.config['MYSQL_HOST'] = 'localhost'
# app.config['MYSQL_USER'] = 'root'
# app.config['MYSQL_PASSWORD'] = 'locked'
# app.config['MYSQL_DB'] = 'ficcidb'

# mysql = MySQL(app)
# UPLOAD_FOLDER = 'C:/Users/Saurav Akolia/Desktop/u/MySqldatabase/Ficci/Documents/'

# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])




UPLOAD_FOLDER = 'C:/Users/Saurav Akolia/Desktop/u/MySqldatabase/Ficci/Documents/'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# @app.route("/",methods=['GET','POST'])
# def Ficci():
# 	return render_template('home.html')

from flask import Flask, request, jsonify


app = Flask(__name__)

@app.route('/', methods=['GET'])
def predict():
	return render_template("home.html")




if __name__ == '__main__':
    app.run(debug=True)