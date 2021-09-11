from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from google.protobuf.message import Error
import re
import os
from model_v2_2 import Model
import time
import requests
import numpy as np
import json
# import mysql

try:
	os.mkdir('dataset')
	# print("Directory dataset created")
except:
	pass

try:
	os.mkdir('test_images')
	# print("Directory test_images created")
except:
	pass

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///test.db'
# app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql:///ubuntu:123@127.0.0.1/test.db'
db = SQLAlchemy(app)
class Table(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    url=db.Column(db.String(200), nullable=False)
    listing=db.Column(db.String(200), nullable=False)
    class_K_mean=db.Column(db.Integer, nullable=False)
    feature_vector=db.Column(db.JSON, nullable=False)
    def __init__(self, url,listing,class_K_mean, feature_vector):
        self.url=url
        self.listing=listing
        self.class_K_mean=class_K_mean
        # self.feature_vector=json.dumps(np.array(feature_vector).tolist())
        self.feature_vector=feature_vector
    def __repr__(self):
        return  f"Object({self.id},'{self.url}','{self.listing}',{self.class_K_mean}, {self.feature_vector})"

trained_model = Model(db)

	
@app.route('/')
def home():
	return "Duplicate listing detection API"

@app.route("/add/", methods=['GET', 'POST'])
def add():
	def check_url(url):
		# Compile the ReGex
		p = re.compile(r'^(?:http|ftp)s?://' # http:// or https://
								 r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|' # domain...
								 r'localhost|' # localhost...
								 r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}|' # ...or ipv4
								 r'\[?[A-F0-9]*:[A-F0-9:]+\]?)' # ...or ipv6
								 r'(?::\d+)?' # optional port
								 r'(?:/?|[/?]\S+)$', re.IGNORECASE)

		# If the string is empty 
		# return false
		if (url == None):
			return False
	
		# Return if the string 
		# matched the ReGex
		if(re.search(p, url)):
			return True
		else:
			return False
	
	# execution_time = time.time()
	try:
		content = request.json
		url_list = content["urls"]
		listing = content["listing"]
		url_list_filtered = []
		for url in url_list:
			if check_url(str(url))==True:
				url_list_filtered.append(url)
	
		print(url_list_filtered)
		dup_found, dup_listing, elapsed_time = trained_model.check_duplicate(url_list_filtered, Table, 1, content['listing'])

		return jsonify(listing=content['listing'], duplicate_listing=dup_listing, duplicate=dup_found, time=elapsed_time), 200
	except Error as e:
		# # print(e)
		return jsonify(label="Error"), 400

@app.route("/duplicate_check/", methods=['GET', 'POST'])
def check():
	# print("here")
	def check_url(url):
		# Compile the ReGex
		p = re.compile(r'^(?:http|ftp)s?://' # http:// or https://
		r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|' # domain...
		r'localhost|' # localhost...
		r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}|' # ...or ipv4
		r'\[?[A-F0-9]*:[A-F0-9:]+\]?)' # ...or ipv6
		r'(?::\d+)?' # optional port
		r'(?:/?|[/?]\S+)$', re.IGNORECASE)
	
		# If the string is empty 
		# return false
		if (url == None):
			return False
	
		# Return if the string 
		# matched the ReGex
		if(re.search(p, url)):
			return True
		else:
			return False
	try:
		# print("tryyyyyyyyyyyyy")
		content = request.json
		url_list = content["urls"]
		url_list_filtered = []
		for url in url_list:
			if check_url(str(url))==True:
				url_list_filtered.append(url)
	
		# print(url_list_filtered)
		result = trained_model.get_duplicate_listings(url_list_filtered, Table)
		return result
	except:
		return jsonify(label="Error"), 400

if __name__ == '__main__':
	# trained_model = Model(db)
	app.run(host='0.0.0.0')