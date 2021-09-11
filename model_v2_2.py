import pandas as pd
from matplotlib import pyplot as plt
import pickle
import numpy as np
import glob
import requests
import cv2
import tensorflow as tf
import tensorflow_hub as hub
import keras
import skimage
import random
import string
import PIL
import re
import json
import time
import os
from skimage import io
from kmeans_model import KMeans
from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial.distance import cdist

class Model:
		def __init__(self, db):
			self.km = KMeans()
			self.effnet_feature_vec = tf.keras.Sequential([
					hub.KerasLayer("https://hub.tensorflow.google.cn/tensorflow/efficientnet/b0/feature-vector/1",
												trainable=False)
			])
			self.effnet_feature_vec.build([None, 224, 224, 3]) # Batch input shape
			model = tf.keras.models.load_model('infused_model_4k7.h5', custom_objects={'KerasLayer': hub.KerasLayer})
			input_layer = tf.keras.Input(shape=(2560))
			output = model.get_layer(index=-3)(input_layer)
			output = model.get_layer(index=-2)(output)
			output = model.get_layer(index=-1)(output)
			self.model = tf.keras.Model(inputs=input_layer, outputs=output)
			# print(self.model.summary())
			self.threshold_1 = 0.993
			self.threshold_2 = 0.98
			self.db = db
	
		# def __init__(self, db):
		# 	self.km = km
		# 	self.effnet_feature_vec = effnet_feature_vec
		# 	self.model = model
		# 	self.threshold_1 = 0.992
		# 	self.threshold_2 = 0.99
		# 	self.db = db
		
		def get_img_from_url(self, url):
			img = io.imread(url)
			# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
			img = img[:,:,0:3]
			img = cv2.resize(img, (224, 224,))/255.0
			return img

		def get_distance(self, x):
			return 1 - cdist(x['feature'].reshape(1,-1), x['chosen_fvs'])[0]/261.17
			# return cdist(x['chosen_fvs'], x['chosen_fvs'])

		def get_count(self, q):
			count_q = q.statement.with_only_columns([func.count()]).order_by(None)
			count = q.session.execute(count_q).scalar()
			return count
		
		def url_to_f_vector(self, url):
			# define your function here, pass an url, return the Kmean class in int type
			img = io.imread(url)
			img=img[:,:,0:3]
			img=cv2.resize(img,(224,224))
			img=(img/255.0).astype('float16')
			return self.effnet_feature_vec.predict(np.array([img]))
		
		def get_duplicate_listings(self, URLs, Table):
			imgs = []
			input_filenames = []
			duration = 0
			for i, URL in enumerate(URLs):
				try:
					imgs.append(self.get_img_from_url(URL))
					input_filenames.append(f'img{i}')
				except:
					pass

			start_time = time.monotonic()
			features = self.effnet_feature_vec(np.array(imgs)).numpy()
			predictions = self.km.predict(features)
			input_filenames = np.array(input_filenames)
			# input_df = pd.DataFrame(list(zip(input_filenames, features, predictions)), columns=['input_filename', 'feature', 'cluster_id'])

			query_return = self.db.session.query(Table).filter(Table.class_K_mean.in_(predictions.tolist()))
			try:
				self.db.create_all()
			except:
				pass

			feature_vectors = []
			listings = []
			class_K_means = []
			URLs = []
			start_load_time = time.monotonic()
			for instance in query_return:
				feature_vectors.append(np.array(json.loads(instance.feature_vector)))
				listings.append(instance.listing)
				class_K_means.append(instance.class_K_mean)
				URLs.append(instance.url)
			feature_vectors = np.array(feature_vectors)
			listings = np.array(listings)
			class_K_means = np.array(class_K_means)
			URLs = np.array(URLs)
			print("Loading time:", time.monotonic() - start_load_time)

			out_dict = {}
			for cluster_id in np.unique(predictions):
				input_imgs = features[predictions==cluster_id]
				in_fns = input_filenames[predictions==cluster_id]
				mask = (class_K_means==cluster_id)
				fvs = feature_vectors[mask]
				lsts = listings[mask]
				urls = URLs[mask]
				if fvs.shape[0] < 10:
					if fvs.shape[0] == 0:
						continue
					else:
						knn = KNeighborsClassifier(n_neighbors=fvs.shape[0])
				else:
					knn = KNeighborsClassifier(n_neighbors=10)
				knn.fit(fvs, np.arange(fvs.shape[0]))
				results = knn.kneighbors(input_imgs)[1]
				for img_id, input_img in enumerate(input_imgs):
					chosen_lsts = lsts[results[img_id]]
					chosen_urls = urls[results[img_id]]
					distances = 1 - cdist(input_img.reshape(1,-1), fvs[results[img_id]])[0]/261.17
					mask = distances > self.threshold_1
					chosen_distances = distances[mask]
					chosen_lsts = chosen_lsts[mask]
					chosen_urls = chosen_urls[mask]
					for i, lst in enumerate(chosen_lsts):
						if lst not in out_dict:
							out_dict[lst] = [1, chosen_distances[i], [[in_fns[img_id], chosen_urls[i], chosen_distances[i]]]]
						else:
							out_dict[lst][1] = (out_dict[lst][1]*out_dict[lst][0] + chosen_distances[i])/(out_dict[lst][0]+1)
							out_dict[lst][0] += 1
							out_dict[lst][2].append([in_fns[img_id], chosen_urls[i], chosen_distances[i]])
			out_dict = dict(sorted(out_dict.items(), key=lambda item: item[1][0], reverse=True))
			out_dict['runtime'] = time.monotonic() - start_time
			# print(json.dumps(out_dict, indent=3))
			return json.dumps(out_dict, indent=3)

		def check_duplicate(self, URLs, Table, _add, input_listing):
			imgs = []
			input_filenames = []
			input_URLs = []
			start_down_time = time.monotonic()
			for i, URL in enumerate(URLs):
				try:
					imgs.append(self.get_img_from_url(URL))
					input_filenames.append(f'img{i}')
					input_URLs.append(URL)
				except:
					pass
			print("Downloading time:", time.monotonic() - start_down_time)
			start_time = time.monotonic()
			input_filenames = np.array(input_filenames)
			features = self.effnet_feature_vec(np.array(imgs)).numpy()
			predictions = self.km.predict(features)
			# input_df = pd.DataFrame(list(zip(input_filenames, features, predictions)), columns=['input_filename', 'feature', 'cluster_id'])
			# print("Predictions:", np.unique(predictions))
			start_query_time = time.monotonic()
			query_return = self.db.session.query(Table).filter(Table.class_K_mean.in_(predictions.tolist()))
			try:
				self.db.create_all()
			except:
				pass
			# print("Query:", time.monotonic() - start_query_time)
			start_proc_time = time.monotonic()
			feature_vectors = []
			listings = []
			class_K_means = []
			URLs = []

			start_load_time = time.monotonic()
			for instance in query_return:
				feature_vectors.append(np.array(json.loads(instance.feature_vector)))
				listings.append(instance.listing)
				class_K_means.append(instance.class_K_mean)
				URLs.append(instance.url)
			feature_vectors = np.array(feature_vectors)
			listings = np.array(listings)
			class_K_means = np.array(class_K_means)
			URLs = np.array(URLs)
			# print("Loading time:", time.monotonic() - start_load_time)

			count_dict = {}
			for cluster_id in np.unique(predictions):
				input_imgs = features[predictions==cluster_id]
				in_fns = input_filenames[predictions==cluster_id]
				mask = (class_K_means==cluster_id)
				fvs = feature_vectors[mask]
				lsts = listings[mask]
				urls = URLs[mask]
				if fvs.shape[0] < 10:
					if fvs.shape[0] == 0:
						continue
					else:
						knn = KNeighborsClassifier(n_neighbors=fvs.shape[0])
				else:
					knn = KNeighborsClassifier(n_neighbors=10)
				knn.fit(fvs, np.arange(fvs.shape[0]))
				results = knn.kneighbors(input_imgs)[1]
				for img_id, input_img in enumerate(input_imgs):
					chosen_lsts = lsts[results[img_id]]
					chosen_urls = urls[results[img_id]]
					distances = 1 - cdist(input_img.reshape(1,-1), fvs[results[img_id]])[0]/261.17
					mask = distances > self.threshold_1
					chosen_lsts_1 = chosen_lsts[mask]
					chosen_distances_1 = distances[mask]
					if chosen_lsts_1.shape[0] > 0:
						# chosen_urls_1 = chosen_urls[mask]
						# print("Dup URL:", chosen_urls_1[np.argmax(chosen_distances_1)])
						return (True, chosen_lsts_1[np.argmax(chosen_distances_1)], time.monotonic() - start_time)
					mask = distances > self.threshold_2
					chosen_lsts = chosen_lsts[mask]
					# chosen_distances = distances[mask]
					chosen_urls = chosen_urls[mask]
					for i, lst in enumerate(chosen_lsts):
						if lst not in count_dict:
							count_dict[lst] = 1
						else:
							return (True, lst, time.monotonic() - start_time)
			# Add to database
			if _add == 1:
				for i, prediction in enumerate(predictions):
					# f_vector = self.url_to_f_vector(input_URLs[i])
					# json_dump = json.dumps(f_vector[0].tolist())
					# print(np.array(json.loads(json_dump)))
					# print(np.array(json.loads(json_dump)).shape)
					# print("feature_vector shape:", len(f_vector[0].tolist()))
					# print('prediction:', int(prediction))
					row = Table(url=input_URLs[i], listing=input_listing, class_K_mean=int(predictions[i]), feature_vector=json.dumps(features[i].tolist()))
					self.db.session.add(row)
				self.db.session.commit()
				print("Committed!")
			# print("Hereeee")
			return (False, 'None', time.monotonic() - start_time)
