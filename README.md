# 1. Setting up virtual environment
## 1.1. Install required packages
```
pip install -r requirements.txt
pip install scikit-learn
```

## 1.2. Install tensorflow gpu
Step 1: Visit https://www.tensorflow.org/install/pip#package-location and choose the tensorflow GPU version that is compatible with your python version

Step 2: Run `python -m pip install <wheel_url>` with `<wheel_url>` is the URL you've just got from step 1.

Step 3: Wait for the installation and finish.

# 2. Hosting API
Use the following command to deploy the API
```
gunicorn --bind 0.0.0.0:5000 wsgi:app
```

# 3. Sending requests and responses
## 3.1. The API has two main functions:
### a. add
This function checks whether the input listing is duplicate with an existing listing in the database or not. If not, the new listing will be added to the database. 

### b. duplicate_check
This function returns all listings that are potentially duplicate with the input listing.

## 3.2. How to send requests:
### Step 1: Run `python request.py`
### Step 2: Choose your option and input the listing
![input](https://user-images.githubusercontent.com/57819211/119126160-e19f4080-ba5c-11eb-8b79-13210ce87bfd.png)

## 3.3 Responses: all the responses are in JSON format
### a. add
If there is an existing duplicate listing in the database:

![image](https://user-images.githubusercontent.com/57819211/121788277-3f572080-cbf6-11eb-8190-e0e09cc209e7.png)

If the input listing is completely new to the database:

![image](https://user-images.githubusercontent.com/57819211/121788444-885ba480-cbf7-11eb-9f57-dcc64cf8249c.png)

### b. duplicate_check
The response will be like:

![output_dup_check](https://user-images.githubusercontent.com/57819211/119129187-c6cecb00-ba60-11eb-8daf-f2cab6ba1946.png)





