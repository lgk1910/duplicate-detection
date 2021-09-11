import requests
import string
import json
import re

BASE = "http://0.0.0.0:5000"

print("Choose one of the following options:")
print("1. add")
print("2. duplicate_check")

option = input("Your choice: ")
if (option!="1" and option!="2"):
    raise Exception("Invalid option")
input_urls = input("URLs (seperated by comma): ")
listing = input("Listing name: ")

import time
start_time = time.monotonic()

url_list = []
for i in re.split(",\s*", input_urls):
    url_list.append(i)

# print('URL list: ' + str(url_list))
try:
    if option=="1":
        res = requests.post(BASE + '/add/', json={"listing": listing, "urls":url_list})
    else:
        res = requests.post(BASE + '/duplicate_check/', json={"listing": listing, "urls":url_list})
    parsed_json_res = res.json()
    print(json.dumps(parsed_json_res, indent=4, sort_keys=True))
except:
    print("Failed to send request")
    
print("Total time:", time.monotonic()-start_time)
