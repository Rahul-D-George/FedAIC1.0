
# THIS IS HIGHLY DEPRECATED, IGNORE


import requests

# Server URL
url = "https://apollo.doc.ic.ac.uk:6296/upload"

# Path to the file you want to upload
file_path = 'rahul_test.txt'

# Set headers for communication
headers = {'Content-Disposition': 'attachment; filename="rahul.txt"'}

# Read the content of the file into server
with open(file_path, 'rb') as file:
    files = {'file': (file_path, file)}
    response = requests.post(url, headers=headers, files=files, verify=False)

# Validate response
print("Response:", response.text)
