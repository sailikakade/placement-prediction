import requests

url = 'http://localhost:5000/predict_api2'

r = requests.post(url,json={'Percentage (X)':2, 'Percentage (XII/Diploma)':9, 'B.E. CGPA':6})


print(r.json())