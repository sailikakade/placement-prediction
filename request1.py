import requests

url = 'http://localhost:5000/predict_api1'

r = requests.post(url,json={'Percentage (X)':2, 'Percentage (XII)':9, 'B.E.Sem 1CGPA':6, 'B.E.Sem 2CGPA':6,'B.E.Sem 3CGPA':6,'B.E.Sem 4CGPA':6,'B.E.Sem 5CGPA':6,'B.E.Sem 6CGPA':6,B.E.CGPA})


print(r.json())