import requests

# Go to: https://publicapis.io/apis
# Go to: py

response = requests.get('https://api.stackexchange.com/2.3/questions?order=desc&sort=activity&site=stackoverflow')
# print(response.json())

# print(response.json()['items'])

for data in response.json()['items']:
    print('=' * 50)
    print('Title: ', data['title'])
    print('Link: ', data['link'])
    print('Answer Count: ', data['answer_count'])
    print('=' * 50, '\n')
