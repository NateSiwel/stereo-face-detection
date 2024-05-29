import requests
import os
from dotenv import load_dotenv
load_dotenv()

server_ip = os.getenv('server_ip')

server_url = f'https://{server_ip}:5000/upload'
api_key = '123456'

class Client():
    def send_test_request(self):
        headers = {'API-Key': api_key}
        response = requests.post(server_url, headers=headers, verify=False)
        return response.json()

if __name__ == '__main__':
    client = Client()
    response = client.send_test_request()
    print(response)

