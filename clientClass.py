import requests
import os
from dotenv import load_dotenv
import json
from types import SimpleNamespace
load_dotenv()

server_ip = os.getenv('server_ip')

server_url = f'https://{server_ip}:5000/upload'
api_key = '123456'

class ClientClass():
    def rectify_frames(self, imgL, imgR):
        headers = {'API-Key': api_key}
        imgL = imgL.tolist()
        imgR = imgR.tolist()
        body = {"imgL":imgL, "imgR":imgR}
        response = requests.post(server_url, headers=headers, json=body, verify=False)
        response = response.json()
        if response['success'] is True:
            return response['imgL'], response['imgR']
        return "Invalid Request"

if __name__ == '__main__':
    client = ClientClass()
    response = client.send_test_request()
    print(response)

