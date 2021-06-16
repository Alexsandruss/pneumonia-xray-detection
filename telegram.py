from telegram_api import Bot
from time import time, sleep
import os
import numpy as np
import cv2
from keras.models import load_model
import base64


def percent_prob(prob):
    return round(float(prob) * 100, 2)


os.system('mkdir -p photos')

with open('token.txt', 'r') as token_file:
    token = token_file.read()
    token = token.replace('\n', '')

IMG_SIZE = 150

model_names = os.listdir('models')
models = [load_model('models/' + model_path) for model_path in model_names]
print(f'{len(models)} MODELS LOADED')

bot = Bot(token)
while True:
    messages = bot.get_last_messages()
    for message in messages:
        try:
            if 'text' in message.keys():
                if '/info' in message["text"]:
                    bot.send_message(
                        message["chat"]["id"],
                        f'Pneumonia prediction bot\nLoaded models: {len(models)}')
            if 'photo' in message.keys():
                photo_url = "https://api.telegram.org/file/bot{}/{}".format(
                    token, bot.get_file(message['photo'][-1]['file_id'])['file_path']
                )
                photo_path = f'photos/photo_{int(time())}.jpg'
                os.system(f'curl -o {photo_path} {photo_url}')
                photo_arr = cv2.imread(photo_path, cv2.IMREAD_GRAYSCALE)
                photo_arr = cv2.resize(photo_arr, (IMG_SIZE, IMG_SIZE))
                photo_arr = photo_arr.reshape((1, IMG_SIZE, IMG_SIZE, 1))

                probabilities = np.zeros(shape=(len(models), ), dtype=np.float64)
                for i, model in enumerate(models):
                    probability = model.predict_proba(photo_arr)
                    probability = float(probability[0])
                    probabilities[i] = probability
                mean_probability = float(probabilities.mean())
                if mean_probability < 0.5:
                    response = 'Pneumonia is found'
                else:
                    response = 'Pneumonia is not found'

                bot.send_message(message["chat"]["id"], response)
        except Exception as e:
            print('EXCEPTION:', e)
    sleep(1)
