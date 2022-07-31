#!/usr/bin/env python3
import struct
import random
import torch
from torch import nn
import websockets
import asyncio
import numpy as np
from PIL import Image
import io
import os
from torchsummary import summary

gamma = 0.2

def img_to_nparray(data):
    img = Image.open(io.BytesIO(data))
    array = np.array(img)
    array = array.reshape(1, 3, 512, 512)
    return array

async def hello():
    print("connecting")
    async with websockets.connect("ws://localhost:8889") as socket:
        await train(socket)
        torch.save(model.state_dict(), "./model/model")
        print("done")

async def train(socket):
    ep = 0.9
    for i in range(1000):
        await socket.send('restart')
        r = 0
        mv_count = 0
        max_mv_count = 200
        state = 0

        await socket.send("state")
        img_data = await socket.recv()
        await socket.send("reward")
        r = await socket.recv()
        r = struct.unpack('f', r)[0]
        array = img_to_nparray(img_data)
        state = torch.from_numpy(array).float().cuda()

        while abs(r) != 1 and mv_count < max_mv_count:
            mv_count += 1
            q = model(state)
            np_q = q.cpu().data.numpy()
            action = np.argmax(np_q)

            if random.random() < ep:
                await socket.send(str(np.random.randint(0, 4)))
            else:
                await socket.send(str(action))
            await socket.send("state")
            img_data2 = await socket.recv()
            await socket.send('reward')
            r = await socket.recv()
            r = struct.unpack('f', r)[0]
            array2 = img_to_nparray(img_data)
            state2 = torch.from_numpy(array2).float().cuda()
            with torch.no_grad():
                q2 = model(state2)
            maxq = torch.max(q2)

            if abs(r) == 1:
                Y = r + gamma * maxq
            else:
                Y = r

            X = torch.Tensor([q.squeeze()[action]])
            X.requires_grad=True
            Y = torch.Tensor([Y])
            loss = loss_fn(X, Y)
            print(loss.data)
            optim.zero_grad()
            loss.backward()
            optim.step()
            state = state2
            if ep > 0.1:
                ep -= 1/1000


model = nn.Sequential(
    nn.Conv2d(3, 7, 3),
    nn.MaxPool2d(3),
    nn.Conv2d(7, 13, 1),
    nn.ReLU(),
    nn.Conv2d(13, 8, 1),
    nn.ReLU(),
    nn.Flatten(),
    nn.Linear(231200, 500),
    nn.ReLU(),
    nn.Linear(500, 500),
    nn.ReLU(),
    nn.Linear(500, 8),
    nn.Softmax(1)
).cuda()
loss_fn = nn.MSELoss()
optim = torch.optim.Adam(model.parameters(), lr=1e-3)

summary(model, (3, 512, 512))

asyncio.run(hello())
