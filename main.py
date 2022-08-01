#!/usr/bin/env python3
import struct
import random
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import websockets
import asyncio
import numpy as np
from PIL import Image
import io
import os
from collections import deque
from torchsummary import summary

gamma = 0.97

def img_to_nparray(data):
    img = Image.open(io.BytesIO(data))
    array = np.array(img)
    array = array.reshape(1, 3, 512, 514)
    return array

async def hello():
    print("connecting")
    async with websockets.connect("ws://localhost:8889") as socket:
        await train(socket)
        torch.save(model.state_dict(), "./model/model")
        print("done")

async def train(socket):
    ep = 1.0
    for i in range(1000):
        await socket.send('restart')
        r = 0
        mv_count = 0
        max_mv_count = 100
        state = 0
        past_states = deque(maxlen=100)

        await socket.send("state")
        img_data = await socket.recv()
        await socket.send("reward")
        r = await socket.recv()
        r = struct.unpack('f', r)[0]
        array = img_to_nparray(img_data)
        state = torch.from_numpy(array).float()#.cuda()

        while abs(r) != 10 and mv_count < max_mv_count:
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
            state2 = torch.from_numpy(array2).float()#.cuda()
            past_states.append((state, action, state2, r, r == 10))

        if (i+1) % 5 == 0 and len(past_states) >= 50:
            print("starting training")
            sample = random.sample(past_states, 10)
            for ps in sample:
                states = torch.cat([s for (s, a, s2, r, d) in sample])
                actions = torch.Tensor([a for (s, a, s2, r, d) in sample])
                states2 = torch.cat([s2 for (s, a, s2, r, d) in sample])
                rewards = torch.Tensor([r for (s, a, s2, r, d) in sample])
                dones = torch.Tensor([d for (s, a, s2, r, d) in sample])

                q = model(states)

                with torch.no_grad():
                    q2 = model(states2)
                    maxq = torch.max(q2)
                    Y = rewards + (gamma * maxq * (1-dones))

                maxq = torch.max(q2)

                actions = actions.long().unsqueeze(dim=1)
                X = q.gather(dim=1, index=actions)
                loss = loss_fn(X, Y)
                writer.add_scalar('loss', loss.data)
                writer.add_scalar('r', r)
                optim.zero_grad()
                loss.backward()
                optim.step()
                print("finished step")
                state = state2
                if ep > 0.1:
                    ep -= 1/100
            print('epoch ' + str(i) + ' done')


model = nn.Sequential(
    nn.Conv2d(3, 8, 3),
    nn.MaxPool2d(3),
    nn.Flatten(),
    nn.ReLU(),
    nn.Linear(231200, 500),
    nn.ReLU(),
    nn.Linear(500, 500),
    nn.ReLU(),
    nn.Linear(500, 8),
    nn.Softmax(1)
)#.cuda()
loss_fn = nn.MSELoss()
optim = torch.optim.Adam(model.parameters(), lr=1e-3)
writer = SummaryWriter()

summary(model, (3, 512, 512))

asyncio.run(hello())
