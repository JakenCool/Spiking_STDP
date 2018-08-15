
# coding: utf-8

# In[1]:


import brian2 as b2
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


import brian2genn
b2.set_device('genn')


# In[4]:

N = 100
eqs ='''
dv/dt = (-v + 1)/(1*ms) :1
'''
G = b2.NeuronGroup(N,eqs,method = "exact")
G.v[:] = [np.random.rand() for i in range(N)]


# In[5]:


statemon_G = b2.StateMonitor(G, 'v', record=True)

net = b2.Network()
net.add(G)
net.add(statemon_G)
net.run(10*ms)

# run(10*ms)
# fig,ax = plt.subplots()
# ax.plot(statemon_G.t/ms,statemon_G.v[2])
# fig.savefig('test-parallel.jpg')

