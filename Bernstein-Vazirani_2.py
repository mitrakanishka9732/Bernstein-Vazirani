#!/usr/bin/env python
# coding: utf-8

# In[23]:


# initialization
import matplotlib.pyplot as plt
import numpy as np

# importing Qiskit
from qiskit import IBMQ, Aer
from qiskit.providers.ibmq import least_busy
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, transpile, assemble

# import basic plot tools
from qiskit.visualization import plot_histogram


# In[24]:


s = '1010'
n = len(s)


# In[25]:


#Create circuit, with n+1 quantum bits, and n classical bits(for output)
circuit = QuantumCircuit(n+1, n)

# put auxiliary in state |->
circuit.x(n)
circuit.h(n)
#circuit.z(n)

# Apply Hadamard gates 
for i in range(n):
    circuit.h(i)
    
circuit.barrier()

#algorithm 
s = s[::-1] # reverse s 
for q in range(n):
    if s[q] == '1':
        circuit.cx(q, n)
        
circuit.barrier()

#Apply Hadamard gates 
for i in range(n):
    circuit.h(i)

# Measurement
for i in range(n):
    circuit.measure(i, i)

circuit.draw(output='mpl')


# In[26]:


# use local simulator
qasm_sim = Aer.get_backend('qasm_simulator')
shots = 1024
qobj = assemble(circuit)
results = qasm_sim.run(qobj).result()
answer = results.get_counts()

plot_histogram(answer)


# In[27]:


# Load our saved IBMQ accounts and get the least busy backend device with less than or equal to 5 qubits
IBMQ.load_account()
provider = IBMQ.get_provider(hub='ibm-q')
provider.backends()
backend = least_busy(provider.backends(filters=lambda x: x.configuration().n_qubits <= 5 and
                                   x.configuration().n_qubits >= 2 and
                                   not x.configuration().simulator and x.status().operational==True))
print("least busy backend: ", backend)


# In[28]:


from qiskit.tools.monitor import job_monitor

shots = 1024
transpiled_circuit = transpile(circuit, backend)
qobj = assemble(transpiled_circuit, shots=shots)
job = backend.run(qobj)

job_monitor(job, interval=2)


# In[30]:


# Get the results from the computation
results = job.result()
answer = results.get_counts()

plot_histogram(answer)


# In[48]:


import qiskit
from qiskit.ignis.mitigation.measurement import (complete_meas_cal,
                                                 CompleteMeasFitter, 
                                                 MeasurementFilter)
from qiskit.providers.aer import noise 


# In[49]:


cal_circuits, state_labels = complete_meas_cal(qr = circuit.qregs[0], circlabel = 'measerrormitigationcal')


# In[50]:


cal_circuits[2].draw(output='mpl')


# In[54]:


len(cal_circuits)


# In[70]:


backend = qiskit.Aer.get_backend('qasm_simulator')
job = qiskit.execute(cal_circuits, backend=backend, shots=1024, optimization_level=0)
cal_results = job.result()


# In[56]:


plot_histogram(cal_results.get_counts(cal_circuits[3]))


# In[71]:


meas_fitter = CompleteMeasFitter(cal_results, state_labels)


# In[72]:


meas_fitter.plot_calibration()


# In[73]:


meas_filter = meas_fitter.filter


# In[74]:


mitigated_results = meas_filter.apply(answer)


# In[ ]:




