import cirq
# from numpy.testing._private.utils import clear_and_catch_warnings
# import numpy as np
# import tensorflow as tf
# import matplotlib.pyplot as plt

print("\n\n-------------------—Beginning of Program-------------------—")

def gate1():
    description = "Example of Hadamard gate acting on one qubit"
    qubit = cirq.NamedQubit("q0")
    circuit = cirq.Circuit(cirq.H(qubit))
    return description, circuit

def gate2():
    description = "Example of a not gate acting on one qubit"
    qubit = cirq.NamedQubit("q0")
    circuit = cirq.Circuit(cirq.X(qubit), cirq.measure(qubit, key='q0'))
    return description, circuit

def gate3 ():
    description = "Example of a controlled-not gate"
    q0, q1= cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.CNOT(q1, q0))
    circuit.append([cirq.measure(q0, key='q0'), cirq.measure(q1, key='q1')])
    return description, circuit

def gate4 ():
    description = ''' Example of a measurement. Note that for the input mixed state √1 (|0⟩ + |1⟩), 
    it is unknown what the result of the measurement will be. All that is known is that the result 
    has equal probability of being |0⟩ or |1⟩'''
    qubit = cirq.NamedQubit("q0")
    circuit = cirq.Circuit(cirq.H(qubit), cirq.measure(qubit, key='q0'))
    return description, circuit

def gate5 ():
    description = "Example of a controlled-not gate"
    return gate3() #This is the same as gate 3 (measuring CNOT)

def gate6 ():
    description = ''' Example of a measurement. Note that for the input mixed state √1 (|0⟩ + |1⟩), 
    it is unknown what the result of the measurement will be. All that is known is that the result 
    has equal probability of being |0⟩ or |1⟩'''
    return gate4() #This is the same as gate 4 (equal superposition)

# Not sure how to do 7 -- seems like a full system
# def gate7 ():
#     description = "Boolean circuit performing function f : F2n → F2m"
#     return

def gate8 ():
    description = "Toffoli Gate"
    q0, q1, q2 = cirq.LineQubit.range(3)
    circuit = cirq.Circuit(cirq.TOFFOLI(q0, q1, q2))
    circuit.append([cirq.measure(q0, key='q0'), cirq.measure(q1, key='q1'), cirq.measure(q2, key='q2')])
    return description, circuit

def gate9 ():
    description = "Decomposition of a Toffoli gate"
    _, circuit = gate8()
    return description, circuit #Not sure how to do this otherwise

def gate10 () :
    description = "Toffoli gate as an and gate"
    q0, q1, q2 = cirq.LineQubit.range(3)
    ops = [
        cirq.amplitude_damp(1.0)(q2), 
        cirq.TOFFOLI(q0, q1, q2),
        cirq.measure(q0, key='x'), 
        cirq.measure(q1, key='y'), 
        cirq.measure(q2, key='0')
    ]
    circuit = cirq.Circuit(ops)
    return description, circuit

def gate11 ():
    description = "A Toffoli gate as an or gate"
    q0, q1, q2 = cirq.LineQubit.range(3)
    ops = [
        cirq.amplitude_damp(1.0)(q2), 
        cirq.X(q0),
        cirq.X(q1),
        cirq.TOFFOLI(q0, q1, q2),
        cirq.measure(q0, key='q0'), 
        cirq.measure(q1, key='q1'), 
        cirq.measure(q2, key='0')
    ]
    circuit = cirq.Circuit(ops)
    return description, circuit

def gate12 ():  #Toffoli gate as fanout.
    description = "Toffoli gate as fanout"
    q0, q1, q2 = cirq.LineQubit.range(3)
    ops = [
        cirq.amplitude_damp(1.0)(q1), #So q1 is always 0 
        cirq.amplitude_damp(1.0)(q2), #So q2 is always 0 
        cirq.X(q1),
        cirq.TOFFOLI(q0, q1, q2),
        cirq.measure(q0, key='x0'), 
        cirq.measure(q1, key='1'), 
        cirq.measure(q2, key='x1')
    ]
    circuit = cirq.Circuit(ops)
    return description, circuit
    

gatesWith1Input = [gate1, gate2, gate4, gate6]
gatesWith2Inputs = [gate3, gate5]
gatesWith3Inputs = [gate8, gate9, gate10, gate11, gate12]

# gatesToRun = gatesWith1Input + gatesWith2Inputs + gatesWith3Inputs
# gatesToRun = gatesWith1Input
# gatesToRun = gatesWith2Inputs
gatesToRun = gatesWith3Inputs
# gatesToRun = [gate12]

count = 1

for gate in gatesToRun:
    print("\n---------------------Circuit {}---------------------".format(count)) 
    description, circuit = gate()
    print("\n{}\n\n{}".format(description, circuit))

    sim = cirq.Simulator()
    res = sim.simulate(circuit, initial_state=0b100)
    print("\n{}\n".format(res))
    count = count + 1