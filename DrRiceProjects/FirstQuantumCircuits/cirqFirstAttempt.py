import cirq
# from numpy.testing._private.utils import clear_and_catch_warnings
import numpy as np
# import tensorflow as tf
# import matplotlib.pyplot as plt

print("\n\n-------------------—Beginning of Program-------------------—")

def gate1(measure = True):
    description = "Example of Hadamard gate acting on one qubit"
    qubit = cirq.NamedQubit("q0")
    circuit = cirq.Circuit(cirq.H(qubit))
    if measure:
        circuit.append(cirq.measure(qubit, key='q0'))
    return description, circuit

def gate2(measure = True):
    description = "Example of a not gate acting on one qubit"
    qubit = cirq.NamedQubit("q0")
    circuit = cirq.Circuit(cirq.X(qubit))
    if measure:
        circuit.append(cirq.measure(qubit, key='q0'))
    return description, circuit

def gate3 (measure = True):
    description = "Example of a controlled-not gate"
    q0, q1= cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.CNOT(q0, q1))
    if measure:
        circuit.append([cirq.measure(q0, key='q0'), cirq.measure(q1, key='q1')])
    return description, circuit

def gate4 (measure = True):
    description = ''' Example of a measurement. Note that for the input mixed state √1 (|0⟩ + |1⟩), 
    it is unknown what the result of the measurement will be. All that is known is that the result 
    has equal probability of being |0⟩ or |1⟩'''
    qubit = cirq.NamedQubit("q0")
    circuit = cirq.Circuit(cirq.H(qubit))
    if measure:
        circuit = circuit.append(cirq.measure(qubit, key='q0'))
    return description, circuit

def gate5 (measure = True):
    description = "Example of a controlled-not gate"
    return gate3(measure) #This is the same as gate 3 (measuring CNOT)

def gate6 (measure = True):
    description = ''' Example of a measurement. Note that for the input mixed state √1 (|0⟩ + |1⟩), 
    it is unknown what the result of the measurement will be. All that is known is that the result 
    has equal probability of being |0⟩ or |1⟩'''
    return gate4(measure) #This is the same as gate 4 (equal superposition)

# Not sure how to do 7 -- seems like a full system
# def gate7 ():
#     description = "Boolean circuit performing function f : F2n → F2m"
#     return

def gate8 (measure = True):
    description = "Toffoli Gate"
    q0, q1, q2 = cirq.LineQubit.range(3)
    circuit = cirq.Circuit(cirq.TOFFOLI(q0, q1, q2))
    if measure:
        circuit.append([cirq.measure(q0, key='q0'), cirq.measure(q1, key='q1'), cirq.measure(q2, key='q2')])
    return description, circuit

def gate9 (measure = True):
    description = "Decomposition of a Toffoli gate"
    _, circuit = gate8(measure)
    return description, circuit #Not sure how to do this otherwise

def gate10 (measure = True) :
    description = "Toffoli gate as an and gate"
    q0, q1, q2 = cirq.LineQubit.range(3)
    ops = [
        cirq.amplitude_damp(1.0)(q2), 
        cirq.TOFFOLI(q0, q1, q2)
    ]
    if measure:
        ops = ops + [
            cirq.measure(q0, key='x'), 
            cirq.measure(q1, key='y'), 
            cirq.measure(q2, key='0')
        ]
    circuit = cirq.Circuit(ops)
    return description, circuit

def gate11 (measure = True):
    description = "A Toffoli gate as an or gate"
    q0, q1, q2 = cirq.LineQubit.range(3)
    ops = [
        cirq.amplitude_damp(1.0)(q2), 
        cirq.X(q0),
        cirq.X(q1),
        cirq.TOFFOLI(q0, q1, q2),
    ]
    if measure:
        ops = ops + [
            cirq.measure(q0, key='q0'), 
            cirq.measure(q1, key='q1'), 
            cirq.measure(q2, key='0')   
        ]
    circuit = cirq.Circuit(ops)
    return description, circuit

def gate12 (measure = True):  #Toffoli gate as fanout.
    description = "Toffoli gate as fanout"
    q0, q1, q2 = cirq.LineQubit.range(3)
    ops = [
        cirq.amplitude_damp(1.0)(q1), #So q1 is always 0 
        cirq.amplitude_damp(1.0)(q2), #So q2 is always 0 
        cirq.X(q1),
        cirq.TOFFOLI(q0, q1, q2),
    ]
    if measure:
        ops += [
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
gatesToRun = gatesWith1Input
# gatesToRun = gatesWith2Inputs
# gatesToRun = gatesWith3Inputs
# gatesToRun = [gate12]

count = 1

states1 = [0b0, 0b1]
states2 = [0b00, 0b10, 0b01, 0b11]
states3 = [0b000, 0b001, 0b011, 0b111, 0b110, 0b101, 0b010, 0b100]
sim = cirq.Simulator()

for gate in gatesToRun:
    print("\n---------------------Circuit {} || Gate: {}---------------------".format(count, gate)) 
    description, circuit = gate(False)
    print("\n{}\n\n{}".format(description, circuit))

    
    for state in states1:
        res = sim.simulate(circuit, initial_state=state)
        # resRun = sim.run(circuit, repetitions=100)
        print("\n{}\nRES: {}\nNotation: {}\nDensity Matrix: {}\n".format(bin(state), res, res.dirac_notation(), np.around(res.final_state_vector, 3)))
    count = count + 1