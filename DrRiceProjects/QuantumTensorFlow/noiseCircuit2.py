import cirq
import numpy as np

class BitAndPhaseFlipChannel(cirq.SingleQubitGate):
    def __init__(self, p: float) -> None:
        self._p = p

    def _mixture_(self):
        ps = [1.0 - self._p, self._p]
        ops = [cirq.unitary(cirq.I), cirq.unitary(cirq.Y)]
        return tuple(zip(ps, ops))

    def _has_mixture_(self) -> bool:
        return True

    def _circuit_diagram_info_(self, args) -> str:
        return f"BitAndPhaseFlip({self._p})"

def threeQubitCorrection(is_one = False):
    q1, q2, q3, q4, q5, extra_qubit = cirq.LineQubit.range(6)
    circuit = cirq.Circuit()
    
    # translates all of the bits into 1's
    if is_one:
        circuit.insert(0, cirq.Moment([
            cirq.X(q1),
            cirq.X(q2),
            cirq.X(q3)
        ]))
    
    #setting the values of the ancilla bits to the correct values
    circuit.append(cirq.CNOT(q1, q4))
    circuit.append(cirq.CNOT(q2, q4))
    circuit.append(cirq.CNOT(q2, q5))
    circuit.append(cirq.CNOT(q3, q5))

    #error correction:
    circuit.append(cirq.CCNOT(q4, q5, extra_qubit)) #if they are both turned to one, then we know that q2 was flipped
    circuit.append(cirq.CNOT(extra_qubit, q2))     #checking so that if both of the ancilla are 1, then only q2 is changed
    circuit.append(cirq.CNOT(extra_qubit, q4))
    circuit.append(cirq.CNOT(extra_qubit, q5))
    circuit.append(cirq.CNOT(q4, q1))              #q4/5 will hold their state if both of them are not one
    circuit.append(cirq.CNOT(q5, q3))              #in which case we correct their respective qubit
    

    circuit.append(cirq.Moment([
        cirq.measure(q1, key="mi1"),
        cirq.measure(q2, key="mi2"),
        cirq.measure(q3, key="mi3")
    ]))

    return circuit


def noisyCircuit(probability = 0.1, measure=True, bitStart = False, depolarize=False, bitPhase=False, bitEnd = False):
    q0 = cirq.NamedQubit('a')
    q1 = cirq.NamedQubit('b')
    q2 = cirq.NamedQubit('c')
    q3 = cirq.NamedQubit('d')
    qreg=list([q0, q1, q2, q3])
    bitFlipInd = 1
    depInd = 1
    bitPhaseFlipInd = 2
    measureInd = 4

    circuit = cirq.Circuit(
        cirq.Moment([cirq.X(q1)]),
        # cirq.Moment([cirq.H.on_each(qreg)]),
        cirq.qft(*qreg),
        cirq.qft(*qreg, inverse=True),
        # cirq.reset(qreg[1]),
    )

    if bitStart:
        depInd += 1
        measureInd += 1
        bitPhaseFlipInd +=  1
        circuit.insert(bitFlipInd, cirq.Moment([cirq.bit_flip(p=probability).on_each(qreg)]))

    if depolarize:
        measureInd += 1
        bitPhaseFlipInd +=  1
        circuit.insert(depInd,  cirq.depolarize(p=probability).on_each(qreg))

    if bitPhase:
        measureInd += 1
        circuit.insert(bitPhaseFlipInd,  BitAndPhaseFlipChannel(p=probability).on_each(qreg[1::2]))

    if measure:
        circuit.insert(measureInd, cirq.Moment([        
            cirq.measure(q0, key='m0'), 
            cirq.measure(q1, key='m1'), 
            cirq.measure(q2, key='m2'), 
            cirq.measure(q3, key='m3'), 
            ]))

    if bitEnd:
       circuit.append(cirq.bit_flip(p=probability).controlled(1).on(*qreg[2:]))

    print("Circuit with multiple channels:\n")
    print(circuit)
    return circuit


def run_noisy():

    probabilities = [0, 0.01, 0.1, 0.2, 0.3, 0.5]
    iterations = 1000
    keys_measure = ["m0", "m1", "m2", "m3"]
    configurations = [(True, False, False, False, False), (True, True, False, False, False), (True, False, True, False, False), (True, False, False, True, False)]
    # measure, bitstart, depolarize, bitphase, cbitflip

    for config in configurations:

        print("\n\nCONFIG :: {}".format(config))

        for count, probability in enumerate(probabilities):
            
            print("PROBABILITY :: {}".format(probability))

            circuit = noisyCircuit(probability=probability, measure=config[0], bitStart=config[1], depolarize=config[2], bitPhase=config[3], bitEnd=config[4]) # 0, 0.01, 0.1, 0.2, 0.3
            sim = cirq.DensityMatrixSimulator()

            result = sim.run(circuit, repetitions=iterations)

            print("A: {} | PROB OF GETTING ONE: %{}".format(result.histogram(key='m0'), round(100 * result.histogram(key='m0')[1] / iterations)))
            print("B: {} | PROB OF GETTING ONE: %{}".format(result.histogram(key='m1'), round(100 * result.histogram(key='m1')[1] / iterations)))
            print("C: {} | PROB OF GETTING ONE: %{}".format(result.histogram(key='m2'), round(100 * result.histogram(key='m2')[1] / iterations)))
            print("D: {} | PROB OF GETTING ONE: %{}".format(result.histogram(key='m3'), round(100 * result.histogram(key='m3')[1] / iterations)))
            print("\n")


def run2_noisy():   #returns 1 if the output after majority voting is 1
    circuit = threeQubitCorrection(is_one = True)
    print(circuit)
    simulator = cirq.DensityMatrixSimulator()
    res = simulator.run(circuit)
    print("RESULTS: \n{}".format(res))
    count1 = 0

    measurement_keys = ["mi1", "mi2", "mi3"]
    for key in measurement_keys:
        if res.histogram(key=key)[1]:
            count1 = count1 + 1

    output = 1 if count1 > 1 else 0
    print("OUTPUT: {}".format(output))
    return output


def run_circuit():
    run2_noisy()