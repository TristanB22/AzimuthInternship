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


probabilities = [0, 0.01, 0.1, 0.2, 0.3, 0.5]
iterations = 10000
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

