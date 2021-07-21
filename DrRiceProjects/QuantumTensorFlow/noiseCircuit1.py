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

def noisyCircuit(probability = 0.1, measure=True, depolarize=True, bitPhase=True, bitEnd = True):
    qreg = cirq.LineQubit.range(4)
    depInd = 1
    bitPhaseFlipInd = 2
    measureInd = 4

    circ = cirq.Circuit(
        cirq.H.on_each(qreg),
        cirq.qft(*qreg),
        cirq.qft(*qreg, inverse=True),
        # cirq.reset(qreg[1]),
    )

    if depolarize:
        measureInd += 1
        bitPhaseFlipInd +=  1
        circ.insert(depInd,  cirq.depolarize(p=probability).on_each(qreg))

    if bitPhase:
        measureInd += 1
        circ.insert(bitPhaseFlipInd,  BitAndPhaseFlipChannel(p=probability).on_each(qreg[1::2]))

    if measure:
        circ.insert(measureInd, cirq.measure(*qreg))

    if bitEnd:
       circ.append(cirq.bit_flip(p=probability).controlled(1).on(*qreg[2:]))



    print("Circuit with multiple channels:\n")
    print(circ)
    return circ


circuit = noisyCircuit(probability=0) # 0, 0.01, 0.1, 0.2, 0.3

sim = cirq.Simulator()

for i in range(100):    
    result = sim.simulate(circuit, initial_state=0b0100)
    # print(result)

