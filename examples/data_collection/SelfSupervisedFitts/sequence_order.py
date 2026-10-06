
def _eulerian_gesture_sequence(n: int, start: int = 0) -> list[int]:
    """
    Returns a gesture ID sequence of length n*(n-1)+1 such that each ordered
    transition i->j (i!=j) appears exactly once (Eulerian circuit).
    """
    adj = {i: [j for j in range(n) if j != i] for i in range(n)}

    stack = [start]
    circuit = []
    while stack:
        v = stack[-1]
        if adj[v]:
            u = adj[v].pop()
            stack.append(u)
        else:
            circuit.append(stack.pop())

    circuit.reverse()

    return circuit

seq = _eulerian_gesture_sequence(5, 0)
print(seq)
print(len(seq))