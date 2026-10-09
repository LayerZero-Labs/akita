"""Arbitrary-precision schoolbook oracle for transmitted lifted A rows."""
import sys

lines = iter(sys.stdin.read().splitlines())
p, q0, degree, rank, width, columns = map(int, next(lines).split())

def row():
    return list(map(int, next(lines).split()))

matrix = [row() for _ in range(rank * width)]
packed = [row() for _ in range(width)]
images = [row() for _ in range(columns * rank)]
challenges = [row() for _ in range(columns)]
quotients = [row() for _ in range(rank)]
carries = [row() for _ in range(rank)]
assert next(lines, None) is None

for i in range(rank):
    residual = [0] * (2 * degree - 1)
    for j in range(width):
        for t, a in enumerate(matrix[i * width + j]):
            for s, b in enumerate(packed[j]):
                residual[t + s] += a * b
    for col in range(columns):
        for t, c in enumerate(challenges[col]):
            if c:
                for s, image in enumerate(images[col * rank + i]):
                    residual[t + s] -= c * image
    quotient = [0] * (degree - 1)
    for t in reversed(range(degree, len(residual))):
        leading = residual[t]
        quotient[t - degree] = leading
        residual[t] = 0
        residual[t - degree] -= leading
        residual[t - degree // 2] += leading
    assert [q % p for q in quotient] == quotients[i], ("QA", i)
    assert residual[:degree] == [q0 * k for k in carries[i]], ("KA", i)
    assert not any(residual[degree:])
