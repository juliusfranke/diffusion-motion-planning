from meshlib.mrmeshpy import Vector3f


def mult_el_wise(a: Vector3f, b: Vector3f) -> Vector3f:
    return Vector3f(a.x * b.x, a.y * b.y, a.z * b.z)
