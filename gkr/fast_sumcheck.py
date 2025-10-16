import numpy as np
from zorch import koalabear
from gkr_utils import chi_weights, chi_eval, mle_eval, chi_cofactor, log2

def generate_round_weights(eval_point):
    weights = [koalabear.ExtendedKoalaBear([[1,0,0,0]])]
    for i in range(len(eval_point)-1, 0, -1):
        R = weights[0] * eval_point[i]
        L = weights[0] - R
        weights.insert(0, koalabear.ExtendedKoalaBear.zeros((L.shape[0]*2,)))
        weights[0][::2] = L
        weights[0][1::2] = R
    return weights

def id_sumcheck_prove(V, eval_point, randomness, hash):
    left_sums = []
    coords = []
    weights = generate_round_weights(eval_point)
    for i in range(log2(V.shape[0])):
        W = weights[i]
        left_sum = W.__class__.sum(W * V[::2], axis=0)
        left_sums.append(left_sum)
        randomness = hash(randomness, left_sum)
        coords.append(koalabear.ExtendedKoalaBear(randomness[:4].value))
        V = V[::2] + coords[-1] * (V[1::2] - V[::2])
    return left_sums, coords

def id_sumcheck_verify(left_sums, eval_point, total, coords, eval_value, randomness, hash):
    shared_cofactor = 1
    for i, (left_sum, coord) in enumerate(zip(left_sums, coords)):
        left_cofactor = shared_cofactor * chi_cofactor(eval_point[i], 0)
        right_cofactor = shared_cofactor * chi_cofactor(eval_point[i], 1)
        new_cofactor = shared_cofactor * chi_cofactor(eval_point[i], coord)
        randomness = hash(randomness, left_sum)
        assert koalabear.ExtendedKoalaBear(randomness[:4].value) == coord
        right_sum = (total - left_sum * left_cofactor) / right_cofactor
        total = (left_sum + coord * (right_sum - left_sum)) * new_cofactor
        shared_cofactor = new_cofactor
    assert total == eval_value
    return True

def deg3_lagrange_weights(x):
    nodes = (0, 1, 2, 3)
    denoms = (-6, 2, -2, 6)  # Π_{m≠k} (k - m) for k = -2,-1,0,1,2
    coeffs = []
    for idx, k in enumerate(nodes):
        num = koalabear.KoalaBear(1)
        for m in nodes:
            if m != k:
                num *= (x - m)
        coeffs.append(num / denoms[idx])
    return tuple(coeffs)

def deg3_sumcheck_prove(V, eval_point, randomness, hash):
    c0 = []
    c2 = []
    c3 = []
    coords = []
    weights = generate_round_weights(eval_point)
    _cls = koalabear.ExtendedKoalaBear
    for i in range(log2(V.shape[0])):
        W = weights[i].reshape(weights[i].shape + (1,) * (V.ndim - 1))
        Vm = V[1::2] - V[::2]
        V2 = V[1::2] + Vm
        V3 = V2 + Vm
        sum0 = _cls.sum(W * V[::2] ** 3, axis=0)
        sum2 = _cls.sum(W * V2 ** 3, axis=0)
        sum3 = _cls.sum(W * V3 ** 3, axis=0)
        c0.append(sum0)
        c2.append(sum2)
        c3.append(sum3)
        randomness = hash(randomness, sum0, sum2, sum3)
        coords.append(koalabear.ExtendedKoalaBear(randomness[:4].value))
        V = V[::2] + coords[-1] * Vm
    return c0, c2, c3, coords

def deg3_sumcheck_verify(c0, c2, c3, eval_point, total, coords, eval_value, randomness, hash):
    shared_cofactor = 1
    for i, (s0, s2, s3, coord) in enumerate(zip(c0, c2, c3, coords)):
        cof0, cof1, cof2, cof3, cofnew = [
            shared_cofactor * chi_cofactor(eval_point[i], x)
            for x in (0, 1, 2, 3, coord)
        ]
        randomness = hash(randomness, s0, s2, s3)
        assert koalabear.ExtendedKoalaBear(randomness[:4].value) == coord
        s1 = (total - s0 * cof0) / cof1
        coeffs = deg3_lagrange_weights(coord)
        total = (
            s0 * coeffs[0] +
            s1 * coeffs[1] +
            s2 * coeffs[2] +
            s3 * coeffs[3]
        ) * cofnew
        shared_cofactor = cofnew
    assert total == eval_value
    return True

def generate_alpha_powers(randomness, count, hash):
    alpha = koalabear.ExtendedKoalaBear(hash(randomness)[:4].value)
    alphapowers = koalabear.ExtendedKoalaBear.zeros((count,))
    alphapowers[0] = 1
    for i in range(1, count):
        alphapowers[i] = alphapowers[i-1] * alpha
    return alphapowers

def mixed_sumcheck_prove(V, eval_point, randomness, hash):
    """
    Assumes that V is two-dimensional, and does a sumcheck of the function
    where only the zeroth element within each value in v gets cubed;
    everything else passes through the identity operator. Works by doing
    two sumchecks: one cubic and one of a random linear combination of
    everything else
    """
    # cubic terms
    c0 = []
    c2 = []
    c3 = []
    # linear rlc terms
    left_sums = []
    coords = []
    weights = generate_round_weights(eval_point)
    V_cube = V[:,0]
    rlc_width = V.shape[1]-1
    V_rlc = koalabear.ExtendedKoalaBear.sum(
        V[:,1:] * generate_alpha_powers(randomness, rlc_width, hash).reshape((1, rlc_width)),
        axis=1
    )
    for i in range(log2(V.shape[0])):
        W = weights[i]
        Vm = V_cube[1::2] - V_cube[::2]
        V2 = V_cube[1::2] + Vm
        V3 = V2 + Vm
        _cls = W.__class__
        sum0 = _cls.sum(W * V_cube[::2] ** 3, axis=0)
        sum2 = _cls.sum(W * V2 ** 3, axis=0)
        sum3 = _cls.sum(W * V3 ** 3, axis=0)
        c0.append(sum0)
        c2.append(sum2)
        c3.append(sum3)
        left_sum = _cls.sum(W * V_rlc[::2], axis=0)
        left_sums.append(left_sum)
        randomness = hash(randomness, sum0, sum2, sum3, left_sum)
        coords.append(koalabear.ExtendedKoalaBear(randomness[:4].value))
        V_cube = V_cube[::2] + coords[-1] * Vm
        V_rlc = V_rlc[::2] + coords[-1] * (V_rlc[1::2] - V_rlc[::2])
    return c0, c2, c3, left_sums, coords

def mixed_sumcheck_verify(c0, c2, c3, left_sums, eval_point, total, coords, eval_value, randomness, hash):
    rlc_width = total.shape[0]-1
    alpha_powers = generate_alpha_powers(randomness, rlc_width, hash)
    total_rlc = koalabear.ExtendedKoalaBear.sum(
        total[1:] * alpha_powers, axis=0
    )
    total_cube = total[0]
    shared_cofactor = 1
    for i, (s0, s2, s3, left_sum, coord) in enumerate(zip(c0, c2, c3, left_sums, coords)):
        cof0, cof1, cof2, cof3, cofnew = [
            shared_cofactor * chi_cofactor(eval_point[i], x)
            for x in (0, 1, 2, 3, coord)
        ]
        randomness = hash(randomness, s0, s2, s3, left_sum)
        assert koalabear.ExtendedKoalaBear(randomness[:4].value) == coord
        s1 = (total_cube - s0 * cof0) / cof1
        coeffs = deg3_lagrange_weights(coord)
        total_cube = (
            s0 * coeffs[0] +
            s1 * coeffs[1] +
            s2 * coeffs[2] +
            s3 * coeffs[3]
        ) * cofnew
        
        right_sum = (total_rlc - left_sum * cof0) / cof1
        total_rlc = (left_sum + coord * (right_sum - left_sum)) * cofnew
        shared_cofactor = cofnew
    eval_cube = eval_value[0]
    eval_rlc = koalabear.ExtendedKoalaBear.sum(
        eval_value[1:] * alpha_powers, axis=0
    )
    assert total_cube == eval_cube
    assert total_rlc == eval_rlc
    return True

def test():
    # dummy hash function
    def hash(*args):
        o = koalabear.KoalaBear.zeros((8,))
        p = 1
        for arg in args:
            buffer = koalabear.KoalaBear(arg.value.reshape((-1,)))
            for i in range(buffer.shape[0]):
                o[i%8] += buffer[i] * p
                p *= 37
        return koalabear.KoalaBear(o)

    eval_point = koalabear.ExtendedKoalaBear([[2,7,18,28], [18,28,45,90], [45,23,53,60], [2,8,7,5]])
    # W = koalabear.KoalaBear([3, 14, 15, 92, 65, 35, 89, 79, 32, 38, 46, 26, 43, 38, 32, 7950])
    W = chi_weights(eval_point)
    V = koalabear.KoalaBear([2, 7,  18, 28, 18, 28, 45, 90, 45, 23, 53, 60,  2,  8,  7,    5])
    # V = koalabear.KoalaBear([1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1])
    total = W.__class__.sum(W * V, axis=0)
    left_sums, coords = id_sumcheck_prove(V, eval_point, hash(W, V), hash)
    W_value = chi_eval(eval_point, coords)
    V_value = mle_eval(V, coords)
    value = W_value * V_value
    assert value == mle_eval(W, coords) * mle_eval(V, coords)
    print("id proof generated")
    assert id_sumcheck_verify(left_sums, eval_point, total, coords, value, hash(W, V), hash)
    print("id proof verified")
    total = W.__class__.sum(W * V**3, axis=0)
    c0, c2, c3, coords = deg3_sumcheck_prove(V, eval_point, hash(W, V), hash)
    W_value = chi_eval(eval_point, coords)
    V_value = mle_eval(V, coords)
    value = W_value * (V_value ** 3)
    print("deg3 proof generated")
    total = W.__class__.sum(W * V**3, axis=0)
    assert deg3_sumcheck_verify(c0, c2, c3, eval_point, total, coords, value, hash(W, V), hash)
    print("deg3 proof verified")
    V2 = koalabear.KoalaBear([[3**i + 7**j for j in range(4)] for i in range(128)])
    eval_point_2 = koalabear.ExtendedKoalaBear([[2,7,18,28], [18,28,45,90], [45,23,53,60], [2,8,7,5], [3,14,15,92], [65,25,89,79], [38,46,26,43]])
    W2 = chi_weights(eval_point_2).reshape((V2.shape[0], 1))
    total2 = W2.__class__.sum(W2 * V2**3, axis=0)
    c0, c2, c3, coords = deg3_sumcheck_prove(V2, eval_point_2, hash(W2, V2), hash)
    W2_value = chi_eval(eval_point_2, coords)
    V2_value = mle_eval(V2, coords)
    value2 = W2_value * (V2_value ** 3)
    print("batch deg3 proof generated")
    assert deg3_sumcheck_verify(c0, c2, c3, eval_point_2, total2, coords, value2, hash(W2, V2), hash)
    print("batch deg3 proof verified")
    mixed_output = W2 * V2
    mixed_output[:,0] = W2[:,0] * V2[:,0] ** 3
    total3 = mixed_output.__class__.sum(mixed_output, axis=0)
    c0, c2, c3, left_sums, coords = mixed_sumcheck_prove(V2, eval_point_2, hash(W2, V2), hash)
    W3_value = chi_eval(eval_point_2, coords)
    V3_value = mle_eval(V2, coords)
    value3 = W3_value * V3_value
    value3[0] = W3_value * V3_value[0] ** 3
    print("batch deg3 proof generated")
    assert mixed_sumcheck_verify(c0, c2, c3, left_sums, eval_point_2, total3, coords, value3, hash(W2, V2), hash)
    print("batch deg3 proof verified")




if __name__ == '__main__':
    test()
