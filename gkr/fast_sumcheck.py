import numpy as np
from zorch import koalabear
from gkr_utils import chi_weights, chi_eval, mle_eval, chi_cofactor, log2

def generate_round_weights(eval_point):
    """
    Generates the weights hypercubes corresponding to a given evaluation
    point, for each round of a sumcheck. Note that because we are using
    Gruen's trick (see section 3.2 here: https://eprint.iacr.org/2024/108 ),
    we only need to go up to a half-size hypercube.
    """
    weights = [koalabear.ExtendedKoalaBear([[1,0,0,0]])]
    for i in range(len(eval_point)-1, 0, -1):
        R = weights[0] * eval_point[i]
        L = weights[0] - R
        weights.insert(0, koalabear.ExtendedKoalaBear.zeros((L.shape[0]*2,)))
        weights[0][::2] = L
        weights[0][1::2] = R
    return weights

def id_sumcheck_prove(V, eval_point, randomness, hash):
    """
    Prover for the sumcheck for an identity function (ie. it converts an
    obligation to prove sum(V * W) into an obligation to prove (V * W)(p).
    Note that here we are assuming that W takes a specific form: it's the
    linear combination that computes P(eval_point) as a function of the
    hypercube P(0,0...0), P(0,0...1) ... P(1,1...1).
    
    This W has the form id(x_0, p_0) * id(x_1, p_1) * ..., where
    id(x, y) = x*y + (1-x)*(1-y). This gives us an optimization: at the i'th
    level in the sumcheck, we ignore the first i+1 terms. This makes our
    adjusted W constant in the variable that the sumcheck is currently
    operating over, and so it makes V * W linear. This means that we only
    need to commit to the left-side and right-side sum per layer.

    On top of this, we add another optimization: given the left-side sum
    and the previous total, you can compute the right-side sum. So we
    only really need to provide the left-side sum.
    """
    left_sums = []
    coords = []
    weights = generate_round_weights(eval_point)
    for i in range(log2(V.shape[0])):
        W = weights[i]
        left_sum = W.__class__.sum(W * V[::2], axis=0)
        left_sums.append(left_sum)
        randomness = hash(randomness, left_sum)
        coords.append(koalabear.ExtendedKoalaBear(randomness[:4].value))
        # "Fold" V into the hypercube of evaluations where the first
        # i coordinates are coords[:i]
        V = V[::2] + coords[-1] * (V[1::2] - V[::2])
    # At the end, V[0] = orig_V(coords)
    return left_sums, coords, V[0]

def id_sumcheck_verify(left_sums, eval_point, total, coords, eval_value, randomness, hash):
    """
    Verifier for an identity sumcheck.
    """
    shared_cofactor = 1
    for i, (left_sum, coord) in enumerate(zip(left_sums, coords)):
        randomness = hash(randomness, left_sum)
        assert koalabear.ExtendedKoalaBear(randomness[:4].value) == coord
        # Computes the right sum from the left sum and the previous-round
        # total (Gruen's trick)
        left_cofactor = shared_cofactor * chi_cofactor(eval_point[i], 0)
        right_cofactor = shared_cofactor * chi_cofactor(eval_point[i], 1)
        new_cofactor = shared_cofactor * chi_cofactor(eval_point[i], coord)
        right_sum = (total - left_sum * left_cofactor) / right_cofactor
        # At each layer, total = sum(V * W), both after i rounds of folding.
        # Hence, the final total should be the evaluation V(p) * W(p)
        total = (left_sum + coord * (right_sum - left_sum)) * new_cofactor
        shared_cofactor = new_cofactor
    assert total == eval_value
    return True

def deg3_lagrange_weights(x):
    """
    Given a deg-3 poly P and a coordinate x, output the linear combination
    that computes P(x) from P(0), P(1), P(2), P(3)
    """
    nodes = (0, 1, 2, 3)
    denoms = (-6, 2, -2, 6)
    coeffs = []
    for idx, k in enumerate(nodes):
        num = koalabear.KoalaBear(1)
        for m in nodes:
            if m != k:
                num *= (x - m)
        coeffs.append(num / denoms[idx])
    return tuple(coeffs)

def deg3_sumcheck_prove(V, eval_point, randomness, hash):
    """
    Prover for the sumcheck for V * W**3, where once again W is the linear
    combination that computes P(eval_point) as a function of the
    hypercube P(0,0...0), P(0,0...1) ... P(1,1...1). Here, we also benefit
    from Gruen's trick, and so the function we are proving is deg-3 in each
    dimension. Hence, we need to provide three values (reminder: the fourth is
    recovered from the previous round total) at each level. We use the sums
    P(coords[:i], x, {(0,0...0), (0,0...1) ...  (1,1...1)}) for x in (0,2,3).
    """
    c0 = []
    c2 = []
    c3 = []
    coords = []
    weights = generate_round_weights(eval_point)
    _cls = koalabear.ExtendedKoalaBear
    for i in range(log2(V.shape[0])):
        # Thanks to Gruen's trick, the weights matrix used is the same for
        # all values of the current coordinate, including 0,2,3, and the
        # evaluation coord we generate this round.
        W = weights[i].reshape(weights[i].shape + (1,) * (V.ndim - 1))
        # slope of V in the current dimension. Lets us easily compute
        # V(coords[:i], 2, ...) and V(coords[:i], 3, ...)
        Vm = V[1::2] - V[::2]
        V2 = V[1::2] + Vm
        V3 = V2 + Vm
        # Compute sums of these three partial hypercubes
        sum0 = _cls.sum(W * V[::2] ** 3, axis=0)
        sum2 = _cls.sum(W * V2 ** 3, axis=0)
        sum3 = _cls.sum(W * V3 ** 3, axis=0)
        c0.append(sum0)
        c2.append(sum2)
        c3.append(sum3)
        randomness = hash(randomness, sum0, sum2, sum3)
        coords.append(koalabear.ExtendedKoalaBear(randomness[:4].value))
        V = V[::2] + coords[-1] * Vm
    return c0, c2, c3, coords, V[0]

def deg3_sumcheck_verify(c0, c2, c3, eval_point, total, coords, eval_value, randomness, hash):
    """
    Verifier for the deg-3 sumcheck
    """
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
    """
    Given some randomness, generate the list of powers
    1, alpha, alpha**2 ... alpha**(count-1)
    """
    alpha = koalabear.ExtendedKoalaBear(hash(randomness)[:4].value)
    alphapowers = koalabear.ExtendedKoalaBear.zeros((count,))
    alphapowers[0] = 1
    for i in range(1, count):
        alphapowers[i] = alphapowers[i-1] * alpha
    return alphapowers

def mixed_sumcheck_prove(V, eval_point, randomness, hash):
    """
    Assumes that V is two-dimensional (N*w), and does a sumcheck along the N
    dimension (meaning, its output is size w), of the function where only the
    zeroth item in each of the N elements is cubed. Everything else passes
    through the identity operator. Works by doing two sumchecks that share
    randomness: one cubic and one of a random linear combination of everything
    else.
    """
    # cubic terms
    c0 = []
    c2 = []
    c3 = []
    # linear random linear combination (rlc) terms
    left_sums = []
    # evaluation coordinate
    coords = []
    weights = generate_round_weights(eval_point)
    # Split V up into the cubic and rlc part
    V_cube = V[:,0]
    rlc_width = V.shape[1]-1
    alpha_powers = generate_alpha_powers(randomness, rlc_width, hash)
    V_rlc = koalabear.ExtendedKoalaBear.sum(
        V[:,1:] * alpha_powers.reshape((1, rlc_width)),
        axis=1
    )
    for i in range(log2(V.shape[0])):
        # In each round, do the cubic thing like above on V_cube, and the
        # linear thing like above on V_rlc, and have them share randomness
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

def mixed_sumcheck_verify(c0, c2, c3, left_sums, eval_point, total,
                          coords, eval_value, randomness, hash):
    """
    Verification for a mixed (cube on first element, linear on later elements)
    sumcheck
    """
    rlc_width = total.shape[0]-1
    alpha_powers = generate_alpha_powers(randomness, rlc_width, hash)
    total_rlc = koalabear.ExtendedKoalaBear.sum(
        total[1:] * alpha_powers, axis=0
    )
    total_cube = total[0]
    shared_cofactor = 1
    for i, (s0, s2, s3, left_sum, coord) in enumerate(zip(c0, c2, c3, left_sums, coords)):
        # Same logic as cubic sumcheck
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
        # Same logic as linear sumcheck 
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

    # Test identity sumcheck
    eval_point = koalabear.ExtendedKoalaBear(
        [[2,7,18,28], [18,28,45,90], [45,23,53,60], [2,8,7,5]]
    )
    W = chi_weights(eval_point)
    V = koalabear.KoalaBear(
        [2, 7,  18, 28, 18, 28, 45, 90, 45, 23, 53, 60,  2,  8,  7,    5]
    )
    total = W.__class__.sum(W * V, axis=0)
    left_sums, coords, V_value = \
            id_sumcheck_prove(V, eval_point, hash(W, V), hash)
    W_value = chi_eval(eval_point, coords)
    value = W_value * V_value
    assert value == mle_eval(W, coords) * mle_eval(V, coords)
    print("id proof generated")
    assert id_sumcheck_verify(
        left_sums, eval_point, total, coords, value, hash(W, V), hash
    )
    print("id proof verified")

    # Test cubic sumcheck
    total = W.__class__.sum(W * V**3, axis=0)
    c0, c2, c3, coords, V_value = \
            deg3_sumcheck_prove(V, eval_point, hash(W, V), hash)
    W_value = chi_eval(eval_point, coords)
    value = W_value * (V_value ** 3)
    print("deg3 proof generated")
    total = W.__class__.sum(W * V**3, axis=0)
    assert deg3_sumcheck_verify(
        c0, c2, c3, eval_point, total, coords, value, hash(W, V), hash
    )
    print("deg3 proof verified")

    # Test batch cubic sumcheck (that is, sumcheck a 128*4 value along the
    # length-128 dimension, so the output is size-4)
    V2 = koalabear.KoalaBear(
        [[3**i + 7**j for j in range(4)] for i in range(128)]
    )
    eval_point_2 = koalabear.ExtendedKoalaBear(
        [[2,7,18,28], [18,28,45,90], [45,23,53,60], [2,8,7,5],
        [3,14,15,92], [65,25,89,79], [38,46,26,43]]
    )
    W2 = chi_weights(eval_point_2).reshape((V2.shape[0], 1))
    total2 = W2.__class__.sum(W2 * V2**3, axis=0)
    c0, c2, c3, coords, V2_value = \
            deg3_sumcheck_prove(V2, eval_point_2, hash(W2, V2), hash)
    W2_value = chi_eval(eval_point_2, coords)
    value2 = W2_value * (V2_value ** 3)
    print("batch deg3 proof generated")
    assert deg3_sumcheck_verify(
        c0, c2, c3, eval_point_2, total2, coords, value2, hash(W2, V2), hash
    )
    print("batch deg3 proof verified")

    # Test a mixed sumcheck
    mixed_output = W2 * V2
    mixed_output[:,0] = W2[:,0] * V2[:,0] ** 3
    total3 = mixed_output.__class__.sum(mixed_output, axis=0)
    c0, c2, c3, left_sums, coords = \
            mixed_sumcheck_prove(V2, eval_point_2, hash(W2, V2), hash)
    W3_value = chi_eval(eval_point_2, coords)
    V3_value = mle_eval(V2, coords)
    value3 = W3_value * V3_value
    value3[0] = W3_value * V3_value[0] ** 3
    print("batch mixed proof generated")
    assert mixed_sumcheck_verify(
        c0, c2, c3, left_sums, eval_point_2, total3,
        coords, value3, hash(W2, V2), hash
    )
    print("batch mixed proof verified")

if __name__ == '__main__':
    test()
