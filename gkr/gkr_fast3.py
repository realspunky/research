from zorch import koalabear
from fast_sumcheck import deg3_sumcheck_prove, deg3_sumcheck_verify, mixed_sumcheck_prove, mixed_sumcheck_verify
import time
from gkr_utils import (
    M, matmul_layer, inner_matmul, inner_matmul_layer, diag_inner, chi_weights, mle_eval, chi_eval, compute_weights, fast_mle_eval,
    fast_point_eval, generate_weights_seed_coords, generate_weights,
    log2, hash as _hash
)

ROUND_COUNT = 32

add_constants = list(range(ROUND_COUNT))

# Here, we do a significantly more complicated and optimized GKR for a more
# optimized version of Poseidon. Specifically, only rounds 0...3 and -4...-1
# are "full", in the other rounds we only cube the first value
def is_inner_round(i):
    return 4 <= i < ROUND_COUNT - 4

def permutation(values):
    orig_shape = values.shape
    values = values.reshape((-1, 16))
    for i in range(ROUND_COUNT):
        if is_inner_round(i):
            values = inner_matmul(values)
            values[:, 0] **= 3
            values += add_constants[i]
        else:
            values = koalabear.matmul(values, M)**3 + add_constants[i]
    return values.reshape(orig_shape)

def hash(*inputs):
    return _hash(*inputs, permutation=permutation)

# Prove evaluations via the GKR algorithm
def gkr_prove(values):
    layers = [values.reshape((values.shape[0]//16, 16))]
    post_matmul_layers = []
    for i in range(ROUND_COUNT):
        if not is_inner_round(i):
            post_matmul_layers.append(koalabear.matmul(layers[-1], M))
            layers.append(post_matmul_layers[-1] ** 3 + add_constants[i])
        else:
            post_matmul_layers.append(inner_matmul(layers[-1]))
            layers.append(post_matmul_layers[-1] + i)
            layers[-1][:,0] = post_matmul_layers[-1][:,0] ** 3 + add_constants[i]
    randomness = hash(layers[-1].reshape(values.shape), values)
    p = generate_weights_seed_coords(randomness, values.shape[0]//16, hash)
    proof = []
    Z_top = fast_mle_eval(layers[-1], p)
    Z = Z_top
    for i in range(ROUND_COUNT-1, -1, -1):
        # Going in, we have an "obligation" to prove that
        # weights * (V**3 + i) sums to Z, initially Z_top
        V = post_matmul_layers[i]
        if not is_inner_round(i):
            c0, c2, c3, p = \
                    deg3_sumcheck_prove(V, p, randomness, hash)
            left_sums = [koalabear.KoalaBear(0)]
            # Now, we have an obligation to prove V(p) = V_p
            pre_matmul_Vp = fast_mle_eval(layers[i], p)
            proof.append((c0, c2, c3, left_sums, p, pre_matmul_Vp))
        else:
            c0, c2, c3, left_sums, p = \
                    mixed_sumcheck_prove(V, p, randomness, hash)
            pre_matmul_Vp = fast_mle_eval(layers[i], p)
            proof.append((c0, c2, c3, left_sums, p, pre_matmul_Vp))
        # Z = V_p
        # V(p) equals the linear combination sum(chi_weights(p) * V). So we
        # compute the coeffs of that linear combination, and we now again
        # have an obligation in the format of the previous layer
        randomness = hash(randomness, c0[-1], c2[-1], c3[-1], left_sums[-1])
    return Z_top, proof

# Verify a GKR proof
def gkr_verify(values, outputs, Z_top, proof):
    randomness = hash(outputs, values)
    num_hashes = values.shape[0] // 16
    prev_p = generate_weights_seed_coords(
        randomness,
        num_hashes,
        hash
    )
    # Verify that the provided Z_top equals sum(outputs * initial_weights)
    assert Z_top == fast_mle_eval(outputs.reshape((num_hashes, 16)), prev_p)
    Z = Z_top
    # Walk through the layers backwards, and verify each proof
    for i, (c0, c2, c3, left_sums, p, pre_matmul_Vp) in zip(range(ROUND_COUNT-1, -1, -1), proof):
        W_p = chi_eval(prev_p, p)
        # Verify the claim of layer * weights = sum, and reduce it to a claim
        # prev_layer(point) = value. In the next step, we use chi_eval to
        # re-interpret this claim as prev_layer * prev_weights = prev_sum
        if is_inner_round(i):
            V_p = inner_matmul(pre_matmul_Vp)
            target = W_p * V_p
            target[0] = W_p * V_p[0] ** 3
            mixed_sumcheck_verify(
                c0, c2, c3, left_sums, prev_p, Z - add_constants[i], p, target, randomness, hash
            )
        else:
            V_p = koalabear.matmul(pre_matmul_Vp, M)
            deg3_sumcheck_verify(
                c0, c2, c3, prev_p, Z - add_constants[i], p, W_p * (V_p ** 3), randomness, hash
            )
        Z = pre_matmul_Vp
        prev_p = p
        # Verify that the first layer equals the inputs
        if i == 0:
            assert pre_matmul_Vp == fast_mle_eval(values.reshape((num_hashes, 16)), p)
        randomness = hash(randomness, c0[-1], c2[-1], c3[-1], left_sums[-1])
    return True


def test():
    values = koalabear.KoalaBear(list(range(524288)))
    t1 = time.time()
    outputs = permutation(values)
    print("Raw execution time:", time.time() - t1)
    t2 = time.time()
    Z_top, proof = gkr_prove(values)
    print("Proof generated")
    print("Generation time:", time.time() - t2)
    import sys
    if '--prove_only' not in sys.argv:
        assert gkr_verify(values, outputs, Z_top, proof)
        print("Verification completed")
    else:
        print("Verification skipped")

if __name__ == '__main__':
    test()
