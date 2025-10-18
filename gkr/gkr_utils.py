from zorch import koalabear

def log2(x):
    return x.bit_length()-1

# Full-round Poseidon matrix
M = 1 / koalabear.KoalaBear([[1+i+j for i in range(16)] for j in range(16)])
# Partial round Poseidon matrix: J + diag
M_inner = (
    1 + koalabear.KoalaBear([[
        i**5+1 if i==j else 0 for i in range(16)] for j in range(16)
    ])
)
diag_inner = koalabear.KoalaBear([i**5+1 for i in range(16)])

U = koalabear.KoalaBear([[256, 2130673793, 1370880, 2102489153, 338607360, 1654136194, 872924922, 654224952, 386106561, 314590459, 483616338, 1732949107, 1979207809, 1805082046, 1006195439, 1583476179], [2130673793, 5548800, 1868525633, 1494912254, 489954722, 977569269, 1786295223, 858461190, 1943567653, 872134593, 947475158, 858275114, 563860436, 793172518, 93060641, 1362168547], [1370880, 1868525633, 429673722, 353244686, 1241628641, 710846392, 110669863, 331498486, 1226619653, 238505038, 1446346149, 927860223, 84026718, 827804594, 2002191756, 983169124], [2102489153, 1494912254, 353244686, 1024862712, 1463621936, 1388030984, 888655320, 488851089, 253689330, 1770462625, 22644809, 1138622964, 1833137130, 189415093, 503279294, 6079972], [338607360, 489954722, 1241628641, 1463621936, 371634228, 437939613, 1828415016, 1268446650, 420990384, 1365042012, 1813679107, 1566442641, 816915237, 1222914643, 249537642, 2044066832], [1654136194, 977569269, 710846392, 1388030984, 437939613, 1435002690, 365780676, 1637563412, 308402342, 1728033648, 455739796, 500301225, 2118318951, 253551011, 306746518, 1614711062], [872924922, 1786295223, 110669863, 888655320, 1828415016, 365780676, 1870233704, 1733037736, 687331065, 753339390, 372881455, 1728989584, 1529568771, 2019986618, 1100739032, 1944619781], [654224952, 858461190, 331498486, 488851089, 1268446650, 1637563412, 1733037736, 228649676, 954952611, 109985054, 1230016890, 449687996, 1277597619, 1979818757, 1067936503, 317103934], [386106561, 1943567653, 1226619653, 253689330, 420990384, 308402342, 687331065, 954952611, 1565076082, 398611158, 877717833, 1275236931, 2360688, 238386851, 1474817653, 1751709231], [314590459, 872134593, 238505038, 1770462625, 1365042012, 1728033648, 753339390, 109985054, 398611158, 1389597271, 782979226, 1427028422, 231876737, 1114252905, 2037546465, 2048709120], [483616338, 947475158, 1446346149, 22644809, 1813679107, 455739796, 372881455, 1230016890, 877717833, 782979226, 2124092218, 2120941769, 1921812342, 1690054539, 33833329, 1188349186], [1732949107, 858275114, 927860223, 1138622964, 1566442641, 500301225, 1728989584, 449687996, 1275236931, 1427028422, 2120941769, 1602134289, 1735537917, 1884224309, 1477413030, 114383587], [1979207809, 563860436, 84026718, 1833137130, 816915237, 2118318951, 1529568771, 1277597619, 2360688, 231876737, 1921812342, 1735537917, 664381607, 1386271884, 576981332, 446888918], [1805082046, 793172518, 827804594, 189415093, 1222914643, 253551011, 2019986618, 1979818757, 238386851, 1114252905, 1690054539, 1884224309, 1386271884, 897159747, 1696598052, 130894148], [1006195439, 93060641, 2002191756, 503279294, 249537642, 306746518, 1100739032, 1067936503, 1474817653, 2037546465, 33833329, 1477413030, 576981332, 1696598052, 2038596422, 570039284], [1583476179, 1362168547, 983169124, 6079972, 2044066832, 1614711062, 1944619781, 317103934, 1751709231, 2048709120, 1188349186, 114383587, 446888918, 130894148, 570039284, 1486512813]])

assert koalabear.matmul(U, M) == koalabear.KoalaBear([[1 if i==j else 0 for i in range(16)] for j in range(16)])

SHA256_OVERRIDE = True

# We store state in our GKR protocol in 1D. Here, we reshape it so [i,j]
# is the j'th wire of the i'th hash, then multiply it by the matrix, then
# reshape back
def matmul_layer(values, matrix=M):
    orig_shape = values.shape
    size = values.shape[-1] // 16
    values = values.reshape(values.shape[:-1] + (size, 16))
    values = koalabear.matmul(values, matrix)
    values = values.reshape(orig_shape)
    return values

def inner_matmul(values):
    values_sum = values.__class__.sum(values, axis=-1)
    return values * diag_inner + values_sum.reshape(values_sum.shape + (1,))

# Same as above, but for partial rounds. Cheaper because the matrix has the
# special form J + diag, so you only need 16 muls and 32 adds (as opposed to
# 256 each for a "full" matrix)
def inner_matmul_layer(values):
    orig_shape = values.shape
    size = values.shape[-1] // 16
    values = values.reshape(values.shape[:-1] + (size, 16))
    values = inner_matmul(values).reshape(orig_shape)
    return values

# Given an evaluation point, compute the linear combination that computes
# that evaluation from evaluations on the hypercube. For example:
# chi_weights([3, 5]) gives [8, -12, -10, 15]. This means that if you have
# a 2D cube (ok fine, normies call that a square) [a, b, c, d], then that
# polynomial evaluated at (3, 5) equals 8*a - 12*b - 10*c + 15*d.
def chi_weights(coords, start_weights=None):
    if start_weights is None:
        weights = koalabear.ExtendedKoalaBear([[1,0,0,0]])
    else:
        weights = start_weights
    for c in coords:
        R = weights * c
        L = weights - R
        weights = koalabear.ExtendedKoalaBear.append(L, R)
    return weights

# Given evaluations on a cube, compute the evaluation at the given coords
def mle_eval(cube, coords):
    for coord in coords:
        b = cube[::2]
        m = cube[1::2] - b
        cube = b + m * coord
    return cube[0]

# More optimized version of the above, suited for the setting where cube is 2D
def fast_mle_eval(cube, coords):
    weights = chi_weights(coords[:-2])
    weights = weights.reshape(weights.shape + (1,) * (cube.ndim - 1))
    qsize = cube.shape[0]//4
    o1 = koalabear.ExtendedKoalaBear.sum(cube[:qsize] * weights, axis=0)
    o2 = koalabear.ExtendedKoalaBear.sum(cube[qsize:qsize*2] * weights, axis=0)
    o3 = koalabear.ExtendedKoalaBear.sum(cube[qsize*2:qsize*3] * weights, axis=0)
    o4 = koalabear.ExtendedKoalaBear.sum(cube[qsize*3:qsize*4] * weights, axis=0)
    b = o1
    m1 = o2 - b
    m2 = o3 - b
    m12 = o4 - (b + m1 + m2)
    return b + m1 * coords[-2] + m2 * coords[-1] + m12 * (coords[-2] * coords[-1])

# A one-dimensional piece of chi_eval below
def chi_cofactor(source_point, eval_point):
    return (source_point * eval_point + (1-source_point) * (1-eval_point))

# Given an evaluation point source_coords, take the cube that would be
# generated by chi_eval(source_coords), and evaluate it at eval_coords,
# without ever actually materializing that cube
def chi_eval(source_coords, eval_coords):
    o = 1
    for s, e in zip(source_coords, eval_coords):
        o *= chi_cofactor(s, e)
    return o

# Some more complicated methods, whose goal is to take the cube generated by
# chi_eval, and evaluate that cube *multiplied by M* at eval_coords
def compute_lower_order_weights(lower_coords):
    return koalabear.matmul(M.swapaxes(0,1), chi_weights(lower_coords))

def compute_weights(coords):
    intermediate = compute_lower_order_weights(coords[:4])
    return chi_weights(coords[4:], start_weights=intermediate)

def fast_point_eval(source_coords, eval_coords):
    intermediate = compute_lower_order_weights(source_coords[:4])
    o = mle_eval(intermediate, eval_coords[:4])
    o *= chi_eval(source_coords[4:], eval_coords[4:])
    return o

# Given randomness, generates a random coord that is then used to generate
# the initial weights for GKR via chi_weights
def generate_weights_seed_coords(randomness, count, hash):
    return [
        koalabear.ExtendedKoalaBear(
            hash(randomness, koalabear.KoalaBear(31337+i)).value[:4]
        )
        for i in range(log2(count))
    ]

# Generate those weights
def generate_weights(randomness, count, hash):
    return chi_weights(generate_weights_seed_coords(randomness, count, hash))

# We'll make our implementation self-contained, and use our own hash in the prover
# and verifier!
def hash16_to_8(inp, permutation):
    return permutation(inp)[...,:8] + inp[...,:8]

def hash(*args, permutation):
    inputs = []
    for arg in args:
        inputs.append(koalabear.KoalaBear(arg.value.reshape((-1,))))
    buffer = koalabear.KoalaBear.append(*inputs)
    if SHA256_OVERRIDE:
        from hashlib import sha256
        d = sha256(buffer.tobytes()).digest()
        return koalabear.KoalaBear([
            int.from_bytes(d[i:i+4], 'little')
            for i in range(0,32,4)
        ])
    buffer_length = buffer.shape[0]
    # padding: buffer -> [length] + buffer + [pad to nearest 16]
    buffer = koalabear.KoalaBear.append(
        koalabear.KoalaBear([buffer_length]),
        buffer,
        koalabear.KoalaBear.zeros(15 - buffer.shape[0] % 16)
    )
    # Merkle tree style hash
    while buffer.shape[0] > 8:
        if buffer.shape[0] % 16 == 8:
            buffer = koalabear.KoalaBear.append(
                buffer,
                koalabear.KoalaBear.zeros(8)
            ) + buffer_length
            # Note: we mixin the total buffer length everywhere to avoid
            # collisions between inputs of different length
        buffer = buffer.reshape((-1, 16))
        buffer = hash16_to_8(buffer, permutation)
        buffer = buffer.reshape((-1,))
    return buffer
