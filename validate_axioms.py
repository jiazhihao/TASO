from itertools import chain, product
import datetime
from multiprocessing.pool import Pool, cpu_count

import z3

import verify
from verify import P, T
from verify import (
    AC_MODE_NONE,
    AC_MODE_SIGMOID,
    AC_MODE_RELU,
    AC_MODE_TANH,
    PD_MODE_SAME,
    PD_MODE_VALID
)


now = datetime.datetime.now


def get_id(x):
    return z3.Z3_get_ast_id(x.ctx_ref(), x.as_ast())

def minimize_core_aux2(s, core):
    mus = []
    ids = set()
    while core != []:
        c = core[0]
        new_core = mus + core[1:]
        is_sat = s.check(new_core)
        if is_sat != z3.unsat:
            mus = mus + [c]
            ids.add(get_id(c))
            core = core[1:]
        else:
            core = s.unsat_core()
            core = [c for c in core if get_id(c) not in ids]
    return mus

def minimize_core(s):
    core = list(s.unsat_core())
    core = minimize_core_aux2(s, core)
    return core


def _shape(T):
    s = ()
    while type(T) is list:
        s = s + (len(T),)
        T = T[0]
    return tuple(s)


class Tensor(object):
    def __init__(self, data, splits=None):
        self.data = data
        self.shape = _shape(data)
        assert all(x > 0 for x in self.shape)
        self.dim = len(self.shape)
        if splits is None:
            self.splits = ((),) * self.dim
        else:
            self.splits = splits

    def __getitem__(self, tup):
        assert len(tup) == self.dim
        x = self.data
        while len(tup) > 0:
            x = x[tup[0]]
            tup = tup[1:]
        return x

    def __setitem__(self, tup, value):
        assert len(tup) == self.dim
        if len(tup) == 0:
            self.data = value
        else:
            x = self.data
            while len(tup) > 1:
                x = x[tup[0]]
                tup = tup[1:]
            x[tup[0]] = value

    @classmethod
    def zeros(cls, s):
        assert all(x > 0 for x in s)
        def _zeros(s):
            return 0 if s == () else [_zeros(s[1:]) for i in range(s[0])]
        return cls(_zeros(s))

    @classmethod
    def tensor(cls, s, prefix='x'):
        assert all(x > 0 for x in s)
        def _tensor(s, prefix):
            return z3.Real(prefix) if s == () else [_tensor(s[1:], '{}_{}'.format(prefix,i)) for i in range(s[0])]
        return cls(_tensor(s,prefix))


class MagicConst(object):
    def __init__(self, _to_tensor):
        self._to_tensor = _to_tensor

    def to_tensor(self, s):
        return self._to_tensor(s)


class BadShapeError(Exception):
    pass


def require(b):
    if not b:
        raise BadShapeError()


_relu = z3.Function("relu", z3.RealSort(), z3.RealSort())


def matmul_0(A,B):
    if isinstance(A, MagicConst):
        assert isinstance(B, Tensor)
        A = A.to_tensor((None, B.shape[0]))
    if isinstance(B, MagicConst):
        assert isinstance(A, Tensor)
        B = B.to_tensor((A.shape[1], None))
    sa = A.shape
    sb = B.shape
    if len(sa) == len(sb) == 2:
        require(sa[1] == sb[0])
        C = Tensor([[sum(A[i,k] * B[k,j] for k in range(sa[1])) for j in range(sb[1])] for i in range(sa[0])])
        C.splits = (A.splits[0], B.splits[1])
    elif len(sa) == len(sb) == 3:
        require(False) # TODO: split matmul into 2D and 3D operators
        require(sa[2] == sb[1])
        require(sa[0] == sb[0])
        C = Tensor([[[sum(A[n,i,k] * B[n,k,j] for k in range(sa[2])) for j in range(sb[2])] for i in range(sa[1])] for n in range(sa[0])])
        C.splits = ((), A.splits[1], B.splits[2])
    else:
        assert False
    return C


def transpose_0(A):
    sa = A.shape
    require(len(sa) == 2)
    C = Tensor([[A[i,j] for i in range(sa[0])] for j in range(sa[1])])
    C.splits = (A.splits[1], A.splits[0])
    return C


def conv2d_0(sx, sy, pad, acti, A, B):
    assert acti in [AC_MODE_NONE, AC_MODE_RELU] # TODO: handle other modes
    assert pad in [PD_MODE_SAME, PD_MODE_VALID]
    sa = A.shape
    if isinstance(B, MagicConst):
        B = B.to_tensor((sa[1], 1, None, None))
    sb = B.shape
    require(len(sa) == len(sb) == 4)
    # require(sa[1] == sb[1]) # non-grouped convolution
    require(sa[1] % sb[1] == 0) # grouped convolution
    group = sa[1] / sb[1]
    require(sb[0] % group == 0) # grouped convolution
    require(sx > 0 and sy > 0)
    if pad == PD_MODE_SAME: # same padding
        ox = (sa[2] + sx - 1) / sx
        oy = (sa[3] + sy - 1) / sy
        if sa[2] % sx == 0:
            totalPadH = max(sb[2] - sx, 0);
        else:
            totalPadH = max(sb[2] - (sa[2] % sx), 0)
        if sa[3] % sy == 0:
            totalPadW = max(sb[3] - sy, 0);
        else:
            totalPadW = max(sb[3] - (sa[3] % sy), 0)
        px = (totalPadH + 1) / 2
        py = (totalPadW + 1) / 2
    elif pad == PD_MODE_VALID: # valid padding
        ox = (sa[2] - sb[2]) / sx + 1
        oy = (sa[3] - sb[3]) / sy + 1
        px = 0
        py = 0
    else:
        assert False

    so = (sa[0], sb[0], ox, oy)
    require(ox > 0 and oy > 0)
    C = Tensor.zeros(so)
    for n in range(so[0]):
        for c in range(so[1]):
            for h in range(so[2]):
                for w in range(so[3]):
                    group_idx = c / (sb[0] / group)
                    value = 0
                    for cin in range(sb[1]):
                        for kh in range(sb[2]):
                            for kw in range(sb[3]):
                                posH = h * sx + kh - px
                                posW = w * sy + kw - py
                                assert -px <= posH <= sa[2] + px, posH
                                assert -py <= posW <= sa[3] + py, (posW, h, w, sx, sy, kh, kw, py, so)
                                if posH >= 0 and posH < sa[2] and posW >= 0 and posW < sa[3]:
                                    value += A[n,cin+group_idx*sb[1],posH,posW] * B[c,cin,kh,kw]
                    C[n,c,h,w] = value if acti == AC_MODE_NONE else _relu(value)
    C.splits = (A.splits[0], B.splits[0], (), ())
    return C


def const_pool_0(kx, ky):
    def to_tensor(s):
        assert len(s) == 4
        assert s[-2:] == (None, None)
        return Tensor(
            [[[[z3.RealVal(1) / z3.RealVal(kx * ky)
                for i4 in range(ky)]
               for i3 in range(kx)]
              for i2 in range(s[1])]
             for i1 in range(s[0])]
        )
    return MagicConst(to_tensor)


def const_iconv_0(kx, ky):
    assert kx % 2 == 1
    assert ky % 2 == 1
    middle = (kx // 2, ky // 2)
    def to_tensor(s):
        assert len(s) == 4
        assert s[-2:] == (None, None)
        return Tensor(
            [[[[1 if (i3,i4) == middle else 0
                for i4 in range(ky)]
               for i3 in range(kx)]
              for i2 in range(s[1])]
             for i1 in range(s[0])]
        )
    return MagicConst(to_tensor)


def const_imm_0():
    def to_tensor(s):
        s = list(s)
        assert len(s) == 2
        if s[0] is None:
            s[0] = s[1]
        if s[1] is None:
            s[1] = s[0]
        assert s[0] == s[1]
        assert s[0] is not None
        s = tuple(s)
        I = Tensor.zeros(s)
        for i in range(s[0]):
            I[i,i] = 1
        return I
    return MagicConst(to_tensor)


def const_one_0():
    def to_tensor(s):
        assert all(x > 0 for x in s)
        def _ones(s):
            return 1 if s == () else [_ones(s[1:]) for i in range(s[0])]
        return Tensor(_ones(s))
    return MagicConst(to_tensor)


def pool2d_avg_0(kx, ky, sx, sy, pad, A):
    assert pad in [PD_MODE_SAME, PD_MODE_VALID]
    sa = A.shape
    require(len(sa) == 4)
    require(sx > 0 and sy > 0)
    if pad == PD_MODE_SAME: # same padding
        ox = (sa[2] + sx - 1) / sx
        oy = (sa[3] + sy - 1) / sy
        if sa[2] % sx == 0:
            totalPadH = max(kx - sx, 0);
        else:
            totalPadH = max(kx - (sa[2] % sx), 0)
        if sa[3] % sy == 0:
            totalPadW = max(ky - sy, 0);
        else:
            totalPadW = max(ky - (sa[3] % sy), 0)
        px = (totalPadH + 1) / 2
        py = (totalPadW + 1) / 2
    elif pad == PD_MODE_VALID: # valid padding
        ox = (sa[2] - kx) / sx + 1
        oy = (sa[3] - ky) / sy + 1
        px = 0
        py = 0
    else:
        assert False

    so = (sa[0], sa[1], ox, oy)
    require(ox > 0 and oy > 0)
    C = Tensor.zeros(so)
    for n in range(so[0]):
        for c in range(so[1]):
            for h in range(so[2]):
                for w in range(so[3]):
                    value = 0
                    for kh in range(kx):
                        for kw in range(ky):
                            posH = h * sx + kh - px
                            posW = w * sy + kw - py
                            assert -px <= posH <= sa[2] + px, posH
                            assert -py <= posW <= sa[3] + py, (posW, h, w, sx, sy, kh, kw, py, so)
                            if posH >= 0 and posH < sa[2] and posW >= 0 and posW < sa[3]:
                                value += A[n,c,posH,posW]
                    C[n,c,h,w] = value / z3.RealVal(kx * ky)
    C.splits = (A.splits[0], A.splits[1], (), ())
    return C


def z3max(x, y):
    if x is None:
        assert y is not None
        return y
    elif y is None:
        return x
    else:
        return z3.If(x > y, x, y)


def pool2d_max_0(kx, ky, sx, sy, pad, A):
    assert pad in [PD_MODE_SAME, PD_MODE_VALID]
    sa = A.shape
    require(len(sa) == 4)
    require(sx > 0 and sy > 0)
    if pad == PD_MODE_SAME: # same padding
        ox = (sa[2] + sx - 1) / sx
        oy = (sa[3] + sy - 1) / sy
        if sa[2] % sx == 0:
            totalPadH = max(kx - sx, 0);
        else:
            totalPadH = max(kx - (sa[2] % sx), 0)
        if sa[3] % sy == 0:
            totalPadW = max(ky - sy, 0);
        else:
            totalPadW = max(ky - (sa[3] % sy), 0)
        px = (totalPadH + 1) / 2
        py = (totalPadW + 1) / 2
    elif pad == PD_MODE_VALID: # valid padding
        ox = (sa[2] - kx) / sx + 1
        oy = (sa[3] - ky) / sy + 1
        px = 0
        py = 0
    else:
        assert False

    so = (sa[0], sa[1], ox, oy)
    require(ox > 0 and oy > 0)
    C = Tensor.zeros(so)
    for n in range(so[0]):
        for c in range(so[1]):
            for h in range(so[2]):
                for w in range(so[3]):
                    value = None
                    for kh in range(kx):
                        for kw in range(ky):
                            posH = h * sx + kh - px
                            posW = w * sy + kw - py
                            assert -px <= posH <= sa[2] + px, posH
                            assert -py <= posW <= sa[3] + py, (posW, h, w, sx, sy, kh, kw, py, so)
                            if posH >= 0 and posH < sa[2] and posW >= 0 and posW < sa[3]:
                                value = z3max(value, A[n,c,posH,posW])
                    C[n,c,h,w] = value
    C.splits = (A.splits[0], A.splits[1], (), ())
    return C


def ewadd_0(A,B):
    sa = A.shape
    sb = B.shape
    require(sa == sb)
    C = Tensor.zeros(sa)
    for ii in product(*[range(n) for n in sa]):
        C[ii] = A[ii] + B[ii]
    if A.splits == B.splits:
        C.splits = A.splits
    return C


def ewmul_0(A,B):
    if isinstance(A, MagicConst):
        assert isinstance(B, Tensor)
        A = A.to_tensor(B.shape)
    if isinstance(B, MagicConst):
        assert isinstance(A, Tensor)
        B = B.to_tensor(A.shape)
    sa = A.shape
    sb = B.shape
    require(sa == sb)
    C = Tensor.zeros(sa)
    for ii in product(*[range(n) for n in sa]):
        C[ii] = A[ii] * B[ii]
    if A.splits == B.splits:
        C.splits = A.splits
    return C


def scalar_mul_0(A,B):
    sa = A.shape
    sb = B.shape
    require(sb == ())
    C = Tensor.zeros(sa)
    for ii in product(*[range(n) for n in sa]):
        C[ii] = A[ii] * B[()]
    C.splits = A.splits
    return C


def relu_0(A):
    sa = A.shape
    C = Tensor.zeros(sa)
    for ii in product(*[range(n) for n in sa]):
        C[ii] = _relu(A[ii])
    C.splits = A.splits
    return C


def concat_0(d, A,B):
    require(A.dim == B.dim and d < A.dim)
    require(all(i == d or A.shape[i] == B.shape[i] for i in range(A.dim)))
    C = Tensor.zeros(tuple(A.shape[i] if i != d else A.shape[i] + B.shape[i] for i in range(A.dim)))
    for ii in product(*[range(n) for n in C.shape]):
        if ii[d] < A.shape[d]:
            C[ii] = A[ii]
        else:
            jj = list(ii)
            jj[d] -= A.shape[d]
            C[ii] = B[tuple(jj)]
    C.splits = tuple(
        (A.shape[d], A.splits[d], B.splits[d]) if i == d else
        A.splits[i] if A.splits[i]==B.splits[i] else
        ()
        for i in range(C.dim)
    )
    return C


def split_0(d, A):
    require(d < A.dim)
    assert A.splits[d] != ()
    s, l, r = A.splits[d]
    C = Tensor.zeros(tuple(A.shape[i] if i != d else s for i in range(A.dim)))
    for ii in product(*[range(n) for n in C.shape]):
        C[ii] = A[ii]
    C.splits = tuple(A.splits[i] if i != d else l for  i in range(A.dim))
    return C


def split_1(d, A):
    require(d < A.dim)
    assert A.splits[d] != ()
    s, l, r = A.splits[d]
    C = Tensor.zeros(tuple(A.shape[i] if i != d else A.shape[i] - s for i in range(A.dim)))
    for ii in product(*[range(n) for n in C.shape]):
        jj = list(ii)
        jj[d] += s
        C[ii] = A[tuple(jj)]
    C.splits = tuple(A.splits[i] if i != d else r for  i in range(A.dim))
    return C


def enlarge_0(kx, ky, A):
    sa = A.shape
    require(len(sa) == 4)
    sc = (sa[0], sa[1], max(sa[2], kx), max(sa[3], ky))
    C = Tensor.zeros(sc)
    dx = (sc[2] - sa[2]) // 2
    dy = (sc[3] - sa[3]) // 2
    for n in range(sa[0]):
        for c in range(sa[1]):
            for h in range(sa[2]):
                for w in range(sa[3]):
                    C[n, c, h + dx, w + dy] = A[n, c, h, w]
    C.splits = (A.splits[0], A.splits[1], (), ()) # TODO: compute split tree for other dimensions?
    return C


# def one():
#     C = Tensor.zeros(())
#     C[()] = 1
#     return C


def eq(S,T):
    assert S.shape == T.shape, (S.shape, T.shape)
    assert S.splits == T.splits, "{} != {}".format(S.splits, T.splits)
    if S == T:
        print "syntactic equality detected"
        return z3.BoolVal(True)
    else:
        return z3.And(*[S[ii] == T[ii] for ii in product(*[range(n) for n in S.shape])])


def body_to_function(variables, body):
    assert body.decl().name() == '='
    assert body.num_args() == 2
    t1, t2 = body.arg(0), body.arg(1)

    def convert(t, values):
        if z3.is_int_value(t):
            return t.as_long()
        if z3.is_app(t):
            func = globals()[t.decl().name()]
            return func(*[convert(t.arg(i), values) for i in range(t.num_args())])
        elif z3.is_var(t):
            return values[z3.get_var_index(t)]

    def function(*vs):
        assert len(vs) == len(variables)
        # TODO: check value types
        return eq(convert(t1, vs[::-1]), convert(t2, vs[::-1]))

    return function


def check_axiom(s):
    """Based on z3.prove, adapted for multiprocessing by forking at the right place"""
    global func
    msg = str(list(s))
    try:
        vs = tuple(
            s[i] if v.sort() == P else
            Tensor.tensor(s[i], 't{}'.format(i))
            for i, v in enumerate(variables)
        )
        assert tuple(x if type(x) is int else x.shape for x in vs) == s
        claim = func(*vs)
        s = z3.Solver()
        s.add(_relu(0) == 0) # assume relu(0) = 0
        s.add(z3.Not(claim))
        r = s.check()
        if r == z3.unsat:
            return "{} proved".format(msg)
        elif r == z3.unknown:
            return "{} failed to prove\n".format(msg)
        elif r == z3.sat:
            return "{} counterexample\n{}".format(msg, s.model())
        else:
            assert False, r
    except BadShapeError:
        return "{} skipped".format(msg)


def print_function(x):
    print x


if __name__ == '__main__':

    if False:
        print "Checking that axioms imply lemmas"
        axioms = verify.axioms
        lemmas = verify.lemmas
        to_assume = [a for a, b in axioms]
        for i, lem in enumerate(lemmas):
            s = z3.Solver()
            for a in to_assume:
                s.add(a)
            s.add(z3.Not(lem))
            print("Checking lemmas[{}]: {}".format(i, lem))
            # print(s)
            if s.check() == z3.unsat:
                print("Proved!")
                to_assume.append(lem)
            else:
                assert False
        print 'Done' + '\n'*2


    if False:
        print "Checking axiom redundancies"
        axioms = verify.axioms
        flags = [z3.Bool('f{}'.format(i)) for i in range(len(axioms))]
        for i, (a,b) in reversed(list(enumerate(axioms))):
            s = z3.Solver()
            s.set("timeout", 10000)
            for j, (aa, bb) in enumerate(axioms):
                if i == j:
                    continue
                s.add(z3.Implies(flags[j], aa))
            s.add(z3.Not(a))
            print("Checking axiom {}".format(i))
            if s.check(flags) == z3.unsat:
                print "Redundant!"
                print axioms[i][0]
                core = minimize_core(s)
                print "core: {}".format(core)
                print s.check(core)
                for x in core:
                    j = int(str(x)[1:])
                    print j, axioms[j][0]
                assert False
            else:
                pass
        print 'Done' + '\n'*2


    if True:
        print "Symbolically checking axioms for small tensors"
        axioms = verify.axioms[-7:] #[35:]
        total_combinations = 0
        print now(), "Checking {} axioms...".format(len(axioms))
        spaces = [b() if b is not None else None for a,b in axioms]
        print now(), "Checking a total of {} combinations...\n".format(sum(len(x) if x is not None else 0 for x in spaces))
        for (a,b), space in zip(axioms, spaces):
            if space is None:
                continue
            print now(), "Checking:\n{}".format(a)
            assert a.is_forall()
            variables = [z3.Const(a.var_name(i), a.var_sort(i)) for i in range(a.num_vars())]
            assert all(len(x) == len(variables) for x in space)
            func = body_to_function(variables, a.body())
            n_proved = 0
            n_skipped = 0
            print "checking {} combinations...".format(len(space))
            total_combinations += len(space)
            pool = Pool(cpu_count()) # fork after computing func
            results = []
            for s in space:
                if False:
                    # this is useful for better error reporting
                    st = check_axiom(s)
                    print st
                    if 'skipped' in st:
                        n_skipped += 1
                    else:
                        n_proved += 1
                else:
                    results.append(pool.apply_async(check_axiom, [s], callback=print_function))
            # get all results, in order to raise exceptions if they occurred
            for r in results:
                st = r.get(10**10)
                if 'skipped' in st:
                    n_skipped += 1
                else:
                    n_proved += 1
            pool.close()
            pool.join()
            print now(), "checked {}, skipped {}\n".format(n_proved, n_skipped)
        print now(), "Done (total of {} combinations)".format(total_combinations)
