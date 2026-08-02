import math
import random


class Config:
    def __init__(self, d_model=16, n_heads=2, n_layers=2, d_ff=32,
                 seq_len=24, lr=0.01, seed=0):
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.seq_len = seq_len
        self.lr = lr
        self.seed = seed


class Tokenizer:
    def __init__(self, corpus):
        self.pad_id = 0
        chars = ['<pad>'] + sorted(set(corpus))
        self.itos = chars
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.vocab_size = len(chars)

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, ids):
        return ''.join('' if i == 0 else self.itos[i] for i in ids)


def linear_forward(x, wt, b):
    B, T, Din = len(x), len(x[0]), len(x[0][0])
    Dout = len(wt)
    out = [[[b[j] for j in range(Dout)] for _ in range(T)] for _ in range(B)]
    for bo in range(B):
        row = out[bo]
        for t in range(T):
            xi = x[bo][t]
            o = row[t]
            for j in range(Dout):
                wj = wt[j]
                acc = o[j]
                for i in range(Din):
                    acc += xi[i] * wj[i]
                o[j] = acc
    return out


def linear_backward(dout, x, wt, b):
    B, T, Din = len(x), len(x[0]), len(x[0][0])
    Dout = len(wt)
    dwt = [[0.0] * Din for _ in range(Dout)]
    db = [0.0] * Dout
    dx = [[[0.0] * Din for _ in range(T)] for _ in range(B)]
    for bo in range(B):
        for t in range(T):
            xi = x[bo][t]
            do = dout[bo][t]
            dxi = dx[bo][t]
            for j in range(Dout):
                v = do[j]
                db[j] += v
                dwj = dwt[j]
                for i in range(Din):
                    dwj[i] += xi[i] * v
                    dxi[i] += v * wt[j][i]
    return dx, dwt, db


def gelu(x):
    return 0.5 * x * (1.0 + math.erf(x / math.sqrt(2.0)))


def gelu_forward(x):
    return [[[gelu(v) for v in row] for row in mat] for mat in x]


def gelu_backward(dout, x):
    c = 1.0 / math.sqrt(2.0 * math.pi)
    out = []
    for b, (mat, dmat) in enumerate(zip(x, dout)):
        rows = []
        for r, (row, drow) in enumerate(zip(mat, dmat)):
            rows.append([dv * (0.5 * (1.0 + math.erf(v / math.sqrt(2.0)))
                               + v * c * math.exp(-0.5 * v * v))
                         for v, dv in zip(row, drow)])
        out.append(rows)
    return out


def layernorm_forward(x, g, b, eps=1e-5):
    B, T, D = len(x), len(x[0]), len(x[0][0])
    y = [[[0.0] * D for _ in range(T)] for _ in range(B)]
    cache = [[None] * T for _ in range(B)]
    for bo in range(B):
        for t in range(T):
            row = x[bo][t]
            mean = sum(row) / D
            var = sum((v - mean) ** 2 for v in row) / D
            inv = 1.0 / math.sqrt(var + eps)
            xhat = [(v - mean) * inv for v in row]
            y[bo][t] = [xhat[i] * g[i] + b[i] for i in range(D)]
            cache[bo][t] = (row, mean, inv, xhat)
    return y, cache


def layernorm_backward(dy, cache, g, b, eps=1e-5):
    B, T, D = len(dy), len(dy[0]), len(dy[0][0])
    dx = [[[0.0] * D for _ in range(T)] for _ in range(B)]
    dg = [0.0] * D
    db = [0.0] * D
    for bo in range(B):
        for t in range(T):
            do = dy[bo][t]
            row, mean, inv, xhat = cache[bo][t]
            dxhat = [do[i] * g[i] for i in range(D)]
            dvar = sum(dxhat[i] * (row[i] - mean) * (-0.5) * (inv ** 3)
                       for i in range(D))
            dmean = sum(dxhat[i] * (-inv) for i in range(D)) + \
                dvar * (-2.0 / D) * sum(row[i] - mean for i in range(D))
            for i in range(D):
                dg[i] += do[i] * xhat[i]
                db[i] += do[i]
                dx[bo][t][i] = (dxhat[i] * inv +
                                dvar * 2.0 * (row[i] - mean) / D +
                                dmean / D)
    return dx, dg, db


def softmax3d(logits):
    B, T, V = len(logits), len(logits[0]), len(logits[0][0])
    probs = [[[0.0] * V for _ in range(T)] for _ in range(B)]
    for bo in range(B):
        for t in range(T):
            row = logits[bo][t]
            m = max(row)
            e = [math.exp(v - m) for v in row]
            s = sum(e)
            probs[bo][t] = [v / s for v in e]
    return probs


def ce_loss_and_grad(probs, yb):
    B, T, V = len(probs), len(probs[0]), len(probs[0][0])
    n = B * T
    loss = 0.0
    d = [[[0.0] * V for _ in range(T)] for _ in range(B)]
    for bo in range(B):
        for t in range(T):
            y = yb[bo][t]
            loss -= math.log(probs[bo][t][y] + 1e-12)
            for v in range(V):
                d[bo][t][v] = (probs[bo][t][v] -
                               (1.0 if v == y else 0.0)) / n
    return loss / n, d


def attention_forward(x1, layer, cfg):
    B, T, d = len(x1), len(x1[0]), cfg.d_model
    H = cfg.n_heads
    dh = d // H
    qkv = linear_forward(x1, layer['wqkv'], layer['bqkv'])
    Q = [[qkv[b][t][:d] for t in range(T)] for b in range(B)]
    K = [[qkv[b][t][d:2 * d] for t in range(T)] for b in range(B)]
    V = [[qkv[b][t][2 * d:] for t in range(T)] for b in range(B)]
    Qh = [[[[Q[b][t][h * dh + hh] for hh in range(dh)]
            for h in range(H)] for t in range(T)] for b in range(B)]
    Kh = [[[[K[b][t][h * dh + hh] for hh in range(dh)]
            for h in range(H)] for t in range(T)] for b in range(B)]
    Vh = [[[[V[b][t][h * dh + hh] for hh in range(dh)]
            for h in range(H)] for t in range(T)] for b in range(B)]
    scale = 1.0 / math.sqrt(dh)
    probs = [[[[0.0] * T for _ in range(T)] for _ in range(H)] for _ in range(B)]
    ctx = [[[[0.0] * dh for _ in range(H)] for _ in range(T)] for _ in range(B)]
    for b in range(B):
        for h in range(H):
            Qb = Qh[b]
            Kb = Kh[b]
            Vb = Vh[b]
            Pb = probs[b][h]
            for i in range(T):
                qi = Qb[i][h]
                row = Pb[i]
                for j in range(i + 1):
                    kj = Kb[j][h]
                    s = 0.0
                    for hh in range(dh):
                        s += qi[hh] * kj[hh]
                    row[j] = s * scale
                m = max(row[:i + 1])
                esum = 0.0
                for j in range(i + 1):
                    e = math.exp(row[j] - m)
                    row[j] = e
                    esum += e
                for j in range(i + 1):
                    row[j] /= esum
                vj = Vb[i][h]
                for hh in range(dh):
                    acc = 0.0
                    for j in range(i + 1):
                        acc += row[j] * Vb[j][h][hh]
                    ctx[b][i][h][hh] = acc
    ctx_flat = [[[ctx[b][t][h][hh] for h in range(H) for hh in range(dh)]
                 for t in range(T)] for b in range(B)]
    attn_out = linear_forward(ctx_flat, layer['wo'], layer['bo'])
    return attn_out, (Qh, Kh, Vh, probs, ctx_flat, x1)


def attention_backward(d_attn_out, cache, layer, cfg):
    Qh, Kh, Vh, probs, ctx_flat, x1 = cache
    B, T, d = len(Qh), len(Qh[0]), cfg.d_model
    H = cfg.n_heads
    dh = d // H
    dx_ctx_flat, dwo, dbo = linear_backward(d_attn_out, ctx_flat,
                                            layer['wo'], layer['bo'])
    dctx = [[[[dx_ctx_flat[b][t][h * dh + hh] for hh in range(dh)]
              for h in range(H)] for t in range(T)] for b in range(B)]
    dQh = [[[[0.0] * dh for _ in range(H)] for _ in range(T)] for _ in range(B)]
    dKh = [[[[0.0] * dh for _ in range(H)] for _ in range(T)] for _ in range(B)]
    dVh = [[[[0.0] * dh for _ in range(H)] for _ in range(T)] for _ in range(B)]
    scale = 1.0 / math.sqrt(dh)
    for b in range(B):
        for h in range(H):
            Qb = Qh[b]
            Vb = Vh[b]
            Kb = Kh[b]
            Pb = probs[b][h]
            dp = [[0.0] * T for _ in range(T)]
            for i in range(T):
                dci = dctx[b][i][h]
                pi = Pb[i]
                for j in range(i + 1):
                    pij = pi[j]
                    if pij == 0.0:
                        continue
                    vj = Vb[j][h]
                    acc = 0.0
                    for hh in range(dh):
                        acc += dci[hh] * vj[hh]
                        dVh[b][j][h][hh] += pij * dci[hh]
                    dp[i][j] = acc
            for i in range(T):
                pi = Pb[i]
                dot = 0.0
                for j in range(i + 1):
                    dot += pi[j] * dp[i][j]
                qi = Qb[i][h]
                for j in range(i + 1):
                    s = pi[j] * (dp[i][j] - dot) * scale
                    if s != 0.0:
                        kj = Kb[j][h]
                        for hh in range(dh):
                            dQh[b][i][h][hh] += s * kj[hh]
                            dKh[b][j][h][hh] += s * qi[hh]
    dqkv = [[[0.0] * (3 * d) for _ in range(T)] for _ in range(B)]
    for b in range(B):
        for t in range(T):
            row = dqkv[b][t]
            for h in range(H):
                for hh in range(dh):
                    row[h * dh + hh] += dQh[b][t][h][hh]
                    row[d + h * dh + hh] += dKh[b][t][h][hh]
                    row[2 * d + h * dh + hh] += dVh[b][t][h][hh]
    dx1, dwqkv, dbqkv = linear_backward(dqkv, x1, layer['wqkv'],
                                        layer['bqkv'])
    return dx1, {'wqkv': dwqkv, 'bqkv': dbqkv, 'wo': dwo, 'bo': dbo}


class MiniLM:
    def __init__(self, cfg, tok):
        self.cfg = cfg
        self.tok = tok
        self.V = tok.vocab_size
        self.d = cfg.d_model
        self.T = cfg.seq_len
        self.n_heads = cfg.n_heads
        assert self.d % cfg.n_heads == 0
        rng = random.Random(cfg.seed)

        def mat(nrow, ncol):
            return [[rng.gauss(0.0, 1.0 / math.sqrt(ncol))
                     for _ in range(ncol)] for _ in range(nrow)]

        self.W_e = [[rng.gauss(0.0, 0.1) for _ in range(self.d)]
                    for _ in range(self.V)]
        self.PE = [[rng.gauss(0.0, 0.05) for _ in range(self.d)]
                   for _ in range(self.T)]
        self.layers = []
        for _ in range(cfg.n_layers):
            layer = {
                'ln1g': [1.0] * self.d,
                'ln1b': [0.0] * self.d,
                'ln2g': [1.0] * self.d,
                'ln2b': [0.0] * self.d,
                'wqkv': mat(3 * self.d, self.d),
                'bqkv': [0.0] * (3 * self.d),
                'wo': mat(self.d, self.d),
                'bo': [0.0] * self.d,
                'wmlp1': mat(cfg.d_ff, self.d),
                'bmlp1': [0.0] * cfg.d_ff,
                'wmlp2': mat(self.d, cfg.d_ff),
                'bmlp2': [0.0] * self.d,
            }
            self.layers.append(layer)
        self.W_h = mat(self.V, self.d)
        self.b_h = [0.0] * self.V
        self.cache = {}

    def params(self):
        params = {'W_e': self.W_e, 'PE': self.PE}
        for i, layer in enumerate(self.layers):
            for key, val in layer.items():
                params['%s_%d' % (key, i)] = val
        params['W_h'] = self.W_h
        params['b_h'] = self.b_h
        return params

    def set_params(self, params):
        def assign(dst, src):
            if isinstance(dst[0], list):
                for i in range(len(dst)):
                    for j in range(len(dst[i])):
                        dst[i][j] = src[i][j]
            else:
                for i in range(len(dst)):
                    dst[i] = src[i]
        cur = self.params()
        for key in cur:
            assign(cur[key], params[key])

    def n_params(self):
        n = 0
        for key, p in self.params().items():
            if isinstance(p[0], list):
                n += sum(len(row) for row in p)
            else:
                n += len(p)
        return n

    def forward(self, xb, yb=None):
        cfg = self.cfg
        B, T = len(xb), len(xb[0])
        d = self.d
        X0 = [[[self.W_e[xb[b][t]][i] + self.PE[t][i] for i in range(d)]
               for t in range(T)] for b in range(B)]
        X = X0
        self.cache = {'X0': X0, 'xb': xb, 'layers': []}
        for layer in self.layers:
            x1, ln1c = layernorm_forward(X, layer['ln1g'], layer['ln1b'])
            attn_out, ac = attention_forward(x1, layer, cfg)
            A = [[[attn_out[b][t][i] + X[b][t][i] for i in range(d)]
                  for t in range(T)] for b in range(B)]
            x2, ln2c = layernorm_forward(A, layer['ln2g'], layer['ln2b'])
            h1 = linear_forward(x2, layer['wmlp1'], layer['bmlp1'])
            h2 = gelu_forward(h1)
            h3 = linear_forward(h2, layer['wmlp2'], layer['bmlp2'])
            H = [[[h3[b][t][i] + A[b][t][i] for i in range(d)]
                  for t in range(T)] for b in range(B)]
            self.cache['layers'].append({
                'X': X, 'x1': x1, 'ln1c': ln1c, 'attn': ac, 'A': A,
                'x2': x2, 'ln2c': ln2c, 'h1': h1, 'h2': h2, 'H': H,
            })
            X = H
        logits = linear_forward(X, self.W_h, self.b_h)
        self.cache['logits'] = logits
        self.cache['H'] = X
        probs = softmax3d(logits)
        self.cache['probs'] = probs
        loss = None
        if yb is not None:
            loss, dlogits = ce_loss_and_grad(probs, yb)
            self.cache['yb'] = yb
            self.cache['dlogits'] = dlogits
            self.cache['loss'] = loss
        return logits, loss

    def backward(self, scale=1.0):
        cfg = self.cfg
        c = self.cache
        d = self.d
        dlogits = c['dlogits']
        B, T, V = len(dlogits), len(dlogits[0]), len(dlogits[0][0])
        if scale != 1.0:
            dlogits = [[[dlogits[b][t][v] * scale for v in range(V)]
                        for t in range(T)] for b in range(B)]
        dx, dw_h, db_h = linear_backward(dlogits, c['H'], self.W_h, self.b_h)
        grads = {
            'W_h': dw_h,
            'b_h': db_h,
            'W_e': [[0.0] * d for _ in range(self.V)],
            'PE': [[0.0] * d for _ in range(self.T)],
        }
        for li in range(len(self.layers) - 1, -1, -1):
            lc = c['layers'][li]
            layer = self.layers[li]
            dA_mlp, dwmlp2, dbmlp2 = linear_backward(
                dx, lc['h2'], layer['wmlp2'], layer['bmlp2'])
            dh2 = gelu_backward(dA_mlp, lc['h1'])
            dx2, dwmlp1, dbmlp1 = linear_backward(
                dh2, lc['x2'], layer['wmlp1'], layer['bmlp1'])
            dA_ln2, dg2, db2 = layernorm_backward(
                dx2, lc['ln2c'], layer['ln2g'], layer['ln2b'])
            dA = [[[dx[b][t][i] + dA_ln2[b][t][i] for i in range(d)]
                   for t in range(T)] for b in range(B)]
            dx1, g_attn = attention_backward(dA, lc['attn'], layer, cfg)
            dX_ln1, dg1, db1 = layernorm_backward(
                dx1, lc['ln1c'], layer['ln1g'], layer['ln1b'])
            dx = [[[dA[b][t][i] + dX_ln1[b][t][i] for i in range(d)]
                   for t in range(T)] for b in range(B)]
            grads['ln1g_%d' % li] = dg1
            grads['ln1b_%d' % li] = db1
            grads['ln2g_%d' % li] = dg2
            grads['ln2b_%d' % li] = db2
            grads['wqkv_%d' % li] = g_attn['wqkv']
            grads['bqkv_%d' % li] = g_attn['bqkv']
            grads['wo_%d' % li] = g_attn['wo']
            grads['bo_%d' % li] = g_attn['bo']
            grads['wmlp1_%d' % li] = dwmlp1
            grads['bmlp1_%d' % li] = dbmlp1
            grads['wmlp2_%d' % li] = dwmlp2
            grads['bmlp2_%d' % li] = dbmlp2
        xb = c['xb']
        dx0 = dx
        for b in range(B):
            for t in range(T):
                tok = xb[b][t]
                row = self.W_e[tok]
                drow = grads['W_e'][tok]
                dpe = grads['PE'][t]
                for i in range(d):
                    drow[i] += dx0[b][t][i]
                    dpe[i] += dx0[b][t][i]
        return grads


class Adam:
    def __init__(self, params, lr, beta1=0.9, beta2=0.999, eps=1e-8,
                 clip=5.0):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.clip = clip
        self.t = 0
        self.m = {k: self._zeros(p) for k, p in params.items()}
        self.v = {k: self._zeros(p) for k, p in params.items()}

    def _zeros(self, p):
        if isinstance(p[0], list):
            return [[0.0] * len(row) for row in p]
        return [0.0] * len(p)

    def _iter(self, p):
        if isinstance(p[0], list):
            for i in range(len(p)):
                for j in range(len(p[i])):
                    yield i, j
        else:
            for i in range(len(p)):
                yield i, None

    def step(self, grads):
        self.t += 1
        beta1 = self.beta1
        beta2 = self.beta2
        lr_t = self.lr * math.sqrt(1.0 - beta2 ** self.t) / \
            (1.0 - beta1 ** self.t)
        for key, g in grads.items():
            p = self.params[key]
            m = self.m[key]
            v = self.v[key]
            for i, j in self._iter(p):
                if j is None:
                    gv = g[i]
                    if gv > self.clip:
                        gv = self.clip
                    elif gv < -self.clip:
                        gv = -self.clip
                    mi = m[i]
                    vi = v[i]
                    mi = mi * beta1 + (1.0 - beta1) * gv
                    vi = vi * beta2 + (1.0 - beta2) * gv * gv
                    m[i] = mi
                    v[i] = vi
                    mh = mi / (1.0 - beta1 ** self.t)
                    vh = vi / (1.0 - beta2 ** self.t)
                    p[i] -= lr_t * mh / (math.sqrt(vh) + self.eps)
                else:
                    gv = g[i][j]
                    if gv > self.clip:
                        gv = self.clip
                    elif gv < -self.clip:
                        gv = -self.clip
                    mi = m[i][j]
                    vi = v[i][j]
                    mi = mi * beta1 + (1.0 - beta1) * gv
                    vi = vi * beta2 + (1.0 - beta2) * gv * gv
                    m[i][j] = mi
                    v[i][j] = vi
                    mh = mi / (1.0 - beta1 ** self.t)
                    vh = vi / (1.0 - beta2 ** self.t)
                    p[i][j] -= lr_t * mh / (math.sqrt(vh) + self.eps)


def sample_from_logits(logits, temperature, top_k, rng):
    if temperature != 1.0:
        logits = [l / temperature for l in logits]
    m = max(logits)
    e = [math.exp(l - m) for l in logits]
    k = min(top_k, len(e))
    cut = sorted(range(len(e)), key=lambda i: e[i], reverse=True)[k - 1]
    cutoff = e[cut]
    masked = [v if v >= cutoff else 0.0 for v in e]
    s = sum(masked)
    r = rng.random()
    acc = 0.0
    for i, p in enumerate(masked):
        if s > 0:
            acc += p / s
        if r <= acc:
            return i
    return len(masked) - 1
