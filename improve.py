import argparse
import json
import math
import random
import time

from llm import (Adam, Config, MiniLM, Tokenizer, sample_from_logits)


CORPUS = [
    ("hi", "hello"),
    ("hello", "hi"),
    ("hey", "hi there"),
    ("how are you", "i am fine"),
    ("good morning", "good morning to you"),
    ("good night", "good night to you"),
    ("what is your name", "my name is llm"),
    ("what are you", "i am a language model"),
    ("what is an llm", "an llm is a language model"),
    ("what can you do", "i can answer questions"),
    ("do you like cake", "yes i love cake"),
    ("what color is the sky", "the sky is blue"),
    ("what color is grass", "grass is green"),
    ("is the sun hot", "yes the sun is hot"),
    ("how many legs does a cat have", "a cat has four legs"),
    ("how many legs does a bird have", "a bird has two legs"),
    ("what does a dog say", "a dog says woof"),
    ("what does a cat say", "a cat says meow"),
    ("what is two plus two", "two plus two is four"),
    ("what is three plus one", "three plus one is four"),
    ("what is five minus two", "five minus two is three"),
    ("what is one plus one", "one plus one is two"),
    ("what is ten divided by two", "ten divided by two is five"),
    ("is ice cold", "yes ice is very cold"),
    ("what is a tree", "a tree has leaves"),
    ("what is a book", "a book has pages"),
    ("can fish fly", "no fish cannot fly"),
    ("what is red", "red is a color"),
    ("do you like books", "yes i like books"),
    ("what color is snow", "snow is white"),
]

STOPWORDS = {
    'the', 'and', 'a', 'an', 'is', 'are', 'of', 'to', 'in', 'on', 'for',
    'was', 'you', 'do', 'does', 'what', 'how', 'i', 'am', 'it', 'my',
    'can', 'yes', 'no', 'very',
}


def build_data():
    corpus = '\n'.join('Q: %s A: %s' % (q, a) for q, a in CORPUS) + '\n'
    return corpus, CORPUS


def keywords_for(reference):
    out = []
    for w in reference.lower().split():
        w = w.strip('.,!?')
        if w not in STOPWORDS and len(w) >= 3:
            out.append(w)
    return out


def reward(prompt, response, reference):
    r = 0.0
    resp = response.lower().strip()
    words = [w.strip('.,!?') for w in resp.split()]
    kws = keywords_for(reference)
    hits = sum(1 for kw in kws if any(kw in w for w in words))
    if kws:
        r += 3.0 * hits / len(kws)
        if hits == len(kws):
            r += 2.0
    target = len(reference)
    if target > 0:
        r += 2.0 * max(0.0, 1.0 - abs(len(response) - target) / target)
    s = response.strip()
    if s and s[0].isalpha():
        r += 0.5
    if s and s[-1] in '.!?':
        r += 0.5
    if len(words) >= 3 and len(set(words)) <= len(words) * 0.5:
        r -= 1.0
    return r


def add_grads(acc, g):
    for key, val in g.items():
        if key not in acc:
            acc[key] = val
        else:
            cur = acc[key]
            if isinstance(val[0], list):
                for i in range(len(val)):
                    row = cur[i]
                    vrow = val[i]
                    for j in range(len(vrow)):
                        row[j] += vrow[j]
            else:
                for i in range(len(val)):
                    cur[i] += val[i]


def make_windows(tok, data_ids, cfg):
    T = cfg.seq_len
    stride = max(1, T // 2)
    windows = []
    for i in range(0, len(data_ids) - T, stride):
        windows.append(data_ids[i:i + T + 1])
    return windows


def train_base(model, opt, windows, cfg, args):
    rng = random.Random(cfg.seed + 1)
    print('  base training (pure-python transformer, %d params)' %
          model.n_params())
    for epoch in range(args.epochs):
        rng.shuffle(windows)
        tot = 0.0
        n = 0
        for b in range(0, len(windows), args.batch_size):
            batch = windows[b:b + args.batch_size]
            _, loss = model.forward([w[:-1] for w in batch],
                                    [w[1:] for w in batch])
            opt.step(model.backward())
            tot += loss
            n += 1
        if epoch == 0 or (epoch + 1) % 5 == 0:
            print('    epoch %3d/%d  loss %.4f' %
                  (epoch + 1, args.epochs, tot / n), flush=True)


def generate(model, tok, prompt, max_new, temperature, top_k, rng):
    ids = tok.encode(prompt)
    cfg = model.cfg
    for _ in range(max_new):
        window = ids[-cfg.seq_len:]
        if len(window) < cfg.seq_len:
            pref = tok.encode(prompt)
            while len(window) < cfg.seq_len:
                window = pref + window
            window = window[-cfg.seq_len:]
        logits, _ = model.forward([window])
        ids.append(sample_from_logits(logits[0][-1], temperature, top_k, rng))
    return ids


def respond(model, tok, prompt, temperature=0.8, top_k=5, rng=None,
            max_new=16):
    rng = rng or random.Random(7)
    ids = generate(model, tok, prompt, max_new, temperature, top_k, rng)
    text = tok.decode(ids)[len(prompt):]
    return text.split('\n')[0].strip()


def eval_rewards(model, tok, pairs, temperature=0.6, samples=2, max_new=16):
    rng = random.Random(99)
    total = 0.0
    for q, a in pairs:
        prompt = 'Q: %s A: ' % q
        rs = [reward(prompt, respond(model, tok, prompt, temperature, 4, rng),
                     a) for _ in range(samples)]
        total += sum(rs) / samples
    return total / len(pairs)


def distill_round_items(model, tok, opt, windows, pairs, args, cfg, rng):
    selected = []
    for q, a in pairs[:args.use_pairs]:
        prompt = 'Q: %s A: ' % q
        best = None
        for _ in range(args.samples):
            resp = respond(model, tok, prompt, 0.9, 5, rng)
            rr = reward(prompt, resp, a)
            if best is None or rr > best[0]:
                best = (rr, resp)
        if best[0] >= args.threshold:
            selected.append((q, a, best[1], best[0]))
    avg_sel = sum(s[3] for s in selected) / len(selected) if selected else 0.0
    pool = list(windows)
    random.Random(0).shuffle(pool)
    for _ in range(args.fine_steps):
        acc = {}
        for seq in pool[:args.batch_size * 3]:
            model.forward([seq[:-1]], [seq[1:]])
            add_grads(acc, model.backward(scale=1.0))
        for q, a, resp, rr in selected:
            seq = tok.encode('Q: %s A: %s' % (q, resp))
            seq = seq[-(cfg.seq_len + 1):]
            if len(seq) < 2:
                continue
            model.forward([seq[:-1]], [seq[1:]])
            add_grads(acc, model.backward(scale=rr / 3.0))
        opt.step(acc)
    return avg_sel, (selected[0][2] if selected else '')


def reinforce_round(model, tok, opt, windows, pairs, args, cfg, rng):
    acc = {}
    all_rewards = []
    samples = []
    for q, a in pairs[:args.use_pairs]:
        prompt = 'Q: %s A: ' % q
        for _ in range(args.samples):
            resp = respond(model, tok, prompt, 1.0, 5, rng)
            rr = reward(prompt, resp, a)
            samples.append((q, resp, rr))
            all_rewards.append(rr)
    baseline = sum(all_rewards) / len(all_rewards) if all_rewards else 0.0
    for q, resp, rr in samples:
        seq = tok.encode('Q: %s A: %s' % (q, resp))
        seq = seq[-(cfg.seq_len + 1):]
        if len(seq) < 2:
            continue
        w = (len(seq) - 1) * (rr - baseline)
        if abs(w) < 1e-9:
            continue
        model.forward([seq[:-1]], [seq[1:]])
        add_grads(acc, model.backward(scale=w))
    pool = list(windows)
    random.Random(0).shuffle(pool)
    for seq in pool[:args.batch_size * 3]:
        model.forward([seq[:-1]], [seq[1:]])
        add_grads(acc, model.backward(scale=1.0))
    opt.step(acc)
    return baseline


def main():
    ap = argparse.ArgumentParser(
        description='A tiny from-scratch LLM that improves itself')
    ap.add_argument('--mode', choices=['all', 'distill', 'reinforce'],
                    default='all')
    ap.add_argument('--rounds', type=int, default=8)
    ap.add_argument('--samples', type=int, default=3)
    ap.add_argument('--fine-steps', type=int, default=2)
    ap.add_argument('--epochs', type=int, default=150)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--use-pairs', type=int, default=8)
    ap.add_argument('--eval-pairs', type=int, default=12)
    ap.add_argument('--threshold', type=float, default=2.0)
    ap.add_argument('--d-model', type=int, default=12)
    ap.add_argument('--n-layers', type=int, default=1)
    ap.add_argument('--n-heads', type=int, default=2)
    ap.add_argument('--d-ff', type=int, default=24)
    ap.add_argument('--seq-len', type=int, default=16)
    ap.add_argument('--lr', type=float, default=0.01)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--chat', action='store_true',
                    help='drop into a REPL with the improved model')
    ap.add_argument('--save', metavar='PATH',
                    help='save trained weights as JSON after the run')
    ap.add_argument('--load', metavar='PATH',
                    help='load weights from a JSON save before running')
    ap.add_argument('--tiny', action='store_true',
                    help='fast smoke-test configuration')
    args = ap.parse_args()

    if args.tiny:
        args.d_model = 8
        args.n_layers = 1
        args.n_heads = 1
        args.d_ff = 16
        args.seq_len = 16
        args.epochs = 5
        args.rounds = 3
        args.samples = 2
        args.fine_steps = 1
        args.use_pairs = 4
        args.eval_pairs = 6

    cfg = Config(d_model=args.d_model, n_heads=args.n_heads,
                 n_layers=args.n_layers, d_ff=args.d_ff,
                 seq_len=args.seq_len, lr=args.lr, seed=args.seed)
    corpus, pairs = build_data()
    tok = Tokenizer(corpus)
    model = MiniLM(cfg, tok)
    if args.load:
        with open(args.load, 'r', encoding='utf-8') as f:
            model.set_params(json.load(f))
        print('loaded weights from %s' % args.load, flush=True)
    opt = Adam(model.params(), lr=cfg.lr)
    data_ids = tok.encode(corpus)
    windows = make_windows(tok, data_ids, cfg)

    print('vocab=%d  params=%d' % (tok.vocab_size, model.n_params()),
          flush=True)
    train_base(model, opt, windows, cfg, args)

    base_full = eval_rewards(model, tok, pairs)
    base_reward = eval_rewards(model, tok, pairs[:args.eval_pairs])
    print('\nbefore self-improvement: avg reward %.3f  (full %d pairs: %.3f)'
          % (base_reward, len(pairs), base_full), flush=True)
    for q, _ in pairs[:4]:
        print('  Q: %s  A: %s' % (q, respond(model, tok, 'Q: %s A: ' % q)),
              flush=True)

    print('\nself-improvement loop (%s):' % args.mode, flush=True)
    history = [base_reward]
    rng = random.Random(cfg.seed + 2)
    for r in range(1, args.rounds + 1):
        t0 = time.perf_counter()
        if args.mode == 'distill' or (args.mode == 'all' and r % 2 == 1):
            avg_sel, best = distill_round_items(model, tok, opt, windows,
                                                pairs, args, cfg, rng)
            desc = 'distill'
        else:
            avg_sel = reinforce_round(model, tok, opt, windows, pairs,
                                      args, cfg, rng)
            best = ''
            desc = 'reinforce'
        avg = eval_rewards(model, tok, pairs[:args.eval_pairs])
        history.append(avg)
        print('  round %2d [%s] reward %.3f  sel %.2f  (+%.3f)  best: %s '
              '(%.1fs)' % (r, desc, avg, avg_sel, avg - history[r - 1],
                           best, time.perf_counter() - t0), flush=True)

    print('\nself-improvement curve (avg reward):')
    for i, h in enumerate(history):
        print('  %d: %.3f' % (i, h))
    print('  gain: %+.3f' % (history[-1] - history[0]), flush=True)

    after_full = eval_rewards(model, tok, pairs)
    print('\nafter self-improvement: avg reward %.3f  (full %d pairs: %.3f)'
          % (history[-1], len(pairs), after_full), flush=True)
    for q, _ in pairs[:8]:
        print('  Q: %s  A: %s' % (q, respond(model, tok, 'Q: %s A: ' % q)),
              flush=True)

    if args.save:
        with open(args.save, 'w', encoding='utf-8') as f:
            json.dump(model.params(), f)
        print('\nsaved weights to %s' % args.save, flush=True)

    if args.chat:
        print('\nchat with the improved model (empty line to quit)')
        while True:
            q = input('you> ').strip()
            if not q:
                break
            print('llm> %s' % respond(model, tok, 'Q: %s A: ' % q))


if __name__ == '__main__':
    main()
