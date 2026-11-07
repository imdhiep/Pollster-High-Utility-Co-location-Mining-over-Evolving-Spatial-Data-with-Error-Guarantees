"""v18 FSQ exact (k<=4) in parallel chunks over a fixed permutation, plus Pollster estimates with the same kernel.
MODE=exact: process perm[START:END] until TLIM seconds, checkpointing acc (dense, rank-indexed num(c)) to OUT.
MODE=pollster: priority samples m in MS per feature; same kernel with weights max(u, tau); saves est arrays + timings.
All counts use numba prange over T threads with per-thread dense accumulators and per-thread dedupe stamps."""
import os, json, time, itertools, faulthandler, subprocess, sys, gc
faulthandler.enable()
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
import numba
from numba import prange
T0 = time.time()
def log(*a):
    print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)
OUT = os.environ.get('V9_OUT', '/kaggle/working')
REL = os.environ.get('V14_REL', '2026-10-15'); NFILES = int(os.environ.get('V14_NFILES', '100'))
K = int(os.environ.get('V14_K', '60')); D = float(os.environ.get('V14_D', '150'))
MODE = os.environ.get('V18_MODE', 'exact'); KMAX = int(os.environ.get('V18_KMAX', '4'))
START = int(os.environ.get('V18_START', '0')); END = int(float(os.environ.get('V18_END', '1e12')))
TLIM = float(os.environ.get('V18_TLIM', '41000')); MS = [int(x) for x in os.environ.get('V18_MS', '300,1000').split(',')]
TAG = os.environ.get('V18_TAG', MODE)
S = dict(rel=REL, K=K, d=D, mode=MODE, kmax=KMAX, start=START, threads=numba.config.NUMBA_NUM_THREADS)
os.makedirs(OUT, exist_ok=True)
def save_meta():
    json.dump(S, open(os.path.join(OUT, f'fsq_{TAG}.json'), 'w'), indent=1, default=float)
def to_xyz(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    return np.c_[np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)]

if os.environ.get('V18_FAKE') != '1':
    # ------------------------------------------------------------------ load FSQ
    from kaggle_secrets import UserSecretsClient
    tok = UserSecretsClient().get_secret("HF_TOKEN")
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'huggingface_hub>=0.24'], check=False)
    from huggingface_hub import HfApi, hf_hub_download
    import pyarrow.parquet as pq, pyarrow.compute as pc
    api = HfApi(token=tok)
    files = sorted(f for f in api.list_repo_files('foursquare/fsq-os-places', repo_type='dataset')
                   if f.startswith(f'release/dt={REL}/places/parquet/') and f.endswith('.parquet'))[:NFILES]
    log('files', len(files))
    lat_l, lon_l, code_l, dict_l, id_l = [], [], [], [], []
    from collections import Counter
    gcount = Counter()
    for i, f in enumerate(files):
        p = hf_hub_download('foursquare/fsq-os-places', f, repo_type='dataset', token=tok, local_dir='/tmp/fsq')
        t = pq.read_table(p, columns=['fsq_place_id', 'latitude', 'longitude', 'date_closed', 'fsq_category_ids'])
        os.remove(p)
        ln = pc.fill_null(pc.list_value_length(t['fsq_category_ids']), 0)
        ok = pc.and_(pc.and_(pc.is_null(t['date_closed']), pc.greater(ln, 0)),
                     pc.and_(pc.is_valid(t['latitude']), pc.is_valid(t['longitude'])))
        t = t.filter(ok)
        if t.num_rows == 0:
            continue
        c0 = pc.dictionary_encode(pc.list_element(t['fsq_category_ids'], 0)).combine_chunks()
        codes = c0.indices.to_numpy(zero_copy_only=False).astype(np.int32); dic = c0.dictionary.to_pylist()
        cnt = np.bincount(codes, minlength=len(dic))
        for j, c in enumerate(dic):
            gcount[c] += int(cnt[j])
        lat_l.append(t['latitude'].to_numpy()); lon_l.append(t['longitude'].to_numpy())
        code_l.append(codes); dict_l.append(dic)
        id_l.append(pd.util.hash_array(t['fsq_place_id'].to_numpy(zero_copy_only=False).astype(object)))
        del t
        if i % 10 == 0:
            log(' file', i, 'rows kept', sum(len(x) for x in lat_l))
    TOP = [c for c, _ in gcount.most_common(K)]; fid = {c: j for j, c in enumerate(TOP)}
    fall = np.concatenate([np.array([fid.get(c, -1) for c in dic], np.int64)[codes] for codes, dic in zip(code_l, dict_l)])
    lat = np.concatenate(lat_l); lon = np.concatenate(lon_l); hid = np.concatenate(id_l)
    del lat_l, lon_l, code_l, dict_l, id_l; gc.collect()
    fin = np.isfinite(lat) & np.isfinite(lon) & (np.abs(lat) <= 90) & (np.abs(lon) <= 180)
    keep = fin & (fall >= 0)
    S['n_open_valid'] = int(fin.sum()); S['n_topK'] = int(keep.sum()); S['top_counts'] = [int(gcount[c]) for c in TOP]
    # category labels
    try:
        cf = [f for f in api.list_repo_files('foursquare/fsq-os-places', repo_type='dataset')
              if f.startswith(f'release/dt={REL}/categories/') and f.endswith('.parquet')]
        cp = hf_hub_download('foursquare/fsq-os-places', cf[0], repo_type='dataset', token=tok, local_dir='/tmp/fsq')
        ct_ = pq.read_table(cp).to_pandas()
        idc = [c for c in ct_.columns if 'category_id' in c][0]; lbc = [c for c in ct_.columns if 'name' in c or 'label' in c][0]
        lab = dict(zip(ct_[idc], ct_[lbc])); S['top_labels'] = [str(lab.get(c, c)) for c in TOP]
    except Exception as e:
        S['top_labels'] = TOP; log('labels failed', repr(e)[:200])
    log('open+valid', S['n_open_valid'], 'in top-K', S['n_topK'], 'labels', S['top_labels'][:10])
    P = to_xyz(lat[keep], lon[keep]) * 6371000.0; F = fall[keep].astype(np.int64)
    U01 = ((hid[keep] >> np.uint64(11)).astype(np.float64) + 0.5) / 9007199254740992.0
    u_all = np.exp(np.sqrt(2.0) * __import__('scipy').special.erfinv(2 * U01 - 1))   # persistent lognormal(0,1) utilities
    del lat, lon, hid, fall; gc.collect()
    N = len(P); R = np.random.default_rng(7).random(N) + 1e-12

if os.environ.get('V18_FAKE') == '1':
    rg = np.random.default_rng(1); n0 = int(os.environ.get('V18_FAKEN', '20000'))
    cen = rg.uniform(-0.02, 0.02, (40, 2)) + np.array([21.0, 105.8])
    ci = rg.integers(0, 40, n0)
    lat = cen[ci, 0] + rg.normal(0, 0.0015, n0); lon = cen[ci, 1] + rg.normal(0, 0.0015, n0)
    P = to_xyz(lat, lon) * 6371000.0; F = rg.integers(0, K, n0).astype(np.int64); u_all = np.exp(rg.normal(0, 1, n0))
    N = len(P); R = np.random.default_rng(7).random(N) + 1e-12
t = time.time(); TREE = cKDTree(P); log('tree built', N, '%.1fs' % (time.time() - t))
U = np.bincount(F, weights=u_all, minlength=K); S['N'] = int(N); S['U'] = U.tolist()

# ------------------------------------------------------------------ pattern ranks (combinatorial number system)
Cb = np.zeros((K + 1, 5), np.int64)
for n_ in range(K + 1):
    for k_ in range(5):
        Cb[n_, k_] = int(__import__('math').comb(n_, k_))
OFF3 = Cb[K, 2]; OFF4 = OFF3 + Cb[K, 3]; NR = int(OFF4 + Cb[K, 4]) if KMAX >= 4 else int(OFF4)
S['NR'] = NR

@numba.njit(inline='always')
def rk2(a, b, Cb):
    if a > b:
        a, b = b, a
    return Cb[b, 2] + Cb[a, 1]

@numba.njit(inline='always')
def rk3(a, b, c, Cb, OFF3):
    if a > b:
        a, b = b, a
    if b > c:
        b, c = c, b
    if a > b:
        a, b = b, a
    return OFF3 + Cb[c, 3] + Cb[b, 2] + Cb[a, 1]

@numba.njit(inline='always')
def rk4(a, b, c, d, Cb, OFF4):
    if a > b:
        a, b = b, a
    if c > d:
        c, d = d, c
    if a > c:
        a, c = c, a
    if b > d:
        b, d = d, b
    if b > c:
        b, c = c, b
    return OFF4 + Cb[d, 4] + Cb[c, 3] + Cb[b, 2] + Cb[a, 1]

@numba.njit(parallel=True)
def star_acc(nodes, wts, ptr, idx, PP, FF, d2, kmax, acc, seen, stamp0, cnt, Cb, OFF3, OFF4):
    T = acc.shape[0]
    for t in prange(T):
        for s in range(t, len(nodes), T):
            o = nodes[s]; st = stamp0 + s; w = wts[s]; fo = FF[o]; c = 0
            a0 = ptr[s]; a1 = ptr[s + 1]
            for i in range(a0, a1):
                v1 = idx[i]; f1 = FF[v1]
                r = rk2(fo, f1, Cb)
                if seen[t, r] != st:
                    seen[t, r] = st; acc[t, r] += w; c += 1
                if kmax < 3:
                    continue
                for j in range(i + 1, a1):
                    v2 = idx[j]; f2 = FF[v2]
                    if f2 == f1:
                        continue
                    dx = PP[v1, 0] - PP[v2, 0]; dy = PP[v1, 1] - PP[v2, 1]; dz = PP[v1, 2] - PP[v2, 2]
                    if dx * dx + dy * dy + dz * dz > d2:
                        continue
                    r = rk3(fo, f1, f2, Cb, OFF3)
                    if seen[t, r] != st:
                        seen[t, r] = st; acc[t, r] += w; c += 1
                    if kmax < 4:
                        continue
                    for k in range(j + 1, a1):
                        v3 = idx[k]; f3 = FF[v3]
                        if f3 == f1 or f3 == f2:
                            continue
                        dx = PP[v1, 0] - PP[v3, 0]; dy = PP[v1, 1] - PP[v3, 1]; dz = PP[v1, 2] - PP[v3, 2]
                        if dx * dx + dy * dy + dz * dz > d2:
                            continue
                        dx = PP[v2, 0] - PP[v3, 0]; dy = PP[v2, 1] - PP[v3, 1]; dz = PP[v2, 2] - PP[v3, 2]
                        if dx * dx + dy * dy + dz * dz > d2:
                            continue
                        r = rk4(fo, f1, f2, f3, Cb, OFF4)
                        if seen[t, r] != st:
                            seen[t, r] = st; acc[t, r] += w; c += 1
            cnt[s] = c

TH = numba.config.NUMBA_NUM_THREADS
ACC = np.zeros((TH, NR)); SEEN = np.full((TH, NR), -1, np.int64); STAMP = [0]

def neigh(nd):
    nb = TREE.query_ball_point(P[nd], D, workers=-1)
    lens = np.fromiter((len(x) for x in nb), np.int64, count=len(nd))
    flat = np.fromiter(itertools.chain.from_iterable(nb), np.int64, count=int(lens.sum()))
    owner = np.repeat(np.arange(len(nd)), lens)
    keep = F[flat] != F[nd][owner]
    flat = flat[keep]; owner = owner[keep]
    ptr = np.r_[0, np.cumsum(np.bincount(owner, minlength=len(nd)))].astype(np.int64)
    return ptr, flat

def run_nodes(nodes, wts, kmax, chunk=20000, deadline=None, on_chunk=None):
    """returns (#done, star time, query time, state entries)."""
    done = 0; tq = 0.0; te = 0.0; st_entries = 0
    for a in range(0, len(nodes), chunk):
        nd = np.ascontiguousarray(nodes[a:a + chunk], np.int64); w = np.ascontiguousarray(wts[a:a + chunk], np.float64)
        t1 = time.time(); ptr, flat = neigh(nd); tq += time.time() - t1
        cnt = np.zeros(len(nd), np.int64)
        t2 = time.time()
        star_acc(nd, w, ptr, flat, P, F, D * D, kmax, ACC, SEEN, STAMP[0], cnt, Cb, OFF3, OFF4)
        te += time.time() - t2; STAMP[0] += len(nd); done += len(nd); st_entries += int(cnt.sum())
        if on_chunk is not None:
            on_chunk(done, tq, te, st_entries)
        if deadline is not None and time.time() > deadline:
            break
    return done, te, tq, st_entries

# warm-up (compiles the parallel kernel)
_w = np.arange(min(200, N), dtype=np.int64); run_nodes(_w, np.zeros(len(_w)), KMAX); ACC[:] = 0.0
log('jit warm-up done, threads', TH, 'NR', NR)

if MODE == 'exact':
    perm = np.random.default_rng(2027).permutation(N)
    end = min(END, N); nodes = perm[START:end]
    S.update(end_target=int(end)); last = [time.time()]
    def ckpt(done, tq, te, st_entries, final=False):
        if final or time.time() - last[0] > 1800:
            np.save(os.path.join(OUT, f'acc_{TAG}.npy'), ACC.sum(0))
            S.update(done_end=int(START + done), nodes_done=int(done), star_time=te, query_time=tq,
                     state_entries=st_entries, ms_per_node=1000.0 * (te + tq) / max(done, 1), elapsed=time.time() - T0)
            save_meta(); last[0] = time.time()
            log('ckpt', json.dumps({k: S[k] for k in ('done_end', 'nodes_done', 'star_time', 'query_time', 'ms_per_node')}))
    done, te, tq, st_entries = run_nodes(nodes, u_all[nodes], KMAX, deadline=T0 + TLIM, on_chunk=ckpt)
    ckpt(done, tq, te, st_entries, final=True)
    S['complete'] = bool(START + done >= end); save_meta(); log('exact done', json.dumps(S)[:600])

if MODE == 'pollster':
    an = np.arange(N); order = np.argsort(F, kind='stable'); bnd = np.searchsorted(F[order], np.arange(K + 1))
    S['pollster'] = {}
    for m in MS:
        t0 = time.time(); sel = []; wt = []
        for f in range(K):
            idx_f = order[bnd[f]:bnd[f + 1]]
            # coordinated threshold sampling (Theorem 1): keep o iff u(o)/r_o > U(f)/m
            tau = float(u_all[idx_f].sum()) / m
            pr = u_all[idx_f] / R[idx_f]; s_ = idx_f[pr > tau]
            sel.append(s_); wt.append(np.maximum(u_all[s_], tau))
        sel = np.concatenate(sel); wt = np.concatenate(wt); ts = time.time() - t0
        for kk in sorted(set([3, KMAX])):
            ACC[:] = 0.0
            done, te, tq, st_entries = run_nodes(sel, wt, kk)
            np.save(os.path.join(OUT, f'thr_est_m{m}_k{kk}.npy'), ACC.sum(0))
            S['pollster'][f'm{m}_k{kk}'] = dict(scheme='threshold', sample_time=ts, star_time=te, query_time=tq, total=ts + te + tq,
                                               sampled=int(len(sel)), state_entries=st_entries)
            save_meta(); log('pollster', m, kk, json.dumps(S['pollster'][f'm{m}_k{kk}']))
    # rank -> feature mask table, for merging/evaluation outside
    masks = np.zeros(NR, np.int64)
    for a_ in range(K):
        for b_ in range(a_ + 1, K):
            masks[Cb[b_, 2] + Cb[a_, 1]] = (1 << a_) | (1 << b_)
            for c_ in range(b_ + 1, K):
                masks[OFF3 + Cb[c_, 3] + Cb[b_, 2] + Cb[a_, 1]] = (1 << a_) | (1 << b_) | (1 << c_)
                if KMAX >= 4:
                    for d_ in range(c_ + 1, K):
                        masks[OFF4 + Cb[d_, 4] + Cb[c_, 3] + Cb[b_, 2] + Cb[a_, 1]] = (1 << a_) | (1 << b_) | (1 << c_) | (1 << d_)
    np.save(os.path.join(OUT, 'rank_masks.npy'), masks); np.save(os.path.join(OUT, 'U.npy'), U)
    log('pollster done')

if os.environ.get('V18_CHECK') == '1':
    ref = np.zeros(NR)
    nodes = np.arange(N); ptr, flat = neigh(nodes)
    for s in range(N):
        nb = flat[ptr[s]:ptr[s+1]]; seenr = set()
        for kk in range(1, KMAX):
            for comb in itertools.combinations(nb, kk):
                fs = [F[s]] + [F[x] for x in comb]
                if len(set(fs)) < len(fs): continue
                ok = all(np.sum((P[x]-P[y])**2) <= D*D for x, y in itertools.combinations(comb, 2))
                if not ok: continue
                fs = sorted(fs)
                if len(fs) == 2: r = Cb[fs[1],2]+Cb[fs[0],1]
                elif len(fs) == 3: r = OFF3+Cb[fs[2],3]+Cb[fs[1],2]+Cb[fs[0],1]
                else: r = OFF4+Cb[fs[3],4]+Cb[fs[2],3]+Cb[fs[1],2]+Cb[fs[0],1]
                seenr.add(int(r))
        for r in seenr: ref[r] += u_all[s]
    if MODE == 'exact':
        got = np.load(os.path.join(OUT, f'acc_{TAG}.npy'))
        log('CHECK maxdiff', float(np.abs(got-ref).max()), 'nonzero', int((ref>0).sum()))
