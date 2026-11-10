"""v19: all experiments on one parallel dense-rank star kernel.
v17: published baselines, utility distributions, sampling ablations, insertion stream with EUCOLOC-U, mixed stream at scale.
v16 (= v15 with array-based mask emission, i.e. no typed-List conversion): maintenance benchmark with an efficient CSR-based Exact-Inc and an exact-incremental lower bound (stars of changed instances only), on TIST (18 monthly snapshots) and Yelp (26 quarterly snapshots).
v13: (a) vary number of features K in {20,40,60}; (b) maintenance on Yelp quarterly snapshots (2012Q1..), Exact-Remine vs Exact-Inc vs Pollster.
v9: (1) scalability in n (city subsets), (2) threshold accuracy with variance-based band + targeted exact refinement,
(3) monthly maintenance on TIST: Exact-Remine vs Exact-Inc (stored participation sets) vs Pollster (coordinated priority
samples with persistent seeds). UPI(c)=num(c)/den(c); num(c)=sum_o u(o)[c in P(o)], P(o)=feature sets of rows containing o.
"""
import os, json, time, itertools, faulthandler
faulthandler.enable()
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
import numba
from numba import types
from numba.typed import Dict, List

T0 = time.time()
def log(*a):
    print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)
ROOT = os.environ.get('V9_ROOT', '/kaggle/input'); OUT = os.environ.get('V9_OUT', '/kaggle/working')
K = int(os.environ.get('V9_K', '60')); D = float(os.environ.get('V9_D', '150'))
FRACS = [float(x) for x in os.environ.get('V9_FRACS', '0.125,0.25,0.5,1.0').split(',')]
MS = [int(x) for x in os.environ.get('V9_MS', '300,1000').split(',')]
K4CAP = float(os.environ.get('V9_K4CAP', '120'))
DYN_M = [int(x) for x in os.environ.get('V9_DYN_M', '300,1000').split(',')]
IDLE = int(os.environ.get('V9_IDLE', '6'))     # months without check-in -> instance deleted
PARTS = os.environ.get('V9_PARTS', 'base')
DELTA_ON = os.environ.get('V19_DELTA', '0') == '1'
CACHE_X = float(os.environ.get('V19_CACHE', '0'))
SCHEME = os.environ.get('V20_SCHEME', 'threshold')          # threshold (default, Theorem 1) | priority
REPS_T = int(os.environ.get('V20_REPS', '1'))                 # timing repetitions in static_eval
DELTA_B = 0.1
S = dict(K=K, d=D, fracs=FRACS, ms=MS, dyn_m=DYN_M, idle=IDLE, scal=[], thr=[], dyn=[])
def save():
    os.makedirs(OUT, exist_ok=True)
    json.dump(S, open(os.path.join(OUT, 'pollster_v22.json'), 'w'), indent=1, default=float)
def find(name):
    for dp, dn, fn in os.walk(ROOT):
        if name in fn:
            return os.path.join(dp, name)
    raise FileNotFoundError(name)

# ------------------------------------------------------------------ data
cities = pd.read_csv(find('dataset_TIST2015_Cities.txt'), sep='\t', header=None,
                     names=['city', 'lat', 'lon', 'cc', 'country', 'type'])
pois = pd.read_csv(find('dataset_TIST2015_POIs.txt'), sep='\t', header=None, usecols=[0, 1, 2, 3],
                   names=['vid', 'lat', 'lon', 'cat'], dtype={'vid': str, 'cat': str}, quoting=3,
                   on_bad_lines='skip', engine='c')
pois['lat'] = pd.to_numeric(pois.lat, errors='coerce'); pois['lon'] = pd.to_numeric(pois.lon, errors='coerce')
pois = pois.dropna().drop_duplicates('vid').reset_index(drop=True)
def to_xyz(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    return np.c_[np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)]
ct = cKDTree(to_xyz(cities.lat.values, cities.lon.values))
city = np.full(len(pois), -1, np.int32)
for a in range(0, len(pois), 500_000):
    dist, ci = ct.query(to_xyz(pois.lat.values[a:a + 500_000], pois.lon.values[a:a + 500_000]))
    city[a:a + 500_000] = np.where(dist * 6371.0 <= 30.0, ci, -1)
vid2row = pd.Series(np.arange(len(pois)), index=pois.vid.values)
NMON = 18
MON = {m: i for i, m in enumerate(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])}
cnt_m = np.zeros((len(pois), NMON), np.float32)
for ch in pd.read_csv(find('dataset_TIST2015_Checkins.txt'), sep='\t', header=None, names=['u', 'vid', 't', 'off'],
                      usecols=[1, 2], chunksize=4_000_000, quoting=3, dtype=str, on_bad_lines='skip'):
    r = ch.vid.map(vid2row); ok = r.notna().values
    t = ch.t.values[ok]; r = r.values[ok].astype(np.int64)
    ts = pd.Series(t)
    yr = pd.to_numeric(ts.str[-4:], errors='coerce').values; mo = ts.str[4:7].map(MON).values
    good = np.isfinite(yr) & np.isfinite(mo)
    mi = ((yr[good] - 2012) * 12 + mo[good] - 3).astype(np.int64)
    mi = np.clip(mi, 0, NMON - 1)
    np.add.at(cnt_m, (r[good], mi), 1.0)
tot = cnt_m.sum(1)
log('loaded POIs', len(pois), 'checkins', int(tot.sum()))
GENERIC = {'Home (private)', 'Office', 'Neighborhood', 'City', 'Road', 'Building', 'Bridge', 'Intersection',
           'Residential Building (Apartment / Condo)', 'Other Great Outdoors', 'States & Municipalities',
           'Housing Development', 'Moving Target', 'Bus Stop', 'Bus Station', 'Subway', 'Train Station', 'Platform'}
XYZ_all = to_xyz(pois.lat.values, pois.lon.values) * 6371000.0
gmask = city >= 0
vc = pois.cat[gmask].value_counts()
TOP = [c for c in vc.index if c not in GENERIC][:K]
fid = {c: j for j, c in enumerate(TOP)}
fall = pois.cat.map(fid).fillna(-1).astype(int).values
IDS = np.nonzero(gmask & (fall >= 0) & (tot > 0))[0]
P = XYZ_all[IDS].copy(); F = fall[IDS].astype(np.int64); CITY = city[IDS]; CM = cnt_m[IDS]
N = len(IDS); TREE = cKDTree(P)
rng = np.random.default_rng(7)
R = rng.random(N) + 1e-12            # persistent per-instance seeds (coordinated samples)
log('instances', N, 'features', K)

def new_dict():
    return Dict.empty(key_type=types.int64, value_type=types.float64)

U64_1 = np.uint64(1); U64_0 = np.uint64(0)
M1 = np.uint64(0x5555555555555555); M2 = np.uint64(0x3333333333333333); M4 = np.uint64(0x0F0F0F0F0F0F0F0F)
H01 = np.uint64(0x0101010101010101)

def build_adj(active):
    """CSR adjacency over active instances; neighbours within D with a different feature; sorted int32 lists."""
    t = time.time(); nodes = np.nonzero(active)[0]
    ptr = np.zeros(N + 1, np.int64); parts = []; lens_all = np.zeros(N, np.int64)
    for a in range(0, len(nodes), 100000):
        nd = nodes[a:a + 100000]
        nb = TREE.query_ball_point(P[nd], D)
        lens = np.fromiter((len(x) for x in nb), np.int64, count=len(nd))
        flat = np.fromiter(itertools.chain.from_iterable(nb), np.int64, count=int(lens.sum()))
        owner = np.repeat(np.arange(len(nd)), lens)
        keep = (F[flat] != F[nd][owner]) & active[flat]
        flat = flat[keep]; owner = owner[keep]
        order = np.lexsort((flat, owner)); flat = flat[order]; owner = owner[order]
        lens_all[nd] = np.bincount(owner, minlength=len(nd))
        parts.append(flat.astype(np.int32))
    ptr[1:] = np.cumsum(lens_all)
    idx = np.concatenate(parts) if parts else np.zeros(0, np.int32)
    # parts are in node order of `nodes` (ascending), so idx is aligned with ptr
    return ptr, idx, time.time() - t

# ------------------------------------------------------------------ shared: (mask, instance) hash set with utility sums
@numba.njit
def pair_insert(hm, hi, mk, inst):
    H = hm.shape[0] - 1
    h = ((mk * 0x9E3779B97F4A7C15) ^ (inst * 0xC2B2AE3D27D4EB4F)) & H
    while True:
        if hm[h] == 0:
            hm[h] = mk; hi[h] = inst
            return 1
        if hm[h] == mk and hi[h] == inst:
            return 0
        h = (h + 1) & H

# ------------------------------------------------------------------ Join-based (Huang et al. 2004), UPI (Yang et al. 2015)
@numba.njit
def is_adj(ptr, idx, x, y):
    lo = ptr[x]; hi_ = ptr[x + 1] - 1
    while lo <= hi_:
        mid = (lo + hi_) >> 1
        v = idx[mid]
        if v == y:
            return True
        if v < y:
            lo = mid + 1
        else:
            hi_ = mid - 1
    return False

@numba.njit
def join_level2(nodes, ptr, idx, FF, outA, outB, pos):
    for a in nodes:
        for i in range(ptr[a], ptr[a + 1]):
            b = idx[i]
            if FF[b] > FF[a]:
                q = pos[0]
                if q < outA.shape[0]:
                    outA[q] = a; outB[q] = b
                pos[0] = q + 1

@numba.njit
def join_level3(A2, B2, g0, g1, ptr, idx, FF, out, pos):
    """T2 rows [g0,g1) grouped by first instance; join rows sharing the first instance (prefix join)."""
    s = g0
    while s < g1:
        a = A2[s]; e = s
        while e < g1 and A2[e] == a:
            e += 1
        for i in range(s, e):
            x = B2[i]
            for j in range(s, e):
                y = B2[j]
                if FF[y] <= FF[x]:
                    continue
                if is_adj(ptr, idx, x, y):
                    q = pos[0]
                    if q < out.shape[0]:
                        out[q, 0] = a; out[q, 1] = x; out[q, 2] = y
                    pos[0] = q + 1
        s = e

@numba.njit
def join_level4(T3, g0, g1, ptr, idx, FF, out, pos):
    s = g0
    while s < g1:
        a = T3[s, 0]; x = T3[s, 1]; e = s
        while e < g1 and T3[e, 0] == a and T3[e, 1] == x:
            e += 1
        for i in range(s, e):
            y1 = T3[i, 2]
            for j in range(s, e):
                y2 = T3[j, 2]
                if FF[y2] <= FF[y1]:
                    continue
                if is_adj(ptr, idx, y1, y2):
                    q = pos[0]
                    if q < out.shape[0]:
                        out[q, 0] = a; out[q, 1] = x; out[q, 2] = y1; out[q, 3] = y2
                    pos[0] = q + 1
        s = e

@numba.njit
def table_participation(T, FF, u, hm, hi, gd):
    """table instances -> participating instances per pattern (dedup through the pair hash set) -> num(c)."""
    k = T.shape[1]; c = 0
    for r in range(T.shape[0]):
        mk = np.int64(0)
        for j in range(k):
            mk |= np.int64(1) << FF[T[r, j]]
        for j in range(k):
            o = T[r, j]
            if pair_insert(hm, hi, mk, o):
                gd[mk] = gd.get(mk, 0.0) + u[o]; c += 1
    return c

def chunked(fn_rows, ncols, total_cap, init=1 << 22):
    """call fn_rows(out, pos) with a growing buffer; returns rows or None if > total_cap."""
    cap = init
    while True:
        out = np.zeros((cap, ncols), np.int32) if ncols > 1 else None
        pos = np.zeros(1, np.int64)
        fn_rows(out, pos)
        if pos[0] <= cap:
            return out[:pos[0]]
        if pos[0] > total_cap:
            return None
        cap = int(pos[0])

def run_join(active, u, kmax, cap_time, row_cap, state_hint):
    """Join-based exact HUCP mining up to size kmax (level-wise prefix join of table instances). Returns gd, info."""
    t0 = time.time(); info = dict(kmax=kmax)
    ptr, idx, t_adj = build_adj(active); info['adj_time'] = t_adj; info['edges'] = int(len(idx))
    nodes = np.nonzero(active)[0].astype(np.int64)
    gd = new_dict()
    hsize = 1 << int(max(20, np.ceil(np.log2(1.5 * max(state_hint, 1)))))
    hm = np.zeros(hsize, np.int64); hi = np.zeros(hsize, np.int32); used = 0; lim = int(0.8 * hsize)
    est2 = int(len(idx) // 2 + 16)
    A2 = np.zeros(est2, np.int32); B2 = np.zeros(est2, np.int32); pos = np.zeros(1, np.int64)
    join_level2(nodes, ptr, idx, F, A2, B2, pos); A2 = A2[:pos[0]]; B2 = B2[:pos[0]]
    info['rows2'] = int(pos[0])
    used += table_participation(np.c_[A2, B2], F, u, hm, hi, gd)
    info['t_level2'] = time.time() - t0
    if kmax >= 3:
        T3parts = []; n3 = 0; L = len(A2); blk = 200000
        bounds = np.r_[0, np.nonzero(np.diff(A2))[0] + 1, L] if L else np.array([0, 0])
        b_i = 0
        while b_i < len(bounds) - 1:
            b_j = min(len(bounds) - 1, b_i + blk)
            g0, g1 = int(bounds[b_i]), int(bounds[b_j])
            def f3(out, pos, g0=g0, g1=g1):
                join_level3(A2, B2, g0, g1, ptr, idx, F, out, pos)
            T = chunked(f3, 3, row_cap - n3, init=1 << 20)
            if T is None:
                info.update(status='rows>cap at level 3', rows3_partial=n3, frac_done=b_i / (len(bounds) - 1),
                            time=time.time() - t0); return None, info
            n3 += len(T); T3parts.append(T)
            if time.time() - t0 > cap_time:
                info.update(status='time cap at level 3', rows3_partial=n3, frac_done=b_j / (len(bounds) - 1),
                            time=time.time() - t0); return None, info
            b_i = b_j
        T3 = np.concatenate(T3parts) if T3parts else np.zeros((0, 3), np.int32); del T3parts
        info['rows3'] = int(len(T3)); del A2, B2
        used += table_participation(T3, F, u, hm, hi, gd)
        info['t_level3'] = time.time() - t0
        if used > lim:
            info.update(status='hash full at level 3', time=time.time() - t0); return None, info
        if kmax >= 4:
            T4n = 0; L3 = len(T3)
            key = T3[:, 0].astype(np.int64) * (N + 1) + T3[:, 1]
            bounds = np.r_[0, np.nonzero(np.diff(key))[0] + 1, L3] if L3 else np.array([0, 0]); del key
            b_i = 0
            while b_i < len(bounds) - 1:
                b_j = min(len(bounds) - 1, b_i + blk)
                g0, g1 = int(bounds[b_i]), int(bounds[b_j])
                def f4(out, pos, g0=g0, g1=g1):
                    join_level4(T3, g0, g1, ptr, idx, F, out, pos)
                T = chunked(f4, 4, row_cap, init=1 << 20)
                if T is None:
                    info.update(status='rows>cap at level 4', rows4_partial=T4n, frac_done=b_i / (len(bounds) - 1),
                                time=time.time() - t0); return None, info
                T4n += len(T)
                used += table_participation(T, F, u, hm, hi, gd)
                if used > lim:
                    info.update(status='hash full at level 4', rows4_partial=T4n, time=time.time() - t0); return None, info
                if time.time() - t0 > cap_time:
                    info.update(status='time cap at level 4', rows4_partial=T4n, frac_done=b_j / (len(bounds) - 1),
                                time=time.time() - t0); return None, info
                b_i = b_j
            info['rows4'] = int(T4n)
    info.update(status='complete', time=time.time() - t0, patterns=len(gd), pairs=int(used))
    return gd, info

# ------------------------------------------------------------------ Clique-Querying (maximal cliques + hash table)
@numba.njit
def popcnt(x):
    x = x - ((x >> U64_1) & M1)
    x = (x & M2) + ((x >> np.uint64(2)) & M2)
    x = (x + (x >> np.uint64(4))) & M4
    return np.int64((x * H01) >> np.uint64(56))

@numba.njit
def ctz(x):
    n = 0
    while (x & U64_1) == U64_0:
        x = x >> U64_1; n += 1
    return n

@numba.njit
def bk(Rb, rl, Pb, Xb, adj, W, v, nb, FF, hm, hi, stats):
    stats[4] += 1
    if stats[4] > stats[5]:
        stats[6] = 1
        return
    pe = True
    for w in range(W):
        if Pb[w] != U64_0:
            pe = False; break
    if pe:
        for w in range(W):
            if Xb[w] != U64_0:
                return
        mk = np.int64(1) << FF[v]
        for i in range(rl):
            mk |= np.int64(1) << FF[nb[Rb[i]]]
        stats[0] += 1
        if rl + 1 > stats[2]:
            stats[2] = rl + 1
        stats[1] += pair_insert(hm, hi, mk, v)
        for i in range(rl):
            stats[1] += pair_insert(hm, hi, mk, nb[Rb[i]])
        return
    best = -1; bc = -1
    for w in range(W):
        word = Pb[w] | Xb[w]
        while word != U64_0:
            low = word & (~word + U64_1)
            b = w * 64 + ctz(low)
            word = word ^ low
            c = 0
            for w2 in range(W):
                c += popcnt(Pb[w2] & adj[b, w2])
            if c > bc:
                bc = c; best = b
    for w in range(W):
        cand = Pb[w] & ~adj[best, w]
        while cand != U64_0:
            low = cand & (~cand + U64_1)
            b = w * 64 + ctz(low)
            cand = cand ^ low
            Rb[rl] = b
            nP = np.empty(W, np.uint64); nX = np.empty(W, np.uint64)
            for w2 in range(W):
                nP[w2] = Pb[w2] & adj[b, w2]; nX[w2] = Xb[w2] & adj[b, w2]
            bk(Rb, rl + 1, nP, nX, adj, W, v, nb, FF, hm, hi, stats)
            if stats[6] == 1:
                return
            Pb[w] = Pb[w] & ~low
            Xb[w] = Xb[w] | low

@numba.njit
def cq_enum(vs, start, rank, ptr, idx, PP, FF, d2, hm, hi, stats):
    """process vs[start:] until the call budget stats[5] is exhausted; returns the index of the first unfinished vertex."""
    for q in range(start, len(vs)):
        v = vs[q]
        a0 = ptr[v]; a1 = ptr[v + 1]; L = a1 - a0
        if L == 0:
            continue
        nb = idx[a0:a1]
        W = (L + 63) // 64
        adj = np.zeros((L, W), np.uint64)
        for i in range(L):
            x = nb[i]
            for j in range(i + 1, L):
                y = nb[j]
                if FF[x] == FF[y]:
                    continue
                dx = PP[x, 0] - PP[y, 0]; dy = PP[x, 1] - PP[y, 1]; dz = PP[x, 2] - PP[y, 2]
                if dx * dx + dy * dy + dz * dz <= d2:
                    adj[i, j // 64] |= U64_1 << np.uint64(j % 64)
                    adj[j, i // 64] |= U64_1 << np.uint64(i % 64)
        Pb = np.zeros(W, np.uint64); Xb = np.zeros(W, np.uint64)
        rv = rank[v]
        for i in range(L):
            if rank[nb[i]] > rv:
                Pb[i // 64] |= U64_1 << np.uint64(i % 64)
            else:
                Xb[i // 64] |= U64_1 << np.uint64(i % 64)
        Rb = np.empty(L + 1, np.int64)
        bk(Rb, 0, Pb, Xb, adj, W, v, nb, FF, hm, hi, stats)
        if stats[6] == 1:
            return q
        if stats[1] > stats[3]:          # hash set too full
            return -1
    return len(vs)

@numba.njit
def gen_candidates(keys, kmax, cand):
    """all sub-patterns (size 2..kmax) of the maximal-clique feature sets (keys)."""
    for key in keys:
        bits = np.empty(64, np.int64); nbits = 0
        for f in range(64):
            if (key >> f) & 1:
                bits[nbits] = f; nbits += 1
        for i in range(nbits):
            for j in range(i + 1, nbits):
                m2 = (np.int64(1) << bits[i]) | (np.int64(1) << bits[j])
                cand[m2] = 0
                if kmax >= 3:
                    for l in range(j + 1, nbits):
                        m3 = m2 | (np.int64(1) << bits[l])
                        cand[m3] = 0
                        if kmax >= 4:
                            for q in range(l + 1, nbits):
                                cand[m3 | (np.int64(1) << bits[q])] = 0

@numba.njit
def cq_query(cands, keys, kptr, kinst, kfeat_ptr, post_ptr, post_keys, FF, u, stamp, nfeat, gd):
    """for each candidate p: keys K >= p from the shortest posting list; per-feature union of objects (Eq. 12)."""
    for ci in range(len(cands)):
        p = cands[ci]
        bestf = -1; bl = 1 << 62
        for f in range(nfeat):
            if (p >> f) & 1:
                l = post_ptr[f + 1] - post_ptr[f]
                if l < bl:
                    bl = l; bestf = f
        num = 0.0
        sid = ci + 1
        for t in range(post_ptr[bestf], post_ptr[bestf + 1]):
            kid = post_keys[t]
            if (keys[kid] & p) != p:
                continue
            for f in range(nfeat):
                if (p >> f) & 1:
                    a = kfeat_ptr[kid * nfeat + f]; b = kfeat_ptr[kid * nfeat + f + 1]
                    for z in range(a, b):
                        o = kinst[z]
                        if stamp[o] != sid:
                            stamp[o] = sid; num += u[o]
        gd[p] = num

def run_cq(active, u, kmax, cap_time, hash_slots):
    t0 = time.time(); info = dict(kmax=kmax)
    ptr, idx, t_adj = build_adj(active); info['adj_time'] = t_adj
    deg = np.diff(ptr)
    order = np.lexsort((np.arange(N), deg)); rank = np.empty(N, np.int64); rank[order] = np.arange(N)
    nodes = np.nonzero(active & (deg > 0))[0]
    perm = np.random.default_rng(11).permutation(nodes).astype(np.int64)
    hm = np.zeros(hash_slots, np.int64); hi = np.zeros(hash_slots, np.int32)
    stats = np.zeros(8, np.int64); stats[3] = int(hash_slots * 0.7)
    B0 = int(os.environ.get('V17_CQBUDGET', str(1 << 22))); done = 0; t1 = time.time(); budget = B0; restarts = 0
    while done < len(perm):
        stats[4] = 0; stats[5] = budget; stats[6] = 0
        q = cq_enum(perm, done, rank, ptr, idx, P, F, D * D, hm, hi, stats)
        if q == -1:
            info.update(status='hash full', done=done, n=len(perm), time=time.time() - t0); return None, info
        if stats[6] == 1 and q == done:      # a single vertex exceeded the budget: restart it with a larger budget
            budget *= 2; restarts += 1
        else:
            done = q
            budget = max(B0, budget // 2)
        if time.time() - t0 > cap_time:
            break
    t_enum = time.time() - t1
    info['restarts'] = restarts; info['final_budget'] = int(budget)
    info.update(max_cliques=int(stats[0]), pairs=int(stats[1]), max_clique_size=int(stats[2]), enum_time=t_enum,
                enum_done=done, enum_n=len(perm))
    if done < len(perm):
        info.update(status='cap in clique enumeration', enum_time_extrap=t_enum * len(perm) / max(done, 1),
                    time=time.time() - t0)
        return None, info
    # participating-instance hash table: key = feature set of a maximal clique, value = objects grouped by feature
    t2 = time.time()
    sl = np.nonzero(hm)[0]; km = hm[sl]; ki = hi[sl].astype(np.int64); del sl
    o = np.lexsort((F[ki], km)); km = km[o]; ki = ki[o]
    keys, kstart = np.unique(km, return_index=True)
    kptr = np.r_[kstart, len(km)].astype(np.int64)
    kid_of = np.repeat(np.arange(len(keys)), np.diff(kptr))
    kfeat = np.zeros(len(keys) * K + 1, np.int64)
    np.add.at(kfeat, kid_of * K + F[ki] + 1, 1)
    kfeat_ptr = np.cumsum(kfeat)
    post = [[] for _ in range(K)]
    fk = (keys[:, None] >> np.arange(K)[None, :]) & 1
    post_ptr = np.r_[0, np.cumsum(fk.sum(0))].astype(np.int64)
    post_keys = np.concatenate([np.nonzero(fk[:, f])[0] for f in range(K)]).astype(np.int64)
    del fk, kid_of, km
    info.update(keys=int(len(keys)), table_entries=int(len(ki)), build_time=time.time() - t2)
    t3 = time.time()
    cd = Dict.empty(key_type=types.int64, value_type=types.int64)
    gen_candidates(keys, kmax, cd)
    cands = np.fromiter(cd.keys(), np.int64, count=len(cd)); del cd
    gd = new_dict(); stamp = np.zeros(N, np.int64)
    cq_query(cands, keys, kptr, ki, kfeat_ptr, post_ptr, post_keys, F, u, stamp, K, gd)
    info.update(candidates=int(len(cands)), query_time=time.time() - t3, status='complete', time=time.time() - t0)
    return gd, info



# =====================================================================================================================
# v19 core: one parallel star kernel (numba prange over threads, per-thread dense rank accumulators and dedupe stamps)
# shared by every method: Exact (=Star), Remine, Exact-Inc, Exact-Inc-LB, EUCOLOC-U, Pollster. Patterns are ranked in
# the combinatorial number system (sizes 2..4 over K features), so per-pattern sums are dense arrays, not hash maps.
# =====================================================================================================================
import math
from numba import prange
QW = [-1]                                  # k-d tree query workers (-1 = all cores; 1 for single-thread tables)
TH_MAX = numba.config.NUMBA_NUM_THREADS

def set_K(Kn):
    global Cb, OFF3, OFF4, NR, NR3, RF, RSIZE, RMASK
    Cb = np.zeros((65, 5), np.int64)
    for n_ in range(65):
        for k_ in range(5):
            Cb[n_, k_] = math.comb(n_, k_)
    OFF3 = int(Cb[Kn, 2]); OFF4 = OFF3 + int(Cb[Kn, 3]); NR = OFF4 + int(Cb[Kn, 4]); NR3 = OFF4
    RF = np.full((NR, 4), Kn, np.int64); RSIZE = np.zeros(NR, np.int64); RMASK = np.zeros(NR, np.int64)
    for a in range(Kn):
        for b in range(a + 1, Kn):
            r = Cb[b, 2] + Cb[a, 1]; RF[r, :2] = (a, b); RSIZE[r] = 2; RMASK[r] = (1 << a) | (1 << b)
            for c in range(b + 1, Kn):
                r = OFF3 + Cb[c, 3] + Cb[b, 2] + Cb[a, 1]; RF[r, :3] = (a, b, c); RSIZE[r] = 3
                RMASK[r] = (1 << a) | (1 << b) | (1 << c)
                for d_ in range(c + 1, Kn):
                    r = OFF4 + Cb[d_, 4] + Cb[c, 3] + Cb[b, 2] + Cb[a, 1]; RF[r] = (a, b, c, d_); RSIZE[r] = 4
                    RMASK[r] = (1 << a) | (1 << b) | (1 << c) | (1 << d_)
    global SEEN
    SEEN = np.full((TH_MAX, NR), -1, np.int64)

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
def star_kernel(nodes, wts, ptr, idx, PP, FF, d2, kmax, T, acc, seen, stamp0, cnt, emit, bufs, fill, off, Cb, OFF3, OFF4):
    """per node o: deduplicated ranks of all clique patterns of o's star (sizes 2..kmax).
    acc[t, r] += w(o) for each; if emit, ranks are appended to the thread buffer bufs[t] (fill[t] may exceed capacity:
    caller retries)."""
    CAP = bufs.shape[1]
    for t in prange(T):
        for s in range(t, len(nodes), T):
            o = nodes[s]; st = stamp0 + s; w = wts[s]; fo = FF[o]; c = 0
            off[s] = fill[t]
            a0 = ptr[s]; a1 = ptr[s + 1]
            for i in range(a0, a1):
                v1 = idx[i]; f1 = FF[v1]
                r = rk2(fo, f1, Cb)
                if seen[t, r] != st:
                    seen[t, r] = st; acc[t, r] += w; c += 1
                    if emit:
                        q = fill[t]
                        if q < CAP:
                            bufs[t, q] = r
                        fill[t] = q + 1
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
                        if emit:
                            q = fill[t]
                            if q < CAP:
                                bufs[t, q] = r
                            fill[t] = q + 1
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
                            if emit:
                                q = fill[t]
                                if q < CAP:
                                    bufs[t, q] = r
                                fill[t] = q + 1
            cnt[s] = c

@numba.njit(parallel=True)
def gather_emit(T, bufs, off, cnt, optr, out):
    for s in prange(len(cnt)):
        t = s % T; a = off[s]; b = optr[s]
        for i in range(cnt[s]):
            out[b + i] = bufs[t, a + i]

@numba.njit(parallel=True)
def acc_csr(nodes, wts, gptr, gr, T, acc):
    """acc[t, r] += w for every rank r stored in the CSR row of each node (rows of `nodes`)."""
    for t in prange(T):
        for s in range(t, len(nodes), T):
            o = nodes[s]; w = wts[s]
            for i in range(gptr[o], gptr[o + 1]):
                acc[t, gr[i]] += w

@numba.njit(parallel=True)
def acc_local(wts, lptr, lr, T, acc):
    """same, for a local CSR (row s = s-th node)."""
    for t in prange(T):
        for s in range(t, len(wts), T):
            w = wts[s]
            for i in range(lptr[s], lptr[s + 1]):
                acc[t, lr[i]] += w

ACC0 = [None]; STAMP = [0]; EMIT_CAP = [int(float(os.environ.get('V20_EMITCAP', str(1 << 24))))]

def nthreads():
    return numba.get_num_threads()

# ---------------------------------------------------------------- v26: grid index over the current snapshot only
# Cells of side d; a cell key packs the three integer cell coordinates (|coordinate| <= 6.4e6 m, so 2^18 cells per axis
# suffice for d >= 50 m). The grid holds the active instances of one snapshot and is rebuilt from them, so no future
# location is used; its build time is reported with every stream.
GRID_ON = os.environ.get('V26_GRID', '0') == '1'
GRIDS = []                       # [(active array, keys, ids)]
G_NB = np.int64(1 << 18); G_OFF = np.int64(1 << 17)

def grid_build(active):
    ids = np.nonzero(active)[0].astype(np.int64)
    c = np.floor(P[ids] / D).astype(np.int64) + G_OFF
    key = (c[:, 0] * G_NB + c[:, 1]) * G_NB + c[:, 2]
    o = np.argsort(key, kind='stable')
    return key[o], ids[o]

@numba.njit
def grid_mark_deleted(gk, gid, dkeys, dids, keep):
    for i in range(len(dids)):
        lo = np.searchsorted(gk, dkeys[i]); hi = np.searchsorted(gk, dkeys[i], side='right')
        for j in range(lo, hi):
            if gid[j] == dids[i]:
                keep[j] = False
                break

def grid_keys(ids):
    c = np.floor(P[ids] / D).astype(np.int64) + G_OFF
    return (c[:, 0] * G_NB + c[:, 1]) * G_NB + c[:, 2]

def grid_update(g, prev_act, act):
    """in-place style update of a grid: drop the deleted instances and merge in the inserted ones (O(n) copying,
    O(changes log n) search); returns new arrays, the old grid stays valid for the previous snapshot."""
    gk, gid = g
    dele = np.nonzero(prev_act & ~act)[0].astype(np.int64); ins = np.nonzero(act & ~prev_act)[0].astype(np.int64)
    keep = np.ones(len(gk), np.bool_)
    if len(dele):
        grid_mark_deleted(gk, gid, grid_keys(dele), dele, keep)
    gk2 = gk[keep]; gid2 = gid[keep]
    if len(ins):
        ki = grid_keys(ins); o = np.argsort(ki, kind='stable'); ki = ki[o]; ins = ins[o]
        pos = np.searchsorted(gk2, ki, side='right')
        gk2 = np.insert(gk2, pos, ki); gid2 = np.insert(gid2, pos, ins)
    return gk2, gid2

GRID_UPD = os.environ.get('V26_GRIDUPD', '1') == '1'

def grid_register(*pairs):
    GRIDS.clear()
    for a, g in pairs:
        if a is not None and g is not None:
            GRIDS.append((a, g[0], g[1]))

def grid_for(active):
    if not GRID_ON or active is None:
        return None
    for a, gk, gid in GRIDS:
        if a is active:
            return gk, gid
    for a, gk, gid in list(GRIDS):
        if a.shape == active.shape and np.array_equal(a, active):
            GRIDS.append((active, gk, gid))          # same snapshot, another array object: remember it
            return gk, gid
    return None

@numba.njit(parallel=True)
def grid_query(nd, PP, FF, gk, gid, dd, d2, NB, OFF, allowed_f, use_allowed, samef, cnt, ptr, out, fill):
    """pass 1 (fill=False): cnt[i] = number of grid instances within dd of nd[i] (different feature unless samef);
    pass 2 (fill=True): write them to out[ptr[i]:ptr[i+1]]."""
    for i in prange(len(nd)):
        o = nd[i]; fo = FF[o]
        cx = np.int64(np.floor(PP[o, 0] / dd)) + OFF; cy = np.int64(np.floor(PP[o, 1] / dd)) + OFF; cz = np.int64(np.floor(PP[o, 2] / dd)) + OFF
        c = 0; q = ptr[i] if fill else 0
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                for dz in range(-1, 2):
                    key = ((cx + dx) * NB + (cy + dy)) * NB + (cz + dz)
                    lo = np.searchsorted(gk, key); hi = np.searchsorted(gk, key, side='right')
                    for j in range(lo, hi):
                        v = gid[j]
                        if not samef and FF[v] == fo:
                            continue
                        if use_allowed and not allowed_f[FF[v]]:
                            continue
                        ex = PP[v, 0] - PP[o, 0]; ey = PP[v, 1] - PP[o, 1]; ez = PP[v, 2] - PP[o, 2]
                        if ex * ex + ey * ey + ez * ez <= d2:
                            if fill:
                                out[q] = v; q += 1
                            c += 1
        cnt[i] = c

@numba.njit(parallel=True)
def near_any(pr, PP, ck, cid, dd, d2, NB, OFF, out):
    """out[i] = True iff some changed instance (cid, sorted by cell key ck) lies within dd of pr[i]."""
    for i in prange(len(pr)):
        o = pr[i]; hit = False
        cx = np.int64(np.floor(PP[o, 0] / dd)) + OFF; cy = np.int64(np.floor(PP[o, 1] / dd)) + OFF; cz = np.int64(np.floor(PP[o, 2] / dd)) + OFF
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                for dz in range(-1, 2):
                    if hit:
                        continue
                    key = ((cx + dx) * NB + (cy + dy)) * NB + (cz + dz)
                    lo = np.searchsorted(ck, key); hi = np.searchsorted(ck, key, side='right')
                    for j in range(lo, hi):
                        v = cid[j]
                        ex = PP[v, 0] - PP[o, 0]; ey = PP[v, 1] - PP[o, 1]; ez = PP[v, 2] - PP[o, 2]
                        if ex * ex + ey * ey + ez * ez <= d2:
                            hit = True
                            break
        out[i] = hit

def probe_share(pr, chg_pts):
    """share of the probe instances within d of a change; one sort of the changes by cell, no tree."""
    t0 = time.time(); ck = grid_keys(chg_pts); o = np.argsort(ck, kind='stable'); out = np.zeros(len(pr), np.bool_)
    t1 = time.time(); near_any(np.ascontiguousarray(pr, np.int64), P, ck[o], chg_pts[o], D, D * D, G_NB, G_OFF, out)
    probe_share.t_sort = t1 - t0; probe_share.t_query = time.time() - t1     # timings, read by the cost model
    return float(out.mean()) if len(pr) else 0.0

def grid_neigh(nd, g, allowed=None, samef=False):
    gk, gid = g; nd = np.ascontiguousarray(nd, np.int64)
    al = np.ones(1, np.bool_) if allowed is None else np.asarray(allowed, np.bool_)
    cnt = np.zeros(len(nd), np.int64); dummy = np.zeros(1, np.int64)
    grid_query(nd, P, F, gk, gid, D, D * D, G_NB, G_OFF, al, allowed is not None, samef, cnt, np.zeros(len(nd) + 1, np.int64), dummy, False)
    ptr = np.r_[0, np.cumsum(cnt)].astype(np.int64); out = np.empty(ptr[-1], np.int64)
    grid_query(nd, P, F, gk, gid, D, D * D, G_NB, G_OFF, al, allowed is not None, samef, cnt, ptr, out, True)
    return ptr, out

def neigh(nd, active=None, allowed=None):
    g = grid_for(active)
    if g is not None:
        return grid_neigh(nd, g, allowed)
    nb = TREE.query_ball_point(P[nd], D, workers=QW[0])
    lens = np.fromiter((len(x) for x in nb), np.int64, count=len(nd))
    flat = np.fromiter(itertools.chain.from_iterable(nb), np.int64, count=int(lens.sum()))
    owner = np.repeat(np.arange(len(nd)), lens)
    keep = F[flat] != F[nd][owner]
    if active is not None:
        keep &= active[flat]
    if allowed is not None:
        keep &= allowed[F[flat]]
    flat = flat[keep]; owner = owner[keep]
    ptr = np.r_[0, np.cumsum(np.bincount(owner, minlength=len(nd)))].astype(np.int64)
    return ptr, flat

def run_enum(nodes, wts, kmax, active=None, allowed=None, emit=False, chunk=20000, cap=None):
    """dense per-pattern sums of wts over participation sets; optionally the participation sets (CSR of int32 ranks).
    Returns acc[NR], time, #done, cnt[#done], (lptr, lranks) or None."""
    T = nthreads(); acc = np.zeros((T, NR)); nodes = np.asarray(nodes, np.int64); wts = np.asarray(wts, np.float64)
    t0 = time.time(); done = 0; cnts = []; outs = []
    for a in range(0, len(nodes), chunk):
        nd = np.ascontiguousarray(nodes[a:a + chunk]); w = np.ascontiguousarray(wts[a:a + chunk])
        ptr, flat = neigh(nd, active, allowed)
        cnt = np.zeros(len(nd), np.int64); off = np.zeros(len(nd), np.int64)
        if STAMP[0] > (1 << 60):
            SEEN[:] = -1; STAMP[0] = 0
        while True:
            capT = EMIT_CAP[0] if emit else 1
            bufs = np.empty((T, capT), np.int32); fill = np.zeros(T, np.int64)
            if emit and not np.any(w):
                if ACC0[0] is None or ACC0[0].shape != (T, NR):
                    ACC0[0] = np.zeros((T, NR))
                accl = ACC0[0]           # weights are zero: nothing is added, the buffer stays zero
            else:
                accl = np.zeros((T, NR)) if emit else acc
            star_kernel(nd, w, ptr, flat, P, F, D * D, kmax, T, accl, SEEN, STAMP[0], cnt, emit, bufs, fill, off,
                        Cb, OFF3, OFF4)
            STAMP[0] += len(nd)
            if not emit or fill.max() <= capT:
                if emit:
                    acc += accl
                break
            EMIT_CAP[0] = int(fill.max() * 1.3) + 1024
        if emit:
            optr = np.r_[0, np.cumsum(cnt)].astype(np.int64); out = np.empty(optr[-1], np.int32)
            gather_emit(T, bufs, off, cnt, optr, out); outs.append(out)
        cnts.append(cnt); done += len(nd)
        if cap is not None and time.time() - t0 > cap:
            break
    cnt = np.concatenate(cnts) if cnts else np.zeros(0, np.int64)
    em = None
    if emit:
        em = (np.r_[0, np.cumsum(cnt)].astype(np.int64), np.concatenate(outs) if outs else np.zeros(0, np.int32))
    return acc.sum(0), time.time() - t0, done, cnt, em

@numba.njit(parallel=True)
def delta_kernel(nodes, ptr, idx, PP, FF, d2, kmax, isins, T, seen, stamp0, cnt, bufs, fill, off, Cb, OFF3, OFF4):
    """per node o: deduplicated ranks of the clique patterns of o's star that contain at least one inserted
    instance (isins); work is proportional to (#inserted neighbours) x (star size)^(kmax-2)."""
    CAP = bufs.shape[1]
    for t in prange(T):
        for s in range(t, len(nodes), T):
            o = nodes[s]; st = stamp0 + s; fo = FF[o]; c = 0
            off[s] = fill[t]
            a0 = ptr[s]; a1 = ptr[s + 1]
            for i in range(a0, a1):
                p = idx[i]
                if not isins[p]:
                    continue
                fp = FF[p]
                r = rk2(fo, fp, Cb)
                if seen[t, r] != st:
                    seen[t, r] = st; q = fill[t]
                    if q < CAP:
                        bufs[t, q] = r
                    fill[t] = q + 1; c += 1
                if kmax < 3:
                    continue
                for j in range(a0, a1):
                    v = idx[j]; fv = FF[v]
                    if j == i or fv == fp:
                        continue
                    dx = PP[p, 0] - PP[v, 0]; dy = PP[p, 1] - PP[v, 1]; dz = PP[p, 2] - PP[v, 2]
                    if dx * dx + dy * dy + dz * dz > d2:
                        continue
                    r = rk3(fo, fp, fv, Cb, OFF3)
                    if seen[t, r] != st:
                        seen[t, r] = st; q = fill[t]
                        if q < CAP:
                            bufs[t, q] = r
                        fill[t] = q + 1; c += 1
                    if kmax < 4:
                        continue
                    for l in range(j + 1, a1):
                        w = idx[l]; fw = FF[w]
                        if l == i or fw == fp or fw == fv:
                            continue
                        dx = PP[p, 0] - PP[w, 0]; dy = PP[p, 1] - PP[w, 1]; dz = PP[p, 2] - PP[w, 2]
                        if dx * dx + dy * dy + dz * dz > d2:
                            continue
                        dx = PP[v, 0] - PP[w, 0]; dy = PP[v, 1] - PP[w, 1]; dz = PP[v, 2] - PP[w, 2]
                        if dx * dx + dy * dy + dz * dz > d2:
                            continue
                        r = rk4(fo, fp, fv, fw, Cb, OFF4)
                        if seen[t, r] != st:
                            seen[t, r] = st; q = fill[t]
                            if q < CAP:
                                bufs[t, q] = r
                            fill[t] = q + 1; c += 1
            cnt[s] = c

@numba.njit(parallel=True)
def union_rows(optr, orr, dptr, dr, nptr, nr):
    """row-wise sorted union of two CSRs; nptr must hold upper bounds (len o + len d); returns exact lengths."""
    lens = np.empty(len(optr) - 1, np.int64)
    for s in prange(len(optr) - 1):
        a = np.sort(np.concatenate((orr[optr[s]:optr[s + 1]], dr[dptr[s]:dptr[s + 1]])))
        w = nptr[s]; last = -1
        for x in a:
            if x != last:
                nr[w] = x; w += 1; last = x
        lens[s] = w - nptr[s]
    return lens

def run_delta(nodes, kmax, active, isins, chunk=20000):
    """participation additions of `nodes` caused by inserted instances; CSR of int32 ranks."""
    T = nthreads(); nodes = np.asarray(nodes, np.int64); cnts = []; outs = []
    for a in range(0, len(nodes), chunk):
        nd = np.ascontiguousarray(nodes[a:a + chunk]); ptr, flat = neigh(nd, active)
        cnt = np.zeros(len(nd), np.int64); off = np.zeros(len(nd), np.int64)
        if STAMP[0] > (1 << 60):
            SEEN[:] = -1; STAMP[0] = 0
        while True:
            bufs = np.empty((T, EMIT_CAP[0]), np.int32); fill = np.zeros(T, np.int64)
            delta_kernel(nd, ptr, flat, P, F, D * D, kmax, isins, T, SEEN, STAMP[0], cnt, bufs, fill, off, Cb, OFF3, OFF4)
            STAMP[0] += len(nd)
            if fill.max() <= EMIT_CAP[0]:
                break
            EMIT_CAP[0] = int(fill.max() * 1.3) + 1024
        optr = np.r_[0, np.cumsum(cnt)].astype(np.int64); out = np.empty(optr[-1], np.int32)
        gather_emit(T, bufs, off, cnt, optr, out); outs.append(out); cnts.append(cnt)
    cnt = np.concatenate(cnts) if cnts else np.zeros(0, np.int64)
    return np.r_[0, np.cumsum(cnt)].astype(np.int64), (np.concatenate(outs) if outs else np.zeros(0, np.int32))


def csr_sum(nodes, wts, gptr, gr):
    T = nthreads(); acc = np.zeros((T, NR))
    acc_csr(np.asarray(nodes, np.int64), np.asarray(wts, np.float64), gptr, gr, T, acc)
    return acc.sum(0)

def local_sum(wts, lptr, lr):
    T = nthreads(); acc = np.zeros((T, NR))
    acc_local(np.asarray(wts, np.float64), lptr, lr, T, acc)
    return acc.sum(0)

def dens(U):
    Ue = np.r_[np.asarray(U, np.float64), 0.0]
    return Ue[RF[:, 0]] + Ue[RF[:, 1]] + Ue[RF[:, 2]] + Ue[RF[:, 3]]

def compare(ex, est, U, kmax, var=None, taus=None, mus=(75, 90, 95), m=None, T=1):
    sel = np.nonzero((RSIZE <= kmax) & (ex > 0))[0]
    den = dens(U)[sel]; upi = ex[sel] / den; e_ = est[sel] / den; err = np.abs(e_ - upi)
    out = dict(patterns=int(len(sel)), mae=float(err.mean()), p99=float(np.quantile(err, .99)), max=float(err.max()))
    for sz in range(2, kmax + 1):
        mk = RSIZE[sel] == sz
        if mk.any():
            out[f's{sz}'] = dict(n=int(mk.sum()), mae=float(err[mk].mean()), max=float(err[mk].max()))
    if var is not None:
        sd = np.sqrt(np.maximum(var[sel], 0)) / den; out['cover2sd'] = float((err <= 2 * sd + 1e-12).mean())
    # distribution of exact UPI and error per decile
    qs = np.quantile(upi, np.linspace(0, 1, 11)); out['upi_q'] = [float(x) for x in qs]
    dec = np.clip(np.searchsorted(qs[1:-1], upi, side='right'), 0, 9); dl = []
    for b in range(10):
        mk = dec == b
        if mk.any():
            dl.append(dict(upi=float(upi[mk].mean()), mae=float(err[mk].mean()), rel=float((err[mk] / upi[mk]).mean()), mx=float(err[mk].max())))
    out['deciles'] = dl
    if m is not None:
        eb = eps_b(upi, m, kmax, K, DELTA_B, T); out['bern_viol'] = int((err > eb).sum()); out['bern_ratio_max'] = float((err / eb).max())
        lam = lam_max(U)[sel]; eb2 = eps2(upi, lam, m, kmax, K, DELTA_B, T)
        out['bern2_viol'] = int((err > eb2).sum()); out['bern2_ratio_max'] = float((err / eb2).max())
        out['eps2_over_eps_median'] = float(np.median(eb2 / eb)); out['lam_q'] = [float(x) for x in np.quantile(lam, [0, .25, .5, .75, 1])]
    thr = {}
    for q in mus:
        mu = float(np.quantile(upi, q / 100)); H = upi >= mu; He = e_ >= mu
        tp = int((H & He).sum()); pr = tp / max(int(He.sum()), 1); rc = tp / max(int(H.sum()), 1)
        d_ = dict(mu=mu, H=int(H.sum()), f1=2 * pr * rc / max(pr + rc, 1e-12), precision=pr, recall=rc)
        if m is not None:
            em = float(eps_b(mu, m, kmax, K, DELTA_B, T)); Hh = e_ >= mu - em
            d_.update(eps_mu=em, Hhat=int(Hh.sum()), Hhat_recall=float((H & Hh).sum() / max(int(H.sum()), 1)),
                      Hhat_precision=float((H & Hh).sum() / max(int(Hh.sum()), 1)),
                      Hhat_in_band=float((Hh & (upi >= mu - 2 * em)).sum() / max(int(Hh.sum()), 1)))
            em2 = eps2(mu, lam, m, kmax, K, DELTA_B, T); Hh2 = e_ >= mu - em2; Hp = e_ >= mu + em; Hp2 = e_ >= mu + em2
            d_.update(eps2_mu_q=[float(x) for x in np.quantile(em2, [0, .5, 1])], eps2_mu_atH=float(np.median(em2[H])) if H.any() else None,
                      Hhat2=int(Hh2.sum()), Hhat2_recall=float((H & Hh2).sum() / max(int(H.sum()), 1)),
                      Hhat2_precision=float((H & Hh2).sum() / max(int(Hh2.sum()), 1)),
                      Hplus=int(Hp.sum()), Hplus_precision=float((H & Hp).sum() / max(int(Hp.sum()), 1)), Hplus_recall=float((H & Hp).sum() / max(int(H.sum()), 1)),
                      Hplus2=int(Hp2.sum()), Hplus2_precision=float((H & Hp2).sum() / max(int(Hp2.sum()), 1)), Hplus2_recall=float((H & Hp2).sum() / max(int(H.sum()), 1)),
                      undecided2=int((Hh2 & ~Hp2).sum()))
        thr[str(q)] = d_
    out['thr'] = thr
    return out

def bern_L(k, Kf, delta=0.1, T=1):
    M = sum(math.comb(Kf - 1, j) for j in range(1, k))
    return math.log(2.0 * Kf * M * T / delta)
def eps_b(x, m, k=3, Kf=60, delta=0.1, T=1):
    """Theorem 1 (Bernstein): |est - true| <= sqrt(2 x L / m) + 2 L / (3 m), x = true UPI (or pi_f)."""
    L = bern_L(k, Kf, delta, T)
    return np.sqrt(2.0 * np.asarray(x, dtype=np.float64) * L / m) + 2.0 * L / (3.0 * m)
def bern_L2(k, Kf, delta=0.1, T=1):
    """union bound over the patterns themselves (sizes 2..k), not over (feature, pattern) pairs"""
    P_ = sum(math.comb(Kf, j) for j in range(2, k + 1))
    return math.log(2.0 * P_ * T / delta)
def lam_max(U):
    """largest utility weight lambda_f(c) = U(f)/sum_{g in c} U(g) of every pattern (known exactly)"""
    Ue = np.r_[np.asarray(U, np.float64), 0.0]
    mx = np.maximum(np.maximum(Ue[RF[:, 0]], Ue[RF[:, 1]]), np.maximum(Ue[RF[:, 2]], Ue[RF[:, 3]]))
    den = Ue[RF[:, 0]] + Ue[RF[:, 1]] + Ue[RF[:, 2]] + Ue[RF[:, 3]]
    return np.where(den > 0, mx / np.maximum(den, 1e-300), 1.0)
def eps2(x, lam, m, k=3, Kf=60, delta=0.1, T=1):
    """pattern-level Bernstein bound: the UPI estimate is one sum of independent terms with range <= lam/m and
    variance <= lam x / m, so |est - UPI| <= sqrt(2 lam x L2 / m) + 2 lam L2 / (3 m)."""
    L2 = bern_L2(k, Kf, delta, T); lam = np.asarray(lam, np.float64)
    return np.sqrt(2.0 * lam * np.asarray(x, dtype=np.float64) * L2 / m) + 2.0 * lam * L2 / (3.0 * m)
# ---------------------------------------------------------------- v26d: bound on the change of a UPI between snapshots
# With persistent seeds, UPI_t(c) - UPI_{t-1}(c) is estimated by a sum over instances of D_o = y_o^t - y_o^{t-1},
# y_o^s = 1[o sampled at s] 1[c in P_s(o)] w_o^s / U_c^s. The D_o are independent (one seed each), |D_o - E D_o| <= b,
# and Var(sum) <= V = E sum Dl_o^2, where Dl_o drops the deterministic terms of heavy instances (u >= tau, always sampled),
# so |Dl_o| <= b' = max lambda / m. V is bounded from the sample: sum D_o^2 has nonnegative terms <= b'^2, so
# V <= (b' sqrt(L1/2) + sqrt(b'^2 L1/2 + Vhat))^2 (one-sided, Maurer). Then |dhat - d| <= sqrt(2 V L2) + 2 b L2 / 3.
# Union bound over the P patterns and the T-1 updates: L1 = ln(2 P (T-1)/delta), L2 = ln(4 P (T-1)/delta).
@numba.njit
def sort_rows_local(ptr, r):
    for o in range(len(ptr) - 1):
        a = ptr[o]; b = ptr[o + 1]
        if b - a > 1:
            r[a:b] = np.sort(r[a:b])

@numba.njit
def cross_acc(ia, ib, wa, wb, pa, ra, pb, rb, X):
    """X[r] += wa[j] * wb[j] for every pattern r in both P_a(ia[j]) and P_b(ib[j]) (rows sorted)."""
    for j in range(len(ia)):
        i = ia[j]; k = ib[j]; x = pa[i]; xe = pa[i + 1]; y = pb[k]; ye = pb[k + 1]; w = wa[j] * wb[j]
        while x < xe and y < ye:
            if ra[x] == rb[y]:
                X[ra[x]] += w; x += 1; y += 1
            elif ra[x] < rb[y]:
                x += 1
            else:
                y += 1

def sample_emit(act, u, m, kmax):
    """threshold sample (persistent seeds) with the participation set of every sampled instance."""
    ua = np.where(act, u, 0.0); Uf = np.bincount(F, weights=ua, minlength=K); tau = np.where(Uf > 0, Uf / m, 0.0)
    sel = np.nonzero(act & (ua > tau[F] * R))[0]; w = np.maximum(ua[sel], tau[F[sel]])
    _, _, _, cnt, (lptr, lr) = run_enum(sel, np.zeros(len(sel)), kmax, active=act, emit=True)
    sort_rows_local(lptr, lr)
    wl = np.where(ua[sel] < tau[F[sel]], w, 0.0)     # light instances only: a heavy one (u >= tau) is always sampled
    return dict(sel=sel, w=w, wl=wl, lptr=lptr, lr=lr, U=Uf, est=local_sum(w, lptr, lr), sq=local_sum(wl * wl, lptr, lr))

def change_bound(a, b, m, kmax, nupd, delta=0.1):
    """estimated change of every UPI from snapshot a to b and its bound (both from sample_emit)."""
    Ua = dens(a['U']); Ub = dens(b['U']); ok = (Ua > 0) & (Ub > 0)
    Ua_ = np.where(ok, Ua, 1.0); Ub_ = np.where(ok, Ub, 1.0)
    common, ia, ib = np.intersect1d(a['sel'], b['sel'], assume_unique=True, return_indices=True)
    X = np.zeros(NR); cross_acc(ia.astype(np.int64), ib.astype(np.int64), a['wl'][ia], b['wl'][ib], a['lptr'], a['lr'], b['lptr'], b['lr'], X)
    vhat = np.maximum(a['sq'] / Ua_ ** 2 + b['sq'] / Ub_ ** 2 - 2.0 * X / (Ua_ * Ub_), 0.0)
    la = lam_max(a['U']); lb = lam_max(b['U']); bb = (la + lb) / m; b1 = np.maximum(la, lb) / m
    P_ = sum(math.comb(K, j) for j in range(2, kmax + 1)); L1 = math.log(2.0 * P_ * nupd / delta); L2 = math.log(4.0 * P_ * nupd / delta)
    V = (b1 * math.sqrt(L1 / 2) + np.sqrt(b1 ** 2 * L1 / 2 + vhat)) ** 2
    e = np.sqrt(2.0 * V * L2) + 2.0 * bb * L2 / 3.0
    dhat = b['est'] / Ub_ - a['est'] / Ua_
    return dhat, e, ok, vhat

def move_eval(a, b, ex_a, ex_b, m, kmax, nupd, delta=0.1, Ua=None, Ub=None):
    """compare certified moves (|dhat| > e) with the true changes of the UPI (exact values ex_a, ex_b)."""
    dhat, e, ok, vhat = change_bound(a, b, m, kmax, nupd, delta)
    sel = np.nonzero((RSIZE <= kmax) & ok & ((ex_a > 0) | (ex_b > 0)))[0]
    da = dens(a['U'] if Ua is None else Ua)[sel]; db = dens(b['U'] if Ub is None else Ub)[sel]
    up_a = ex_a[sel] / da; up_b = ex_b[sel] / db; d = up_b - up_a; dh = dhat[sel]; ee = e[sel]; err = np.abs(dh - d)
    cert = np.abs(dh) > ee; right = cert & (np.sign(dh) == np.sign(d)) & (d != 0)
    lam = np.maximum(lam_max(a['U'])[sel], lam_max(b['U'])[sel])
    naive = eps2(up_a, lam, m, kmax, K, delta, nupd + 1) + eps2(up_b, lam, m, kmax, K, delta, nupd + 1)
    out = dict(n=int(len(sel)), viol=int((err > ee).sum()), ratio_max=float((err / np.maximum(ee, 1e-300)).max()) if len(sel) else 0.0,
               e_q=[float(x) for x in np.quantile(ee, [0.1, 0.5, 0.9])] if len(sel) else [], naive_q=[float(x) for x in np.quantile(naive, [0.1, 0.5, 0.9])] if len(sel) else [],
               err_max=float(err.max()) if len(sel) else 0.0, n_cert=int(cert.sum()), n_cert_right=int(right.sum()),
               n_cert_naive=int((np.abs(dh) > naive).sum()), n_cert_naive_right=int(((np.abs(dh) > naive) & (np.sign(dh) == np.sign(d)) & (d != 0)).sum()),
               sum_abs_d=float(np.abs(d).sum()), sum_abs_d_cert=float(np.abs(d[right]).sum()))
    for g in (0.005, 0.01, 0.02, 0.05):
        big = np.abs(d) >= g
        out[f'g{g}'] = [int(big.sum()), int((big & right).sum()), int((big & (np.abs(dh) > naive)).sum())]
    return out

MOVE_ON = os.environ.get('V26_MOVE', '1') == '1'
MOVE_MS = [int(x) for x in os.environ.get('V26_MOVE_MS', '300,1000').split(',') if x]

def eps_hoef(m, k=3, Kf=60, delta=0.1, T=1):
    return float(math.sqrt(bern_L(k, Kf, delta, T) / (2.0 * m)))
def eps_vc(m, k=3, Kf=60, delta=0.1, c=0.5):
    d = 1 + int(math.floor(math.log2(sum(math.comb(Kf - 1, j) for j in range(1, k)))))
    return float(math.sqrt(c * (d + math.log(Kf / delta)) / m))

# ------------------------------------------------------------------ Pollster (priority sampling, persistent seeds)
def pollster_est(active, u, m, kmax, Rv=None, stored=None, recompute=None, changes=None, ages=None, tnow=0):
    Rv = R if Rv is None else Rv
    t0 = time.time(); sel_all = []; wt_all = []; tau_all = []; taus = np.zeros(K)
    an = np.nonzero(active)[0]; order = np.argsort(F[an], kind='stable'); an = an[order]
    bnd = np.searchsorted(F[an], np.arange(K + 1))
    for f in range(K):
        idx_f = an[bnd[f]:bnd[f + 1]]
        uf = u[idx_f]
        if SCHEME == 'threshold':
            # coordinated threshold (Poisson PPS) sampling: o is kept iff u(o)/r_o > tau_f = U(f)/m, i.e. r_o < min(1, m u(o)/U(f))
            Uf = float(uf.sum()); tau = Uf / m if Uf > 0 else 0.0
            if tau > 0:
                mk = (uf / Rv[idx_f]) > tau; s_ = idx_f[mk]; w_ = np.maximum(uf[mk], tau)
            else:
                s_ = idx_f[:0]; w_ = np.zeros(0)
        elif len(idx_f) <= m:
            s_, w_, tau = idx_f, uf.astype(np.float64), 0.0
        else:
            pr = uf / Rv[idx_f]; part = np.argpartition(-pr, m)
            s_ = idx_f[part[:m]]; tau = float(pr[part[m]]); w_ = np.maximum(u[s_], tau)
        sel_all.append(s_); wt_all.append(w_); tau_all.append(np.full(len(s_), tau)); taus[f] = tau
    sel = np.concatenate(sel_all); wt = np.concatenate(wt_all); tau = np.concatenate(tau_all)
    t_samp = time.time() - t0
    delta = None; isins = None
    if stored is not None and changes is not None:
        # winners within d of an inserted (delta update) or deleted (full recompute) instance; timed as part of Pollster
        ins_, del_ = changes; held = np.fromiter(stored.keys(), np.int64, count=len(stored))
        recompute = np.zeros(N, bool); delta = np.zeros(N, bool)
        for arr, mask in ((del_, recompute), (ins_, delta)):
            if len(arr) and len(held):
                tr = cKDTree(P[arr]); hit = tr.query_ball_point(P[held], D, workers=QW[0], return_length=True) > 0
                mask[held[hit]] = True
        if not DELTA_ON:
            recompute |= delta; delta = None
        else:
            isins = np.zeros(N, bool); isins[ins_] = True
    if stored is None:
        need = sel
    else:
        need = np.array([o for o in sel.tolist() if (o not in stored) or (recompute is not None and recompute[o])], np.int64)
    t1 = time.time()
    newP = {}
    if len(need):
        _, _, _, cnt, (lptr, lr) = run_enum(need, np.zeros(len(need)), kmax, active=active, emit=True)
        for i, o in enumerate(need.tolist()):
            newP[o] = lr[lptr[i]:lptr[i + 1]]
    n_delta = 0
    if stored is not None and delta is not None and isins is not None:
        dn = np.array([o for o in sel.tolist() if (o in stored) and (o not in newP) and delta[o]], np.int64)
        n_delta = len(dn)
        if n_delta:
            dptr, dr = run_delta(dn, kmax, active, isins)
            olds = [stored[o] for o in dn.tolist()]
            ol = np.fromiter((len(x) for x in olds), np.int64, count=n_delta)
            optr = np.r_[0, np.cumsum(ol)].astype(np.int64); orr = np.concatenate(olds).astype(np.int32) if ol.sum() else np.zeros(0, np.int32)
            nptr = optr + dptr; nr = np.empty(nptr[-1], np.int32)
            lens = union_rows(optr, orr, dptr, dr, nptr, nr)
            for i, o in enumerate(dn.tolist()):
                newP[o] = nr[nptr[i]:nptr[i] + lens[i]].copy()
    t_star = time.time() - t1
    keepP = {}
    if stored is not None:
        for o in sel.tolist():
            if o in stored and o not in newP:
                keepP[o] = stored[o]
    keepP.update(newP)
    cache_entries = 0
    if stored is not None and ages is not None and CACHE_X > 0:
        # keep participation sets of recently evicted winners that are still valid (no insertion or deletion within d)
        for o in sel.tolist():
            ages[o] = tnow
        inval = (recompute if recompute is not None else np.zeros(N, bool)) | (delta if delta is not None else np.zeros(N, bool))
        cand = [o for o in stored if o not in keepP and not inval[o]]
        cap = int(CACHE_X * len(sel))
        if len(cand) > cap:
            cand.sort(key=lambda o: -ages.get(o, -1)); cand = cand[:cap]
        cache = {o: stored[o] for o in cand}
        cache_entries = int(sum(len(x) for x in cache.values()))
        for o in list(ages.keys()):
            if o not in keepP and o not in cache:
                del ages[o]
    else:
        cache = {}
    t2 = time.time()
    rows = [keepP[o] for o in sel.tolist()]
    lens = np.fromiter((len(x) for x in rows), np.int64, count=len(rows))
    lptr = np.r_[0, np.cumsum(lens)].astype(np.int64); lr = np.concatenate(rows) if rows else np.zeros(0, np.int32)
    est = local_sum(wt, lptr, lr)
    vw = np.where(u[sel] < tau, tau * (tau - u[sel]), 0.0)
    var = local_sum(vw, lptr, lr)
    t_acc = time.time() - t2
    keepP.update(cache)
    return est, var, keepP, dict(sample_time=t_samp, star_time=t_star, acc_time=t_acc, recomputed=int(len(need)), delta_updated=int(n_delta), cache_entries=cache_entries,
                                 sampled=int(len(sel)), state_entries=int(lens.sum()), taus=taus,
                                 total_time=time.time() - t0)

# ===================================================================================================== v21: delta-driven maintenance
# Pollster maintenance that touches only (i) changed instances, (ii) instances whose key u/r or utility lies between the old
# and the new threshold (found in per-feature sorted indexes), and (iii) sampled instances within d of an insertion or a
# deletion.  Estimates are kept in accumulators: A[c] = sum of u over sampled heavy instances (u >= tau) with c in P(o),
# B[f, c] = number of sampled light instances of feature f with c in P(o); est = A + B^T tau.  The sample and the estimates
# are identical to a rebuild with the same seeds (checked at every snapshot).

@numba.njit
def acc_apply(sign, heavy, wu, fidx, ptr, ranks, A, B):
    for i in range(len(sign)):
        s = sign[i]
        for j in range(ptr[i], ptr[i + 1]):
            r = ranks[j]
            if heavy[i]:
                A[r] += s * wu[i]
            else:
                B[fidx[i], r] += s

def pollster_rebuild(act, u, m, kmax, Rv=None):
    """Pollster-Rebuild: threshold sample and estimates from scratch (same seeds unless Rv is given)."""
    Rv = R if Rv is None else Rv
    t0 = time.time()
    ua = np.where(act, u, 0.0); Uf = np.bincount(F, weights=ua, minlength=K)
    tau = np.where(Uf > 0, Uf / m, 0.0)
    sel = np.nonzero(act & (ua > tau[F] * Rv))[0]
    w = np.maximum(u[sel], tau[F[sel]])
    t1 = time.time()
    est = run_enum(sel, w, kmax, active=act)[0]
    return est, dict(sample_time=t1 - t0, star_time=time.time() - t1, total_time=time.time() - t0, sampled=int(len(sel)))

class PMaint:
    """coordinated threshold sample with stored participation sets, maintained from deltas."""
    def __init__(self, act, u, m, kmax, rebuild_frac=0.7, resort_frac=0.5, auto=False):
        self.m = m; self.kmax = kmax; self.rf = rebuild_frac; self.sf = resort_frac
        self.act = act.copy(); self.u = np.where(act, u, 0.0).astype(np.float64)
        self.stale = False; self.est_rb = None; self.rng = np.random.default_rng(5)
        tb = time.time(); self.full(); ns = max(int(self.ins.sum()), 1); self.t_full = time.time() - tb
        self.ce = self.t_star_full / ns; self.cr = 0.6 * self.ce; self.cq = 0.05 * self.ce
        # v26d: calibration factors, measured time / modelled time, for each mode (EMA over the updates of that mode)
        self.kd = max(1.0, (self.t_full - getattr(self, 't_idx', 0.0)) / max(self.t_star_full, 1e-9)); self.kr = 1.0; self.kn = 1.0
        if auto:     # measure the per-star cost of a rebuild once (not counted as maintenance)
            _, irb = pollster_rebuild(self.act, self.u, m, kmax); self.cr = irb['star_time'] / max(irb['sampled'], 1)
            self.kr = max(1.0, irb['total_time'] / max(irb['star_time'], 1e-9))

    # ---------------------------------------------------------------- from scratch (initial build and fallback)
    def full(self):
        act, u = self.act, self.u
        self.Uf = np.bincount(F, weights=u, minlength=K)
        self.tau = np.where(self.Uf > 0, self.Uf / self.m, 0.0)
        self.ins = act & (u > self.tau[F] * R)
        self.heavy = self.ins & (u >= self.tau[F])
        sel = np.nonzero(self.ins)[0]
        tb_ = time.time(); _, _, _, cnt, (lptr, lr) = run_enum(sel, np.zeros(len(sel)), self.kmax, active=act, emit=True); self.t_star_full = time.time() - tb_
        self.Pst = {int(o): lr[lptr[i]:lptr[i + 1]].copy() for i, o in enumerate(sel)}
        self.A = np.zeros(NR); self.B = np.zeros((K, NR))
        acc_apply(np.ones(len(sel)), self.heavy[sel], u[sel], F[sel], lptr, lr, self.A, self.B)
        tb_ = time.time(); self.build_index(); self.t_idx = time.time() - tb_

    def build_index(self):
        ids = np.nonzero(self.act)[0]; f = F[ids]
        key = self.u[ids] / R[ids]
        o = np.lexsort((key, f)); self.ik_ids = ids[o].astype(np.int32 if N < 2**31 else np.int64); self.ik_key = key[o]; self.ik_bnd = np.searchsorted(f[o], np.arange(K + 1))
        o = np.lexsort((self.u[ids], f)); self.iu_ids = ids[o].astype(self.ik_ids.dtype); self.iu_u = self.u[ids][o]; self.iu_bnd = np.searchsorted(f[o], np.arange(K + 1))
        self.dirty = np.zeros(N, bool); self.dids = np.zeros(0, np.int64); self.nd = 0; self.nact = len(ids)

    def estimate(self):
        return self.est_rb if self.stale else self.A + self.B.T @ self.tau

    # ---------------------------------------------------------------- v26: cost-based choice between delta and rebuild
    def auto_step(self, ins, dele, chg, act, u, probe=2000, back=0.8):
        """Both modes give the same sample and estimates (persistent seeds), so each update can use the cheaper one.
        Cost model (per sampled instance): delta = c_q (find changed stars) + c_e * h (re-enumerate and store the changed
        stars), rebuild = c_r (enumerate all sampled stars without storing them); h is the share of sampled instances within
        d of a change, estimated from a random subset of `probe` sampled instances. Leaving rebuild mode costs one stored
        enumeration of all stars (full()); it is taken when delta is predicted to be cheaper by the factor `back`."""
        t0 = time.time()
        chg_pts = np.concatenate([ins, dele]).astype(np.int64)
        # in rebuild mode, extrapolate h from the last probe, h = 1 - (1 - h_last)^(changes / changes_last), and probe
        # only when the extrapolation does not favour rebuilding clearly (by a factor 2)
        skip = False
        if self.stale and getattr(self, 'h_last', None) is not None and self.nchg_last > 0 and len(chg_pts) > 0:
            h_ex = 1.0 - (1.0 - min(self.h_last, 0.999)) ** (len(chg_pts) / self.nchg_last)
            n_ = max(getattr(self, 'nsel_rb', 1), 1)
            md_ = (self.cq + self.ce * h_ex) * n_
            skip = self.kd * md_ + self._t_ne(len(chg_pts), n_) + self.t_full / 3.0 > 2.0 * back * self.kr * self.cr * n_
        if skip:
            h = h_ex
        else:
            held = self.held_ids(); n_ = max(len(held), 1)
            if len(chg_pts) and len(held):
                pr = held if len(held) <= probe else self.rng.choice(held, probe, replace=False)
                h = probe_share(pr, chg_pts)
                # locating the changed stars: one pass over the changes plus one test per held instance (measured on the probe)
                self.ts_per = probe_share.t_sort / len(chg_pts); self.tq_per = probe_share.t_query / len(pr)
            else:
                h = 0.0
            self.h_last = h; self.nchg_last = len(chg_pts)
        t_probe = time.time() - t0
        md = (self.cq + self.ce * h) * n_; mr = self.cr * n_
        t_ne = self._t_ne(len(chg_pts), n_)
        pred_d = self.kd * md + t_ne; pred_r = self.kr * mr
        # leaving rebuild mode costs one full stored build (measured), amortized over 3 updates
        go_delta = (pred_d + self.t_full / 3.0 < back * pred_r) if self.stale else (pred_d < pred_r)
        if go_delta and self.stale:
            tb = time.time(); self.act = act.copy(); self.u = np.where(act, u, 0.0).astype(np.float64); self.stale = False; self.full()
            ns = max(int(self.ins.sum()), 1); self.ce = 0.5 * self.ce + 0.5 * self.t_star_full / ns; self.t_full = time.time() - tb
            info = dict(mode='full', star_time=time.time() - tb)
        elif go_delta:
            tb = time.time(); info = self.step(ins, dele, chg, act, u); t_act = time.time() - tb
            if info.get('mode') == 'delta':
                if info['recomputed'] > 50:
                    self.ce = 0.5 * self.ce + 0.5 * info['star_time'] / info['recomputed']
                self.cq = 0.5 * self.cq + 0.5 * info['near_time'] / n_
                t_nr = info.get('near_time', 0.0)
                if t_ne > 0:
                    self.kn = 0.5 * self.kn + 0.5 * self.kn * t_nr / t_ne
                self.kd = max(0.5, 0.5 * self.kd + 0.5 * (t_act - t_nr) / max(md, 1e-9))
        else:
            tb = time.time(); est, irb = pollster_rebuild(act, u, self.m, self.kmax); t_act = time.time() - tb
            self.cr = 0.5 * self.cr + 0.5 * irb['star_time'] / max(irb['sampled'], 1)
            self.kr = 0.5 * self.kr + 0.5 * t_act / max(self.cr * n_, 1e-9)
            self.stale = True; self.est_rb = est; self.act = act; self.u = u     # references only: copied when leaving rebuild mode
            self.nsel_rb = irb['sampled']; info = dict(mode='rebuild', star_time=irb['star_time'])
        info.update(probe_skip=bool(skip), probe_h=h, probe_time=t_probe, pred_delta=pred_d, pred_rebuild=pred_r, total_time=time.time() - t0,
                    c_e=self.ce, c_r=self.cr, c_q=self.cq)
        return info

    def _t_ne(self, nchg, n_):
        """predicted time to locate the changed stars, calibrated (kn) on the measured near_time of incremental updates."""
        if nchg == 0:
            return 0.0
        return getattr(self, 'kn', 1.0) * (getattr(self, 'ts_per', 0.0) * nchg + getattr(self, 'tq_per', 0.0) * n_)

    def held_ids(self):
        if self.stale:
            return self._rb_sel()
        return np.fromiter(self.Pst.keys(), np.int64, count=len(self.Pst))

    def _rb_sel(self):
        ua = np.where(self.act, self.u, 0.0); Uf = np.bincount(F, weights=ua, minlength=K); tau = np.where(Uf > 0, Uf / self.m, 0.0)
        return np.nonzero(self.act & (ua > tau[F] * R))[0]

    def state_entries(self):
        return int(sum(len(x) for x in self.Pst.values()))

    # ---------------------------------------------------------------- one batch of changes
    def step(self, ins, dele, chg, act, u):
        t0 = time.time(); m = self.m
        ch = np.unique(np.concatenate([ins, dele, chg]).astype(np.int64))
        un = np.where(act[ch], u[ch], 0.0); uo = self.u[ch]
        self.Uf += np.bincount(F[ch], weights=un - uo, minlength=K)
        self.Uf[np.abs(self.Uf) < 1e-9] = 0.0
        tau_o = self.tau.copy(); tau_n = np.where(self.Uf > 0, self.Uf / m, 0.0)
        # apply the new utilities / activity to the changed instances
        self.u[ch] = un; self.act[ch] = act[ch]
        nd = ch[~self.dirty[ch]]; self.dirty[nd] = True; self.dids = np.concatenate([self.dids, nd]); self.nd += len(nd)
        # candidates: all dirty instances, plus clean instances whose key or utility lies between the two thresholds
        cands = [self.dids]
        for f in np.nonzero(tau_n != tau_o)[0]:
            lo, hi = (tau_o[f], tau_n[f]) if tau_o[f] < tau_n[f] else (tau_n[f], tau_o[f])
            a, b = self.ik_bnd[f], self.ik_bnd[f + 1]
            if b > a:
                i0 = a + np.searchsorted(self.ik_key[a:b], lo, 'left'); i1 = a + np.searchsorted(self.ik_key[a:b], hi, 'right')
                cands.append(self.ik_ids[i0:i1])
            a, b = self.iu_bnd[f], self.iu_bnd[f + 1]
            if b > a:
                i0 = a + np.searchsorted(self.iu_u[a:b], lo, 'left'); i1 = a + np.searchsorted(self.iu_u[a:b], hi, 'right')
                cands.append(self.iu_ids[i0:i1])
        C = np.unique(np.concatenate([c_.astype(np.int64) for c_ in cands])) if cands else np.zeros(0, np.int64)
        uC = self.u[C]; aC = self.act[C]
        old_in = self.ins[C].copy(); old_heavy = self.heavy[C].copy()
        new_in = aC & (uC > tau_n[F[C]] * R[C]); new_heavy = new_in & (uC >= tau_n[F[C]])
        self.ins[C] = new_in; self.heavy[C] = new_heavy
        t_samp = time.time() - t0
        # sampled instances whose star changed (an inserted or deleted neighbour, in either snapshot)
        t1 = time.time()
        near = np.zeros(0, np.int64)
        chg_pts = np.concatenate([ins, dele]).astype(np.int64)
        held = np.fromiter(self.Pst.keys(), np.int64, count=len(self.Pst))
        newcomers = C[new_in & ~old_in]
        cand_held = np.union1d(held, newcomers)
        if len(chg_pts) and len(cand_held):
            tr = cKDTree(P[chg_pts]); hit = tr.query_ball_point(P[cand_held], D, workers=QW[0], return_length=True) > 0
            near = cand_held[hit]
            near = near[self.ins[near]]
        t_near = time.time() - t1
        need = np.union1d(near, newcomers)
        nsel = int(self.ins.sum()) if len(need) > self.rf * max(len(held), 1) else 0
        if nsel and len(need) > self.rf * nsel:
            # most sampled stars changed: recompute everything (same sample, same output)
            t2 = time.time(); self.tau = tau_n; self.full()
            return dict(sample_time=t_samp, near_time=t_near, star_time=time.time() - t2, acc_time=0.0, recomputed=int(self.ins.sum()),
                        candidates=int(len(C)), mode='full', total_time=time.time() - t0, sampled=int(self.ins.sum()))
        t2 = time.time()
        if len(need):
            _, _, _, cnt, (lptr, lr) = run_enum(need, np.zeros(len(need)), self.kmax, active=self.act, emit=True)
            newP = {int(o): lr[lptr[i]:lptr[i + 1]] for i, o in enumerate(need)}
        else:
            newP = {}
        t_star = time.time() - t2
        # instances whose contribution changes: membership/class/utility changes in C, and recomputed sets (vectorized filter)
        t3 = time.time()
        X = np.union1d(C, need)
        ic = np.minimum(np.searchsorted(C, X), max(len(C) - 1, 0)); inC = (len(C) > 0) & (C[ic] == X) if len(C) else np.zeros(len(X), bool)
        was_in = np.where(inC, old_in[ic] if len(C) else False, True)
        was_h = np.where(inC, old_heavy[ic] if len(C) else False, self.heavy[X])
        ih = np.minimum(np.searchsorted(ch, X), max(len(ch) - 1, 0)); inch = (len(ch) > 0) & (ch[ih] == X) if len(ch) else np.zeros(len(X), bool)
        uoX = np.where(inch, uo[ih] if len(ch) else 0.0, self.u[X])
        is_in = self.ins[X]; is_h = self.heavy[X]; unX = self.u[X]
        ir = np.minimum(np.searchsorted(need, X), max(len(need) - 1, 0)); rec = (need[ir] == X) if len(need) else np.zeros(len(X), bool)
        upd = rec | (was_in != is_in) | (was_h != is_h) | (is_h & (uoX != unX))
        Xc = X[upd]; was_in = was_in[upd]; was_h = was_h[upd]; uoX = uoX[upd]; is_in = is_in[upd]; is_h = is_h[upd]; unX = unX[upd]; rec = rec[upd]
        rm = np.nonzero(was_in)[0]; ad = np.nonzero(is_in)[0]
        rem_o = Xc[rm].tolist(); rem_h = was_h[rm]; rem_u = uoX[rm]; rem_P = [self.Pst[o] for o in rem_o]
        add_o = Xc[ad].tolist(); add_h = is_h[ad]; add_u = unX[ad]
        add_P = [newP[o] if r_ else self.Pst[o] for o, r_ in zip(add_o, rec[ad].tolist())]
        for lst_o, lst_h, lst_u, lst_P, sgn in ((rem_o, rem_h, rem_u, rem_P, -1.0), (add_o, add_h, add_u, add_P, 1.0)):
            if len(lst_o):
                lens = np.fromiter((len(x) for x in lst_P), np.int64, count=len(lst_P))
                ptr = np.r_[0, np.cumsum(lens)].astype(np.int64)
                rk = np.concatenate(lst_P).astype(np.int32) if lens.sum() else np.zeros(0, np.int32)
                oo = np.asarray(lst_o, np.int64)
                acc_apply(np.full(len(oo), sgn), np.asarray(lst_h, np.bool_), np.asarray(lst_u, np.float64), F[oo], ptr, rk, self.A, self.B)
        for o in rem_o:
            self.Pst.pop(o, None)
        for o, pp in zip(add_o, add_P):
            self.Pst[o] = pp
        self.tau = tau_n
        t_acc = time.time() - t3
        # keep the sorted indexes fresh (amortized)
        t4 = time.time()
        if self.nd > self.sf * max(self.nact, 1):
            self.build_index()
        t_idx = time.time() - t4
        return dict(sample_time=t_samp, near_time=t_near, star_time=t_star, acc_time=t_acc, index_time=t_idx,
                    recomputed=int(len(need)), candidates=int(len(C)), changed_contrib=int(len(add_o) + len(rem_o)),
                    mode='delta', total_time=time.time() - t0, sampled=int(len(self.Pst)))

# ------------------------------------------------------------------ Exact-Inc CSR (int32 ranks)
@numba.njit
def rebuild_csr(gptr, gr, pos_new, nptr, nr, drop):
    Nn = len(gptr) - 1
    lens = np.empty(Nn, np.int64)
    for o in range(Nn):
        if pos_new[o] >= 0:
            lens[o] = nptr[pos_new[o] + 1] - nptr[pos_new[o]]
        elif drop[o]:
            lens[o] = 0
        else:
            lens[o] = gptr[o + 1] - gptr[o]
    optr = np.empty(Nn + 1, np.int64); optr[0] = 0
    for o in range(Nn):
        optr[o + 1] = optr[o] + lens[o]
    out = np.empty(optr[Nn], np.int32)
    for o in range(Nn):
        a = optr[o]
        if pos_new[o] >= 0:
            b = nptr[pos_new[o]]
            for i in range(lens[o]):
                out[a + i] = nr[b + i]
        elif not drop[o]:
            b = gptr[o]
            for i in range(lens[o]):
                out[a + i] = gr[b + i]
    return optr, out

def build_store(nodes, kmax, active):
    _, t, _, cnt, (lptr, lr) = run_enum(nodes, np.zeros(len(nodes)), kmax, active=active, emit=True)
    pos_new = np.full(N, -1, np.int64); pos_new[nodes] = np.arange(len(nodes))
    gptr, gr = rebuild_csr(np.zeros(N + 1, np.int64), np.zeros(0, np.int32), pos_new, lptr, lr, np.zeros(N, bool))
    return gptr, gr, t

def neighbours_of(nodes):
    if len(nodes) == 0:
        return np.zeros(0, np.int64)
    parts = [nodes.astype(np.int64)]
    if GRID_ON and GRIDS:
        seen_ = set()
        for a, gk, gid in GRIDS:
            if id(gk) in seen_:
                continue
            seen_.add(id(gk)); parts.append(grid_neigh(nodes, (gk, gid), samef=True)[1])
        return np.unique(np.concatenate(parts))
    for a in range(0, len(nodes), 50000):
        nb = TREE.query_ball_point(P[nodes[a:a + 50000]], D, workers=QW[0])
        lens = np.fromiter((len(x) for x in nb), np.int64, count=len(nb))
        parts.append(np.fromiter(itertools.chain.from_iterable(nb), np.int64, count=int(lens.sum())))
    return np.unique(np.concatenate(parts))

def set_data(Pn, Fn, Kn, seed=7):
    global P, F, TREE, N, K, R
    P = np.ascontiguousarray(Pn); F = np.asarray(Fn, np.int64); K = Kn; N = len(P); TREE = cKDTree(P)
    R = np.random.default_rng(seed).random(N) + 1e-12
    set_K(Kn)

set_K(K)
_w = np.arange(min(300, N), dtype=np.int64)
_a, _, _, _c, (_lp, _lr) = run_enum(_w, np.ones(len(_w)), 4, emit=True); csr_sum(_w[:1], np.ones(1), np.zeros(N + 1, np.int64), np.zeros(1, np.int32))
local_sum(np.ones(len(_w)), _lp, _lr); build_store(_w, 3, np.ones(N, bool)); pollster_est(np.ones(N, bool), np.ones(N), 5, 3)
_isn = np.zeros(N, bool); _isn[_w[::3]] = True; _dp, _dr = run_delta(_w, 4, np.ones(N, bool), _isn)
union_rows(_lp[:len(_w) + 1], _lr, _dp, _dr, _lp[:len(_w) + 1] + _dp, np.empty(_lp[len(_w)] + _dp[-1], np.int32))
_pm = PMaint(np.ones(N, bool), np.ones(N), 5, 3); _ch = np.arange(min(50, N)); _u2 = np.ones(N); _u2[_ch] = 2.0
_pm.step(np.zeros(0, np.int64), np.zeros(0, np.int64), _ch, np.ones(N, bool), _u2); del _pm
log('v19 jit warm-up done; threads', nthreads(), 'NR', NR)


# =====================================================================================================================
# v19 experiments (all on the parallel kernel above; the `base` table is single-threaded for every method)
# =====================================================================================================================
S['v19'] = {}; S['scheme'] = SCHEME; S['reps_t'] = REPS_T; NAMES = list(TOP)
def save():
    os.makedirs(OUT, exist_ok=True)
    json.dump(S, open(os.path.join(OUT, f"pollster_v22_{PARTS.replace(',', '_')}.json"), 'w'), indent=1, default=float)

def dict_to_dense(gd):
    m2r = {int(m): i for i, m in enumerate(RMASK.tolist())}
    out = np.zeros(NR)
    for k_, v in gd.items():
        r = m2r.get(int(k_))
        if r is not None:
            out[r] = v
    return out

def maxdiff(a, b):
    return float(np.abs(a - b).max() / max(np.abs(a).max(), 1e-12))

def pol_row(act, u, ex, U, kmax, ms):
    out = {}
    for m in ms:
        est, var, _, info = pollster_est(act, u, m, kmax)
        info.pop('taus', None)
        if ex is not None:
            info['acc'] = compare(ex, est, U, kmax, var=var, m=m)
        out[str(m)] = info
    return out

# ------------------------------------------------------------------ base: published exact miners, single thread
if 'base' in PARTS:
    numba.set_num_threads(1); QW[0] = 1
    S['v19']['base'] = []
    u_all = np.log1p(tot[IDS]).astype(np.float64)
    ucities = np.unique(CITY); corder = rng.permutation(ucities)
    BCAP = float(os.environ.get('V17_BCAP', '2400')); ROWCAP = int(float(os.environ.get('V17_ROWCAP', '6e8')))
    for frac in [float(x) for x in os.environ.get('V17_FRACS', '0.125,0.25,0.5,1.0').split(',')]:
        for kmax in (3, 4):
            if kmax == 4 and frac not in [float(x) for x in os.environ.get('V17_K4FRACS', '0.125,0.25').split(',')]:
                continue
            try:
                cs = set(corder[:max(1, int(round(frac * len(ucities))))].tolist())
                act = np.isin(CITY, list(cs)); nodes = np.nonzero(act)[0]
                U = np.bincount(F[act], weights=u_all[act], minlength=K)
                ex, t_ex, _, c_ex, _ = run_enum(nodes, u_all[nodes], kmax, active=act)
                row = dict(frac=frac, kmax=kmax, n=int(len(nodes)), star_time=t_ex, exact_state=int(c_ex.sum()),
                           patterns=int(((RSIZE <= kmax) & (ex > 0)).sum()))
                gd_j, ij = run_join(act, u_all, kmax, BCAP, ROWCAP, row['exact_state'])
                row['join'] = ij
                if gd_j is not None:
                    row['join_maxdiff'] = maxdiff(ex, dict_to_dense(gd_j))
                del gd_j
                row['pollster'] = pol_row(act, u_all, ex, U, kmax, MS)
                log('base', json.dumps({k_: v for k_, v in row.items() if k_ != 'pollster'})[:900],
                    json.dumps({m: dict(t=v['total_time'], mae=v['acc']['mae'], f1=v['acc']['thr']['90']['f1'])
                                for m, v in row['pollster'].items()}))
                S['v19']['base'].append(row); save()
            except Exception:
                import traceback; tb = traceback.format_exc(); log('ERROR base\n' + tb); S['v19']['base'].append(dict(frac=frac, kmax=kmax, error=tb)); save()
    S['v19']['base_cq'] = []
    for frac in [float(x) for x in os.environ.get('V17_CQFRACS', '0.125,0.25,0.5,1.0').split(',')]:
        try:
            cs = set(corder[:max(1, int(round(frac * len(ucities))))].tolist())
            act = np.isin(CITY, list(cs)); nodes = np.nonzero(act)[0]
            ex, t_ex, _, c_ex, _ = run_enum(nodes, u_all[nodes], 3, active=act)
            slots = 1 << int(min(29, max(22, np.ceil(np.log2(3 * c_ex.sum())))))
            gd_c, ic = run_cq(act, u_all, 3, float(os.environ.get('V17_CQCAP', '1800')), slots)
            if gd_c is None and ic.get('status') == 'hash full':
                gd_c, ic = run_cq(act, u_all, 3, float(os.environ.get('V17_CQCAP', '1800')), slots * 2)
            row = dict(frac=frac, kmax=3, n=int(len(nodes)), star_time=t_ex, cq=ic)
            if gd_c is not None:
                row['cq_maxdiff'] = maxdiff(ex, dict_to_dense(gd_c))
            log('base cq', json.dumps(row)[:900]); S['v19']['base_cq'].append(row); save()
            del gd_c
        except Exception:
            import traceback; tb = traceback.format_exc(); log('ERROR cq\n' + tb); S['v19']['base_cq'].append(dict(frac=frac, error=tb)); save()
    numba.set_num_threads(TH_MAX); QW[0] = -1

# ------------------------------------------------------------------ static evaluation (full exact k3 and k4)
def static_eval(tag, act, u, ms, kmaxs=(3, 4)):
    nodes = np.nonzero(act)[0]; U = np.bincount(F[act], weights=u[act], minlength=K)
    row = dict(tag=tag, n=int(len(nodes)), K=int(K), d=float(D))
    for kmax in kmaxs:
        tex = []
        for rep in range(REPS_T):
            ex, t_ex, _, c_ex, _ = run_enum(nodes, u[nodes], kmax, active=act); tex.append(t_ex)
        t_ex = float(np.median(tex))
        row[f'exact_k{kmax}'] = dict(time=t_ex, times=tex, state=int(c_ex.sum()), patterns=int(((RSIZE <= kmax) & (ex > 0)).sum()))
        pr = {}
        for m in ms:
            tp_ = []
            for rep in range(REPS_T):
                est, var, _, info = pollster_est(act, u, m, kmax); tp_.append(info['total_time'])
            info['times'] = tp_; info['total_time'] = float(np.median(tp_)); taus = info.pop('taus')
            info['acc'] = compare(ex, est, U, kmax, var=var, m=m)
            info['speedup'] = t_ex / max(info['total_time'], 1e-9); info['eps'] = float(eps_b(info['acc']['thr']['90']['mu'], m, kmax, K))
            info['eps_hoef'] = eps_hoef(m, kmax, K)
            pr[str(m)] = info
        row[f'pollster_k{kmax}'] = pr
        log(tag, f'k{kmax}', json.dumps(dict(n=len(nodes), ex=t_ex, st=int(c_ex.sum()), **{
            m: dict(t=v['total_time'], sp=v['speedup'], st=v['state_entries'], mae=v['acc']['mae'], p99=v['acc']['p99'],
                    mx=v['acc']['max'], eps=v['eps'], cov=v['acc'].get('cover2sd'), f1=v['acc']['thr']['90']['f1'])
            for m, v in pr.items()})))
    return row

if 'tile' in PARTS:
    S['v19']['tile'] = []
    P0 = P.copy(); F0 = F.copy(); u0 = np.log1p(tot[IDS]).astype(np.float64); K0 = K
    for T_ in [int(x) for x in os.environ.get('V19_TILES', '1,2,4,8').split(',')]:
        try:
            Ps = []
            for r_ in range(T_):
                th = np.radians(2.0 * r_); c, s_ = np.cos(th), np.sin(th)
                Ps.append(np.c_[c * P0[:, 0] - s_ * P0[:, 1], s_ * P0[:, 0] + c * P0[:, 1], P0[:, 2]])
            set_data(np.vstack(Ps), np.tile(F0, T_), K0); del Ps
            row = static_eval('tile x%d' % T_, np.ones(N, bool), np.tile(u0, T_), MS); row['tiles'] = T_
            S['v19']['tile'].append(row); save()
        except Exception:
            import traceback; tb = traceback.format_exc(); log('ERROR tile\n' + tb); S['v19']['tile'].append(dict(tiles=T_, error=tb)); save()
    set_data(P0, F0, K0)

if 'K' in PARTS:
    S['v19']['vary_K'] = []
    P0 = P.copy(); F0 = F.copy(); K0 = K
    for Kx in [int(x) for x in os.environ.get('V13_KS', '20,40').split(',')]:
        try:
            topx = [c for c in vc.index if c not in GENERIC][:Kx]; fx = {c: j for j, c in enumerate(topx)}
            fallx = pois.cat.map(fx).fillna(-1).astype(int).values
            idsx = np.nonzero(gmask & (fallx >= 0) & (tot > 0))[0]
            set_data(XYZ_all[idsx], fallx[idsx], Kx)
            row = static_eval('K=%d' % Kx, np.ones(N, bool), np.log1p(tot[idsx]).astype(np.float64), MS); row['Kx'] = Kx
            S['v19']['vary_K'].append(row); save()
        except Exception:
            import traceback; tb = traceback.format_exc(); log('ERROR K\n' + tb); S['v19']['vary_K'].append(dict(K=Kx, error=tb)); save()
    set_data(P0, F0, K0)

if 'dvar' in PARTS:
    S['v19']['dvar'] = []; D0 = D
    for dd in [float(x) for x in os.environ.get('V17_DS', '100,200,300').split(',')]:
        try:
            D = dd
            row = static_eval('d=%g' % dd, np.ones(N, bool), np.log1p(tot[IDS]).astype(np.float64), MS,
                              kmaxs=(3, 4) if dd <= 200 else (3,))
            S['v19']['dvar'].append(row); save()
        except Exception:
            import traceback; tb = traceback.format_exc(); log('ERROR dvar\n' + tb); S['v19']['dvar'].append(dict(d=dd, error=tb)); save()
    D = D0

# ------------------------------------------------------------------ utility distributions + sampling ablation (k=3)
if 'util' in PARTS:
    S['v19']['util'] = []
    try:
        act_all = np.ones(N, bool); nodes_all = np.arange(N, dtype=np.int64); tot_i = tot[IDS].astype(np.float64)
        rgu = np.random.default_rng(2027)
        UTILS = dict(log=np.log1p(tot_i), raw=tot_i, uniform=np.ones(N),
                     lognormal=np.exp(rgu.normal(0, 2.0, N)), pareto=rgu.pareto(1.1, N) + 1.0)
        gptr, gr, t_store = build_store(nodes_all, 3, act_all); log('util store', len(gr), t_store)
        REPS = int(os.environ.get('V17_REPS', '10')); idx_by_f = [np.nonzero(F == f)[0] for f in range(K)]
        sel3 = RSIZE <= 3
        UMETH = os.environ.get('V26_UMETH', 'pps,threshold,priority,uniform,uniform_sn').split(',')
        UKEEP = os.environ.get('V26_UTILS', 'log,raw,uniform,lognormal,pareto').split(',')
        for uname, uu in UTILS.items():
            if uname not in UKEEP:
                continue
            U = np.bincount(F, weights=uu, minlength=K)
            exf = [csr_sum(idx_by_f[f], uu[idx_by_f[f]], gptr, gr) for f in range(K)]
            ex = np.sum(exf, axis=0)
            row = dict(utility=uname, res={})
            for m in MS:
                for meth in UMETH:
                    reps = []
                    for r_ in range(REPS):
                        rr = np.random.default_rng(1000 * r_ + m); Rv = rr.random(N) + 1e-12
                        est = np.zeros(NR); pimax = 0.0; fviol = 0; fratio = 0.0; nsamp = 0
                        for f in range(K):
                            idx_f = idx_by_f[f]; nf = len(idx_f)
                            if nf == 0:
                                continue
                            if meth == 'pps':
                                s_ = rr.choice(idx_f, size=m, replace=True, p=uu[idx_f] / U[f]); w_ = np.full(m, U[f] / m)
                            elif meth == 'threshold':
                                tau = U[f] / m; mk_ = (uu[idx_f] / Rv[idx_f]) > tau; s_ = idx_f[mk_]; w_ = np.maximum(uu[s_], tau)
                            elif meth == 'uniform_sn':
                                mm = min(m, nf); s_ = rr.choice(idx_f, size=mm, replace=False); w_ = uu[s_] * U[f] / max(uu[s_].sum(), 1e-300)
                            elif meth == 'priority':
                                if nf <= m:
                                    s_ = idx_f; w_ = uu[idx_f]
                                else:
                                    pk = uu[idx_f] / Rv[idx_f]; part = np.argpartition(-pk, m)
                                    s_ = idx_f[part[:m]]; tau = float(pk[part[m]]); w_ = np.maximum(uu[s_], tau)
                            else:
                                mm = min(m, nf); s_ = rr.choice(idx_f, size=mm, replace=False); w_ = uu[s_] * nf / mm
                            ef = csr_sum(s_, w_, gptr, gr); est += ef; nsamp += len(np.unique(s_))
                            mk = sel3 & (exf[f] > 0)
                            if mk.any():
                                dif = np.abs(ef[mk] - exf[f][mk]) / U[f]; pimax = max(pimax, float(dif.max()))
                                ebf = eps_b(exf[f][mk] / U[f], m, 3, K, DELTA_B); fviol += int((dif > ebf).sum()); fratio = max(fratio, float((dif / ebf).max()))
                        e = compare(ex, est, U, 3, mus=(90,), m=m)
                        reps.append(dict(mae=e['mae'], max=e['max'], f1=e['thr']['90']['f1'], pi_max=pimax, fviol=fviol, fratio=fratio,
                                         upi_viol=e['bern_viol'], upi_ratio=e['bern_ratio_max'], hhat_recall=e['thr']['90']['Hhat_recall'],
                                         size=nsamp, hhat_precision=e['thr']['90']['Hhat_precision'], upi2_viol=e['bern2_viol'], upi2_ratio=e['bern2_ratio_max'],
                                         hhat2_precision=e['thr']['90']['Hhat2_precision'], hhat2_recall=e['thr']['90']['Hhat2_recall'],
                                         hplus_precision=e['thr']['90']['Hplus_precision'], hplus_recall=e['thr']['90']['Hplus_recall'],
                                         hplus2_precision=e['thr']['90']['Hplus2_precision'], hplus2_recall=e['thr']['90']['Hplus2_recall'],
                                         e2e=e['eps2_over_eps_median'], eps_mu=e['thr']['90']['eps_mu'], eps2_mu_med=e['thr']['90']['eps2_mu_q'][1]))
                    agg = {k_: float(np.mean([x[k_] for x in reps])) for k_ in reps[0]}
                    agg.update(max_over_reps=float(max(x['max'] for x in reps)), pi_max_over_reps=float(max(x['pi_max'] for x in reps)),
                               eps_bound=eps_vc(m, 3, K), bound_violations=int(sum(x['pi_max'] > eps_vc(m, 3, K) for x in reps)),
                               eps_hoef=eps_hoef(m, 3, K), hoef_violations=int(sum(x['pi_max'] > eps_hoef(m, 3, K) for x in reps)),
                               bern_reps_viol=int(sum(x['fviol'] > 0 for x in reps)), bern_fratio_max=float(max(x['fratio'] for x in reps)),
                               bern_upi_reps_viol=int(sum(x['upi_viol'] > 0 for x in reps)), bern2_upi_reps_viol=int(sum(x['upi2_viol'] > 0 for x in reps)))
                    row['res'][f'{meth}_{m}'] = agg; log('util', uname, meth, m, json.dumps(agg))
            S['v19']['util'].append(row); save()
        del gptr, gr
    except Exception:
        import traceback; tb = traceback.format_exc(); log('ERROR util\n' + tb); S['v19']['util'].append(dict(error=tb)); save()

# ------------------------------------------------------------------ streams
@numba.njit(parallel=True)
def eu_kernel(nodes, ptr, idx, PP, FF, d2, kmax, isins, T, bv, br, fill, Cb, OFF3, OFF4):
    """cliques containing an inserted instance, generated once from their smallest inserted member; emits
    (member, rank) pairs into per-thread buffers."""
    CAP = bv.shape[1]
    for t in prange(T):
        for s in range(t, len(nodes), T):
            o = nodes[s]; fo = FF[o]
            for i in range(ptr[s], ptr[s + 1]):
                v1 = idx[i]
                if isins[v1] and v1 < o:
                    continue
                r = rk2(fo, FF[v1], Cb); q = fill[t]
                if q + 2 <= CAP:
                    bv[t, q] = o; br[t, q] = r; bv[t, q + 1] = v1; br[t, q + 1] = r
                fill[t] = q + 2
                if kmax < 3:
                    continue
                for j in range(i + 1, ptr[s + 1]):
                    v2 = idx[j]
                    if FF[v2] == FF[v1] or (isins[v2] and v2 < o):
                        continue
                    dx = PP[v1, 0] - PP[v2, 0]; dy = PP[v1, 1] - PP[v2, 1]; dz = PP[v1, 2] - PP[v2, 2]
                    if dx * dx + dy * dy + dz * dz > d2:
                        continue
                    r = rk3(fo, FF[v1], FF[v2], Cb, OFF3); q = fill[t]
                    if q + 3 <= CAP:
                        bv[t, q] = o; br[t, q] = r; bv[t, q + 1] = v1; br[t, q + 1] = r; bv[t, q + 2] = v2; br[t, q + 2] = r
                    fill[t] = q + 3
                    if kmax < 4:
                        continue
                    for k in range(j + 1, ptr[s + 1]):
                        v3 = idx[k]
                        if FF[v3] == FF[v1] or FF[v3] == FF[v2] or (isins[v3] and v3 < o):
                            continue
                        ok = True
                        for vv in (v1, v2):
                            dx = PP[vv, 0] - PP[v3, 0]; dy = PP[vv, 1] - PP[v3, 1]; dz = PP[vv, 2] - PP[v3, 2]
                            if dx * dx + dy * dy + dz * dz > d2:
                                ok = False
                        if not ok:
                            continue
                        r = rk4(fo, FF[v1], FF[v2], FF[v3], Cb, OFF4); q = fill[t]
                        if q + 4 <= CAP:
                            bv[t, q] = o; br[t, q] = r; bv[t, q + 1] = v1; br[t, q + 1] = r
                            bv[t, q + 2] = v2; br[t, q + 2] = r; bv[t, q + 3] = v3; br[t, q + 3] = r
                        fill[t] = q + 4

@numba.njit
def sort_rows(gptr, gr):
    for o in range(len(gptr) - 1):
        a = gptr[o]; b = gptr[o + 1]
        if b - a > 1:
            gr[a:b] = np.sort(gr[a:b])

@numba.njit(parallel=True)
def eu_check(pv, pr, gptr, gr, isnew):
    for i in prange(len(pv)):
        v = pv[i]; r = pr[i]; lo = gptr[v]; hi_ = gptr[v + 1] - 1; found = False
        while lo <= hi_:
            mid = (lo + hi_) >> 1
            if gr[mid] == r:
                found = True; break
            if gr[mid] < r:
                lo = mid + 1
            else:
                hi_ = mid - 1
        isnew[i] = not found

@numba.njit
def eu_merge(gptr, gr, pv, pr, cntv):
    Nn = len(gptr) - 1
    optr = np.empty(Nn + 1, np.int64); optr[0] = 0
    for o in range(Nn):
        optr[o + 1] = optr[o] + (gptr[o + 1] - gptr[o]) + cntv[o]
    out = np.empty(optr[Nn], np.int32); j = 0
    for o in range(Nn):
        a = gptr[o]; b = gptr[o + 1]; w = optr[o]; e = j + cntv[o]
        while a < b or j < e:
            if j >= e or (a < b and gr[a] <= pr[j]):
                out[w] = gr[a]; a += 1
            else:
                out[w] = pr[j]; j += 1
            w += 1
    return optr, out

EU_CAP = [1 << 22]
def eucoloc_insert(ins, act, u, prev_u, prev_act, gptr, gr, ex, kmax):
    t0 = time.time()
    old = np.nonzero(prev_act & act & (u != prev_u))[0]
    if len(old):
        ex += csr_sum(old, u[old] - prev_u[old], gptr, gr)
    isins = np.zeros(N, bool); isins[ins] = True; T = nthreads(); keys = []
    for a in range(0, len(ins), 20000):
        nd = np.ascontiguousarray(ins[a:a + 20000], np.int64); ptr, flat = neigh(nd, act)
        while True:
            bv = np.empty((T, EU_CAP[0]), np.int64); br = np.empty((T, EU_CAP[0]), np.int64); fill = np.zeros(T, np.int64)
            eu_kernel(nd, ptr, flat, P, F, D * D, kmax, isins, T, bv, br, fill, Cb, OFF3, OFF4)
            if fill.max() <= EU_CAP[0]:
                break
            EU_CAP[0] = int(fill.max() * 1.3) + 1024
        kk = np.concatenate([bv[t, :fill[t]] * NR + br[t, :fill[t]] for t in range(T)])
        keys.append(np.unique(kk))
    kk = np.unique(np.concatenate(keys)) if keys else np.zeros(0, np.int64)
    pv = kk // NR; pr = (kk % NR).astype(np.int32)
    isnew = np.zeros(len(pv), np.bool_); eu_check(pv, pr, gptr, gr, isnew)
    pv = pv[isnew]; pr = pr[isnew]
    ex += np.bincount(pr, weights=u[pv], minlength=NR)
    gptr, gr = eu_merge(gptr, gr, pv, pr, np.bincount(pv, minlength=N).astype(np.int64))
    return gptr, gr, time.time() - t0, int(len(pv))

STREAM_AUTO = os.environ.get('V26_SAUTO', '1') == '1'

def run_stream(CM_, NMON_, IDLE_, LABEL, insert_only, eucoloc, kmax=3, remine_every=1, win=0):
    S['v19'].setdefault(LABEL, [])
    try:
        cum = np.cumsum(CM_, axis=1); first = np.where(CM_.sum(1) > 0, np.argmax(CM_ > 0, axis=1), NMON_)
        lastseen = np.full(N, -10**6); prev_act = None; prev_u = None
        gptr = np.zeros(N + 1, np.int64); gr = np.zeros(0, np.int32); ex_inc = np.zeros(NR)
        eu = None; pol = {m: None for m in DYN_M}; pola = {m: None for m in DYN_M}; prev_status = {}; mu = None; ages = {m: {} for m in DYN_M}
        mv_prev = {}; ref_prev = None
        for t in range(NMON_):
            lastseen = np.where(CM_[:, t] > 0, t, lastseen)
            act = (first <= t) if insert_only else ((first <= t) & (t - lastseen < IDLE_))
            if win > 0:     # utility over a sliding window of `win` months: it decreases when check-ins stop
                u = np.log1p(cum[:, t] - (cum[:, t - win] if t >= win else 0.0)).astype(np.float64)
            else:
                u = np.log1p(cum[:, t]).astype(np.float64)
            U = np.bincount(F[act], weights=u[act], minlength=K)
            row = dict(snapshot=t, active=int(act.sum())); nodes = np.nonzero(act)[0]
            ex_rm = None
            if t % remine_every == 0 or t in (0, NMON_ - 1):
                ex_rm, t_rm, _, c_rm, _ = run_enum(nodes, u[nodes], kmax, active=act)
                row.update(remine_time=t_rm, exact_state=int(c_rm.sum()))
            # Exact-Inc
            ti = time.time()
            if prev_act is None:
                ins = nodes; dele = np.zeros(0, np.int64); aff = nodes
            else:
                ins = np.nonzero(act & ~prev_act)[0]; dele = np.nonzero(prev_act & ~act)[0]
                aff = neighbours_of(np.r_[ins, dele]); aff = aff[(prev_act[aff] | act[aff])]
            old = aff[prev_act[aff]] if prev_act is not None else np.zeros(0, np.int64)
            if len(old):
                ex_inc -= csr_sum(old, prev_u[old], gptr, gr)
            new = aff[act[aff]]
            _, _, _, cnt, (nptr, nr) = run_enum(new, np.zeros(len(new)), kmax, active=act, emit=True)
            ex_inc += local_sum(u[new], nptr, nr)
            e5 = np.zeros(0, np.int64)
            if prev_act is not None:
                affm = np.zeros(N, bool); affm[aff] = True
                e5 = np.nonzero(act & prev_act & ~affm & (u != prev_u))[0]
                if len(e5):
                    ex_inc += csr_sum(e5, u[e5] - prev_u[e5], gptr, gr)
            pos_new = np.full(N, -1, np.int64); pos_new[new] = np.arange(len(new)); drop = np.zeros(N, bool); drop[aff] = True
            gptr, gr = rebuild_csr(gptr, gr, pos_new, nptr, nr, drop)
            row.update(inc_time=time.time() - ti, inserted=int(len(ins)), deleted=int(len(dele)), affected=int(len(aff)),
                       affected_share=float(len(aff) / max(act.sum(), 1)), e5=int(len(e5)), inc_state=int(len(gr)))
            if eucoloc:
                if eu is None:
                    eg = gptr.copy(); er = gr.copy(); sort_rows(eg, er); eu = dict(gptr=eg, gr=er, ex=ex_inc.copy())
                    row['eucoloc_time'] = row['inc_time']
                else:
                    eu['gptr'], eu['gr'], te, nnew = eucoloc_insert(ins, act, u, prev_u, prev_act, eu['gptr'], eu['gr'], eu['ex'], kmax)
                    row.update(eucoloc_time=te, eucoloc_newpairs=nnew)
            if prev_act is None:
                row['inc_lb_time'] = row['inc_time']
            else:
                _, a1, _, _, _ = run_enum(ins, u[ins], kmax, active=act)
                a2 = run_enum(dele, prev_u[dele], kmax, active=prev_act)[1] if len(dele) else 0.0
                row['inc_lb_time'] = a1 + a2
            ref = ex_inc
            if ex_rm is not None:
                row['inc_maxdiff'] = maxdiff(ex_rm, ex_inc)
                if eucoloc:
                    row['eucoloc_maxdiff'] = maxdiff(ex_rm, eu['ex'])
            sel = np.nonzero((RSIZE <= kmax) & (ref > 0))[0]; den = dens(U)[sel]; upi = ref[sel] / den
            if mu is None:
                mu = float(np.quantile(upi, .9))
            Hs = upi >= mu; recomp = np.zeros(N, bool); recomp[aff] = True
            chg = (ins, dele) if prev_act is not None else None
            row['pollster'] = {}
            chg_d = np.nonzero(act & prev_act & (u != prev_u))[0] if prev_act is not None else None   # delta input (not timed)
            for m in DYN_M:
                for mode in (('coord', 'auto', 'indep') if STREAM_AUTO else ('coord', 'indep')):
                    if mode == 'auto':
                        if pola[m] is None:
                            tb = time.time(); pola[m] = PMaint(act, u, m, kmax, auto=True); info = dict(mode='init', total_time=time.time() - tb)
                        else:
                            info = pola[m].auto_step(ins, dele, chg_d, act, u)
                            est_rb, _ = pollster_rebuild(act, u, m, kmax); info['maint_vs_rebuild_maxdiff'] = maxdiff(est_rb, pola[m].estimate())
                        est = pola[m].estimate()
                    elif mode == 'coord':
                        if pol[m] is None:
                            tb = time.time(); pol[m] = PMaint(act, u, m, kmax); info = dict(mode='init', total_time=time.time() - tb)
                        else:
                            info = pol[m].step(ins, dele, chg_d, act, u)
                        est = pol[m].estimate(); info['state_entries'] = pol[m].state_entries(); info['sampled'] = len(pol[m].Pst)
                    else:
                        Rv = np.random.default_rng(10007 * (t + 1) + m).random(N) + 1e-12
                        est, info = pollster_rebuild(act, u, m, kmax, Rv=Rv)
                    c_ = compare(ref, est, U, kmax, mus=(90,), m=m, T=NMON_)
                    info.update(mae=c_['mae'], max=c_['max'], f1=c_['thr']['90']['f1'], bern_viol=c_['bern_viol'],
                                bern_ratio_max=c_['bern_ratio_max'], hhat_recall=c_['thr']['90']['Hhat_recall'],
                                hhat_precision=c_['thr']['90']['Hhat_precision'], mu=c_['thr']['90']['mu'], eps_mu=c_['thr']['90'].get('eps_mu'))
                    info.update(bern2_viol=c_['bern2_viol'], bern2_ratio_max=c_['bern2_ratio_max'],
                                **{k_: c_['thr']['90'][k_] for k_ in ('Hhat2', 'Hhat2_precision', 'Hhat2_recall', 'H', 'Hplus2', 'Hplus2_precision', 'Hplus2_recall')})
                    if mode == 'coord' and prev_act is not None:
                        est_rb, irb = pollster_rebuild(act, u, m, kmax)
                        info['rebuild_time'] = irb['total_time']; info['maint_vs_rebuild_maxdiff'] = maxdiff(est_rb, est)
                    if mode == 'coord' and t == NMON_ - 1:
                        top = sel[np.argsort(-upi)[:10]]
                        info['top10'] = [dict(names=[NAMES[f_] for f_ in range(K) if (RMASK[r_] >> f_) & 1], upi=float(ref[r_] / dens(U)[r_]),
                                              est=float(est[r_] / dens(U)[r_])) for r_ in top]
                    Es = (est[sel] / den) >= mu; key = f'{mode}_{m}'
                    if key in prev_status:
                        psel, pH, pE = prev_status[key]
                        lut = np.full(NR, -1, np.int64); lut[psel] = np.arange(len(psel)); ix = lut[sel]; okm = ix >= 0
                        pHv = np.where(okm, pH[np.maximum(ix, 0)], False); pEv = np.where(okm, pE[np.maximum(ix, 0)], False)
                        same = okm & (Hs == pHv)
                        info['spurious_flip_rate'] = float((same & (Es != pEv)).sum() / max(okm.sum(), 1))
                        info['true_flip_rate'] = float((okm & ~same).sum() / max(okm.sum(), 1))
                        info['reported_flip_rate'] = float((okm & (Es != pEv)).sum() / max(okm.sum(), 1))
                        if t == NMON_ - 1:
                            nm = lambda r_: '+'.join(NAMES[f_] for f_ in range(K) if (RMASK[r_] >> f_) & 1)
                            tf = sel[okm & ~same][:8]; sf = sel[same & (Es != pEv)][:8]
                            info['flip_examples'] = dict(true=[nm(r_) for r_ in tf], spurious=[nm(r_) for r_ in sf],
                                                         n_true=int((okm & ~same).sum()), n_spurious=int((same & (Es != pEv)).sum()),
                                                         n_reported=int((okm & (Es != pEv)).sum()), n_both=int((okm & ~same & (Es != pEv)).sum()))
                    prev_status[key] = (sel, Hs, Es); info['mis_rate'] = float((Es != Hs).mean())
                    row['pollster'][key] = info
            if MOVE_ON:
                for m_ in MOVE_MS:
                    tb_ = time.time(); cur_ = sample_emit(act, u, m_, kmax); t_s = time.time() - tb_
                    if m_ in mv_prev:
                        tb_ = time.time(); mv_ = move_eval(mv_prev[m_], cur_, ref_prev, ref, m_, kmax, NMON_ - 1)
                        mv_.update(emit_time=t_s, eval_time=time.time() - tb_); row.setdefault('moves', {})[str(m_)] = mv_
                        log(LABEL + ' moves m=%d' % m_, json.dumps(mv_))
                    mv_prev[m_] = cur_
                ref_prev = ref.copy()
            S['v19'][LABEL].append(row); save()
            log(LABEL + ' t=%d' % t, json.dumps({k_: row[k_] for k_ in row if k_ != 'pollster'}),
                json.dumps({k_: dict(t=v['total_time'], rec=v.get('recomputed'), st=v.get('state_entries'), rb=v.get('rebuild_time'), md=v.get('maint_vs_rebuild_maxdiff'), mae=v['mae'], mx=v['max'],
                                     f1=v['f1'], sfr=v.get('spurious_flip_rate')) for k_, v in row['pollster'].items()}))
            prev_act = act; prev_u = u
    except Exception:
        import traceback; tb = traceback.format_exc(); log('ERROR stream\n' + tb); S['v19'][LABEL].append(dict(error=tb)); save()

def eu_warm():
    T = nthreads(); z = np.zeros(1, np.int64)
    eu_kernel(z, np.zeros(2, np.int64), np.zeros(0, np.int64), P, F, D * D, 3, np.zeros(N, bool), T,
              np.empty((T, 8), np.int64), np.empty((T, 8), np.int64), np.zeros(T, np.int64), Cb, OFF3, OFF4)
    isn = np.zeros(1, np.bool_); eu_check(z, np.zeros(1, np.int32), np.zeros(N + 1, np.int64), np.zeros(0, np.int32), isn)
    eu_merge(np.zeros(N + 1, np.int64), np.zeros(0, np.int32), np.zeros(0, np.int64), np.zeros(0, np.int32), np.zeros(N, np.int64))
    sort_rows(np.zeros(N + 1, np.int64), np.zeros(0, np.int32))
    probe_share(np.zeros(1, np.int64), np.zeros(1, np.int64))      # compile the probe kernel outside the timings
eu_warm()

if 'mixed' in PARTS:
    run_stream(CM.copy(), NMON, IDLE, 'tist_mixed', insert_only=False, eucoloc=False, kmax=int(os.environ.get('V21_STREAMK', '3')))
if 'ins' in PARTS:
    run_stream(CM.copy(), NMON, IDLE, 'tist_insert', insert_only=True, eucoloc=True, kmax=int(os.environ.get('V21_STREAMK', '3')))
if 'win' in PARTS:
    run_stream(CM.copy(), NMON, IDLE, 'tist_window', insert_only=False, eucoloc=False, win=int(os.environ.get('V20_WIN', '6')))
if 'cqsmall' in PARTS:
    # CQ in settings where it completes: smaller distance thresholds, one thread, compared with Star
    numba.set_num_threads(1); QW[0] = 1; D0 = D; S['v19']['cqsmall'] = []
    u_all = np.log1p(tot[IDS]).astype(np.float64)
    ucities = np.unique(CITY); _r7 = np.random.default_rng(7); _r7.random(len(IDS)); corder = _r7.permutation(ucities)   # same city order as `base`
    for dd in [float(x) for x in os.environ.get('V20_CQD', '25,50,75,100').split(',')]:
        for frac in [float(x) for x in os.environ.get('V20_CQFRAC', '0.125').split(',')]:
            try:
                D = dd
                cs = set(corder[:max(1, int(round(frac * len(ucities))))].tolist())
                act = np.isin(CITY, list(cs)); nodes = np.nonzero(act)[0]
                ex, t_ex, _, c_ex, _ = run_enum(nodes, u_all[nodes], 3, active=act)
                slots = 1 << int(min(29, max(22, np.ceil(np.log2(3 * max(c_ex.sum(), 1))))))
                gd_c, ic = run_cq(act, u_all, 3, float(os.environ.get('V20_CQCAP', '3600')), slots)
                if gd_c is None and ic.get('status') == 'hash full':
                    gd_c, ic = run_cq(act, u_all, 3, float(os.environ.get('V20_CQCAP', '3600')), slots * 2)
                row = dict(d=dd, frac=frac, n=int(len(nodes)), star_time=t_ex, star_state=int(c_ex.sum()),
                           patterns=int(((RSIZE <= 3) & (ex > 0)).sum()), cq=ic)
                if gd_c is not None:
                    row['cq_maxdiff'] = maxdiff(ex, dict_to_dense(gd_c))
                gd_j, ij = run_join(act, u_all, 3, 2400.0, int(6e8), row['star_state']); row['join'] = ij
                if gd_j is not None:
                    row['join_maxdiff'] = maxdiff(ex, dict_to_dense(gd_j))
                del gd_c, gd_j
                log('cqsmall', json.dumps(row)[:900]); S['v19']['cqsmall'].append(row); save()
            except Exception:
                import traceback; tb = traceback.format_exc(); log('ERROR cqsmall\n' + tb); S['v19']['cqsmall'].append(dict(d=dd, frac=frac, error=tb)); save()
    D = D0; numba.set_num_threads(TH_MAX); QW[0] = -1
if 'big' in PARTS:
    TILE = int(os.environ.get('V17_TILE', '8')); P0 = P.copy(); F0 = F.copy(); CM0 = CM.copy(); Ps = []
    for r_ in range(TILE):
        th = np.radians(2.0 * r_); c, s_ = np.cos(th), np.sin(th)
        Ps.append(np.c_[c * P0[:, 0] - s_ * P0[:, 1], s_ * P0[:, 0] + c * P0[:, 1], P0[:, 2]])
    set_data(np.vstack(Ps), np.tile(F0, TILE), K); del Ps
    S['v19']['big_tiles'] = TILE; log('big: tiles', TILE, 'instances', N)
    run_stream(np.tile(CM0, (TILE, 1)), NMON, IDLE, 'big_mixed', insert_only=False, eucoloc=False,
               remine_every=int(os.environ.get('V17_REMINE_EVERY', '1')))


# ===================================================================================================== v22: review instrumentation
# (1) instr: index build time, peak memory per method, degree of sampled instances, mu quantiles, Cor.2 sample sizes,
#     cost of verifying the uncertain band, participation index (u = 1) from per-feature samples
# (2) churn: TIST mixed stream with a fraction q of the instances following their real insert/delete dynamics; Exact-Count
#     (a counting maintainer), per-snapshot index rebuild, change detection with hysteresis, Cor.2 sample size on the stream
S.setdefault('v22', {})

def _rss():
    try:
        d = {}
        with open('/proc/self/status') as fh:
            for l in fh:
                if l.startswith('VmRSS') or l.startswith('VmHWM'):
                    d[l.split(':')[0]] = int(l.split()[1]) * 1024
        return d.get('VmRSS', 0), d.get('VmHWM', 0)
    except Exception:
        return 0, 0

def _reset_hwm():
    try:
        with open('/proc/self/clear_refs', 'w') as fh:
            fh.write('5')
        return True
    except Exception:
        return False

def mem_run(fn, *a, **kw):
    import gc as _gc
    _gc.collect(); ok = _reset_hwm(); base, _ = _rss()
    out = fn(*a, **kw)
    rss, hwm = _rss()
    return out, dict(peak_extra=int(hwm - base), rss_extra=int(rss - base), base=int(base), hwm_reset=bool(ok))

def m_for_rho(mu, kmax, rho=1.0 / 3.0, T=1):
    lo, hi = 1, 4
    while eps_b(mu, hi, kmax, K, DELTA_B, T) > rho * mu:
        hi *= 2
    while lo < hi:
        mid = (lo + hi) // 2
        if eps_b(mu, mid, kmax, K, DELTA_B, T) <= rho * mu:
            hi = mid
        else:
            lo = mid + 1
    return int(lo)

def degrees(active):
    deg = np.zeros(N, np.int64); nodes = np.nonzero(active)[0]
    for a in range(0, len(nodes), 200000):
        nd = nodes[a:a + 200000]; ptr, _ = neigh(nd, active); deg[nd] = np.diff(ptr)
    return deg

# ---------------------------------------------------------------------------------------------- Exact-Count (hash of counts)
@numba.njit
def ht_apply(hk, hc, keys, cnts, sign, uval, num, NRn):
    mask = len(hk) - 1; nnew = 0
    for i in range(len(keys)):
        k = keys[i]; c = cnts[i]
        h = ((k ^ (k >> 31)) * 0x2545F4914F6CDD1D) & mask
        while True:
            kk = hk[h]
            if kk == -1:
                hk[h] = k; hc[h] = 0; nnew += 1
                break
            if kk == k:
                break
            h = (h + 1) & mask
        old = hc[h]; new = old + sign * c; hc[h] = new
        x = k // NRn; r = k - x * NRn
        if old == 0 and new > 0:
            num[r] += uval[x]
        elif old > 0 and new == 0:
            num[r] -= uval[x]
    return nnew

@numba.njit(parallel=True)
def ht_util(hk, hc, du, NRn, T, acc):
    n = len(hk)
    for t in prange(T):
        for i in range(t, n, T):
            if hc[i] > 0:
                k = hk[i]; x = k // NRn; d = du[x]
                if d != 0.0:
                    acc[t, k - x * NRn] += d

@numba.njit
def ht_compact(hk, hc, nk, nc):
    mask = len(nk) - 1; live = 0
    for i in range(len(hk)):
        if hc[i] > 0:
            k = hk[i]; h = ((k ^ (k >> 31)) * 0x2545F4914F6CDD1D) & mask
            while nk[h] != -1:
                h = (h + 1) & mask
            nk[h] = k; nc[h] = hc[i]; live += 1
    return live

class ExactCount:
    """stores, for every (instance, pattern), the number of row instances of the pattern that contain the instance; an
    insertion or deletion enumerates the cliques that contain it once (from their smallest changed member), and a utility
    change is applied to the patterns of the instance through the table."""
    def __init__(self, bits=26):
        self.hk = np.full(1 << bits, -1, np.int64); self.hc = np.zeros(1 << bits, np.int32); self.used = 0; self.num = np.zeros(NR)
    def _grow(self):
        live = int((self.hc > 0).sum()); bits = max(24, int(np.ceil(np.log2(max(live, 1) * 2.5))))
        nk = np.full(1 << bits, -1, np.int64); nc = np.zeros(1 << bits, np.int32)
        self.used = ht_compact(self.hk, self.hc, nk, nc); self.hk, self.hc = nk, nc
    def apply(self, chg, active, sign, uval, kmax, chunk=4000):
        isins = np.zeros(N, bool); isins[chg] = True; T = nthreads(); npairs = 0
        for a in range(0, len(chg), chunk):
            nd = np.ascontiguousarray(chg[a:a + chunk], np.int64); ptr, flat = neigh(nd, active)
            while True:
                bv = np.empty((T, EU_CAP[0]), np.int64); br = np.empty((T, EU_CAP[0]), np.int64); fill = np.zeros(T, np.int64)
                eu_kernel(nd, ptr, flat, P, F, D * D, kmax, isins, T, bv, br, fill, Cb, OFF3, OFF4)
                if fill.max() <= EU_CAP[0]:
                    break
                EU_CAP[0] = int(fill.max() * 1.3) + 1024
            kk = np.concatenate([bv[t, :fill[t]] * NR + br[t, :fill[t]] for t in range(T)])
            del bv, br
            uk, cc = np.unique(kk, return_counts=True); npairs += len(uk)
            if (self.used + len(uk)) > 0.6 * len(self.hk):
                self._grow()
                while (self.used + len(uk)) > 0.6 * len(self.hk):
                    nk = np.full(2 * len(self.hk), -1, np.int64); nc = np.zeros(2 * len(self.hk), np.int32)
                    self.used = ht_compact(self.hk, self.hc, nk, nc); self.hk, self.hc = nk, nc
            self.used += ht_apply(self.hk, self.hc, uk, cc.astype(np.int64), sign, uval, self.num, NR)
        return npairs
    def utility(self, du):
        T = nthreads(); acc = np.zeros((T, NR)); ht_util(self.hk, self.hc, du, NR, T, acc); self.num += acc.sum(0)
    def live(self):
        return int((self.hc > 0).sum())

def _warm_count():
    ec = ExactCount(bits=10); z = np.arange(min(50, N), dtype=np.int64)
    ec.apply(z, np.ones(N, bool), 1, np.ones(N), 3); ec.utility(np.zeros(N)); ec.apply(z[:5], np.ones(N, bool), -1, np.ones(N), 3)
    nb_ = max(11, int(np.ceil(np.log2(max(ec.live(), 1) * 2.5)))); nk = np.full(1 << nb_, -1, np.int64); nc = np.zeros(1 << nb_, np.int32); ht_compact(ec.hk, ec.hc, nk, nc)
_warm_count()

# ---------------------------------------------------------------------------------------------- (1) instr
if 'probe' in PARTS:
    IR = S['v22'].setdefault('instr', {})
    try:
        u0 = np.log1p(tot[IDS]).astype(np.float64); act_all = np.ones(N, bool); nodes_all = np.arange(N, dtype=np.int64)
        U = np.bincount(F, weights=u0, minlength=K)
        tt = []
        for _ in range(3):
            tb = time.time(); _tr = cKDTree(P); tt.append(time.time() - tb); del _tr
        IR['index_build'] = dict(n=int(N), times=tt, median=float(np.median(tt)))
        P0_ = P.copy(); Ps = []
        for r_ in range(8):
            th = np.radians(2.0 * r_); c_, s_ = np.cos(th), np.sin(th)
            Ps.append(np.c_[c_ * P0_[:, 0] - s_ * P0_[:, 1], s_ * P0_[:, 0] + c_ * P0_[:, 1], P0_[:, 2]])
        Pt = np.vstack(Ps); del Ps; tb = time.time(); _tr = cKDTree(Pt); IR['index_build_x8'] = dict(n=int(len(Pt)), time=time.time() - tb); del _tr, Pt
        log('instr index', json.dumps(IR)); save()
        tb = time.time(); deg = degrees(act_all); IR['degree_time'] = time.time() - tb
        IR['deg_all'] = dict(mean=float(deg.mean()), p50=float(np.median(deg)), p99=float(np.quantile(deg, .99)), max=int(deg.max()))
        for kmax in (3, 4):
            KR = IR.setdefault(f'k{kmax}', {})
            (ex, t_ex, _, c_ex, _), mem = mem_run(run_enum, nodes_all, u0[nodes_all], kmax, active=act_all)
            KR['exact'] = dict(time=t_ex, state=int(c_ex.sum()), mem=mem); log('instr exact', kmax, t_ex, json.dumps(mem))
            acc0 = compare(ex, ex, U, kmax, mus=(90, 95, 99))
            mus = {q: acc0['thr'][q]['mu'] for q in acc0['thr']}; KR['mu'] = mus
            mlist = [300, 1000, 2000] + sorted({m_for_rho(mus['90'], kmax), m_for_rho(mus['95'], kmax), m_for_rho(mus['99'], kmax)})
            KR['m_cor2'] = dict(q90=m_for_rho(mus['90'], kmax), q95=m_for_rho(mus['95'], kmax), q99=m_for_rho(mus['99'], kmax),
                                q90_rho02=m_for_rho(mus['90'], kmax, 0.2))
            KR['pollster'] = {}
            for m in mlist:
                if kmax == 4 and m > 12000:
                    continue
                (est, var, _, info), mem = mem_run(pollster_est, act_all, u0, m, kmax); info.pop('taus', None)
                acc = compare(ex, est, U, kmax, var=var, m=m, mus=(90, 95, 99))
                tau = U / m; mk = act_all & (u0 > tau[F] * R)
                info.update(mem=mem, acc=acc, deg_sampled=float(deg[mk].mean()), deg_sampled_wt=float((deg[mk] * np.maximum(1.0, tau[F[mk]] / np.maximum(u0[mk], 1e-300))).sum() / max(np.maximum(1.0, tau[F[mk]] / np.maximum(u0[mk], 1e-300)).sum(), 1e-300)),
                            star_per_inst=info['star_time'] / max(info['sampled'], 1), exact_per_inst=t_ex / N, speedup=t_ex / max(info['total_time'], 1e-9))
                # verifying the uncertain band exactly: stars of all instances of the band's features, restricted to them
                if m in (1000, 2000) or m == KR['m_cor2']['q90']:
                    ver = {}
                    for q in ('90', '95', '99'):
                        mu = mus[q]; em = float(eps_b(mu, m, kmax, K, DELTA_B)); sel = np.nonzero((RSIZE <= kmax) & (ex > 0))[0]
                        e_ = est[sel] / dens(U)[sel]; band = sel[(e_ >= mu - em) & (e_ < mu + em)]
                        fm = int(np.bitwise_or.reduce(RMASK[band])) if len(band) else 0
                        allowed = np.array([(fm >> f) & 1 == 1 for f in range(K)])
                        nv = np.nonzero(allowed[F])[0]
                        if len(band) == 0:
                            ver[q] = dict(band=0)
                            continue
                        _, t_v, _, _, _ = run_enum(nv, u0[nv], kmax, active=act_all, allowed=allowed)
                        ver[q] = dict(band=int(len(band)), features=int(allowed.sum()), instances=int(len(nv)), time=t_v,
                                      hhat=int((e_ >= mu - em).sum()), H=int((ex[sel] / dens(U)[sel] >= mu).sum()))
                    info['verify'] = ver
                KR['pollster'][str(m)] = info
                log('instr', kmax, m, json.dumps(dict(t=info['total_time'], mem=mem['peak_extra'], degS=info['deg_sampled'], mae=acc['mae'],
                                                     thr={q: (v['mu'], v.get('eps_mu'), v.get('Hhat_precision'), v['f1']) for q, v in acc['thr'].items()},
                                                     ver=info.get('verify'))))
                save()
            # memory of the maintained states
            (gs, mem) = mem_run(build_store, nodes_all, kmax, act_all); KR['exact_inc_store'] = dict(entries=int(len(gs[1])), mem=mem); del gs
            pm, mem = mem_run(PMaint, act_all, u0, 300, kmax); KR['pmaint300'] = dict(entries=pm.state_entries(), mem=mem); del pm
            save()
            # participation index (u = 1): per-feature participation ratios from the stored sets; PI = min over features
            gptr, gr, _ = build_store(nodes_all, kmax, act_all); idx_by_f = [np.nonzero(F == f)[0] for f in range(K)]
            nf = np.array([len(x) for x in idx_by_f], np.float64); sz = RSIZE <= kmax
            pi_ex = np.full(NR, np.inf)
            for f in range(K):
                if nf[f] == 0:
                    continue
                rf = csr_sum(idx_by_f[f], np.ones(int(nf[f])), gptr, gr) / nf[f]; mk = sz & (((RMASK >> f) & 1) == 1)
                pi_ex[mk] = np.minimum(pi_ex[mk], rf[mk])
            selp = np.nonzero(sz & np.isfinite(pi_ex) & (pi_ex > 0))[0]; pe = pi_ex[selp]
            PIR = dict(patterns=int(len(selp)), q=[float(x) for x in np.quantile(pe, [.5, .9, .95, .99, 1.0])], res={})
            for m in (300, 1000, 2000):
                reps = []
                for rep in range(int(os.environ.get('V22_PIREPS', '5'))):
                    Rv = np.random.default_rng(900 + 37 * rep + m).random(N); pi_es = np.full(NR, np.inf)
                    nsamp = 0; fv = 0; fr = 0.0
                    for f in range(K):
                        if nf[f] == 0:
                            continue
                        p_ = min(1.0, m / nf[f]); s_ = idx_by_f[f][Rv[idx_by_f[f]] < p_]; nsamp += len(s_)
                        rf = csr_sum(s_, np.full(len(s_), 1.0 / p_), gptr, gr) / nf[f]; mk = sz & (((RMASK >> f) & 1) == 1)
                        pi_es[mk] = np.minimum(pi_es[mk], rf[mk])
                    e_ = pi_es[selp]; e_ = np.where(np.isfinite(e_), e_, 0.0); err = np.abs(e_ - pe); eb = eps_b(pe, m, kmax, K, DELTA_B)
                    d_ = dict(mae=float(err.mean()), max=float(err.max()), viol=int((err > eb).sum()), ratio=float((err / eb).max()), sampled=int(nsamp))
                    for q in (90, 95, 99):
                        mu = float(np.quantile(pe, q / 100)); H = pe >= mu; He = e_ >= mu; em = float(eps_b(mu, m, kmax, K, DELTA_B)); Hh = e_ >= mu - em
                        tp = int((H & He).sum()); pr = tp / max(int(He.sum()), 1); rc = tp / max(int(H.sum()), 1)
                        d_[str(q)] = dict(mu=mu, f1=2 * pr * rc / max(pr + rc, 1e-12), eps_mu=em, Hhat=int(Hh.sum()), H=int(H.sum()),
                                          hhat_recall=float((H & Hh).sum() / max(int(H.sum()), 1)), hhat_precision=float((H & Hh).sum() / max(int(Hh.sum()), 1)),
                                          in_band=float((Hh & (pe >= mu - 2 * em)).sum() / max(int(Hh.sum()), 1)))
                    reps.append(d_)
                PIR['res'][str(m)] = reps; log('instr PI', kmax, m, json.dumps(reps[0]))
            KR['pi'] = PIR; del gptr, gr; save()
    except Exception:
        import traceback; tb_ = traceback.format_exc(); log('ERROR instr\n' + tb_); IR['error'] = tb_; save()

# ---------------------------------------------------------------------------------------------- (2) churn sweep
def run_stream_v22(CM_, NMON_, IDLE_, LABEL, churn_q=1.0, ms=(300, 1000), kmax=3, exact_count=True, hfrac=(0.0, 0.1, 0.25)):
    out = S['v22'].setdefault(LABEL, [])
    try:
        cum = np.cumsum(CM_, axis=1); first = np.where(CM_.sum(1) > 0, np.argmax(CM_ > 0, axis=1), NMON_)
        dyn = np.random.default_rng(99).random(N) < churn_q; act0 = None
        lastseen = np.full(N, -10**6); prev_act = None; prev_u = None
        gptr = np.zeros(N + 1, np.int64); gr = np.zeros(0, np.int32); ex_inc = np.zeros(NR)
        pol = {m: None for m in ms}; prev_status = {}; mu = None; ec = ExactCount(bits=27) if exact_count else None; est_rb_m = {}
        mv_prev = {}; ref_prev = None
        for t in range(NMON_):
            lastseen = np.where(CM_[:, t] > 0, t, lastseen)
            act_r = (first <= t) & (t - lastseen < IDLE_)
            if act0 is None:
                act0 = act_r.copy()
            act = np.where(dyn, act_r, act0)
            u = np.log1p(cum[:, t]).astype(np.float64)
            U = np.bincount(F[act], weights=u[act], minlength=K)
            row = dict(snapshot=t, active=int(act.sum())); nodes = np.nonzero(act)[0]
            tb = time.time(); _tr = cKDTree(P[nodes]); row['index_rebuild_time'] = time.time() - tb; del _tr
            if GRID_ON:
                tb = time.time(); g_cur = grid_update(g_prev, prev_act, act) if (GRID_UPD and prev_act is not None) else grid_build(act); row['grid_time'] = time.time() - tb
                grid_register((act, g_cur), (prev_act, g_prev if prev_act is not None else None))
            ex_rm, t_rm, _, c_rm, _ = run_enum(nodes, u[nodes], kmax, active=act)
            row.update(remine_time=t_rm, exact_state=int(c_rm.sum()))
            ti = time.time()
            if prev_act is None:
                ins = nodes; dele = np.zeros(0, np.int64); aff = nodes
            else:
                ins = np.nonzero(act & ~prev_act)[0]; dele = np.nonzero(prev_act & ~act)[0]
                aff = neighbours_of(np.r_[ins, dele]); aff = aff[(prev_act[aff] | act[aff])]
            old = aff[prev_act[aff]] if prev_act is not None else np.zeros(0, np.int64)
            if len(old):
                ex_inc -= csr_sum(old, prev_u[old], gptr, gr)
            new = aff[act[aff]]
            _, _, _, cnt, (nptr, nr) = run_enum(new, np.zeros(len(new)), kmax, active=act, emit=True)
            ex_inc += local_sum(u[new], nptr, nr)
            e5 = np.zeros(0, np.int64)
            if prev_act is not None:
                affm = np.zeros(N, bool); affm[aff] = True
                e5 = np.nonzero(act & prev_act & ~affm & (u != prev_u))[0]
                if len(e5):
                    ex_inc += csr_sum(e5, u[e5] - prev_u[e5], gptr, gr)
            pos_new = np.full(N, -1, np.int64); pos_new[new] = np.arange(len(new)); drop = np.zeros(N, bool); drop[aff] = True
            gptr, gr = rebuild_csr(gptr, gr, pos_new, nptr, nr, drop)
            row.update(inc_time=time.time() - ti, inserted=int(len(ins)), deleted=int(len(dele)), affected=int(len(aff)),
                       affected_share=float(len(aff) / max(act.sum(), 1)), e5=int(len(e5)), inc_state=int(len(gr)),
                       inc_maxdiff=maxdiff(ex_rm, ex_inc))
            if prev_act is None:
                row['inc_lb_time'] = row['inc_time']
            else:
                _, a1, _, _, _ = run_enum(ins, u[ins], kmax, active=act) if len(ins) else (None, 0.0, None, None, None)
                a2 = run_enum(dele, prev_u[dele], kmax, active=prev_act)[1] if len(dele) else 0.0
                row['inc_lb_time'] = a1 + a2
            if ec is not None:
                tc = time.time()
                if prev_act is None:
                    npi = ec.apply(nodes, act, 1, u, kmax); npd = 0
                else:
                    npd = ec.apply(dele, prev_act, -1, prev_u, kmax) if len(dele) else 0
                    ec.utility(np.where(act & prev_act, u - prev_u, 0.0))
                    npi = ec.apply(ins, act, 1, u, kmax) if len(ins) else 0
                row.update(count_time=time.time() - tc, count_maxdiff=maxdiff(ex_rm, ec.num), count_pairs_touched=int(npi + npd),
                           count_live=ec.live(), count_slots=int(len(ec.hk)))
            ref = ex_rm
            sel = np.nonzero((RSIZE <= kmax) & (ref > 0))[0]; den = dens(U)[sel]; upi = ref[sel] / den
            if mu is None:
                mu = float(np.quantile(upi, .9))
            Hs = upi >= mu
            chg_d = np.nonzero(act & prev_act & (u != prev_u))[0] if prev_act is not None else None
            row['pollster'] = {}
            for m in ms:
                em = float(eps_b(mu, m, kmax, K, DELTA_B, NMON_))
                for mode in ('coord', 'auto', 'indep'):
                    if mode == 'auto':
                        if (m, 'auto') not in pol:
                            tb = time.time(); pol[(m, 'auto')] = PMaint(act, u, m, kmax, auto=True); info = dict(mode='init', total_time=time.time() - tb)
                        else:
                            info = pol[(m, 'auto')].auto_step(ins, dele, chg_d, act, u)
                        est = pol[(m, 'auto')].estimate()
                        if prev_act is not None:
                            info['maint_vs_rebuild_maxdiff'] = maxdiff(est_rb_m[m], est)
                    elif mode == 'coord':
                        if pol[m] is None:
                            tb = time.time(); pol[m] = PMaint(act, u, m, kmax); info = dict(mode='init', total_time=time.time() - tb)
                        else:
                            info = pol[m].step(ins, dele, chg_d, act, u)
                        est = pol[m].estimate(); info['state_entries'] = pol[m].state_entries(); info['sampled'] = len(pol[m].Pst)
                        if prev_act is not None:
                            est_rb, irb = pollster_rebuild(act, u, m, kmax); est_rb_m[m] = est_rb
                            info['rebuild_time'] = irb['total_time']; info['maint_vs_rebuild_maxdiff'] = maxdiff(est_rb, est)
                    else:
                        Rv = np.random.default_rng(10007 * (t + 1) + m).random(N) + 1e-12
                        est, info = pollster_rebuild(act, u, m, kmax, Rv=Rv)
                    c_ = compare(ref, est, U, kmax, mus=(90,), m=m, T=NMON_)
                    info.update(mae=c_['mae'], max=c_['max'], f1=c_['thr']['90']['f1'], bern_viol=c_['bern_viol'], bern_ratio_max=c_['bern_ratio_max'],
                                hhat_precision=c_['thr']['90']['Hhat_precision'], hhat_recall=c_['thr']['90']['Hhat_recall'], eps_mu_fixed=em, mu_fixed=mu)
                    e_ = est[sel] / den; Es = e_ >= mu; key = f'{mode}_{m}'
                    hst = {}; hlt = {}
                    em2a = eps2(mu, lam_max(U)[sel], m, kmax, K, DELTA_B, NMON_)
                    HOPTS = list(hfrac) + ['eps2', 'cert', 'cert2']
                    if key in prev_status:
                        psel, pH, pE, pHst, pHlt = prev_status[key]
                        lut = np.full(NR, -1, np.int64); lut[psel] = np.arange(len(psel)); ix = lut[sel]; okm = ix >= 0
                        pHv = np.where(okm, pH[np.maximum(ix, 0)], False); pEv = np.where(okm, pE[np.maximum(ix, 0)], False)
                        same = okm & (Hs == pHv); truec = okm & ~same
                        info['spurious_flip_rate'] = float((same & (Es != pEv)).sum() / max(okm.sum(), 1))
                        info['true_flip_rate'] = float(truec.sum() / max(okm.sum(), 1))
                        cd = {}
                        for hf in HOPTS:
                            h = em / 2 if hf == 'eps2' else (em if hf == 'cert' else (em2a if hf == 'cert2' else hf * mu))
                            plt_ = np.where(okm, pHlt[str(hf)][np.maximum(ix, 0)], Hs)
                            certn = (e_ >= mu + h) | (e_ < mu - h)
                            if hf in ('cert', 'cert2'):
                                # tri-state: +1 certified in, -1 certified out, 0 never certified; report only flips of a certified status
                                prevs = np.where(okm, pHst[str(hf)][np.maximum(ix, 0)], 0).astype(np.int8)
                                news = np.where(e_ >= mu + h, 1, np.where(e_ < mu - h, -1, 0)).astype(np.int8)
                                cur_s = np.where(news != 0, news, prevs).astype(np.int8); hst[str(hf)] = cur_s
                                rep_ = okm & (news != 0) & (prevs != 0) & (news != prevs)
                                cur = cur_s > 0; prevh = prevs > 0
                            else:
                                prevh = np.where(okm, pHst[str(hf)][np.maximum(ix, 0)], Es)
                                cur = np.where(e_ >= mu + h, True, np.where(e_ < mu - h, False, prevh)); hst[str(hf)] = cur
                                rep_ = okm & (cur != prevh)
                            hlt[str(hf)] = np.where(certn, Hs, plt_)
                            cd[str(hf)] = dict(n_true=int(truec.sum()), n_rep=int(rep_.sum()), n_both=int((rep_ & truec).sum()),
                                               n_correct_lastcert=int((rep_ & (Hs != plt_)).sum()),
                                               n_wrong_status=int((cur != Hs).sum()), n_certified=int(certn.sum()), n_patterns=int(len(sel)))
                        info['change_detection'] = cd
                    else:
                        for hf in HOPTS:
                            h = em / 2 if hf == 'eps2' else (em if hf == 'cert' else (em2a if hf == 'cert2' else hf * mu))
                            if hf in ('cert', 'cert2'):
                                hst[str(hf)] = np.where(e_ >= mu + h, 1, np.where(e_ < mu - h, -1, 0)).astype(np.int8)
                            else:
                                hst[str(hf)] = np.where(e_ >= mu + h, True, np.where(e_ < mu - h, False, Es))
                            hlt[str(hf)] = Hs.copy()
                    prev_status[key] = (sel, Hs, Es, hst, hlt)
                    row['pollster'][key] = info
            if MOVE_ON:
                for m_ in MOVE_MS:
                    tb_ = time.time(); cur_ = sample_emit(act, u, m_, kmax); t_s = time.time() - tb_
                    if m_ in mv_prev:
                        tb_ = time.time(); mv_ = move_eval(mv_prev[m_], cur_, ref_prev, ref, m_, kmax, NMON_ - 1)
                        mv_.update(emit_time=t_s, eval_time=time.time() - tb_); row.setdefault('moves', {})[str(m_)] = mv_
                        log(LABEL + ' moves m=%d' % m_, json.dumps(mv_))
                    mv_prev[m_] = cur_
                ref_prev = ref.copy()
            out.append(row); save()
            log(LABEL + ' t=%d' % t, json.dumps({k_: row[k_] for k_ in row if k_ != 'pollster'}),
                json.dumps({k_: dict(t=v['total_time'], rb=v.get('rebuild_time'), md=v.get('maint_vs_rebuild_maxdiff'), mae=v['mae'],
                                     f1=v['f1'], sfr=v.get('spurious_flip_rate')) for k_, v in row['pollster'].items()}))
            prev_act = act; prev_u = u
            if GRID_ON:
                g_prev = g_cur
        del ec
    except Exception:
        import traceback; tb_ = traceback.format_exc(); log('ERROR stream v22\n' + tb_); out.append(dict(error=tb_)); save()

if 'churn' in PARTS:
    for q_ in [float(x) for x in os.environ.get('V22_QS', '0.01,0.05,0.2,1').split(',')]:
        msq = [int(x) for x in os.environ.get('V22_MS', '300,1000').split(',')]
        if q_ >= 1.0:
            msq = msq + [int(x) for x in os.environ.get('V22_MBIG', '7400').split(',') if x]
        run_stream_v22(CM.copy(), NMON, IDLE, 'churn_%g' % q_, churn_q=q_, ms=msq)


# ===================================================================================================== v23: progressive certified refinement
# Persistent seeds make T_f(m) = {o : u(o)/r_o > U(f)/m} nested in m. Pollster-Prog starts every feature at m0, computes a
# confidence interval for every pattern from per-feature Bernstein intervals (valid for all levels of the schedule m0 2^j at
# once, union bound over the levels), and doubles m only for features that occur in an undecided pattern; only entering
# instances get a star. A feature whose threshold falls below its smallest utility is exact.
S.setdefault('v23', {})

@numba.njit
def pr_acc(sign, heavy, wu, fi, ptr, ranks, A2, B2):
    for i in range(len(heavy)):
        f = fi[i]
        for j in range(ptr[i], ptr[i + 1]):
            r = ranks[j]
            if heavy[i]:
                A2[f, r] += sign * wu[i]
            else:
                B2[f, r] += sign

@numba.njit
def pr_convert(mask, wu, f, ptr, ranks, A2, B2):
    for i in range(len(mask)):
        if mask[i]:
            for j in range(ptr[i], ptr[i + 1]):
                r = ranks[j]
                B2[f, r] -= 1.0
                A2[f, r] += wu[i]

def pr_interval(ph, m, Lp):
    a = np.sqrt(2.0 * Lp / m); b = 2.0 * Lp / (3.0 * m)
    hi = ((a + np.sqrt(a * a + 4.0 * (ph + b))) / 2.0) ** 2
    q = ph - b
    lo = np.where(q > 0, ((-a + np.sqrt(a * a + 4.0 * np.maximum(q, 0.0))) / 2.0) ** 2, 0.0)
    return lo, np.minimum(hi, 1.0)

RULE = os.environ.get('V23_RULE', 'min')

def prog_refine(act, u, kmax, mu, m0, jmax=None, delta=DELTA_B, exf=None, budget=None, tag=''):
    t_start = time.time()
    U = np.bincount(F[act], weights=u[act], minlength=K); nodes_all = np.nonzero(act)[0]
    ids_f, key_f, umin_f = [], [], np.zeros(K)
    for f in range(K):
        idx = nodes_all[F[nodes_all] == f]
        if len(idx) == 0:
            ids_f.append(idx); key_f.append(np.zeros(0)); umin_f[f] = np.inf; continue
        kk = u[idx] / R[idx]; o = np.argsort(-kk, kind='stable'); ids_f.append(idx[o]); key_f.append(kk[o]); umin_f[f] = u[idx].min()
    jex = np.array([int(np.ceil(np.log2(max(U[f] / umin_f[f] / m0, 1.0)))) if np.isfinite(umin_f[f]) else 0 for f in range(K)])
    J = int(jex.max()) if jmax is None else int(jmax)
    M = sum(math.comb(K - 1, j) for j in range(1, kmax)); Lp = math.log(2.0 * K * M * (J + 1) / delta)
    sel = np.nonzero((RSIZE >= 2) & (RSIZE <= kmax))[0]; den = dens(U)[sel]
    pats = []; pidx = []
    pos = np.full(NR, -1, np.int64); pos[sel] = np.arange(len(sel))
    for f in range(K):
        pf = sel[((RMASK[sel] >> f) & 1) == 1]; pats.append(pf); pidx.append(pos[pf])
    lev = np.zeros(K, np.int64); cnt = np.zeros(K, np.int64); exact = np.zeros(K, bool)
    A2 = np.zeros((K, NR)); B2 = np.zeros((K, NR)); batches = [[] for _ in range(K)]; stars = 0
    rounds = []; need = np.ones(K, bool); first = True
    while True:
        tr = time.time()
        if not first:
            lev = np.where(need, np.minimum(lev + 1, J), lev)
        m_f = m0 * (2.0 ** lev); tau = np.where(U > 0, U / m_f, 0.0)
        ent_ids = []; ent_f = []
        for f in np.nonzero(need)[0]:
            if len(ids_f[f]) == 0:
                continue
            for b_ in batches[f]:
                mk = (~b_['heavy']) & (b_['u'] >= tau[f])
                if mk.any():
                    pr_convert(mk, b_['u'], f, b_['ptr'], b_['ranks'], A2, B2); b_['heavy'] = b_['heavy'] | mk
            c_new = int(np.searchsorted(-key_f[f], -tau[f], side='left'))
            if c_new > cnt[f]:
                ent_ids.append(ids_f[f][cnt[f]:c_new]); ent_f.append(np.full(c_new - cnt[f], f))
                cnt[f] = c_new
            exact[f] = tau[f] <= umin_f[f]
        if ent_ids:
            ei = np.concatenate(ent_ids); _, _, _, cn, (lp, lr) = run_enum(ei, np.zeros(len(ei)), kmax, active=act, emit=True)
            stars += len(ei); fi = F[ei]; heavy = u[ei] >= tau[fi]
            pr_acc(1.0, heavy, u[ei], fi, lp, lr, A2, B2)
            # split into per-feature batches (for later light->heavy conversion)
            o = np.argsort(fi, kind='stable'); bnd = np.searchsorted(fi[o], np.arange(K + 1))
            lens = np.diff(lp)
            for f in np.unique(fi):
                ii = o[bnd[f]:bnd[f + 1]]
                li = lens[ii]; pt = np.r_[0, np.cumsum(li)].astype(np.int64)
                rk = lr[np.repeat(lp[ii] - pt[:-1], li) + np.arange(pt[-1])] if pt[-1] else np.zeros(0, np.int32)
                batches[f].append(dict(u=u[ei[ii]], heavy=heavy[ii].copy(), ptr=pt, ranks=rk))
        # intervals
        LO = np.zeros(len(sel)); HI = np.zeros(len(sel))
        for f in range(K):
            if len(pats[f]) == 0 or U[f] == 0:
                continue
            ph = (A2[f, pats[f]] + tau[f] * B2[f, pats[f]]) / U[f]
            if exact[f]:
                lo, hi = ph, ph
            else:
                lo, hi = pr_interval(ph, m_f[f], Lp)
            lam = U[f] / den[pidx[f]]
            LO[pidx[f]] += lam * lo; HI[pidx[f]] += lam * hi
        cin = LO >= mu; cout = HI < mu; und = ~(cin | cout)
        if RULE == 'all':
            fm = int(np.bitwise_or.reduce(RMASK[sel[und]])) if und.any() else 0
            infeat = np.array([(fm >> f) & 1 == 1 for f in range(K)])
            need = infeat & ~exact & (lev < J)
        else:
            levx = np.r_[np.where(exact | (lev >= J), 1 << 40, lev), 1 << 40]
            rf = RF[sel[und]]; ml = levx[rf].min(1); need = np.zeros(K, bool)
            for jj in range(4):
                g = rf[:, jj]; mk_ = (levx[g] == ml) & (ml < (1 << 40)); need[g[mk_]] = True
        el = time.time() - t_start
        rd = dict(round=len(rounds), t=el, t_round=time.time() - tr, stars=int(stars), sampled=int(cnt.sum()), undecided=int(und.sum()),
                  cert_in=int(cin.sum()), cert_out=int(cout.sum()), n_refine=int(need.sum()), n_exact=int(exact.sum()), max_lev=int(lev.max()))
        rounds.append(rd); log('prog', tag, json.dumps(rd))
        first = False
        if not need.any() or (budget is not None and el > budget):
            break
    res = dict(m0=m0, J=J, Lp=Lp, rounds=rounds, total_time=time.time() - t_start, stars=int(stars), final_sampled=int(cnt.sum()),
               lev=lev.tolist(), cnt=cnt.tolist(), exact=exact.tolist())
    # cost of independent (fresh-seed) progressive sampling with the same level sequence: a new sample at every level
    indep = 0
    for f in range(K):
        for j in range(int(lev[f]) + 1):
            if len(ids_f[f]):
                indep += int(np.searchsorted(-key_f[f], -(U[f] / (m0 * 2.0 ** j)), side='left'))
    res['indep_stars'] = indep
    res['final'] = dict(cin=cin, cout=cout, und=und, sel=sel)
    if os.environ.get('V23_DEBUG') == '1':
        res['est_total'] = (A2 + tau[:, None] * B2).sum(0)
    # oracle (Theorem): j*(c) = smallest uniform level at which the deterministic intervals O_f = {x : |x - pi_f| <= eps(x) + eps(pi_f)}
    # around the TRUE per-feature probabilities classify c; oracle level of f = max over patterns containing f of j*(c)
    if exf is not None:
        pit = [None] * K
        for f in range(K):
            pit[f] = exf[f][pats[f]] / U[f] if U[f] > 0 else np.zeros(len(pats[f]))
        upi_t = np.zeros(len(sel))
        for f in range(K):
            upi_t[pidx[f]] += (U[f] / den[pidx[f]]) * pit[f]
        jstar = np.full(len(sel), J, np.int64); done_ = np.zeros(len(sel), bool)
        for jl in range(J + 1):
            mj = m0 * 2.0 ** jl; LOt = np.zeros(len(sel)); HIt = np.zeros(len(sel))
            for f in range(K):
                if len(pats[f]) == 0 or U[f] == 0:
                    continue
                if U[f] / mj <= umin_f[f]:
                    lo_, hi_ = pit[f], pit[f]
                else:
                    ep = np.sqrt(2.0 * pit[f] * Lp / mj) + 2.0 * Lp / (3.0 * mj)
                    hi_ = pr_interval(pit[f] + ep, mj, Lp)[1]; lo_ = pr_interval(np.maximum(pit[f] - ep, 0.0), mj, Lp)[0]
                lam = U[f] / den[pidx[f]]
                LOt[pidx[f]] += lam * lo_; HIt[pidx[f]] += lam * hi_
            dec = (LOt >= mu) | (HIt < mu)
            newd = dec & ~done_; jstar[newd] = jl; done_ |= dec
            if done_.all():
                break
        olev = np.zeros(K, np.int64)
        for f in range(K):
            if len(pidx[f]):
                olev[f] = int(jstar[pidx[f]].max())
        ostars = 0
        for f in range(K):
            if len(ids_f[f]):
                ostars += int(np.searchsorted(-key_f[f], -(U[f] / (m0 * 2.0 ** olev[f])), side='left'))
        res['oracle'] = dict(lev=olev.tolist(), stars=int(ostars), undecidable=int((~done_).sum()), ratio=float(stars / max(ostars, 1)),
                             lev_viol=int((lev > olev).sum()))
        Ht = upi_t >= mu
        res['check'] = dict(H=int(Ht.sum()), in_viol=int((cin & ~Ht).sum()), out_viol=int((cout & Ht).sum()), und_final=int(und.sum()))
    return res

if 'progref' in PARTS:
    PR = S['v23'].setdefault('progref', {})
    try:
        u0 = np.log1p(tot[IDS]).astype(np.float64); act_all = np.ones(N, bool); nodes_all = np.arange(N, dtype=np.int64)
        U0 = np.bincount(F, weights=u0, minlength=K)
        for kmax in [int(x) for x in os.environ.get('V23_KS', '3,4').split(',')]:
            KR = PR.setdefault(f'k{kmax}', {})
            ex, t_ex, _, c_ex, _ = run_enum(nodes_all, u0, kmax, active=act_all); KR['exact_time'] = t_ex
            exf = []
            for f in range(K):
                idx = np.nonzero(F == f)[0]; exf.append(run_enum(idx, u0[idx], kmax, active=act_all)[0])
            selx = np.nonzero((RSIZE <= kmax) & (ex > 0))[0]; upi = ex[selx] / dens(U0)[selx]
            for q in [int(x) for x in os.environ.get('V23_QS', '90,95,99').split(',')]:
                mu = float(np.quantile(upi, q / 100))
                mc = m_for_rho(mu, kmax)
                (est, var, _, info) = pollster_est(act_all, u0, mc, kmax)
                cmp_ = compare(ex, est, U0, kmax, m=mc, mus=(q,))
                fixed = dict(m=mc, time=info['total_time'], sampled=info.get('sampled'), hhat_precision=cmp_['thr'][str(q)]['Hhat_precision'],
                             hhat=cmp_['thr'][str(q)]['Hhat'], H=cmp_['thr'][str(q)]['H'])
                for m0 in [int(x) for x in os.environ.get('V23_M0', '500').split(',')]:
                    r_ = prog_refine(act_all, u0, kmax, mu, m0, exf=exf, budget=float(os.environ.get('V23_BUDGET', '5400')), tag=f'k{kmax} q{q} m0={m0}')
                    fin = r_.pop('final'); r_.pop('est_total', None)
                    # undecided below 10% / 1% / 0 of |H|
                    Hn = r_['check']['H'] if 'check' in r_ else 1
                    thr_t = {}
                    for frac in (0.1, 0.01, 0.0):
                        ok_ = [rd['t'] for rd in r_['rounds'] if rd['undecided'] <= frac * Hn]
                        thr_t[str(frac)] = ok_[0] if ok_ else None
                    r_.update(mu=mu, exact_time=t_ex, fixed_cor2=fixed, t_undecided=thr_t)
                    KR[f'q{q}_m0{m0}'] = r_; save()
                    log('progref', kmax, q, m0, json.dumps({k_: v for k_, v in r_.items() if k_ not in ('rounds', 'lev', 'cnt', 'exact')}))
            del exf
    except Exception:
        import traceback; tb_ = traceback.format_exc(); log('ERROR progref\n' + tb_); PR['error'] = tb_; save()

if 'prtest' in PARTS:
    u0 = np.log1p(tot[IDS]).astype(np.float64); act_all = np.ones(N, bool)
    for kmax in (3, 4):
        for m0 in (20, 50):
            r_ = prog_refine(act_all, u0, kmax, 0.5, m0, jmax=0)
            est = pollster_est(act_all, u0, m0, kmax)[0]
            log('prtest', kmax, m0, 'maxdiff', maxdiff(est, r_['est_total']), 'stars', r_['stars'])
            for jm in (1, 2):
                r2 = prog_refine(act_all, u0, kmax, 1e-9, m0, jmax=jm)
                e2 = pollster_est(act_all, u0, m0 * 2 ** jm, kmax)[0]
                log('prtest lev', jm, 'maxdiff', maxdiff(e2, r2['est_total']), r2['lev'][:3])

if 'pioverlap' in PARTS:
    PO = S['v23'].setdefault('pioverlap', {})
    try:
        u0 = np.log1p(tot[IDS]).astype(np.float64); U0 = np.bincount(F, weights=u0, minlength=K); nf = np.bincount(F, minlength=K).astype(np.float64)
        for m in (300, 1000, 2000):
            pu = np.minimum(1.0, m * u0 / U0[F]); pc_ = np.minimum(1.0, m / nf[F])
            su = R < pu; sc = R < pc_
            PO[str(m)] = dict(util=int(su.sum()), count=int(sc.sum()), inter=int((su & sc).sum()), union=int((su | sc).sum()),
                              jaccard=float((su & sc).sum() / max((su | sc).sum(), 1)), expected_inter=float(np.minimum(pu, pc_).sum()))
            log('pioverlap', m, json.dumps(PO[str(m)]))
        save()
    except Exception:
        import traceback; tb_ = traceback.format_exc(); log('ERROR pioverlap\n' + tb_); PO['error'] = tb_; save()


# ===================================================================================================== v21: FSQ-OS real monthly deltas
def splitmix64(x):
    x = (x + np.uint64(0x9E3779B97F4A7C15)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    z = x.copy()
    z = ((z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    z = ((z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    return z ^ (z >> np.uint64(31))

def fsq_load_release(api, tok, rel, top=None, nfiles=100):
    """open places with valid coordinates; first category; returns hid (uint64), lat, lon, category strings or top-K codes."""
    import pyarrow.parquet as pq, pyarrow.compute as pc
    from huggingface_hub import hf_hub_download
    from collections import Counter
    files = sorted(f for f in api.list_repo_files('foursquare/fsq-os-places', repo_type='dataset')
                   if f.startswith(f'release/dt={rel}/places/parquet/') and f.endswith('.parquet'))[:nfiles]
    H, LA, LO, FC = [], [], [], []; gcount = Counter(); dicts = []
    for i, f in enumerate(files):
        p = hf_hub_download('foursquare/fsq-os-places', f, repo_type='dataset', token=tok, local_dir='/tmp/fsq')
        t = pq.read_table(p, columns=['fsq_place_id', 'latitude', 'longitude', 'date_closed', 'fsq_category_ids']); os.remove(p)
        ln = pc.fill_null(pc.list_value_length(t['fsq_category_ids']), 0)
        ok = pc.and_(pc.and_(pc.is_null(t['date_closed']), pc.greater(ln, 0)),
                     pc.and_(pc.is_valid(t['latitude']), pc.is_valid(t['longitude'])))
        t = t.filter(ok)
        if t.num_rows == 0:
            continue
        c0 = pc.dictionary_encode(pc.list_element(t['fsq_category_ids'], 0)).combine_chunks()
        codes = c0.indices.to_numpy(zero_copy_only=False).astype(np.int32); dic = c0.dictionary.to_pylist()
        lat = t['latitude'].to_numpy(); lon = t['longitude'].to_numpy()
        hid = pd.util.hash_array(t['fsq_place_id'].to_numpy(zero_copy_only=False).astype(object))
        fin = np.isfinite(lat) & np.isfinite(lon) & (np.abs(lat) <= 90) & (np.abs(lon) <= 180)
        if top is None:
            cnt = np.bincount(codes, minlength=len(dic))        # counted before the coordinate check, as in the static run
            for j, c in enumerate(dic):
                gcount[c] += int(cnt[j])
            H.append(hid[fin]); LA.append(lat[fin]); LO.append(lon[fin]); FC.append(codes[fin]); dicts.append(dic)
        else:
            fm = np.array([top.get(c, -1) for c in dic], np.int64)[codes]
            k_ = fin & (fm >= 0)
            H.append(hid[k_]); LA.append(lat[k_]); LO.append(lon[k_]); FC.append(fm[k_].astype(np.int8))
        del t
        if i % 20 == 0:
            log(' fsq', rel, 'file', i, 'rows', sum(len(x) for x in H))
    if top is None:
        TOPL = [c for c, _ in gcount.most_common(K)]; topd = {c: j for j, c in enumerate(TOPL)}
        fm = [np.array([topd.get(c, -1) for c in dic], np.int64)[codes] for codes, dic in zip(FC, dicts)]
        H = [h[f_ >= 0] for h, f_ in zip(H, fm)]; LA = [a[f_ >= 0] for a, f_ in zip(LA, fm)]; LO = [a[f_ >= 0] for a, f_ in zip(LO, fm)]
        FC = [f_[f_ >= 0].astype(np.int8) for f_ in fm]
        return np.concatenate(H), np.concatenate(LA), np.concatenate(LO), np.concatenate(FC), topd
    return np.concatenate(H), np.concatenate(LA), np.concatenate(LO), np.concatenate(FC), top

AUTO_FSQ = os.environ.get('V26_AUTO', '1') == '1'
if 'fsqdelta' in PARTS:
    S['v19']['fsqdelta'] = {}; FD = S['v19']['fsqdelta']
    try:
        NREL = int(os.environ.get('V21_NREL', '3')); LASTREL = os.environ.get('V14_REL', '2026-10-15')
        KMS = [int(x) for x in os.environ.get('V21_KMAX', '3,4').split(',')]; MSD = [int(x) for x in os.environ.get('V21_MS', '300,1000,2000').split(',')]
        LBCAP = float(os.environ.get('V21_LBCAP', '1800'))
        if os.environ.get('V21_FAKE') == '1':
            # synthetic releases from the TIST instances: churn of ~3% per release (test of the code path only)
            rgf = np.random.default_rng(5); base = np.arange(N); rels = ['r0', 'r1', 'r2']; REL_ = []
            alive = rgf.random(N) < 0.9
            for t_ in range(NREL):
                if t_ > 0:
                    flip = rgf.random(N) < 0.03; alive = alive ^ flip
                ids = base[alive]
                REL_.append((ids.astype(np.uint64) * np.uint64(2654435761) + np.uint64(17), P[ids].copy(), F[ids].astype(np.int8)))
            FD['releases'] = rels[:NREL]
            hids = [r[0] for r in REL_]; xyz = [r[1] for r in REL_]; fcs = [r[2] for r in REL_]
            lab = list(NAMES)
        else:
            import subprocess, sys
            from kaggle_secrets import UserSecretsClient
            tok = UserSecretsClient().get_secret("HF_TOKEN")
            subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'huggingface_hub>=0.24'], check=False)
            from huggingface_hub import HfApi
            api = HfApi(token=tok)
            allf = api.list_repo_files('foursquare/fsq-os-places', repo_type='dataset')
            dts = sorted({f.split('/')[1][3:] for f in allf if f.startswith('release/dt=') and '/places/parquet/' in f})
            dts = [x for x in dts if x <= LASTREL][-NREL:]; FD['releases'] = dts; log('fsqdelta releases', dts)
            hids, xyz, fcs = [None] * NREL, [None] * NREL, [None] * NREL
            h_, la_, lo_, fc_, topd = fsq_load_release(api, tok, dts[-1])
            hids[-1], xyz[-1], fcs[-1] = h_, to_xyz(la_, lo_) * 6371000.0, fc_; del la_, lo_
            for i_ in range(NREL - 1):
                h_, la_, lo_, fc_, _ = fsq_load_release(api, tok, dts[i_], top=topd)
                hids[i_], xyz[i_], fcs[i_] = h_, to_xyz(la_, lo_) * 6371000.0, fc_; del la_, lo_
            lab = sorted(topd, key=topd.get)
        FD['n_release'] = [int(len(h)) for h in hids]; log('fsqdelta release sizes', FD['n_release'])
        # universe of instance versions: (place id, feature, location); a move or recategorization is a deletion plus an insertion
        keys = []
        for h, x, f_ in zip(hids, xyz, fcs):
            q = np.round(x).astype(np.int64).view(np.uint64)
            k_ = splitmix64(h ^ splitmix64(q[:, 0] ^ splitmix64(q[:, 1] ^ splitmix64(q[:, 2] ^ f_.astype(np.uint64)))))
            keys.append(k_)
        allk = np.concatenate(keys); uk, first, inv = np.unique(allk, return_index=True, return_inverse=True)
        Pu = np.concatenate(xyz)[first]; Fu = np.concatenate(fcs)[first].astype(np.int64); Hu = np.concatenate(hids)[first]
        offs = np.r_[0, np.cumsum([len(k_) for k_ in keys])]
        ACT = []
        for i_ in range(NREL):
            a_ = np.zeros(len(uk), bool); a_[inv[offs[i_]:offs[i_ + 1]]] = True; ACT.append(a_)
        del allk, keys, xyz, inv; gc_ = __import__('gc'); gc_.collect()
        FD['universe'] = int(len(uk)); log('fsqdelta universe', len(uk))
        tb = time.time(); set_data(Pu, Fu, K); FD['tree_time'] = time.time() - tb
        U01 = ((Hu >> np.uint64(11)).astype(np.float64) + 0.5) / 9007199254740992.0
        from scipy.special import erfinv
        uU = np.exp(np.sqrt(2.0) * erfinv(2 * U01 - 1))               # same persistent utilities as the static FSQ run
        R = ((splitmix64(Hu ^ np.uint64(0x5DEECE66D)) >> np.uint64(11)).astype(np.float64) + 0.5) / 9007199254740992.0  # persistent seeds
        NAMES = lab; del Pu, Fu, U01
        FD['steps'] = []; PMS = {}; mv_prev = {}; ref_prev = None
        for t_ in range(NREL):
            act = ACT[t_]; row = dict(release=FD['releases'][t_], active=int(act.sum()))
            nodes = np.nonzero(act)[0]; U = np.bincount(F[act], weights=uU[act], minlength=K)
            if GRID_ON:
                tb = time.time(); g_cur = grid_update(g_prev, ACT[t_ - 1], act) if (GRID_UPD and t_ > 0) else grid_build(act); row['grid_time'] = time.time() - tb; log('fsqdelta', t_, 'grid', row['grid_time'])
                grid_register((act, g_cur), (ACT[t_ - 1], g_prev) if t_ > 0 else (None, None))
            ex3, t3, _, c3, _ = run_enum(nodes, uU[nodes], 3, active=act); row.update(remine_k3=t3, exact_state_k3=int(c3.sum()))
            log('fsqdelta', t_, 'remine k3', t3)
            if t_ > 0:
                prev = ACT[t_ - 1]
                ins = np.nonzero(act & ~prev)[0]; dele = np.nonzero(prev & ~act)[0]; chg = np.zeros(0, np.int64)   # delta input
                row.update(inserted=int(len(ins)), deleted=int(len(dele)))
                for kk in KMS:
                    a1 = run_enum(ins, uU[ins], kk, active=act, cap=LBCAP); a2 = run_enum(dele, uU[dele], kk, active=prev, cap=LBCAP)
                    row[f'lb_k{kk}'] = dict(time=a1[1] + a2[1], done_ins=int(a1[2]), n_ins=int(len(ins)), done_del=int(a2[2]), n_del=int(len(dele)),
                                            extrapolated=a1[1] * len(ins) / max(a1[2], 1) + a2[1] * len(dele) / max(a2[2], 1))
                    log('fsqdelta', t_, 'LB k%d' % kk, json.dumps(row[f'lb_k{kk}']))
            for kk in KMS:
                for m in MSD:
                    key = f'm{m}_k{kk}'
                    if t_ == 0:
                        tb = time.time(); PMS[key] = PMaint(act, uU, m, kk); info = dict(mode='init', total_time=time.time() - tb)
                        if AUTO_FSQ:
                            PMS[key + '_auto'] = PMaint(act, uU, m, kk, auto=True)
                    else:
                        info = PMS[key].step(ins, dele, chg, act, uU)
                        est_rb, irb = pollster_rebuild(act, uU, m, kk)
                        info['rebuild_time'] = irb['total_time']; info['rebuild_sample_time'] = irb['sample_time']
                        if AUTO_FSQ:
                            ia = PMS[key + '_auto'].auto_step(ins, dele, chg, act, uU)
                            ia['maxdiff'] = maxdiff(est_rb, PMS[key + '_auto'].estimate()); info['auto'] = ia
                    est = PMS[key].estimate(); info['state_entries'] = PMS[key].state_entries(); info['sampled'] = len(PMS[key].Pst)
                    if t_ > 0:
                        info['maint_vs_rebuild_maxdiff'] = maxdiff(est_rb, est); del est_rb
                    if kk == 3:
                        c_ = compare(ex3, est, U, 3, mus=(90,), m=m, T=NREL)
                        info['acc'] = dict(mae=c_['mae'], p99=c_['p99'], max=c_['max'], f1=c_['thr']['90']['f1'], mu=c_['thr']['90']['mu'],
                                           eps_mu=c_['thr']['90'].get('eps_mu'), viol=c_['bern_viol'], ratio=c_['bern_ratio_max'],
                                           hhat_recall=c_['thr']['90']['Hhat_recall'], hhat_precision=c_['thr']['90']['Hhat_precision'],
                                           viol2=c_['bern2_viol'], ratio2=c_['bern2_ratio_max'], e2e=c_['eps2_over_eps_median'],
                                           **{k_: c_['thr']['90'][k_] for k_ in ('Hhat', 'Hhat2', 'Hhat2_precision', 'Hhat2_recall', 'H', 'Hplus', 'Hplus_precision',
                                                                                 'Hplus_recall', 'Hplus2', 'Hplus2_precision', 'Hplus2_recall', 'eps2_mu_q')})
                    if t_ == NREL - 1:
                        np.save(os.path.join(OUT, f'fsqd_est_{key}.npy'), est)
                    row[key] = info; log('fsqdelta', t_, key, json.dumps({k_: v for k_, v in info.items() if k_ != 'acc'}), json.dumps(info.get('acc')))
            if MOVE_ON:
                for m_ in MOVE_MS:
                    tb_ = time.time(); cur_ = sample_emit(act, uU, m_, 3); t_s = time.time() - tb_
                    if m_ in mv_prev:
                        tb_ = time.time(); mv_ = move_eval(mv_prev[m_], cur_, ref_prev, ex3, m_, 3, NREL - 1)
                        mv_.update(emit_time=t_s, eval_time=time.time() - tb_); row.setdefault('moves', {})[str(m_)] = mv_
                        log('fsqdelta moves m=%d' % m_, json.dumps(mv_))
                    mv_prev[m_] = cur_
                ref_prev = ex3.copy()
            if t_ == NREL - 1:
                np.save(os.path.join(OUT, 'fsqd_ex3_last.npy'), ex3); np.save(os.path.join(OUT, 'fsqd_U_last.npy'), U)
                np.save(os.path.join(OUT, 'fsqd_rmask.npy'), RMASK)
            FD['steps'].append(row); save()
            if GRID_ON:
                g_prev = g_cur
    except Exception:
        import traceback; tb_ = traceback.format_exc(); log('ERROR fsqdelta\n' + tb_); FD['error'] = tb_; save()

# ------------------------------------------------------------------ Yelp (static k3/k4 full exact + quarterly stream)
if 'yelp' in PARTS:
    try:
        from json import loads as jloads
        from collections import Counter
        bus = pd.read_json(find('yelp_academic_dataset_business.json'), lines=True)
        bus = bus[['business_id', 'latitude', 'longitude', 'stars', 'categories']].dropna()
        bus = bus[np.isfinite(bus.latitude) & np.isfinite(bus.longitude)].reset_index(drop=True)
        GEN = {'Restaurants', 'Food', 'Shopping', 'Nightlife', 'Bars', 'Beauty & Spas', 'Health & Medical', 'Home Services',
               'Local Services', 'Automotive', 'Event Planning & Services', 'Active Life', 'Arts & Entertainment',
               'Hotels & Travel', 'Professional Services', 'Public Services & Government', 'Pets', 'Religious Organizations',
               'Education', 'Financial Services', 'Local Flavor', 'Mass Media'}
        cl = bus.categories.map(lambda s: [x.strip() for x in str(s).split(',') if x.strip() and x.strip() not in GEN])
        cc = Counter(c for l in cl for c in l); topy = [c for c, _ in cc.most_common(60)]; fy = {c: j for j, c in enumerate(topy)}; NAMES = list(topy)
        feat = cl.map(lambda l: next((fy[c] for c in l if c in fy), -1)).values; keep = np.nonzero(feat >= 0)[0]
        bid2i = {b: i for i, b in enumerate(bus.business_id.values[keep])}
        Q0 = 2012 * 4; rb = []; rq = []
        with open(find('yelp_academic_dataset_review.json'), 'rb') as fh:
            for line in fh:
                r = jloads(line); i = bid2i.get(r['business_id'])
                if i is None:
                    continue
                dd = r['date']; q = int(dd[:4]) * 4 + (int(dd[5:7]) - 1) // 3 - Q0
                rb.append(i); rq.append(max(q, 0))
        rb = np.asarray(rb, np.int64); rq = np.asarray(rq, np.int64); NQ = int(rq.max())
        CMy = np.zeros((len(keep), NQ), np.float32); ok = rq < NQ; np.add.at(CMy, (rb[ok], rq[ok]), 1.0)
        nrev = np.bincount(rb, minlength=len(keep)).astype(np.float64)
        Py = to_xyz(bus.latitude.values[keep], bus.longitude.values[keep]) * 6371000.0
        set_data(Py, feat[keep], len(topy)); log('yelp: businesses', N, 'reviews', int(nrev.sum()), 'quarters', NQ)
        uy = np.log1p(nrev) * bus.stars.values[keep]
        S['v19']['yelp_static'] = static_eval('yelp', np.ones(N, bool), uy, MS); save()
        run_stream(CMy, NQ, int(os.environ.get('V13_IDLE', '4')), 'yelp_dyn', insert_only=False, eucoloc=False)
    except Exception:
        import traceback; tb = traceback.format_exc(); log('ERROR yelp\n' + tb); S['v19']['yelp_error'] = tb; save()
log('v19 done')


if 'fsqprog' in PARTS:
    FP = S['v23'].setdefault('fsqprog', {})
    try:
        import subprocess, sys
        from kaggle_secrets import UserSecretsClient
        tok = UserSecretsClient().get_secret("HF_TOKEN")
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'huggingface_hub>=0.24'], check=False)
        from huggingface_hub import HfApi
        api = HfApi(token=tok)
        h_, la_, lo_, fc_, topd = fsq_load_release(api, tok, os.environ.get('V14_REL', '2026-10-15'))
        Pf = to_xyz(la_, lo_) * 6371000.0; del la_, lo_
        set_data(Pf, fc_.astype(np.int64), K); del Pf
        U01 = ((h_ >> np.uint64(11)).astype(np.float64) + 0.5) / 9007199254740992.0
        from scipy.special import erfinv
        uf = np.exp(np.sqrt(2.0) * erfinv(2 * U01 - 1)); del U01
        actf = np.ones(N, bool); nodes_f = np.arange(N, dtype=np.int64); Uf = np.bincount(F, weights=uf, minlength=K)
        kmax = 3; log('fsqprog loaded', N)
        exf = []; t0_ = time.time()
        for f in range(K):
            idx = np.nonzero(F == f)[0]; exf.append(run_enum(idx, uf[idx], kmax, active=actf)[0])
        FP['exact_time'] = time.time() - t0_; ex = np.sum(exf, axis=0); log('fsqprog exact', FP['exact_time'])
        selx = np.nonzero((RSIZE <= kmax) & (ex > 0))[0]; upi = ex[selx] / dens(Uf)[selx]
        for q in [int(x) for x in os.environ.get('V23_QS', '90,95,99').split(',')]:
            mu = float(np.quantile(upi, q / 100)); mc = m_for_rho(mu, kmax)
            (est, var, _, info) = pollster_est(actf, uf, mc, kmax); cmp_ = compare(ex, est, Uf, kmax, m=mc, mus=(q,))
            fixed = dict(m=mc, time=info['total_time'], hhat_precision=cmp_['thr'][str(q)]['Hhat_precision'], hhat=cmp_['thr'][str(q)]['Hhat'], H=cmp_['thr'][str(q)]['H'])
            for m0 in [int(x) for x in os.environ.get('V23_M0', '500').split(',')]:
                r_ = prog_refine(actf, uf, kmax, mu, m0, jmax=int(os.environ['V23_JMAX']) if os.environ.get('V23_JMAX') else None, exf=exf, budget=float(os.environ.get('V23_BUDGET', '5400')), tag=f'fsq k3 q{q} m0={m0}')
                r_.pop('final'); r_.pop('est_total', None)
                Hn = r_['check']['H']; thr_t = {}
                for frac in (0.1, 0.01, 0.0):
                    ok_ = [rd['t'] for rd in r_['rounds'] if rd['undecided'] <= frac * Hn]
                    thr_t[str(frac)] = ok_[0] if ok_ else None
                r_.update(mu=mu, exact_time=FP['exact_time'], fixed_cor2=fixed, t_undecided=thr_t)
                FP[f'q{q}_m0{m0}'] = r_; save()
                log('fsqprog', q, m0, json.dumps({k_: v for k_, v in r_.items() if k_ not in ('rounds', 'lev', 'cnt', 'exact')}))
    except Exception:
        import traceback; tb_ = traceback.format_exc(); log('ERROR fsqprog\n' + tb_); FP['error'] = tb_; save()


# ===================================================================================================== v24: top-k tracking + FSQ timing repeats
if 'topk' in PARTS:
    TK = S.setdefault('v24', {}).setdefault('topk', {})
    try:
        KS_ = [int(x) for x in os.environ.get('V24_TOPK', '10,50,100').split(',')]
        MS_ = [int(x) for x in os.environ.get('V24_MS', '300,1000,2000').split(',')]
        MI_ = [int(x) for x in os.environ.get('V24_MI', '300,2000').split(',')]
        for kmax in [int(x) for x in os.environ.get('V24_KMAX', '3,4').split(',')]:
            rows = TK.setdefault(f'k{kmax}', [])
            cum = np.cumsum(CM, axis=1); first = np.where(CM.sum(1) > 0, np.argmax(CM > 0, axis=1), NMON)
            lastseen = np.full(N, -10**6); prevtop = {}
            for t in range(NMON):
                lastseen = np.where(CM[:, t] > 0, t, lastseen)
                act = (first <= t) & (t - lastseen < IDLE)
                u = np.log1p(cum[:, t]).astype(np.float64)
                U = np.bincount(F[act], weights=u[act], minlength=K); nodes = np.nonzero(act)[0]
                ex, t_rm, _, _, _ = run_enum(nodes, u[nodes], kmax, active=act)
                sel = np.nonzero((RSIZE <= kmax) & (ex > 0))[0]; den = dens(U)[sel]; upi = ex[sel] / den
                oex = np.argsort(-upi, kind='stable')
                row = dict(snapshot=t, active=int(act.sum()), remine=t_rm, n_pat=int(len(sel)), res={})
                cfgs = [('coord', m) for m in MS_] + [('indep', m) for m in MI_] + [('exact', 0)]
                for mode, m in cfgs:
                    if mode == 'exact':
                        e_ = upi; tt = t_rm
                    else:
                        Rv = None if mode == 'coord' else np.random.default_rng(10007 * (t + 1) + m).random(N) + 1e-12
                        est, info = pollster_rebuild(act, u, m, kmax, Rv=Rv); e_ = est[sel] / den; tt = info['total_time']
                    oe = np.argsort(-e_, kind='stable'); key = f'{mode}_{m}'; r_ = dict(time=tt)
                    for kk in KS_:
                        te = set(sel[oex[:kk]].tolist()); tr = set(sel[oe[:kk]].tolist())
                        d_ = dict(overlap=len(te & tr) / kk, maxerr_top=float(np.max(np.abs(e_[oex[:kk]] - upi[oex[:kk]]))),
                                  rank_shift_max=int(np.max(np.argsort(oe)[oex[:kk]])) if mode != 'exact' else kk - 1)
                        pk = (key, kk)
                        if pk in prevtop:
                            ptr = prevtop[pk]; pte = prevtop[('exact_0', kk)]
                            ent = tr - ptr; lft = ptr - tr; ent_e = te - pte; lft_e = pte - te
                            d_.update(changed=len(ent), changed_true=len(ent & ent_e) + len(lft & lft_e),
                                      changed_total=len(ent) + len(lft), exact_changed=len(ent_e))
                        prevtop[pk] = tr
                        r_[f'top{kk}'] = d_
                    row['res'][key] = r_
                rows.append(row); save()
                log('topk k%d t=%d' % (kmax, t), json.dumps({k_: {kk: v[kk] for kk in v if kk.startswith('top')} for k_, v in row['res'].items()})[:1500])
    except Exception:
        import traceback; tb_ = traceback.format_exc(); log('ERROR topk\n' + tb_); TK['error'] = tb_; save()

if 'fsqrep' in PARTS:
    FR = S.setdefault('v24', {}).setdefault('fsqrep', {})
    try:
        import subprocess, sys
        FR['env'] = dict(python=sys.version.split()[0], numba=numba.__version__, numpy=np.__version__,
                         cpu=open('/proc/cpuinfo').read().split('model name')[1].split('\n')[0].strip(': \t'), nproc=os.cpu_count(), threads=nthreads())
        from kaggle_secrets import UserSecretsClient
        tok = UserSecretsClient().get_secret("HF_TOKEN")
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'huggingface_hub>=0.24'], check=False)
        from huggingface_hub import HfApi
        api = HfApi(token=tok)
        h_, la_, lo_, fc_, topd = fsq_load_release(api, tok, os.environ.get('V14_REL', '2026-10-15'))
        Pf = to_xyz(la_, lo_) * 6371000.0; del la_, lo_
        tb = time.time(); set_data(Pf, fc_.astype(np.int64), K); FR['tree_time'] = time.time() - tb; FR['n'] = int(N)
        U01 = ((h_ >> np.uint64(11)).astype(np.float64) + 0.5) / 9007199254740992.0
        from scipy.special import erfinv
        uf = np.exp(np.sqrt(2.0) * erfinv(2 * U01 - 1)); del U01, Pf
        Rh = ((splitmix64(h_ ^ np.uint64(0x5DEECE66D)) >> np.uint64(11)).astype(np.float64) + 0.5) / 9007199254740992.0
        from scipy.stats import spearmanr
        FR['seed_util_spearman_stream'] = float(spearmanr(uf, Rh)[0]); FR['seed_util_spearman_static'] = float(spearmanr(uf, R)[0]); del Rh
        log('fsqrep corr', FR['seed_util_spearman_stream'], FR['seed_util_spearman_static']); save()
        actf = np.ones(N, bool); FR['res'] = {}
        for rep in range(int(os.environ.get('V24_REPS', '2'))):
            for kk in (3, 4):
                for m_ in [int(x) for x in os.environ.get('V24_FMS', '300,1000,2000').split(',')]:
                    info = pollster_est(actf, uf, m_, kk)[3]; info.pop('taus', None)
                    FR['res'].setdefault(f'm{m_}_k{kk}', []).append(dict(total=info['total_time'], sample=info['sample_time'], star=info['star_time'],
                                                                         acc=info['acc_time'], sampled=info['sampled'], state=info['state_entries']))
                    log('fsqrep', rep, m_, kk, json.dumps(FR['res'][f'm{m_}_k{kk}'][-1])); save()
    except Exception:
        import traceback; tb_ = traceback.format_exc(); log('ERROR fsqrep\n' + tb_); FR['error'] = tb_; save()
