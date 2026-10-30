"""Read the DICOM study and series UIDs of CQ500 scans without downloading the whole archive (HTTP range requests on the
released zip from the official host). Only the zip directory and the first 64 KB of the first DICOM file of each series are
read (header elements before the pixel data); nothing is kept but the UIDs and file counts.
Usage: python fetch_uids_range.py <json_dir> [scan ...]   (default: every scan JSON in <json_dir> without series_uids)
Output: <json_dir>/uids/<scan>.json"""
import sys, os, io, json, glob, zipfile, ssl, http.client, threading, time
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pydicom
from cq500_select import choose

S3 = 'https://s3.ap-south-1.amazonaws.com/qure.headct.study/'
HEADER_BYTES = 1 << 16
_local = threading.local()


def _conn(host):
    """One persistent HTTPS connection per thread (through the HTTPS proxy when one is configured)."""
    c = getattr(_local, 'conn', None)
    if c is None:
        ctx = ssl.create_default_context(cafile=os.environ.get('SSL_CERT_FILE') or None)
        px = os.environ.get('HTTPS_PROXY') or os.environ.get('https_proxy')
        if px:
            p = urlparse(px)
            c = http.client.HTTPSConnection(p.hostname, p.port, context=ctx, timeout=120)
            c.set_tunnel(host, 443)
        else:
            c = http.client.HTTPSConnection(host, 443, context=ctx, timeout=120)
        _local.conn = c
    return c


def _request(method, url, headers=None):
    u = urlparse(url)
    for attempt in range(6):
        try:
            c = _conn(u.hostname)
            c.request(method, u.path, headers=headers or {})
            r = c.getresponse()
            data = r.read()
            return r, data
        except Exception:
            try:
                _local.conn.close()
            except Exception:
                pass
            _local.conn = None
            if attempt == 5:
                raise
            time.sleep(1.5 * (attempt + 1))


class RangeFile(io.RawIOBase):
    """Seekable read-only view of a remote file through HTTP Range requests, with a block cache."""

    def __init__(self, url, block=1 << 16):
        self.url, self.block, self.pos, self.cache, self.bytes_read = url, block, 0, {}, 0
        r, _ = _request('HEAD', url)
        assert r.status == 200, f'HEAD {r.status}'
        self.size = int(r.getheader('Content-Length'))

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off)
        return self.pos

    def _fetch(self, i0, i1):
        a = i0 * self.block; b = min(self.size, (i1 + 1) * self.block) - 1
        r, data = _request('GET', self.url, {'Range': f'bytes={a}-{b}'})
        assert r.status == 206 and len(data) == b - a + 1, f'range read {r.status} {len(data)}'
        self.bytes_read += len(data)
        for i in range(i0, i1 + 1):
            self.cache[i] = data[(i - i0) * self.block:(i - i0 + 1) * self.block]

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = max(0, min(n, self.size - self.pos))
        if n == 0:
            return b''
        i0, i1 = self.pos // self.block, (self.pos + n - 1) // self.block
        miss = [i for i in range(i0, i1 + 1) if i not in self.cache]
        if miss:
            self._fetch(min(miss), max(miss))
        buf = b''.join(self.cache[i] for i in range(i0, i1 + 1))
        o = self.pos - i0 * self.block
        out = buf[o:o + n]
        self.pos += len(out)
        return out

    def readinto(self, b):
        d = self.read(len(b)); b[:len(d)] = d; return len(d)


def uids_for(name):
    rf = RangeFile(S3 + 'CQ500-CT-' + name.split('-')[-1] + '.zip')
    z = zipfile.ZipFile(rf)
    sk, bk, allser, ok = choose(z)
    out = {}
    for k, n in allser.items():
        files = sorted(f for f in z.namelist() if f.startswith(k + '/') and f.endswith('.dcm'))
        with z.open(files[0]) as fh:
            head = fh.read(HEADER_BYTES)
        h = pydicom.dcmread(io.BytesIO(head), stop_before_pixels=True, force=True)
        out[k.rsplit('/', 1)[-1]] = dict(series_uid=str(getattr(h, 'SeriesInstanceUID', '')), study_uid=str(getattr(h, 'StudyInstanceUID', '')), n=n)
        assert out[k.rsplit('/', 1)[-1]]['series_uid'] and out[k.rsplit('/', 1)[-1]]['study_uid'], 'UID not found in the first 64 KB'
    return dict(name=name, soft_series=sk.rsplit('/', 1)[-1] if sk else None, series_uids=out,
                via='HTTP range read of the released archive (directory and the first 64 KB of one file per series)',
                zip_bytes=rf.size, bytes_read=rf.bytes_read)


def job(jd, nm):
    dst = os.path.join(jd, 'uids', nm + '.json')
    if os.path.exists(dst):
        return nm, None
    r = uids_for(nm)
    tmp = dst + '.tmp'
    json.dump(r, open(tmp, 'w'), indent=1)
    os.replace(tmp, dst)
    return nm, r


if __name__ == '__main__':
    jd = sys.argv[1]
    os.makedirs(os.path.join(jd, 'uids'), exist_ok=True)
    names = sys.argv[2:]
    if not names:
        for f in sorted(glob.glob(os.path.join(jd, 'CQ500-CT-*.json'))):
            if f.endswith('.error.json'):
                continue
            d = json.load(open(f))
            if not d.get('series_uids'):
                names.append(d['name'])
    workers = int(os.environ.get('UID_WORKERS', '8'))
    with ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(job, jd, nm) for nm in names]
        for fu in as_completed(futs):
            try:
                nm, r = fu.result()
                if r:
                    print(nm, len(r['series_uids']), 'series;', round(r['bytes_read'] / 1e6, 2), 'of', round(r['zip_bytes'] / 1e6, 1), 'MB read', flush=True)
            except Exception as e:
                print('ERR', repr(e)[:300], flush=True)
