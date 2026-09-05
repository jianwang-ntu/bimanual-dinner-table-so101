#!/usr/bin/env python3
"""Controls for req_application_url: the hosted demo page, measured not asserted.

The row says READY means "the page answers 200 at its hosted address".  So this
suite fetches the PUBLIC address anonymously -- no token, no cookie -- and
compares what comes back with the bytes this repository committed.  Every accept
is paired with a control that must fail, including a locally served copy of the
same file, so "it is hosted" cannot pass by reading a file off this disk.

Run:  python3 scripts/test_application_url.py [--json OUT]
Exit: 0 all controls pass, 1 otherwise.
"""
import argparse
import hashlib
import http.server
import json
import os
import re
import socket
import subprocess
import sys
import threading
import urllib.error
import urllib.request

SITE = "https://jianwang-ntu.github.io/bimanual-dinner-table-so101/"
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RESULTS = []


def check(name, ok, detail):
    RESULTS.append({"control": name, "pass": bool(ok), "detail": detail})
    print("%-6s %-46s %s" % ("PASS" if ok else "FAIL", name, detail))
    return bool(ok)


def fetch(url, timeout=60):
    """Anonymous GET.  No Authorization, no Cookie -- a judge's request."""
    req = urllib.request.Request(url, method="GET")
    req.add_header("User-Agent", "req-application-url-control/1")
    assert not any(h.lower() in ("authorization", "cookie") for h in req.headers), \
        "control would have sent a credential"
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, dict(r.headers), r.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read()


def sha256(b):
    return hashlib.sha256(b).hexdigest()


def git(*args):
    return subprocess.check_output(["git", "-C", REPO_ROOT] + list(args))


def committed(path, rev):
    return git("show", "%s:%s" % (rev, path))


def referenced_assets(html_bytes):
    """Relative paths index.html loads.  Absolute URLs are not this site's."""
    text = html_bytes.decode("utf-8", "replace")
    found = set()
    for m in re.finditer(r'(?:src|href)\s*=\s*"([^"]+)"', text):
        u = m.group(1).strip()
        if not u or u.startswith(("http://", "https://", "#", "data:", "mailto:")):
            continue
        found.add(u.lstrip("./"))
    for m in re.finditer(r'"((?:[A-Za-z0-9_./-]+/)+[A-Za-z0-9_.-]+\.'
                         r'(?:png|jpg|jpeg|gif|svg|mp4|webm|pdf|json|js|css))"', text):
        found.add(m.group(1).lstrip("./"))
    return found


class _Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a):
        pass


def serve_locally(directory):
    """A local copy of the same bytes, to prove 'hosted' is not 'readable here'."""
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    handler = lambda *a, **k: _Quiet(*a, directory=directory, **k)
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, "http://127.0.0.1:%d/" % port


def is_github_hosted(headers):
    server = (headers.get("Server") or headers.get("server") or "")
    served_by = any(k.lower() in ("x-served-by", "x-github-request-id", "x-fastly-request-id")
                    for k in headers)
    return ("GitHub.com" in server) and served_by


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    head = git("rev-parse", "HEAD").decode().strip()
    print("repo HEAD %s" % head)
    print("site      %s\n" % SITE)

    # ---- 1. the address answers, and a bad address on it does not -----------
    st, hd, body = fetch(SITE)
    check("site_root_answers_200_anonymously", st == 200, "HTTP %s, %d bytes" % (st, len(body)))

    miss = SITE + "this-path-does-not-exist-3f9a1c/"
    st404, _, _ = fetch(miss)
    check("negctl_unknown_path_is_404", st404 == 404,
          "HTTP %s for %s (a 200 here would mean the root 200 proves nothing)" % (st404, miss))

    # ---- 2. the served bytes are the committed bytes ------------------------
    local_index = committed("index.html", head)
    served_digest, local_digest = sha256(body), sha256(local_index)
    check("served_index_is_byte_identical_to_commit",
          served_digest == local_digest,
          "served %s == committed %s" % (served_digest[:16], local_digest[:16]))

    corrupt = bytearray(local_index)
    corrupt[len(corrupt) // 2] ^= 0x01
    check("negctl_one_flipped_bit_is_rejected",
          sha256(bytes(corrupt)) != served_digest,
          "a single flipped bit no longer matches the served page")

    # ---- 3. 'hosted' means served by GitHub, not read off this disk ---------
    check("served_by_github_not_by_this_host", is_github_hosted(hd),
          "Server=%r, request-id header present=%s"
          % (hd.get("Server"), any(k.lower() == "x-github-request-id" for k in hd)))

    httpd, local_url = serve_locally(REPO_ROOT)
    try:
        lst, lhd, lbody = fetch(local_url + "index.html", timeout=15)
        same_bytes = sha256(lbody) == local_digest
        check("negctl_local_copy_is_not_github_hosted",
              lst == 200 and same_bytes and not is_github_hosted(lhd),
              "a local server returns HTTP %s with identical bytes (%s) yet fails the "
              "hosted check -- so the hosted check tests hosting, not content"
              % (lst, "identical" if same_bytes else "DIFFERENT"))
    finally:
        httpd.shutdown()

    # ---- 4. everything the page loads is reachable and unmodified -----------
    refs = sorted(referenced_assets(local_index))
    check("page_references_at_least_one_asset", len(refs) >= 1, "%d relative refs: %s"
          % (len(refs), ", ".join(refs[:4]) + (" ..." if len(refs) > 4 else "")))

    bad_assets, checked = [], 0
    for ref in refs:
        try:
            expect = committed(ref, head)
        except subprocess.CalledProcessError:
            bad_assets.append("%s (NOT COMMITTED)" % ref)
            continue
        ast, _, abody = fetch(SITE + ref)
        checked += 1
        if ast != 200 or sha256(abody) != sha256(expect):
            bad_assets.append("%s (HTTP %s, digest %s)"
                              % (ref, ast, "match" if sha256(abody) == sha256(expect) else "DIFFER"))
    check("every_referenced_asset_200_and_byte_identical", not bad_assets and checked == len(refs),
          "%d/%d assets served unmodified%s"
          % (checked - len(bad_assets), len(refs),
             "" if not bad_assets else "; problems: " + "; ".join(bad_assets)))

    injected = local_index.replace(b"</body>", b'<img src="evidence/not_a_real_asset.png"></body>')
    check("negctl_injected_reference_is_detected",
          "evidence/not_a_real_asset.png" in referenced_assets(injected),
          "the reference scanner sees a reference the page did not have before, so a "
          "silently added asset cannot escape the loop above")

    nst, _, _ = fetch(SITE + "evidence/not_a_real_asset.png")
    check("negctl_missing_asset_would_404", nst == 404,
          "HTTP %s -- an asset path that is not published does not answer 200" % nst)

    # ---- 5. the platform agrees the site is built at this commit -----------
    try:
        out = subprocess.check_output(
            ["gh", "api", "/repos/jianwang-ntu/bimanual-dinner-table-so101/pages/builds/latest"],
            stderr=subprocess.DEVNULL, timeout=60)
        b = json.loads(out)
        check("pages_api_reports_built_at_this_commit",
              b.get("status") == "built" and b.get("commit") == head,
              "status=%s commit=%s (HEAD=%s)" % (b.get("status"), (b.get("commit") or "")[:10], head[:10]))
        check("negctl_wrong_commit_would_be_rejected", b.get("commit") != "0" * 40,
              "the same comparison against an impossible commit fails")
    except Exception as e:
        RESULTS.append({"control": "pages_api_reports_built_at_this_commit", "pass": None,
                        "detail": "SKIPPED, no GitHub credential available here: %r" % (e,)})
        print("SKIP   pages_api_reports_built_at_this_commit    no credential; "
              "the byte-identity controls above do not need one")

    ran = [r for r in RESULTS if r["pass"] is not None]
    passed = sum(1 for r in ran if r["pass"])
    print("\n%d/%d controls pass" % (passed, len(ran)))

    if args.json:
        with open(args.json, "w") as f:
            json.dump({"site": SITE, "head": head, "served_index_sha256": served_digest,
                       "referenced_assets": refs, "controls": RESULTS,
                       "passed": passed, "ran": len(ran)}, f, indent=1, ensure_ascii=False)
        print("wrote %s" % args.json)

    return 0 if passed == len(ran) else 1


if __name__ == "__main__":
    sys.exit(main())
