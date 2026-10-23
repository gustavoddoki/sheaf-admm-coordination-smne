from pathlib import Path
import hashlib,json
r=Path(__file__).resolve().parent
m=json.loads((r/"PUBLICATION_MANIFEST.json").read_text(encoding="utf-8"))
bad=[x["path"] for x in m["files"] if not (r/x["path"]).is_file() or hashlib.sha256((r/x["path"]).read_bytes()).hexdigest()!=x["sha256"]]
print("PASS" if not bad else "FAIL: "+str(bad))
raise SystemExit(bool(bad))
