from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import json, math, webbrowser
from threading import Timer
from web_predict import load, predict, validate, REGIONS

MODEL, CARD = load()
HTML = Path(__file__).with_name('web_ui.html').read_bytes()
class Handler(BaseHTTPRequestHandler):
    def send(self, status, body, kind='application/json; charset=utf-8'):
        self.send_response(status); self.send_header('Content-Type', kind); self.send_header('Content-Length', str(len(body))); self.end_headers(); self.wfile.write(body)
    def do_GET(self):
        if self.path == '/': self.send(200, HTML, 'text/html; charset=utf-8')
        elif self.path == '/api/model': self.send(200, json.dumps(CARD, ensure_ascii=False).encode())
        else: self.send(404, b'{}')
    def do_POST(self):
        if self.path != '/api/predict': return self.send(404,b'{}')
        try:
            n=int(self.headers.get('Content-Length','0'))
            if not 0<n<=4096: raise ValueError('Dữ liệu không hợp lệ')
            d=json.loads(self.rfile.read(n)); a=float(d['age']); b=float(d['bmi']); c=float(d['children']); s=d['sex']; t=d['smoker']; r=d['region']
            if not all(math.isfinite(x) for x in (a,b,c)) or not a.is_integer() or not c.is_integer(): raise ValueError('Tuổi và số con phải là số nguyên; chỉ số khối cơ thể phải là số hữu hạn.')
            validate(a,b,c)
            if s not in ['male','female'] or t not in ['yes','no'] or r not in ("north", "central", "south"): raise ValueError('Lựa chọn không hợp lệ.')
            p=sum(predict(MODEL,int(a),s,b,int(c),t,region) for region in REGIONS)/len(REGIONS)
            q=sum(predict(MODEL,int(a),s,b,int(c),'no' if t=='yes' else 'yes',region) for region in REGIONS)/len(REGIONS)
            self.send(200,json.dumps({'cost':p,'alternative':q}).encode())
        except (ValueError, TypeError, KeyError, OverflowError) as e: self.send(400,json.dumps({'error':str(e)},ensure_ascii=False).encode())
        except Exception: self.send(500,json.dumps({'error':'Không thể dự đoán. Kiểm tra mô hình và thư viện.'},ensure_ascii=False).encode())
if __name__=='__main__':
    for port in range(8000,8011):
        try:
            server=ThreadingHTTPServer(('127.0.0.1',port),Handler)
            break
        except OSError:
            continue
    else:
        raise SystemExit('Khong tim duoc cong trong 8000-8010.')
    url=f'http://127.0.0.1:{port}'
    print(url, flush=True)
    Timer(0.6,lambda:webbrowser.open(url)).start()
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally: server.server_close()
