import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
REGIONS=['northeast','northwest','southeast','southwest']
def load():
    return tuple(json.loads((ROOT/'models'/f).read_text(encoding='utf-8')) for f in ['web_weights.json','model_card.json'])
def validate(a,b,c):
    if not (18<=a<=64 and 15<=b<=55 and 0<=c<=5): raise ValueError('Thông tin nằm ngoài phạm vi cho phép.')
def predict(m,a,s,b,c,t,r):
    smoker=int(t=='yes'); obese=int(b>=30)
    x=[a,b,c,obese,smoker*obese,smoker*b,a*a]
    x=[(v-u)/z for v,u,z in zip(x,m['mean'],m['scale'])]+[int(s=='male'),smoker]+[int(r==k) for k in REGIONS[1:]]
    return m['intercept']+sum(v*w for v,w in zip(x,m['coef']))
