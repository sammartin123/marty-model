"""Inject a game's JSON into the page template.
usage: python3 src/build_page.py out/<GAME>.json site/index.html"""
import sys, json, os
from teams import colours
J = json.load(open(sys.argv[1]))
m = J['meta']
m.setdefault('colors', colours(m['home'], m['away']))
m.setdefault('data_through', 'the most recent completed game')
t = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'template.html')).read()
os.makedirs(os.path.dirname(os.path.abspath(sys.argv[2])), exist_ok=True)
open(sys.argv[2], 'w').write(t.replace('/*DATA*/', json.dumps(J, separators=(',', ':')).replace('</', '<\\/')))
print('wrote', sys.argv[2], os.path.getsize(sys.argv[2]))
