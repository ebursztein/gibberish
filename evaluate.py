import json
import pickle
from tabulate import tabulate
from gibberish.gibberish import GibberishDetector
from time import time

def main():
  data = json.load(open('data/test/test.json'))
  gb = GibberishDetector()
  rows = []
  s = time()
  for ex in data:
    verdict, reason = gb.is_text(ex['text'], debug=True)
    rows.append((ex['text'][:50], ex['is_text'], verdict, reason))
  print(f"Time: {time() - s}")
  print(tabulate(rows, headers=['String', 'label',  'verdict', 'reason'], floatfmt=".2f"))

if __name__ == "__main__":
  main()