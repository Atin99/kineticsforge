import re
p = r'c:\project 5\kineticsforge\kaggle_deploy_2\acct1_cell.py'
with open(p, 'r', encoding='utf-8') as f:
    code = f.read()
# Replace print( with log( except in the def log line
lines = code.split('\n')
out = []
for line in lines:
    if 'def log(msg)' in line:
        out.append(line)
    elif 'print(msg' in line:
        out.append(line)
    else:
        out.append(line.replace('print(f"', 'log(f"').replace('print("', 'log("'))
with open(p, 'w', encoding='utf-8') as f:
    f.write('\n'.join(out))
n = sum(1 for l in out if 'log(' in l and 'def log' not in l)
print(f"Done: {n} log() calls")
