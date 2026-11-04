from pathlib import Path
from collections import defaultdict
import hashlib,json,os,re,statistics,sys
root=Path(sys.argv[2])
if sys.argv[1]=='select':
 groups=defaultdict(list)
 for name in sys.argv[3:]:
  for line in (root/name).read_text().splitlines():
   m=re.search(r'\[onehot_screen\] repeat=(\d+) mode=(\S+) blocks=(\d+) columns=(\d+) prefetch=(\w+) elapsed_s=([\d.]+)',line)
   if m:
    repeat,mode,b,c,p,t=m.groups();groups[(name,mode,int(b),int(c),int(p=='true'))].append(float(t))
 rows=[dict(log=k[0],mode=k[1],blocks=k[2],columns=k[3],prefetch=k[4],samples=v,mean_s=statistics.mean(v)) for k,v in groups.items()]
 assert rows and all(len(x['samples'])==2 for x in rows),rows
 rows.sort(key=lambda x:x['mean_s'])
 # Full proofs cover each arithmetic family; the top two five-prime panels
 # test whether close screening ranks survive the full-size workload.
 selected=[]
 for family,count in [('ntt-exact',1),('ntt-panel6',1),('ntt-panel5',2),('ntt-binary5',1)]:
  selected.extend([x for x in rows if x['mode']==family][:count])
 best_plan=selected[0]['log']
 # Pick the globally fastest plan, then screen finalists of all families in it.
 candidates=[x for x in rows if x['mode'] not in {'accumulation','ntt-legacy'}]
 best_plan=candidates[0]['log']
 selected=[]
 for family,count in [('ntt-exact',1),('ntt-panel6',1),('ntt-panel5',2),('ntt-binary5',1)]:
  selected.extend([x for x in rows if x['mode']==family and x['log']==best_plan][:count])
 tags=[]
 for row in selected:
  mode=row['mode']
  tag=mode if mode=='ntt-exact' else f"{mode}@{row['blocks']}x{row['columns']}x{row['prefetch']}"
  if tag not in tags: tags.append(tag)
 order=','.join(['accumulation','ntt-legacy',*tags,'best','accumulation','best'])
 result={'rows':rows,'selected':selected,'best_simd_log':best_plan,'order':order}
 (root/'screen-summary.json').write_text(json.dumps(result,indent=2)+'\n')
 (root/'selected-order.txt').write_text(order+'\n')
 print(json.dumps(result),flush=True)
elif sys.argv[1] in {'verify','ci-verify'}:
 ci=sys.argv[1]=='ci-verify'
 nv=sys.argv[3];name=sys.argv[4] if len(sys.argv)>4 else f'nv{nv}'
 log=(root/f'{name}.log').read_text();trials=[]
 assert ' INFO Done' in log
 for section in log.split('[experiment] trial=')[1:]:
  match=re.match(r'(\d+) onehot_commit=(\S+)(?: requested=(\S+))?',section)
  index,mode,requested=match.groups()
  prefix=r'\[onehot_fp128(?:_k16)?\] '
  def number(label): return float(re.search('^'+prefix+label+r': ([\d.]+)',section,re.M)[1])
  assert re.search(prefix+'verify multi threaded OK:',section)
  assert re.search(prefix+'verify single threaded OK:',section)
  io=re.search(r'commit_io_read_bytes: (\d+)',section)
  trials.append({'trial':int(index),'mode':mode,'requested':requested,'commit_s':number('commit'),'prove_s':number('prove'),'storage_read_bytes':int(io[1]) if io else None,'both_verified':True})
 assert len(trials)>=2
 if ci:
  assert [t['trial'] for t in trials]==list(range(8)), 'unexpected CI trial indices'
  expected=['accumulation','ntt-panel5@8x48x0']*4
  # Warm up each method, then interleave three samples per method.
  assert [t['mode'] for t in trials]==expected, 'unexpected CI kernel sequence'
  for t,section in zip(trials,log.split('[experiment] trial=')[1:]):
   if t['mode']=='accumulation':
    assert 'one hot commit schedule' in section and '[onehot_ntt]' not in section, 'accumulation kernel was not observed'
    t['observed_kernel']='accumulation'
   else:
    assert '[onehot_ntt] primes=5 blocks=8 tile=48 binary=false ' in section, 'five-prime NTT kernel was not observed'
    assert 'one hot commit schedule' not in section, 'NTT trial used accumulation'
    t['observed_kernel']='ntt-panel5@8x48x0'
   t['warmup']=t['trial']<2
 result={'nv':int(nv),'trials':trials}
 for ext in ['proof','commitment']:
  files=sorted(root.glob(f'{name}-*.{ext}'));assert len(files)==len(trials),(ext,len(files),len(trials))
  data=files[0].read_bytes();assert data and all(p.read_bytes()==data for p in files),ext
  result[ext+'_bytes']=len(data);result[ext+'_sha256']=hashlib.sha256(data).hexdigest();result[ext+'_all_identical']=True
 if nv=='38' and not ci:
  # Historical hashes belong to the benchmark revision. Compare variants
  # within this run so policy updates on main can change the proof bytes.
  a=statistics.mean(x['commit_s'] for x in trials if x['mode']=='accumulation')
  repeated=[x for x in trials if x['requested']=='best']
  assert len(repeated)==2 and len({x['mode'] for x in repeated})==1,repeated
  n=statistics.mean(x['commit_s'] for x in repeated)
  result.update(accumulation_mean_s=a,best_mean_s=n,best_mode=repeated[0]['mode'],speedup=a/n,ntt_time_reduction_percent=100*(1-n/a))
 if ci:
  measured=[t for t in trials if not t['warmup']]
  a=statistics.median(t['commit_s'] for t in measured if t['mode']=='accumulation')
  n=statistics.median(t['commit_s'] for t in measured if t['mode'].startswith('ntt'))
  result.update(accumulation_median_s=a,ntt_median_s=n,speedup=a/n,ntt_time_reduction_percent=100*(1-n/a),
                measured_revision=os.environ.get('GITHUB_SHA'),threads=16,k_chunk=16,setup_mode='recursive',
                comparison='Same head binary; explicit accumulation versus five-prime NTT; three measured samples per method after one warmup each')
  report=f'''
### Onehot NTT commitment comparison

nv={nv}, K=16, recursive FP128; 16 threads. Same head binary, witness and setup.
Three interleaved samples per method after one warmup per method; medians below.
This compares kernels within the head revision, separately from the merge-base table.

| Observed commit kernel | Median commit time |
| --- | ---: |
| Accumulation | {a:.3f} s |
| Five-prime NTT (8 blocks, 48 columns) | {n:.3f} s |

Commit time reduction: **{100*(1-n/a):.1f}%**, speedup **{a/n:.2f}x**.
Setup/preprocessing excluded; witness conversion included. No witness gadget decomposition.
All eight proofs passed both verifier thread policies; serialized commitments and proofs match.
Kernel execution markers were checked for every trial; fallback fails the benchmark.
Raw logs, CPU fingerprint, saved bytes and trial JSON are in the `onehot-ntt-ci` artifact.
'''
  (root/'ci-report.md').write_text(report)
 (root/f'{name}-summary.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result),flush=True)
else:raise ValueError(sys.argv[1])
