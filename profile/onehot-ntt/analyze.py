from pathlib import Path
from collections import defaultdict
import hashlib,json,re,statistics,sys
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
elif sys.argv[1]=='verify':
 nv=sys.argv[3];name=sys.argv[4] if len(sys.argv)>4 else f'nv{nv}'
 log=(root/f'{name}.log').read_text();trials=[]
 assert ' INFO Done' in log
 for section in log.split('[experiment] trial=')[1:]:
  match=re.match(r'(\d+) onehot_commit=(\S+)(?: requested=(\S+))?',section)
  index,mode,requested=match.groups()
  def number(label): return float(re.search(r'^\[onehot_fp128\] '+label+r': ([\d.]+)',section,re.M)[1])
  assert '[onehot_fp128] verify multi threaded OK:' in section
  assert '[onehot_fp128] verify single threaded OK:' in section
  io=re.search(r'commit_io_read_bytes: (\d+)',section)
  trials.append({'trial':int(index),'mode':mode,'requested':requested,'commit_s':number('commit'),'prove_s':number('prove'),'storage_read_bytes':int(io[1]) if io else None,'both_verified':True})
 assert len(trials)>=2
 result={'nv':int(nv),'trials':trials}
 for ext in ['proof','commitment']:
  files=sorted(root.glob(f'{name}-*.{ext}'));assert len(files)==len(trials),(ext,len(files),len(trials))
  data=files[0].read_bytes();assert all(p.read_bytes()==data for p in files),ext
  result[ext+'_bytes']=len(data);result[ext+'_sha256']=hashlib.sha256(data).hexdigest();result[ext+'_all_identical']=True
 if nv=='38':
  # Historical hashes belong to the benchmark revision. Compare variants
  # within this run so policy updates on main can change the proof bytes.
  a=statistics.mean(x['commit_s'] for x in trials if x['mode']=='accumulation')
  repeated=[x for x in trials if x['requested']=='best']
  assert len(repeated)==2 and len({x['mode'] for x in repeated})==1,repeated
  n=statistics.mean(x['commit_s'] for x in repeated)
  result.update(accumulation_mean_s=a,best_mean_s=n,best_mode=repeated[0]['mode'],speedup=a/n,ntt_time_reduction_percent=100*(1-n/a))
 (root/f'{name}-summary.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result),flush=True)
else:raise ValueError(sys.argv[1])
