"""Compare energy, gradients, full proposals, diagnostics and RNG to a git baseline.

CPU only. Baseline is read from local git objects; no network or model loading.
"""
import copy,importlib.util,json,random,sys,inspect,subprocess,tempfile
from pathlib import Path
from dataclasses import replace
from unittest.mock import patch
import numpy as np
import torch
from rfd3.inference.symmetry import scaffold_core_guidance as core, scaffold_guidance as pose, motif_mobility as motion
repo=Path(__file__).resolve().parents[3]
output=Path(sys.argv[1]);output.mkdir(parents=True,exist_ok=True)
baseline=sys.argv[2] if len(sys.argv)>2 else '1d997cc711219ed7f151db133a1e85c7b419cb99'
temporary=tempfile.TemporaryDirectory(prefix='mosaic-cache-regression-');root=Path(temporary.name)
for filename,module in [('baseline_core.py','scaffold_core_guidance.py'),('baseline_scaffold.py','scaffold_guidance.py')]:
 (root/filename).write_bytes(subprocess.check_output(['git','-C',str(repo),'show',baseline+':models/rfd3/src/rfd3/inference/symmetry/'+module]))
torch.set_num_threads(2)
def load(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
old=load('rfd3.inference.symmetry._cache_old_core',root/'baseline_core.py');oldpose=load('rfd3.inference.symmetry._cache_old_pose',root/'baseline_scaffold.py')
oldpose.ScaffoldGuidanceEnergy=pose.ScaffoldGuidanceEnergy
fixture=load('_cache_fixture',repo/'tests/rfd3_mosaic/unit/test_motif_mobility.py').MotifMobilityTestCase()
def state():return torch.get_rng_state().clone(),repr(np.random.get_state()),repr(random.getstate())
def check_state(a):
 b=state();assert torch.equal(a[0],b[0]) and a[1:]==b[1:]
def core_call(m,*a,**kw):
 fn=m.apply_scaffold_core_guidance;kw={k:v for k,v in kw.items() if k in inspect.signature(fn).parameters};return fn(*a,**kw)
rows=[];mutations=[];exceptions=[];motion_rows=[]
for dtype in (torch.float32,torch.float64):
 for n in (5,12,23):
  f={'atom_to_token_map':torch.arange(2*n),'asym_id':torch.arange(2).repeat_interleave(n),'residue_index':torch.arange(n).repeat(2),'is_ca':torch.ones(2*n,dtype=torch.bool),'is_protein':torch.ones(2*n,dtype=torch.bool)}
  fixed=torch.zeros(2*n,dtype=torch.bool);fixed[[0,n-1,n,2*n-1]]=True;top=core.build_scaffold_core_topology(f,fixed)
  cfg=core.ScaffoldCoreGuidanceConfig(intra_chain_weight=1.,sequence_separation=3)
  torch.manual_seed(29+n);x=torch.randn(1,2*n,3,dtype=dtype)*.2;x[0,:,0]+=torch.arange(n).repeat(2)*3.8;x[0,n:,1]+=7
  for mode in ('cold','warm','residue_mutation','mask_data_mutation','index_mutation','threshold_change'):
   if mode=='residue_mutation':top.chains[0].residue_indices[1]+=2
   if mode=='mask_data_mutation':top.chains[0].generated_ca_mask.data[1]=False
   if mode=='index_mutation':top.chains[0].ca_atom_indices[1:3]=top.chains[0].ca_atom_indices[1:3].flip(0)
   if mode=='threshold_change':cfg=replace(cfg,sequence_separation=4)
   a=x[0].clone().requires_grad_(True);b=x[0].clone().requires_grad_(True)
   ea=old.scaffold_core_energy(a,top,cfg);eb=core.scaffold_core_energy(b,top,cfg)
   assert ea.detached_dict()==eb.detached_dict(),mode
   ea.total.backward();eb.total.backward();assert torch.equal(a.grad,b.grad),mode
   mutations.append({'dtype':str(dtype),'n':n,'case':mode,'energy_gradient_equal':True})
  for diagnostic in (False,True):
   a=x.clone();b=x.clone()
   for progress in (.02,.2,.5,.8,.95):
    rng=state();a,da=core_call(old,a,top,progress=progress,config=cfg,record_rejected_trial_energies=diagnostic);b,db=core_call(core,b,top,progress=progress,config=cfg,record_rejected_trial_energies=diagnostic)
    assert torch.equal(a,b) and da==db;check_state(rng);rows.append({'dtype':str(dtype),'n':n,'progress':progress,'diag':diagnostic,'same':True})
  for bad in (x[0],x.repeat(2,1,1)):
   e=[]
   for m in (old,core):
    try:core_call(m,bad,top,progress=.5,config=cfg)
    except Exception as err:e.append((type(err).__name__,str(err)))
   assert len(e)==2 and e[0]==e[1];exceptions.append(e[0])
pose_rows=[]
for seed in (17,29):
 xyz,transforms,controller,top,axis,cfg,it,ic,features=fixture._joint_packing_mobility_case()
 a=copy.deepcopy(controller);b=copy.deepcopy(controller)
 for progress in (.15,.3,.5,.7):
  kwargs=dict(progress=progress,topology=top,axis=axis,principal_axes=(axis.direction,),config=cfg,apply_update=True,proposal_selection_seed=seed,record_objective_gradients=False)
  kwargs={k:v for k,v in kwargs.items() if k in inspect.signature(b.update_orbits_from_scaffold).parameters}
  rng=state()
  with patch.object(motion,'scaffold_orbit_energy',oldpose.scaffold_orbit_energy):outa=a.update_orbits_from_scaffold(xyz,**kwargs)
  outb=b.update_orbits_from_scaffold(xyz,**kwargs)
  assert torch.equal(outa,outb) and a.diagnostics()==b.diagnostics();check_state(rng)
  motion_rows.append({'seed':seed,'progress':progress,'committed':b.last_update_applied,'same':True})
 for case in ('warm','junction_data','fixed_indices','generated_indices'):
  if case=='junction_data':top.junction_pairs.data[0,1]=top.generated_ca_atom_indices[-1]
  if case=='fixed_indices':top.fixed_ca_atom_indices[:2]=top.fixed_ca_atom_indices[:2].flip(0)
  if case=='generated_indices':top.generated_ca_atom_indices[:2]=top.generated_ca_atom_indices[:2].flip(0)
  u=xyz[0].detach().clone().requires_grad_(True);v=u.detach().clone().requires_grad_(True)
  ea=oldpose.scaffold_orbit_energy(u,xyz,top,axis,principal_axis=axis.direction,config=cfg);eb=pose.scaffold_orbit_energy(v,xyz,top,axis,principal_axis=axis.direction,config=cfg)
  assert ea.detached_dict()==eb.detached_dict();ea.total.backward();eb.total.backward();assert torch.equal(u.grad,v.grad)
  pose_rows.append({'seed':seed,'case':case,'same_energy_gradient':True})
# Accepted control and invalid projector behavior, beyond rejected proposals.
f={'atom_to_token_map':torch.arange(10),'asym_id':torch.arange(2).repeat_interleave(5),'residue_index':torch.arange(5).repeat(2),'is_ca':torch.ones(10,dtype=torch.bool),'is_protein':torch.ones(10,dtype=torch.bool)}
fixed=torch.tensor([True,False,False,False,True]*2);top=core.build_scaffold_core_topology(f,fixed)
x=torch.tensor([[[0.,0.,0.],[2.,8.,0.],[4.,8.,0.],[6.,8.,0.],[8.,0.,0.],[0.,10.,0.],[2.,10.,0.],[4.,10.,0.],[6.,10.,0.],[8.,10.,0.]]])
cfg=core.ScaffoldCoreGuidanceConfig(routing_ownership_weight=1.,clash_weight=0.,continuity_weight=0.,maximum_token_step=.4)
a,da=old.apply_scaffold_core_guidance(x,top,progress=.5,config=cfg);b,db=core.apply_scaffold_core_guidance(x,top,progress=.5,config=cfg)
assert db['accepted'] and torch.equal(a,b) and da==db
projector_checks=[]
for kind in ('shape','integer','nan','inf','raises'):
 def project(x):
  if kind=='shape':return x[:,:1]
  if kind=='integer':return x.to(torch.int64)
  if kind=='nan':return x*float('nan')
  if kind=='inf':return x*float('inf')
  raise ValueError('diagnostic projector error')
 outcomes=[]
 for m in (old,core):
  try:
   value,info=m.apply_scaffold_core_guidance(x,top,progress=.5,config=cfg,projector=project)
   outcomes.append(('returned',bool(torch.equal(value,x)),repr(info)))
  except Exception as e:outcomes.append((type(e).__name__,str(e)))
 assert outcomes[0]==outcomes[1],kind
 projector_checks.append({'case':kind,'same':True})
report={'accepted_control':True,'invalid_projectors':projector_checks,'core_proposal_steps':rows,'core_energy_gradient_cases':mutations,'motion_steps':motion_rows,'boundary_energy_gradient_cases':pose_rows,'invalid_shape_exceptions':exceptions,'CPU_only':True}
(output/'equivalence.json').write_text(json.dumps(report,indent=2));print({k:len(v) for k,v in report.items() if isinstance(v,list)})
