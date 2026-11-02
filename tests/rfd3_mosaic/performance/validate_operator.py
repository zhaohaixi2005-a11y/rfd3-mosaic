"""Deterministic-input regression against the frozen scalar implementation.

Run with --device cpu or, inside a GPU allocation, --device cuda.
Only finite masks/coordinates are part of the inference contract. No checkpoint
or untrusted serialized object is loaded by this test.
"""
import argparse,importlib.util,json,copy
from pathlib import Path
import torch
from rfd3.model.layers.blocks import PositionPairDistEmbedder,SinusoidalDistEmbed
from rfd3.model.layers.chunked_pairwise import ChunkedPairwiseEmbedder,ChunkedPositionPairDistEmbedder,ChunkedSinusoidalDistEmbed
p=Path(__file__).with_name('legacy_chunked_pairwise.py')
spec=importlib.util.spec_from_file_location('frozen_scalar_chunk',p);legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)

def run(device):
    torch.set_num_threads(2);torch.manual_seed(20261101);results=[]
    with torch.no_grad():
        for batch,n,k in [(1,0,0),(1,1,1),(2,17,5),(1,127,7),(2,128,9),(1,129,11),(2,257,13)]:
            for masks in ['empty','disjoint','partial','full']:
                for cached in [False,True]:
                    obj=ChunkedPairwiseEmbedder(16,ChunkedSinusoidalDistEmbed(SinusoidalDistEmbed(16).to(device)),ChunkedPositionPairDistEmbedder(PositionPairDistEmbedder(16).to(device)),process_single_l=torch.nn.Linear(128,16,bias=False).to(device),process_single_m=torch.nn.Linear(128,16,bias=False).to(device),process_z=torch.nn.Linear(128,16,bias=False).to(device),pair_mlp=torch.nn.Sequential(torch.nn.ReLU(),torch.nn.Linear(16,16,bias=False)).to(device))
                    coord=torch.zeros(n,dtype=torch.bool,device=device);seq=coord.clone()
                    if masks=='disjoint':coord[::2]=True;seq[1::2]=True
                    if masks=='partial':coord[::3]=True;seq[::5]=True
                    if masks=='full':coord[:]=True;seq[:]=True
                    f={'motif_pos':torch.randn(n,3,device=device),'ref_pos':torch.randn(n,3,device=device),'is_motif_atom_with_fixed_coord':coord,'is_motif_atom_with_fixed_seq':seq,'ref_space_uid':torch.arange(n,device=device)//3}
                    tokens=max(1,(n+2)//3);kw={'f':f,'indices':torch.randint(-3,max(n+3,4),(batch,n,k),device=device),'C_L':torch.randn(n,128,device=device),'Z_init_II':torch.randn(tokens,tokens,128,device=device),'tok_idx':torch.arange(n,device=device)//3}
                    if cached:obj.cache_static_projections(kw['C_L'],kw['Z_init_II'])
                    errors=[]
                    for phase in ['initial','moved','next_sample']:
                        if phase=='moved':f['motif_pos']=f['motif_pos']+torch.randn(n,3,device=device)*.4;f['ref_pos']=f['ref_pos']+torch.randn(n,3,device=device)*.2
                        if phase=='next_sample':
                            kw['C_L']=torch.randn_like(kw['C_L']);kw['Z_init_II']=torch.randn_like(kw['Z_init_II'])
                            if cached:obj.cache_static_projections(kw['C_L'],kw['Z_init_II'])
                        before={key:value.clone() for key,value in f.items()};old=legacy.ChunkedPairwiseEmbedder.forward_chunked(obj,**kw);new=obj.forward_chunked(**kw)
                        torch.testing.assert_close(old,new,rtol=2e-5,atol=2e-5)
                        for key,value in f.items():assert torch.equal(value,before[key]),key
                        assert torch.isfinite(new).all();errors.append(float((old-new).abs().max()) if old.numel() else 0.)
                    results.append({'batch':batch,'atoms':n,'keys':k,'masks':masks,'cached':cached,'errors':errors})
    return {'device':device,'passed':True,'cases':len(results),'phases_per_case':3,'maximum_error':max(max(x['errors']) for x in results),'cases_detail':results}
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--device',default='cpu');parser.add_argument('--output',required=True);a=parser.parse_args()
    result=run(a.device);Path(a.output).write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='cases_detail'}))
