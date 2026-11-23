#!/usr/bin/env python3
"""Generates the lens artwork in lens-tryon-deploy/lenses/ (RGBA PNG: fibres, limbal ring, transparent pupil hole).
The pupil hole edge sits at HOLE (fraction of the lens radius); index.html stretches it to the wearer's real pupil."""
import numpy as np, os, sys
from PIL import Image
N=320; HOLE=.34
OUT=os.path.join(os.path.dirname(__file__),"..","lens-tryon-deploy","lenses")
LENSES=[  # name, inner (collarette), mid, outer, ring, fibre contrast, opacity
 ("hazel",     (206,150,62),(150,128,60),(92,112,62),(38,32,18),.9,.9),
 ("ocean-blue",(168,208,236),(96,158,216),(38,98,172),(14,28,62),.85,.92),
 ("emerald",   (188,196,96),(96,170,90),(28,124,72),(10,42,26),.85,.92),
 ("grey",      (214,214,210),(168,176,184),(112,122,134),(40,45,52),.8,.92),
 ("amethyst",  (206,180,232),(160,128,206),(104,72,166),(38,22,60),.85,.92),
 ("honey",     (244,196,96),(214,150,62),(172,112,42),(62,36,10),.85,.9),
 ("turquoise", (170,232,214),(84,196,196),(22,150,162),(10,50,62),.85,.92),
 ("lime",      (214,232,110),(160,206,70),(96,170,40),(24,48,10),.85,.92),
 ("crimson",   (240,70,60),(214,36,40),(150,12,24),(28,0,0),1,1),
 ("moon-white",(250,250,252),(240,242,246),(222,226,232),(110,112,124),.5,1),
]
def sm(a,b,x): t=np.clip((x-a)/(b-a),0,1); return t*t*(3-2*t)
def ang_noise(rng,theta,freq,r=None,twist=0):
    """Smooth periodic noise along the angle; optional slow drift with radius so fibres are not perfectly straight."""
    tab=rng.random(freq); t=(theta/(2*np.pi)+.5)
    if r is not None: t=t+twist*r
    t=(t%1)*freq; i=np.floor(t).astype(int); f=t-i; w=(1-np.cos(f*np.pi))*.5
    return tab[i%freq]*(1-w)+tab[(i+1)%freq]*w
def make(spec,seed):
    name,inner,mid,outer,ring,fib,opac=spec; rng=np.random.default_rng(seed)
    y,x=np.mgrid[0:N,0:N]; a=(x+.5)/N*2-1; b=(y+.5)/N*2-1; r=np.hypot(a,b); th=np.arctan2(b,a)
    # fibres: fine radial streaks at several widths, slightly wavy
    wav=.0035*np.sin(r*11+ang_noise(rng,th,9)*6)
    f=(.34*ang_noise(rng,th+wav*6,260,r,.02)+.26*ang_noise(rng,th,520,r,-.015)+.22*ang_noise(rng,th,90,r,.06)+.18*ang_noise(rng,th,28,r,.1))
    f=np.clip((f-.5)*1.5+.5,0,1)
    crypt=sm(.6,.9,ang_noise(rng,th,40,r,.3))*sm(.45,.6,r)*(1-sm(.72,.88,r))
    # colour zones with ragged borders (the collarette reaches out in spikes)
    spike=ang_noise(rng,th,64)*.5+ang_noise(rng,th,170)*.5
    k1=sm(.40+.14*spike,.56+.2*spike,r); k2=sm(.6,.9,r+.06*(f-.5))
    col=np.zeros((N,N,3))
    for c in range(3):
        v=inner[c]+(mid[c]-inner[c])*k1; v=v+(outer[c]-v)*k2
        col[...,c]=v
    col*=(1+(f-.5)*.95*fib-crypt*.25*fib)[...,None]
    # limbal ring: dark band whose inner edge breaks into fine teeth, like a printed lens
    teeth=ang_noise(rng,th,300)*.6+ang_noise(rng,th,75)*.4
    rg=sm(.80-.10*teeth,.95,r)*.85
    for c in range(3): col[...,c]=col[...,c]+(ring[c]-col[...,c])*rg
    col*=(.95+.1*rng.random((N,N)))[...,None]
    # alpha: clear pupil hole with a feathered, slightly irregular edge; printed-dot fade at the rim
    jag=ang_noise(rng,th,48)*.6+ang_noise(rng,th,150)*.4
    al=sm(HOLE-.02+.05*jag,HOLE+.12+.08*jag,r)*(.84+.16*f if opac<1 else 1)*opac
    al*=1-sm(.94,1.0,r)
    al[r>1]=0
    img=np.dstack([np.clip(col,0,255),al*255]).astype(np.uint8)
    Image.fromarray(img,"RGBA").save(os.path.join(OUT,name+".png"),optimize=True)
os.makedirs(OUT,exist_ok=True)
for i,s in enumerate(LENSES): make(s,1000+i*17)
