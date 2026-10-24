"""Contract: exact duration identity vs numerical/continuous response evidence.
All current conditions retained. Log colorbars encode ppm and ordered target width.
Symlog axes show the actual linear-to-log threshold; unresolved cases have no y value.
"""
from pathlib import Path
import sys,json,hashlib
import numpy as np
R=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(R/'code/figures'))
import main_figures as g
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, LinearSegmentedColormap
from matplotlib.ticker import NullLocator
g.OUT.mkdir(parents=True,exist_ok=True)
scan=json.loads((R/'data/figures/parameter_scan.json').read_text())['cases']
ode=json.loads((R/'data/brownian_finite_rate.json').read_text())['rows']
def D4(z):return z*z*(4*z+7)/(40*(z+1)**2*(2*z+1)**2*(2*z+3)**2*(2*z*z+4*z+1)**2)
def L(z):return (14*z**3+37*z*z+26*z+5)/(5*(z+1)*(2*z+1)*(2*z*z+4*z+1))
def reset_gap(theta):
    a=1/(1+17*theta/18);b=a*theta;zp=(1+b)/a
    h=((5/18)*L(zp)+zp/2-5/18)/zp**2/a
    return (5/18)*a*D4(zp)/((a*5/18+b/4)*(1+b)**2*(1-b*h)**2)
W,H=183,166
fig=plt.figure(figsize=(W/25.4,H/25.4))
axes=[g.mm_axes(fig,box,W,H) for box in [(20,108,61,44),(108,108,61,44),(20,39,61,44),(108,39,61,44)]]
for ax,key in zip(axes,'abcd'):
    g.label(ax,key,'');ax.tick_params(labelsize=7.5)
def reset_axis(ax,threshold=True):
    ax.set_xscale('symlog',base=10,linthresh=1e-5,linscale=1)
    ax.set_xlim(0,170)
    ax.set_xticks([0,1e-5,1e-3,1e-1,10])
    ax.set_xticklabels(['0',r'$10^{-5}$',r'$10^{-3}$',r'$10^{-1}$',r'$10^1$'])
    ax.xaxis.set_minor_locator(NullLocator())
    if threshold: ax.axvline(1e-5,color=g.LIGHT,lw=.5,ls=':',zorder=0)
    # There is no axis break: symlog is continuous at its linear threshold.
ax=axes[0]
theta=np.r_[0,np.geomspace(1e-6,100,250)]
y=reset_gap(theta);assert np.all(y>0)
ax.plot(theta,y,color=g.INK)
nodes=np.array([0,.001,.01,.1,1,10]);ax.plot(nodes,reset_gap(nodes),'o',color=g.INK,ms=3)
reset_axis(ax);ax.set_yscale('log');ax.set_ylim(1e-23,5e-5)
ax.set_yticks([1e-6,1e-12,1e-18])
ax.set_xlabel(r'Reset strength $\theta$');ax.set_ylabel(r'Four-duration difference $\Delta J_4$')
ax.text(.20,.15,'Durations 1–3: exact match\nAt unit arguments: '+r'$\Delta J_4>0$',transform=ax.transAxes,fontsize=7.7)
ax=axes[1]
alphas=[1.25,1.5,1.75,1.9,1.99,1.999,2];gammas=[1,10,100,1000,10000]
mat=np.array([[next(z['relative_ppm_finest_cell'] for z in scan if z['alpha']==a and z['gamma']==v and z['width']==1/32 and z['theta']==0) for v in gammas] for a in alphas])
assert np.all(mat>0)
cmap=plt.colormaps['cividis'];norm=LogNorm(vmin=.2,vmax=200)
mesh=ax.pcolormesh(np.arange(6)-.5,np.arange(8)-.5,mat,cmap=cmap,norm=norm,shading='flat',rasterized=False)
ax.set_xlim(-.5,4.5);ax.set_ylim(6.5,-.5)
for i in range(7):
    for j in range(5):
        val=mat[i,j];rgb=cmap(norm(val))[:3];bright=np.dot(rgb,[.2126,.7152,.0722])
        ax.text(j,i,f'{val:.2f}' if val<10 else f'{val:.1f}',ha='center',va='center',fontsize=7.2,color=g.INK if bright>.52 else 'white')
ax.set_xticks(range(5));ax.set_xticklabels(['1','10',r'$10^2$',r'$10^3$',r'$10^4$'])
ax.set_yticks(range(7));ax.set_yticklabels(['1.25','1.5','1.75','1.9','1.99','1.999','2'])
ax.set_xlabel(r'Switching rate $\gamma$');ax.set_ylabel(r'Transport exponent $\alpha$');ax.tick_params(length=0)
cbax=g.mm_axes(fig,[172,108,2.5,44],W,H)
cb=fig.colorbar(mesh,cax=cbax,ticks=[.2,2,20,200]);cb.ax.set_yticklabels(['0.2','2','20','200'])
cb.ax.tick_params(labelsize=6.5,pad=1,length=2);cb.ax.minorticks_off();cb.outline.set_linewidth(.5)
cb.solids.set_rasterized(False)
cb.ax.text(.5,1.10,'ppm',ha='center',transform=cb.ax.transAxes,fontsize=7.3)
ax=axes[2];widths=[1/128,1/64,1/32,1/16,1/8,1/4]
wnorm=LogNorm(vmin=widths[0],vmax=widths[-1]);wcmap=LinearSegmentedColormap.from_list('width_purple_red',plt.colormaps['PuRd'](np.linspace(.30,.95,256)))
for w in widths:
    vals=[next(z for z in ode if z['width']==w and z['gamma']==v) for v in gammas]
    lo=np.array([z['relative_ppm_enclosure'][0] for z in vals]);hi=np.array([z['relative_ppm_enclosure'][1] for z in vals]);yy=(lo+hi)/2
    ax.errorbar(gammas,yy,yerr=[yy-lo,hi-yy],marker='o',ms=3,color=wcmap(wnorm(w)),lw=.85,capsize=1.5)
ax.set_xscale('log');ax.set_yscale('log');ax.set_ylim(.4,360);ax.set_xlim(.7,16000)
ax.set_xlabel(r'Switching rate $\gamma$');ax.set_ylabel('Relative mean difference (ppm)')
ax.set_xticks([1,100,10000]);ax.xaxis.set_minor_locator(NullLocator())
ax.annotate('192.66 ppm',xy=(1,192.66252),xytext=(1.8,255),textcoords='data',arrowprops=dict(arrowstyle='-',lw=.5,color=g.GRAY),fontsize=7.5)
wax=g.mm_axes(fig,[83,39,2.5,44],W,H)
wcb=fig.colorbar(plt.cm.ScalarMappable(norm=wnorm,cmap=wcmap),cax=wax,orientation='vertical',ticks=widths)
wcb.ax.set_yticklabels(['1/128','1/64','1/32','1/16','1/8','1/4']);wcb.ax.tick_params(labelsize=6.8,pad=1,length=2);wcb.ax.minorticks_off();wcb.outline.set_linewidth(.5)
wcb.solids.set_rasterized(False)
wcb.ax.text(.5,1.12,'w',transform=wcb.ax.transAxes,ha='center',fontsize=8)
ax=axes[3];unresolved=[];plotted=[]
for gamma,marker,style in [(100,'o','-'),(10000,'s','--')]:
    rows=sorted([z for z in scan if z['alpha']==1.999 and z['width']==1/32 and z['gamma']==gamma and z['theta'] in [0,1e-6,1e-4,.001,.01,.1,1,10,100]],key=lambda z:z['theta'])
    assert len(rows)==9
    good=[z for z in rows if z['grid_resolved_by_prespecified_rule']]
    unresolved.extend((z['theta'],gamma) for z in rows if not z['grid_resolved_by_prespecified_rule'])
    plotted.extend(good)
    ax.plot([z['theta'] for z in good],[z['relative_ppm_finest_cell'] for z in good],style,marker=marker,ms=3,color=g.INK,mfc='white' if gamma==10000 else g.INK,label=r'$\gamma='+('10^2' if gamma==100 else '10^4')+'$')
reset_axis(ax);ax.set_yscale('log');ax.set_ylim(1e-4,100)
ax.tick_params(axis='x',labelbottom=False,bottom=False);ax.spines['bottom'].set_visible(False)
ax.set_ylabel('Relative mean difference (ppm)',labelpad=1);ax.legend(loc='upper right',fontsize=7.5)
status=g.mm_axes(fig,[108,32,61,5],W,H);reset_axis(status,threshold=False)
status.set(ylim=(-.5,.5),yticks=[]);status.tick_params(labelsize=7.5);status.set_facecolor('#F2F4F5')
status.spines['left'].set_visible(False);status.spines['top'].set_visible(False)
for x in [10,100]:status.plot(x,0,'x',ms=4,color=g.INK)
status.text(.02,.5,'Unresolved at both rates',transform=status.transAxes,fontsize=6.7,va='center')
status.set_xlabel(r'Reset strength $\theta$',fontsize=8)
assert sorted(unresolved)==[(10,100),(10,10000),(100,100),(100,10000)]
g.save(fig,'fig4_family_response')
