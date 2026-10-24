"""Current range map plus the complete original 1558-cell enclosures."""
import main_figures as g
import numpy as np
from fractions import Fraction
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def figS7():
    W,H=183,145;fig=plt.figure(figsize=(W/25.4,H/25.4))
    boxes=[[18,85,69,43],[108,85,69,43],[18,18,69,43],[108,18,69,43]]
    axes=[g.mm_axes(fig,b,W,H) for b in boxes]
    a=axes[0];g.label(a,'a','Current certified ranges')
    colors=['#3B6FB6','#8C5BA6','#009E73'];starts=[1.99651,1.9,1.995055]
    for y,lower,name,color in zip([3,2,1],starts,g.NAMES,colors):
        a.plot([lower,2],[y,y],lw=3,color=color,solid_capstyle='butt')
        a.scatter([lower],[y],s=13,color=color,zorder=4)
        a.text(1.035,y+.16,r'$\alpha\geq'+str(lower)+'$',fontsize=8,color=color)
    a.set(xlim=(1,2.035),xticks=[1,1.5,1.9,2],ylim=(.55,3.55),yticks=[1,2,3],
          yticklabels=list(reversed(g.NAMES)),xlabel=r'$\alpha$')
    output=[]
    for i,(ax,pair,name) in enumerate(zip(axes[1:],g.IDS,g.NAMES)):
        g.label(ax,'bcd'[i],name+' original near-Brownian bounds')
        src=g.load('raw/round2/interval_'+pair+'_95.json');rows=list(reversed(src['rows']))
        edges=[1000*(2-float(rows[0]['alpha_hi']))]
        lo=[];hi=[]
        for r in rows:
            edges.append(1000*(2-float(r['alpha_lo'])));lo.append(r['gap'][0]*1e4);hi.append(r['gap'][1]*1e4)
        ax.fill_between(edges,lo+[lo[-1]],hi+[hi[-1]],step='post',color='#DDE2E6',lw=0)
        ax.step(edges,lo+[lo[-1]],where='post',color=g.GRAY,lw=.55)
        ax.step(edges,hi+[hi[-1]],where='post',color=g.GRAY,lw=.55)
        # Endpoints are separately enclosed in the original source file.
        aa=src['A_metadata']['brownian_enclosure'];bb=src['B_metadata']['brownian_enclosure']
        brown=[float(aa[0])-float(bb[1]),float(aa[1])-float(bb[0])]
        ax.plot([0],[sum(brown)*.5e4],marker='o',ms=4,mfc='white',mec=g.ORTH,mew=.8,zorder=5)
        ax.axhline(0,color=g.INK,lw=.5)
        ax.set(xlim=(-.06,float(edges[-1])*1.03),xlabel=r'$2-\alpha$  ($10^{-3}$)',
               ylabel=r'$\Delta\ell_\alpha$  ($10^{-4}$)')
        ax.yaxis.set_major_locator(MaxNLocator(4))
        m=min(abs(r['gap'][1 if i==0 else 0]) for r in rows)
        ax.text(.04,.85 if i==0 else .08,'All '+str(len(rows))+' original cells',transform=ax.transAxes,fontsize=8)
        output.append({'pair':pair,'rows':src['rows'],'count':len(rows)})
    g.save(fig,'figS7_full_coefficient_enclosures')
    g.dump('figS7_source.json',{'current_lower_endpoints':starts,'original_bands':output})
if __name__=='__main__':
    g.OUT.mkdir(exist_ok=True,parents=True)
    g.fig2();figS7()
