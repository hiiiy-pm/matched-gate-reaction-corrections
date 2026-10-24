"""current Figures 1--3 and S4, with explicit inherited/fresh evidence provenance."""
from pathlib import Path
from decimal import Decimal, localcontext, ROUND_FLOOR, ROUND_CEILING
from fractions import Fraction
import csv
import hashlib
import json
import sys

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, FancyArrowPatch, Polygon
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.patches import Patch
from matplotlib.collections import Collection, QuadMesh
from matplotlib.ticker import NullLocator, MaxNLocator
from matplotlib import transforms

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
DATA=ROOT/'data/figures'
OUT=ROOT/'figures'
A, B = '#0072B2', '#D55E00'
INK, GRAY, LIGHT, PALE = '#202830', '#737B82', '#CCD1D5', '#EFF2F4'
SAME, DIFFERENT = '#E2F1EC', '#202830'
ORTH = '#8064A2'
IDS = ['C18_0064', 'C18_0189', 'C20_0201']
NAMES, MARKERS = ['Pair I', 'Pair II', 'Pair III'], ['o', 's', '^']
mpl.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8.3,
    'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 7.5,
    'mathtext.fontset': 'custom', 'mathtext.rm': 'Arial',
    'mathtext.sf': 'Arial', 'mathtext.it': 'Arial:italic', 'mathtext.bf': 'Arial:bold',
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': .5, 'lines.linewidth': .85,
    'xtick.major.width': .6, 'ytick.major.width': .6,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'text.color': INK, 'axes.labelcolor': INK, 'axes.edgecolor': INK,
    'xtick.color': INK, 'ytick.color': INK, 'legend.frameon': False,
    'pdf.fonttype': 42, 'svg.fonttype': 'none',
    'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.dpi': 600,
})

def load(name):
    return json.loads((DATA / name).read_text(encoding='utf-8'))

def dump(name, obj):
    (DATA / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

def table(name):
    with (DATA / name).open(encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def label(ax, letter, title, pad=7):
    trans = transforms.ScaledTranslation(-14/72, pad/72, ax.figure.dpi_scale_trans)
    ax.text(0, 1, letter, transform=ax.transAxes+trans, fontsize=9,
            fontweight='bold', va='bottom', clip_on=False)
    ax.set_title(title, loc='left', pad=pad, fontweight='normal')

def mm_axes(fig, box, width, height):
    x, y, w, h = box
    return fig.add_axes([x/width, y/height, w/width, h/height])

def canvas(fig, box, width, height, letter, title):
    ax = mm_axes(fig, box, width, height)
    ax.set(xlim=(0, box[2]), ylim=(0, box[3]))
    ax.axis('off')
    label(ax, letter, title)
    return ax

def prepare_figure(fig):
    """Keep scientific values fixed while making lettering legible at print size."""
    fig.canvas.draw()
    for text in fig.findobj(Text):
        if text.get_text():
            text.set_text(text.get_text().replace('neighbour', 'neighbor').replace('labelled', 'labeled'))
            text.set_fontsize(max(9.0, text.get_fontsize()))
            if text.get_text() in tuple('abcdef') and text.get_fontweight() in ('bold', 700):
                text.set_text('(' + text.get_text() + ')')
    for line in fig.findobj(Line2D):
        if line.get_linewidth() > 0:
            line.set_linewidth(max(.6, line.get_linewidth()))
        if line.get_marker() in ('o', 's', '^', 'v', '<', '>', 'D', 'd', 'p', 'h', 'H', '*'):
            line.set_markersize(max(3.2, line.get_markersize()))
        if line.get_markeredgewidth() > 0:
            line.set_markeredgewidth(max(.6, line.get_markeredgewidth()))
    for patch in fig.findobj(Patch):
        if patch.get_linewidth() > 0:
            patch.set_linewidth(max(.6, patch.get_linewidth()))
    for collection in fig.findobj(Collection):
        collection.set_rasterized(False)
        if not isinstance(collection, QuadMesh):
            widths = np.asarray(collection.get_linewidths())
            collection.set_linewidths(np.where(widths > 0, np.maximum(widths, .6), widths))
    fig.canvas.draw()


def save(fig, stem):
    prepare_figure(fig)
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT/(stem+'.pdf'),facecolor='white')
    plt.close(fig)

def certificate_rows():
    out = []
    for fn in ['current_rate_certificate.json', 'second_rate_certificate.json']:
        src = load('certificates/' + fn)
        assert src['all_one_percent']
        out.extend(src['rows'])
    return out

def outward(iv):
    return (np.nextafter(float(iv['lower_decimal']), -np.inf),
            np.nextafter(float(iv['upper_decimal']), np.inf))


def fraction_interval(iv):
    return [Fraction(iv['lower_decimal']), Fraction(iv['upper_decimal'])]

def ratio_interval(numerator, denominator, scale=1, shift=0):
    assert denominator[0] * denominator[1] > 0
    vals = [(x/y-shift)*scale for x in numerator for y in denominator]
    return [min(vals), max(vals)]

def decimal_outward(vals, digits=5):
    scale = 10**digits
    lo = vals[0].numerator*scale // vals[0].denominator
    hi = -(-vals[1].numerator*scale // vals[1].denominator)
    return [f'{lo/scale:.{digits}f}', f'{hi/scale:.{digits}f}']

def ppm_interval(row):
    t = row['models']
    mean = [(Fraction(t['A']['tau_interval'][k]) + Fraction(t['B']['tau_interval'][k]))/2
            for k in ['lower_decimal', 'upper_decimal']]
    iv = ratio_interval(fraction_interval(row['delta_tau_interval']), mean, 10**6)
    return decimal_outward(iv), [str(x) for x in iv]

def checkmark(ax,x,y,color='white',lw=1.8):
    ax.plot([x-1.7,x-.4,x+2.1],[y,y-1.5,y+1.9],color=color,lw=lw,
            solid_capstyle='round',zorder=5)

def fig1():
    d = load('pair_ii_diagonal_duration_exact.json')
    masks = {k: set(v) for k,v in d['masks'].items()}
    changed = masks['A'] ^ masks['B']
    assert changed == {5,7,8,10}
    exact = {r['id']:r for r in load('core_fig1_source_data.json')['all_seven_exact_records']}
    assert exact[IDS[0]]['joint11_gap']=='2/5445'
    assert exact[IDS[2]]['joint11_gap']=='13/6534'
    row = next(r for r in certificate_rows() if r['pair']==IDS[1] and r['gamma']==2560000)
    ppm, exact_ppm = ppm_interval(row)
    W,H = 183,154
    fig = plt.figure(figsize=(W/25.4,H/25.4))
    a = canvas(fig,[8,107,170,36],W,H,'a','')
    theta = np.pi/2 - np.arange(18)*2*np.pi/18
    for gate,cx,color in [('A',45,A),('B',126,B)]:
        cy,radius=19,12.7
        xy=np.c_[cx+radius*np.cos(theta),cy+radius*np.sin(theta)]
        a.plot(np.r_[xy[:,0],xy[0,0]],np.r_[xy[:,1],xy[0,1]],color=LIGHT,lw=.7)
        for j,(x,y) in enumerate(xy):
            a.add_patch(Circle((x,y),1.1,facecolor=color if j in masks[gate] else 'white',
                               edgecolor=color,lw=.8,zorder=3))
            if j in changed:
                a.add_patch(Circle((x,y),1.8,facecolor='none',edgecolor=INK,lw=.7,zorder=2))
            if j in changed or j % 3 == 0:
                a.text(cx+18.8*np.cos(theta[j]),cy+18.8*np.sin(theta[j]),str(j),
                       ha='center',va='center',fontsize=8.5,fontweight='bold' if j in changed else 'normal')
        a.text(cx,cy+1.8,'Mask '+gate,ha='center',fontsize=9,fontweight='bold',color=color)
        a.text(cx,cy-3,r'$p=1/2$',ha='center',fontsize=8.5)
    a.text(85,40,'Pair II   |   Same Q, particle and target',ha='center',fontsize=8.5)
    a.text(85,-5,'A only: 5, 8     /     B only: 7, 10',ha='center',fontsize=8)
    a.text(0,25,'Open',fontsize=7.5)
    a.add_patch(Circle((2,21),1.1,facecolor=GRAY,edgecolor=GRAY,lw=.7))
    a.text(0,15,'Closed',fontsize=7.5)
    a.add_patch(Circle((2,11),1.1,facecolor='white',edgecolor=GRAY,lw=.7))
    b = canvas(fig,[8,54,170,32],W,H,'b','')
    xs=[33,49,65,85,101,117,133,156]
    heads=[r'$C_2$',r'$C_3$',r'$C_4$','1','2','3','4',r'$\Delta\ell_\alpha$']
    b.text(49,43,'Fixed clock times',ha='center',fontsize=8)
    b.text(109,43,'Consecutive complete durations',ha='center',fontsize=8)
    b.text(156,43,'Response',ha='center',fontsize=8)
    for x,h in zip(xs,heads):
        b.text(x,36,h,ha='center',fontsize=8.5)
    b.plot([25,73],[40,40],color=LIGHT,lw=.5,clip_on=False)
    b.plot([77,141],[40,40],color=LIGHT,lw=.5,clip_on=False)
    matched = [[True,True,False,True,False,False,False],
               [True,True,False,True,True,True,False],
               [True,True,False,True,False,False,False]]
    witnesses=[{4:'2/5445'}, {6:'11/1764000'}, {4:'13/6534'}]
    for i,(name,y) in enumerate(zip(NAMES,[27,15.8,4.7])):
        b.text(0,y,name,ha='left',va='center',fontsize=8.5,fontweight='bold' if i==1 else 'normal')
        if i==1:
            b.add_patch(Rectangle((23.5,y-5.5),145,11,fill=False,edgecolor=INK,lw=.65))
        for j,x in enumerate(xs[:-1]):
            same=matched[i][j]
            implied=i in [0,2] and j in [5,6]
            b.add_patch(Rectangle((x-7.2,y-4.7),14.4,9.4,facecolor=SAME if same else (PALE if implied else DIFFERENT),lw=0))
            if same:
                checkmark(b,x,y+.3,color='#447869',lw=1.3)
            else:
                b.text(x,y,'→' if implied else '≠',ha='center',va='center',
                       fontsize=13,fontweight='bold',color=GRAY if implied else 'white')

        b.text(xs[-1],y,'< 0' if i==0 else '> 0',ha='center',va='center',fontsize=10.5,fontweight='bold')
    b.text(0,-6.5,'First contrasts:  I  2/5445     II  11/1764000     III  13/6534',fontsize=8.5)
    b.text(0,-11.5,'Ticks: exact match     ≠ direct difference     → implied by marginalization',fontsize=7.8)
    c = mm_axes(fig,[17,10,47,23],W,H)
    label(c,'c','')
    cov=[7/450,-7/1800]
    c.bar([0,1],cov,width=.44,color=[A,B],zorder=3)
    c.axhline(0,color=INK,lw=.5,zorder=4)
    c.set(xlim=(-.65,1.65),ylim=(-.012,.021),xticks=[0,1],xticklabels=['A','B'],
          yticks=[0,.01,.02],ylabel=r'$\mathrm{Cov}(T_1,T_4)$')
    c.text(0,cov[0]+.0015,'7/450',ha='center',fontsize=8,color=A)
    c.text(1,cov[1]-.0016,'−7/1800',ha='center',va='top',fontsize=8,color=B)
    c.tick_params(labelsize=7.5)
    d_ax=mm_axes(fig,[105,10,65,23],W,H)
    label(d_ax,'d','')
    lo,hi=map(float,ppm);mid=(lo+hi)/2
    d_ax.axvline(0,color=LIGHT,lw=.6)
    d_ax.errorbar([mid],[.45],xerr=[[mid-lo],[hi-mid]],fmt='none',color=ORTH,capsize=4,elinewidth=1.5)
    d_ax.set(xlim=(-.002,.07),ylim=(0,1),yticks=[],xticks=[0,.02,.04,.06],xlabel='Certified relative mean gap (ppm)')
    d_ax.spines['left'].set_visible(False)
    d_ax.text(.5,.83,ppm[0]+'–'+ppm[1]+' ppm',transform=d_ax.transAxes,ha='center',fontsize=9,fontweight='bold')
    d_ax.text(.5,1.12,'Pair II · α = 1.999 · γ = 2.56 × 10⁶',transform=d_ax.transAxes,ha='center',fontsize=7.5)
    save(fig,'fig1_observation_hierarchy')
    dump('fig1_source.json',{'masks':d['masks'],'changed_sites':sorted(changed),
        'matrix_matches':matched,'matrix_columns':['C2','C3','C4','duration1','duration2','duration3','duration4'],
        'matrix_numeric_witnesses':witnesses,
        'C4_derivative_orders':[[1,1,3],[1,1,5],[1,1,2]],
        'covariance':['7/450','-7/1800'],
        'ppm_definition':'1e6*(tau_A-tau_B)/((tau_A+tau_B)/2)',
        'ppm_display_outward':ppm,'ppm_exact_rational_enclosure':exact_ppm,
        'finite_rate_certificate':row,
        'scope':'Pair II duration identities: either stationary entrance. Pair I/III nonzero adjacent on/off probes imply higher-order mismatch by marginalization.'})

def amplitude_data():
    rows = table('figS4_amplitude_convergence_all93.csv')
    assert len(rows)==93
    best = {}
    for pair in IDS:
        rr = [r for r in rows if r['pair']==pair]
        assert len(rr)==31
        best[pair] = []
        for rate in sorted({float(r['gamma']) for r in rr}):
            x = max((r for r in rr if float(r['gamma'])==rate and r['method']=='cell'),
                    key=lambda r:int(r['n']))
            best[pair].append(x)
        for r in rr:
            assert np.isclose(float(r['normalized_gap']), float(r['gap'])/float(r['epsilon']), rtol=3e-14)
    return rows, best


def fig2():
    raw,best=amplitude_data()
    certs=certificate_rows()
    W,H=183,137
    fig=plt.figure(figsize=(W/25.4,H/25.4))
    axes=[];bands=[];selected={};displayed_cert=[]
    for i,(pair,name) in enumerate(zip(IDS,NAMES)):
        a=mm_axes(fig,[16+i*59,91,43,33],W,H);axes.append(a)
        label(a,'abc'[i],'')
        if i==1:
            extended=load('continuous_pairII.json')['rows']
            edges=[float(Fraction(r['alpha_lo'])) for r in extended]+[2.]
            lower=[r['lower_bound']*1e4 for r in extended]
            a.step(edges,lower+[lower[-1]],where='post',color=A,lw=1.2,zorder=4)
            a.plot([r['anchor'] for r in extended],[r['nominal_galerkin_contrast']*1e4 for r in extended],
                   color=GRAY,ls='--',lw=.9,zorder=3)
            point=next(r for r in load('table1_source.json')['rows'] if r['pair']==pair)
            plo,phi=[float(Fraction(v))*1e4 for v in point['coefficient_gap_rational']]
            pm=(plo+phi)/2
            a.errorbar([1.999],[pm],yerr=[[pm-plo],[phi-pm]],fmt='none',color=ORTH,capsize=3,elinewidth=1.3,zorder=6)
            a.axhline(0,color=INK,lw=.5)
            a.set(xlim=(1.899,2.001),xticks=[1.9,1.95,2],xlabel=r'$\alpha$',
                  ylim=(-.1,1.0),yticks=[0,.3,.6,.9],ylabel=r'$\Delta\ell_\alpha$  ($10^{-4}$)')
            a.tick_params(labelsize=8.8)
            a.text(.05,.89,'Pair II',transform=a.transAxes,fontsize=8.5,fontweight='bold')
            a.text(.05,1.09,r'Certified: $1.9\leq\alpha<2$',transform=a.transAxes,fontsize=8,va='bottom')
            a.text(.05,.43,r'$m_{\min}>2.72\times10^{-5}$',transform=a.transAxes,fontsize=7.7)
            bands.append({'pair':pair,'segments':20,'rows':extended,'minimum_bound_display_lower':'2.72e-5',
                          'bound_type':'ONE_SIDED_LOWER','nominal_curve_status':'NOT_CERTIFIED'})
            continue
        src=json.loads((DATA/f'raw/round2/interval_{pair}_95.json').read_text())
        rows=list(reversed(src['rows']))
        # One path with exact step edges removes rendering seams without smoothing bounds.
        edges=[];low=[];high=[]
        for k,r in enumerate(rows):
            xl=Decimal(2)-Decimal(r['alpha_hi']);xr=Decimal(2)-Decimal(r['alpha_lo'])
            if k:
                assert xl==Decimal(2)-Decimal(rows[k-1]['alpha_lo'])
            else:
                edges.append(float(1000*xl))
            edges.append(float(1000*xr))
            l,u=r['gap']
            assert l<=u and (u<0 if i==0 else l>0)
            low.append(l*1e4);high.append(u*1e4)
        a.fill_between(edges,low+[low[-1]],high+[high[-1]],step='post',
                       color='#DDE2E6',linewidth=0,zorder=2)
        a.step(edges,low+[low[-1]],where='post',color=GRAY,lw=.5,zorder=3)
        a.step(edges,high+[high[-1]],where='post',color=GRAY,lw=.5,zorder=3)
        point=next(r for r in load('table1_source.json')['rows'] if r['pair']==pair)
        plo,phi=[float(Fraction(v))*1e4 for v in point['coefficient_gap_rational']]
        pm=(plo+phi)/2;px=1000*(2-float(point['alpha']))
        a.errorbar([px],[pm],yerr=[[pm-plo],[phi-pm]],fmt='none',color='#8C5BA6',capsize=3,elinewidth=1.3,zorder=6)
        a.axhline(0,color=INK,lw=.5)
        a.axvline(edges[-1],color=GRAY,lw=.5,ls=(0,(2,2)),zorder=1)
        a.set(xlim=(0,5.1),xticks=[0,2,4],xlabel=r'$2-\alpha$  ($10^{-3}$)',
              ylim=[(-6.3,1.5),(-.17,1.5),(-1.3,12.8)][i],
              ylabel=r'$\Delta\ell_\alpha$  ($10^{-4}$)')
        a.tick_params(labelsize=8.8)
        a.yaxis.set_major_locator(MaxNLocator(4))
        a.text(.05,.12 if i==0 else .90,name,transform=a.transAxes,fontsize=8.5,fontweight='bold')
        a.text(.05,1.09,r'Limit: $\alpha\geq'+src['lower'].rstrip('0')+'$',
               transform=a.transAxes,fontsize=8,va='bottom')
        bound=np.array([-u for u in high] if i==0 else low)
        k=int(np.argmin(bound));m=bound[k]/1e4
        m_exact=abs(Fraction.from_float(rows[k]['gap'][1 if i==0 else 0]))
        exponent=Decimal(str(float(m_exact))).adjusted()
        power=Fraction(10)**exponent
        mantissa=decimal_outward([m_exact/power,m_exact/power],3)[0]
        nearest=high[k] if i==0 else low[k]
        x=.5*(edges[k]+edges[k+1])
        # The tiny vertical mark has its true data length; the printed bound rounds down.
        a.annotate(r'$m_{\min}\geq'+mantissa+r'\times10^{'+str(exponent)+'}$',
                   xy=(x,nearest),xycoords='data',xytext=(.05 if i!=1 else .98,.72 if i==2 else (.90 if i==0 else .58)),
                   textcoords='axes fraction',fontsize=7.7,ha='left' if i!=1 else 'right',
                   )
        a.annotate('',xy=(x,nearest),xytext=(x,0),
                   arrowprops={'arrowstyle':'<->','lw':.5,'color':INK,
                               'mutation_scale':2,'shrinkA':0,'shrinkB':0})
        bands.append({'pair':pair,'segments':src['segments'],'rows':src['rows'],
                      'minimum_absolute_bound':m,'minimum_bound_exact':str(m_exact),
                      'minimum_bound_display_lower':mantissa+'e'+str(exponent),
                      'polygon_edges_scaled':edges})
    fig.text(16/W,77/H,'a, c: shaded enclosures.  b: lower-bound steps and a noncertified dashed estimate.',fontsize=7.7)
    fig.text(16/W,71/H,'Purple capped bars: separate point certificates. Horizontal scales differ.',fontsize=7.7)
    d=mm_axes(fig,[16,17,161,47],W,H);axes.append(d)
    label(d,'d','')
    d.axhspan(-1,1,color='#F0F3F5',zorder=0)
    for y in [-1,0,1]:
        d.axhline(y,color=LIGHT if y else INK,lw=.5,ls=(0,(3,2)) if y else '-')
    shades=['#3B6FB6','#8C5BA6','#009E73']
    for pi,(pair,name,marker,color) in enumerate(zip(IDS,NAMES,MARKERS,shades)):
        dodge=1.10**(pi-1)
        cc=sorted((r for r in certs if r['pair']==pair),key=lambda r:r['gamma'])
        den=fraction_interval(cc[0]['true_leading_amplitude_interval'])
        assert cc[0]['true_leading_amplitude_interval']==cc[1]['true_leading_amplitude_interval']
        rr=best[pair];xs=[];ys=[];los=[];his=[]
        for r in rr:
            q=Fraction(r['normalized_gap'])
            iv=ratio_interval([q,q],den,100,1)
            lo,hi=map(float,iv);mid=(lo+hi)/2
            xs.append(float(r['gamma']));ys.append(mid);los.append(mid-lo);his.append(hi-mid)
        d.errorbar(np.array(xs)*dodge,ys,yerr=[los,his],color=color,fmt='none',
                   capsize=1.8,elinewidth=.5,zorder=3)
        estimate=[float(r['relative_amplitude_mismatch_percent']) for r in rr]
        d.plot(np.array(xs)*dodge,estimate,color=color,lw=.8,marker=marker,ms=4,
               mfc='white',mew=.8,zorder=4)
        selected[pair]=[{'gamma':r['gamma'],'n':r['n'],'normalized_gap':r['normalized_gap'],
                         'relative_deviation_interval_percent':[ys[k]-los[k],ys[k]+his[k]]}
                        for k,r in enumerate(rr)]
        for r in cc:
            iv=ratio_interval(fraction_interval(r['d_gamma_interval']),den,100,1)
            lo,hi=map(float,iv);mid=(lo+hi)/2
            d.errorbar([r['gamma']*dodge],[mid],yerr=[[mid-lo],[hi-mid]],
                       color=color,fmt='none',capsize=3,elinewidth=1.3,zorder=5)
            displayed_cert.append({'pair':pair,'gamma':r['gamma'],
                'relative_deviation_percent_rational':[str(x) for x in iv],
                'display_midpoint_is_not_an_estimator':True,
                'd_gamma_interval':r['d_gamma_interval'],
                'C_infinity_interval':r['true_leading_amplitude_interval']})
    d.set(xscale='log',xlim=(7e3,3.7e6),ylim=(-1.12,2.6),xticks=[1e4,1e5,1e6],
          yticks=[-1,0,1,2],xlabel=r'Switching rate $\gamma$',
          ylabel=r'$100(d_\gamma/C_\infty-1)$  (%)')
    d.set_xticklabels([r'$10^4$',r'$10^5$',r'$10^6$'],fontsize=9)
    d.xaxis.set_minor_locator(NullLocator())
    pairhandles=[Line2D([],[],color=c,marker=m,mfc='white',ms=4,lw=.8,label=n+'  (α = '+al+')')
                 for c,m,n,al in zip(shades,MARKERS,NAMES,['1.999','1.999','1.997'])]
    lg=d.legend(handles=pairhandles,loc='upper right',bbox_to_anchor=(1,1),fontsize=7.7,
                handlelength=1.7,labelspacing=.4)
    d.add_artist(lg)
    d.legend(handles=[Line2D([],[],color=INK,marker='o',mfc='white',lw=0,ms=4,label='Grid estimate / numerical amplitude'),
                      Line2D([],[],color=INK,marker='|',lw=0,ms=8,mew=1.3,label='Continuous-model enclosure')],
             loc='lower left',fontsize=7.6,handlelength=1.3,labelspacing=.35)
    save(fig,'fig2_response_certification')
    assert sum(r['segments'] for r in bands)==1218
    dump('fig2_source.json',{'coefficient_bands':bands,'main_grid_selection':selected,
        'selection_rule':'Maximum n among executed cell grids per pair and rate.',
        'input_grid_count':93,'main_grid_count':15,'supplement_grid_count':93,
        'continuous_certificates':displayed_cert,
        'relative_deviation':'100*(d_gamma/C_infinity-1); independent denominator interval propagated by exact rational arithmetic.',
        'center_definition':'Midpoint of the displayed ratio interval, not a point estimate of the true amplitude.',
        'band_definition':'Pairs I/III: original two-sided enclosures. Pair II: 20 full-cell lower bounds; dashed nominal contrast is not certified as the exact coefficient.'})

def fig3():
    data=load('pair_ii_spectral_allocation.json')
    assert data['all_pass'] and Fraction(data['exact_delta_total'])==0
    assert Fraction(data['exact_delta_weighted_total'])*4608==2
    with localcontext() as ctx:
        ctx.prec=100
        un=np.array([float(sum((Decimal(r['delta']) for r in data['rows'][:j+1]),Decimal(0))*4608) for j in range(10)])
        wt=np.array([float(sum((Decimal(r['delta_weighted']) for r in data['rows'][:j+1]),Decimal(0))*4608) for j in range(10)])
    assert max(abs(un[-2:]))<1e-60 and max(abs(wt[-2:]-2))<1e-13
    un[-2:]=0;wt[-2:]=2
    W,H=183,147
    fig=plt.figure(figsize=(W/25.4,H/25.4))
    a=canvas(fig,[13,101,70,34],W,H,'a','')
    a.add_patch(Polygon([[3,5],[45,5],[60,18],[18,18]],closed=True,
                       facecolor='#EDF1F3',edgecolor=LIGHT,lw=.5,zorder=1))
    origin=(17,9);proj=(43,12);tip=(43,30)
    for start,end,col in [(origin,proj,GRAY),(proj,tip,ORTH),(origin,tip,INK)]:
        a.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=7,lw=.9,color=col,zorder=3))
    a.plot([40,40,43],[12,15,15],color=GRAY,lw=.5,zorder=4)
    a.text(3,-2,'Cyclic space',fontsize=8.5)
    a.text(21,1.5,r'$\mathsf{P}_g\Pi v$',fontsize=9,color=GRAY)
    a.text(46,29,r'$\Pi v$',fontsize=9)
    a.text(47,20,r'$(I-\mathsf{P}_g)\Pi v$',fontsize=8.2,color=ORTH)
    a.text(5,34,r'$\mathsf{P}_g$: cyclic projection',fontsize=9.5)
    # Plane and arrows are qualitative; no measured angles or projections are implied.
    d=mm_axes(fig,[110,108,66,27],W,H)
    label(d,'b','')
    tot=data['exact_totals']
    dv=(Fraction(tot['A']['visible'])-Fraction(tot['B']['visible']))*4608
    du=(Fraction(tot['A']['weighted_total'])-Fraction(tot['B']['weighted_total']))*4608
    assert dv==0 and du==2
    d.axvline(0,color=LIGHT,lw=.5)
    d.barh(0,float(du),height=.42,color=ORTH)
    d.plot(0,1,'o',ms=3,color=GRAY)
    d.text(.08,1,'0',va='center',fontsize=11,color=GRAY)
    d.text(2.10,0,'2',va='center',fontsize=12,fontweight='bold',color=ORTH)
    d.set(yticks=[1,0],yticklabels=[r'$\Delta V$',r'$\Delta U$'],xticks=[0,1,2],
          xlim=(-.1,2.7),ylim=(-.6,1.6),xlabel=r'A $-$ B  (units of 1/4608)')
    d.spines['left'].set_visible(False);d.tick_params(axis='y',length=0)
    b=mm_axes(fig,[16,18,66,57],W,H)
    c=mm_axes(fig,[110,18,66,57],W,H)
    insets=[]
    rates=[float(r['lambda']) for r in data['rows']]
    mu_labels=[f'{v:.4f}'.replace('-','−') for v in rates]
    for ax,letter,vals,ylabel,endtext in [
        (b,'c',un,r'Cumulative $\Delta d$  (units of 1/4608)','Total = 0'),
        (c,'d',wt,r'Cumulative $\mu\Delta d$  (units of 1/4608)','Total = 1/2304')]:
        label(ax,letter,'',pad=35)
        ax.axhline(0,color=LIGHT,lw=.5)
        edges=np.arange(11)-.5
        ax.step(edges,np.r_[vals,vals[-1]],where='post',color=INK,lw=.9)
        ax.plot(np.arange(10),vals,'o',color=INK,ms=2.7,mew=.4)
        ax.set(xlim=(-.5,9.5),xticks=np.arange(10),xlabel='Last included eigenvalue index k',ylabel=ylabel)
        ax.set_ylim(-470,390) if letter=='c' else ax.set_ylim(-270,220)
        ax.tick_params(labelsize=7.3)
        ax.yaxis.set_major_locator(MaxNLocator(4))
        top=ax.secondary_xaxis('top')
        top.set_xticks([0,2,4,6,8],labels=[mu_labels[j] for j in [0,2,4,6,8]])
        top.tick_params(labelsize=7,rotation=25,length=2,pad=1.5)
        top.set_xlabel(r'Intermediate generator eigenvalue $\mu_j$',fontsize=8,labelpad=2)
        top.spines['top'].set_linewidth(.5)
        z=ax.inset_axes([.65,.08,.31,.21])
        insets.append(z)
        z.axhline(0,color=LIGHT,lw=.5)
        # Draw only the magnified segment. Avoid invisible clipped curves outside the inset.
        z.step([7.65,8.5,9.4],[vals[8],vals[9],vals[9]],where='post',color=INK,lw=.85)
        z.plot([8,9],vals[8:],'o',color=INK,ms=2.6)
        z.set(xlim=(7.65,9.4),ylim=(-.4,2.6),xticks=[8,9],yticks=[0,2])
        z.tick_params(labelsize=7,length=1.5,pad=3)
        z.text(.5,1.12,'endpoint zoom',ha='center',transform=z.transAxes,fontsize=7.2)
        ax.annotate(endtext,xy=(9,vals[-1]),xytext=(.98,.91),textcoords='axes fraction',
                    ha='right',va='center',fontsize=8,
                    arrowprops={'arrowstyle':'->','lw':.5,'color':GRAY,
                                'connectionstyle':'angle,angleA=0,angleB=90,rad=3'})
    save(fig,'fig3_spectral_cancellation')
    dump('fig3_source.json',{'source':'pair_ii_spectral_allocation.json',
        'definition_unweighted':'4608*cumulative sum_j (d_A(mu_j)-d_B(mu_j))',
        'definition_weighted':'4608*cumulative sum_j mu_j*(d_A(mu_j)-d_B(mu_j))',
        'unweighted':un.tolist(),'weighted':wt.tolist(),'exact_endpoints':[0,2],
        'rates':rates,'common_unweighted_sum':'3269/10368','endpoint_zoom_ylim':[-.4,2.6],
        'exact_decomposition':tot,
        'bar_axis':'negative derivative in units 1/4608; internal labels give signed V and U coefficients.',
        'schematic':'Qualitative K_g plane, projection and orthogonal residual; lengths and angles are not measured.'})

def figS4():
    rows,best=amplitude_data()
    fig=plt.figure(figsize=(7.0078740157480315,5.511811023622048))
    gs=fig.add_gridspec(2,3,left=.11,right=.975,bottom=.17,top=.85,hspace=.80,wspace=.43)
    axes=[];slopes={}
    for i,(pair,mark) in enumerate(zip(IDS,MARKERS)):
        ax=fig.add_subplot(gs[0,i]);axes.append(ax)
        label(ax,'abc'[i],NAMES[i]+'  ·  rate dependence')
        rr=[r for r in rows if r['pair']==pair]
        for method,filled in [('cell',True),('lattice',False)]:
            ss=[r for r in rr if r['method']==method]
            x=np.array([float(r['epsilon']) for r in ss])
            y=np.abs([float(r['relative_amplitude_mismatch_percent']) for r in ss])
            assert np.all(x>0) and np.all(y>0)
            ax.scatter(x,y,marker=mark,s=11,facecolors=INK if filled else 'none',
                       edgecolors=INK,alpha=.3,linewidths=.5,zorder=2)
        br=sorted(best[pair],key=lambda r:float(r['epsilon']))
        x=np.array([float(r['epsilon']) for r in br]);y=np.array([abs(float(r['relative_amplitude_mismatch_percent'])) for r in br])
        ax.plot(x,y,color=INK,marker=mark,ms=3,lw=.8)
        ax.plot(x,y[0]*x/x[0],color=GRAY,lw=.7,ls=(0,(3,2)))
        ax.set(xscale='log',yscale='log',xlim=(8e-5,2e-3),ylim=(.08,2.8),
               xticks=[1e-4,1e-3],yticks=[.1,1],xlabel='ε',ylabel='|dγ/Cpred − 1|  (%)')
        ax.set_xticklabels([r'$10^{-4}$',r'$10^{-3}$']);ax.set_yticklabels(['0.1','1'])
        ax.xaxis.set_minor_locator(NullLocator());ax.yaxis.set_minor_locator(NullLocator())
        fitted=float(np.polyfit(np.log(x),np.log(y),1)[0])
        ax.text(.5,-.45,f'Numerical slope {fitted:.3f}',ha='center',transform=ax.transAxes,fontsize=7.4)
        ax.text(.5,-.57,'Dashed: slope 1 guide',ha='center',transform=ax.transAxes,fontsize=7.2)
        slopes[pair]={'fitted_log_slope':fitted,'consecutive_log_slopes':(np.diff(np.log(y))/np.diff(np.log(x))).tolist(),
                      'guide_anchor':'finest cell at largest gamma','guide_is_fit':False}
    for i,(pair,mark) in enumerate(zip(IDS,MARKERS)):
        ax=fig.add_subplot(gs[1,i]);axes.append(ax)
        label(ax,'def'[i],NAMES[i]+'  ·  spatial refinement')
        for method,filled in [('cell',True),('lattice',False)]:
            ss=sorted((r for r in rows if r['pair']==pair and float(r['gamma'])==2560000 and r['method']==method),
                      key=lambda r:int(r['n']))
            ax.plot([int(r['n']) for r in ss],[float(r['relative_amplitude_mismatch_percent']) for r in ss],
                    color=INK,marker=mark,ms=3,mfc=INK if filled else 'white',mew=.7,
                    lw=.8,ls='-' if filled else (0,(3,2)))
        ax.set(xscale='log',xlabel='Grid size n',ylabel='dγ/Cpred − 1  (%)',
               xticks=[262144,1048576],xlim=(2e5,2.4e6))
        ax.set_xticklabels(['2¹⁸','2²⁰']);ax.xaxis.set_minor_locator(NullLocator())
        ax.ticklabel_format(axis='y',style='plain',useOffset=False)
        ax.yaxis.set_major_locator(MaxNLocator(4))
    fig.text(.11,.965,'All 93 calculations: rate scaling and spatial refinement',fontsize=9,fontweight='bold')
    fig.text(.11,.925,'Circles / squares / triangles: Pairs I / II / III; filled / open: cell / lattice.',fontsize=7.5)
    fig.text(.11,.064,'Top: all grids, with finest-cell connections. Dashed unit-slope guide is anchored, not fitted.',
             fontsize=7.2)
    fig.text(.11,.025,'Bottom: every executed grid at γ = 2.56 × 10⁶. Cpred is an independent numerical reference.',
             fontsize=7.2)
    save(fig,'figS4_amplitude_convergence')
    dump('figS4_source.json',{'all_input_rows':rows,'input_count':93,'plotted_top_count':93,
        'all_grid_records_retained':True,'finest':best,'empirical_log_slopes':slopes})

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    jobs={'1':fig1,'2':fig2,'3':fig3,'S4':figS4}
    for key in sys.argv[1:] or list(jobs):
        jobs[key]()
if __name__=='__main__':
    main()
