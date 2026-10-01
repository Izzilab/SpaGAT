"""Redraw complete gene PCC distributions for the archived Figure 3 cohorts."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

ROOT = Path(__file__).resolve().parents[1]

def portable_pdf(output, specs):
    """Vector PDF fallback requiring only the same ReportLab used for Figure 2."""
    from reportlab.pdfgen import canvas
    output.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(output), pagesize=(600, 220))
    c.setTitle('Complete gene-wise residual PCC distributions for Figure 3 cohorts')
    for panel, (dataset, title, gene, n_genes, n_cells) in enumerate(specs):
        rows = pd.read_csv(ROOT/f'results/figure3_cohorts/{dataset}_gene_PCC.csv')
        assert len(rows) == n_genes and rows.gene.is_unique
        assert rows.n_cells.eq(n_cells).all() and rows.PCC_status.eq('defined').all()
        xs = np.sort(rows.PCC.to_numpy())
        assert np.isfinite(xs).all()
        selected = float(rows.loc[rows.gene.eq(gene), 'PCC'].iloc[0])
        percentile = np.searchsorted(xs, selected, side='right')/n_genes
        x0, y0, width, height = 46+185*panel, 41, 160, 137
        lo, hi = min(-.05, xs[0]), max(.85, xs[-1])
        px = lambda x: x0+(x-lo)/(hi-lo)*width
        py = lambda y: y0+y*height
        c.setFont('Helvetica-Bold', 9.5)
        c.drawCentredString(x0+width/2, 204, f'({chr(65+panel)}) {title}')
        c.setFont('Helvetica', 8.5)
        c.drawCentredString(x0+width/2, 191, f'{n_genes:,} genes; {n_cells:,} cells')
        for y in [0, .25, .5, .75, 1]:
            c.setStrokeGray(.87); c.setLineWidth(.4); c.line(x0,py(y),x0+width,py(y))
            if panel == 0:
                c.setFillGray(0); c.setFont('Helvetica',7.5)
                c.drawRightString(x0-5,py(y)-2.5,f'{y:.2f}')
        c.setStrokeGray(0); c.setLineWidth(.65)
        c.line(x0,y0,x0+width,y0); c.line(x0,y0,x0,py(1.02))
        for x in [0,.2,.4,.6,.8]:
            c.line(px(x),y0,px(x),y0-3); c.setFont('Helvetica',7.5)
            c.drawCentredString(px(x),y0-13,f'{x:.1f}')
        path=c.beginPath(); path.moveTo(px(lo),py(0)); previous=0
        for rank,x in enumerate(xs,1):
            path.lineTo(px(x),py(previous)); previous=rank/n_genes
            path.lineTo(px(x),py(previous))
        path.lineTo(px(hi),py(1)); c.setLineWidth(1.1); c.drawPath(path)
        c.setDash(2,2); c.setStrokeGray(.4); c.setLineWidth(.7)
        c.line(px(selected),y0,px(selected),py(percentile)); c.setDash()
        c.setFillGray(0); c.circle(px(selected),py(percentile),1.5,fill=1,stroke=0)
        c.setFont('Helvetica',8); c.drawRightString(px(selected)-5,py(.28),gene)
        c.drawRightString(px(selected)-5,py(.28)-10,f'PCC = {selected:.3f}')
        c.setFont('Helvetica',8.5); c.drawCentredString(x0+width/2,14,'Gene-wise residual PCC')
        if panel == 0:
            c.saveState(); c.translate(10,y0+height/2); c.rotate(90)
            c.drawCentredString(0,0,'Cumulative fraction of genes'); c.restoreState()
    c.showPage(); c.save(); print(output.resolve())

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    specs=[('Mouse','Mouse','Gfap',254,3855),('CancerousLiver','Carcinoma liver','APOE',1000,30000),
           ('NormalLiver','Normal liver','GLUL',1000,332828)]
    if plt is None:
        if a.output.suffix.lower() != '.pdf':
            raise ValueError('Without matplotlib, use a .pdf output path.')
        portable_pdf(a.output, specs)
        return
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,
                         'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(1,3,figsize=(8.2,3.05),sharex=True,sharey=True)
    for i,(ax,(ds,title,gene,ng,nc)) in enumerate(zip(axs,specs)):
        d=pd.read_csv(ROOT/f'results/figure3_cohorts/{ds}_gene_PCC.csv')
        assert len(d)==ng and d.gene.is_unique and d.n_cells.eq(nc).all()
        assert d.PCC_status.eq('defined').all() and np.isfinite(d.PCC).all()
        xs=np.sort(d.PCC.to_numpy());ys=np.arange(1,ng+1)/ng
        selected=float(d.loc[d.gene.eq(gene),'PCC'].iloc[0])
        percentile=np.searchsorted(xs,selected,side='right')/ng
        ax.step([-.05]+xs.tolist()+[.85],[0]+ys.tolist()+[1],where='post',color='black',lw=1.4)
        ax.vlines(selected,0,percentile,color='.4',linestyle='--',lw=.9)
        ax.plot(selected,percentile,'o',ms=3.5,color='black',clip_on=False)
        ax.text(selected-.035,.26,gene+'\nPCC = '+f'{selected:.3f}',ha='right',va='center',fontsize=9)
        ax.set_title(f'({chr(65+i)}) {title}\n{ng:,} genes; {nc:,} cells',fontsize=9.5,pad=10)
        ax.set_xlim(-.05,.85);ax.set_ylim(0,1.04)
        ax.set_xticks([0,.2,.4,.6,.8]);ax.set_yticks([0,.25,.5,.75,1])
        ax.tick_params(labelsize=8,length=3);ax.grid(axis='y',color='.90',lw=.5);ax.set_axisbelow(True)
        ax.set_xlabel('Gene-wise residual PCC',fontsize=9)
    axs[0].set_ylabel('Cumulative fraction of genes',fontsize=9)
    fig.subplots_adjust(left=.085,right=.985,bottom=.19,top=.79,wspace=.2)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(a.output)
    plt.close(fig)
    print(a.output.resolve())

if __name__=='__main__':
    main()
