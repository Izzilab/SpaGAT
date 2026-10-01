"""Reproduce Figure 2 from its saved, audited source values.

Dependency: reportlab (pip install reportlab).
Example: python plot_Figure2.py --data Figure2_source_values.csv --output Picture2.pdf
Means and sample SDs are checked against seeds 123, 456, and 789 before plotting.
This script performs no model training or inference.
"""
from pathlib import Path
import argparse, csv, math, statistics
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor, white
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

here=Path(__file__).resolve().parent
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--data',type=Path,default=here/'Figure2_source_values.csv')
parser.add_argument('--output',type=Path,default=here/'Figure2_independent_test.pdf')
args=parser.parse_args()

# Embed Arial when available. Helvetica is the portable PDF fallback.
font_pairs=[
    (Path('/System/Library/Fonts/Supplemental/Arial.ttf'),Path('/System/Library/Fonts/Supplemental/Arial Bold.ttf')),
    (Path('C:/Windows/Fonts/arial.ttf'),Path('C:/Windows/Fonts/arialbd.ttf')),
    (Path('/usr/share/fonts/truetype/msttcorefonts/Arial.ttf'),Path('/usr/share/fonts/truetype/msttcorefonts/Arial_Bold.ttf')),
    (Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'),Path('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf')),
]
for regular,bold in font_pairs:
    if regular.exists() and bold.exists():
        pdfmetrics.registerFont(TTFont('Helvetica',str(regular)))
        pdfmetrics.registerFont(TTFont('Helvetica-Bold',str(bold)))
        break

models=['SpaGAT','GITIII_official','GAT','SPICE_adapted','LightGBM']
labels=['SpaGAT','GITIII','GAT','SPICE-adapted','LightGBM']
colors=['#2D6BAD','#888888','#888888','#888888','#888888']
metrics=['median_gene_pcc','mse','ev_zero_percent']
metric_labels=['Median gene-wise PCC','MSE','EV (%)']
with args.data.open(newline='',encoding='utf-8') as stream:
    rows=list(csv.DictReader(stream))
assert len(rows)==30, 'Expected two datasets x five methods x three metrics.'
data={}
for row in rows:
    key=(row['dataset'],row['method'],row['metric'])
    assert key not in data, f'Duplicate row: {key}'
    vals=[float(row[f'seed{s}']) for s in (123,456,789)]
    mean,sd=statistics.mean(vals),statistics.stdev(vals)
    assert math.isclose(mean,float(row['mean']),rel_tol=1e-10,abs_tol=1e-12)
    assert math.isclose(sd,float(row['sample_SD']),rel_tol=1e-10,abs_tol=1e-12)
    data[key]=(mean,sd,vals)

W,H=880,438
out=args.output
out.parent.mkdir(parents=True,exist_ok=True)
c=canvas.Canvas(str(out),pagesize=(W,H),pageCompression=1)
c.setTitle('SpaGAT independent-test benchmarks')
c.setAuthor('Jiahui Wu and Valerio Izzi')
dark=HexColor('#242424');grey=HexColor('#686868');grid=HexColor('#E4E5E7')
panel_x=[139,392,645];panel_w=213
row_headers=[408,197];tops=[350,139];axis_y=[249,38]
configs={
('SEA_AD',0):(.195,.343,[.20,.24,.28,.32],lambda x:f'{x:.2f}'),
('SEA_AD',1):(.288,.319,[.29,.30,.31],lambda x:f'{x:.2f}'),
('SEA_AD',2):(8,16,[8,10,12,14,16],lambda x:f'{x:.0f}'),
('Mouse',0):(.133,.266,[.14,.18,.22,.26],lambda x:f'{x:.2f}'),
('Mouse',1):(.1518,.1653,[.152,.156,.160,.164],lambda x:f'{x:.3f}'),
('Mouse',2):(6.5,13.9,[7,9,11,13],lambda x:f'{x:.0f}')}

for ri,(d,title,cohort) in enumerate([
    ('SEA_AD','SEA-AD','4 held-out donors  |  54,762 receivers  |  140 genes'),
    ('Mouse','Mouse','1 held-out animal  |  31 sections  |  117,257 receivers  |  254 genes')]):
    c.setFillColor(dark);c.setFont('Helvetica-Bold',13.5);c.drawString(18,row_headers[ri],title)
    for ci,metric in enumerate(metrics):
        x0=panel_x[ci];x1=x0+panel_w;y0=axis_y[ri];top=tops[ri]
        lo,hi,ticks,fmt=configs[d,ci]
        px=lambda v:x0+(v-lo)/(hi-lo)*panel_w
        c.setFillColor(dark);c.setFont('Helvetica-Bold',13);c.drawString(x0-19,top+30,chr(65+ri*3+ci))
        c.setFont('Helvetica-Bold',10.5);c.drawString(x0,top+31,metric_labels[ci])
        # Direction arrows are vector paths, independent of font glyph support.
        ax=x0+pdfmetrics.stringWidth(metric_labels[ci],'Helvetica-Bold',10.5)+9
        ay=top+34.5
        c.setStrokeColor(dark);c.setLineWidth(.9)
        c.line(ax,ay-4,ax,ay+4)
        tip=ay-4 if ci==1 else ay+4
        tail=tip+3 if ci==1 else tip-3
        c.line(ax-2.5,tail,ax,tip);c.line(ax,tip,ax+2.5,tail)
        for t in ticks:
            x=px(t);c.setStrokeColor(grid);c.setLineWidth(.6);c.line(x,y0,x,top+12)
            c.setFillColor(grey);c.setFont('Helvetica',9);c.drawCentredString(x,y0-14,fmt(t))
        c.setStrokeColor(grey);c.setLineWidth(.6);c.line(x0,y0,x1,y0)
        for mi,m in enumerate(models):
            y=top-mi*22
            mean,sd,vals=data[d,m,metric]
            assert all(lo<v<hi for v in vals) and lo<mean-sd<mean+sd<hi
            col=HexColor(colors[mi])
            if ci==0:
                c.setFillColor(dark);c.setFont('Helvetica-Bold' if mi==0 else 'Helvetica',10.3)
                c.drawRightString(x0-27,y-3.2,labels[mi])
            c.setStrokeColor(col);c.setLineWidth(1.25)
            c.line(px(mean-sd),y,px(mean+sd),y)
            c.line(px(mean-sd),y-3,px(mean-sd),y+3)
            c.line(px(mean+sd),y-3,px(mean+sd),y+3)
            c.setFillColor(col);c.circle(px(mean),y,3.5,stroke=0,fill=1)
            for v,j in zip(vals,[-1,0,1]):
                c.setFillColor(white);c.setStrokeColor(col);c.setLineWidth(.9)
                c.circle(px(v),y-6.0+j*1.8,1.75,stroke=1,fill=1)

c.showPage();c.save()
print(out.resolve())
