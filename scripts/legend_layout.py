"""Keep legends inside axes in a reserved band above the plotted data."""
import numpy as np
from matplotlib.ticker import FixedLocator


def reserve_legend_space(fig):
    labels=['PG-VPS','Fixed 0.65','Fixed 0.95','PS','MPPI','iCEM','CEM','Smooth-MPPI']
    markers=['o','s','^','s','^','D','v','x']
    for ax in fig.axes:
        for line in ax.lines:
            label=line.get_label()
            if label in labels:
                idx=labels.index(label);n=len(line.get_xdata())
                line.set_linewidth(1.65 if label=='PG-VPS' else 1.3)
                line.set_marker(markers[idx]);line.set_markersize(3.2)
                line.set_markerfacecolor('white');line.set_markeredgewidth(.8)
                line.set_markevery((0,1) if n<=10 else ((idx%3,3) if n<=30 else (idx*3,max(1,n//9))))
                if n==8 and label in ['PS','MPPI','Smooth-MPPI']:
                    line.set_markevery((['PS','MPPI','Smooth-MPPI'].index(label),3))
                if label=='PS':line.set_color('#666666')
        ax.xaxis.label.set_fontsize(10.5)
        ax.yaxis.label.set_fontsize(10.5)
        ax.title.set_fontsize(10.5)
        ax.tick_params(labelsize=9.5)
        for text in ax.texts:text.set_fontsize(10)
        if ax.get_aspect() == 1.0:
            slot=ax.get_subplotspec().get_position(fig)
            ratio=slot.width*fig.get_figwidth()/(slot.height*fig.get_figheight())
            lo0,hi0=ax.get_ylim();left,right=ax.get_xlim()
            width=max(right-left,(hi0-lo0)*ratio);center=(left+right)/2
            ax.set_xlim(center-width/2,center+width/2)
        old = ax.get_legend()
        if old is None:
            continue
        handles, labels = ax.get_legend_handles_labels()
        if not labels:
            continue
        old.remove()
        cols = 3 if len(labels) in (3, 5, 6) else 2
        legend = ax.legend(handles, labels, loc='upper right', ncol=cols,
                           fontsize=9, frameon=False, borderaxespad=.55,
                           handlelength=2.3, columnspacing=.85,
                           handletextpad=.45, labelspacing=.3)
        fig.canvas.draw()
        while cols > 1 and legend.get_window_extent().width > .95 * ax.get_window_extent().width:
            cols -= 1
            legend.remove()
            legend = ax.legend(handles, labels, loc='upper right', ncol=cols,
                               fontsize=9, frameon=False, borderaxespad=.55,
                               handlelength=2.3, columnspacing=.85,
                               handletextpad=.45, labelspacing=.3)
            fig.canvas.draw()
        # Measure in display coordinates; reserve space without changing samples.
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        fraction = legend.get_window_extent(renderer).height / ax.get_window_extent(renderer).height + .075
        fraction = min(.48, max(.15, fraction))
        lo, hi = ax.get_ylim()
        ticks = ax.get_yticks()
        scale = ax.yaxis.get_transform()
        bounds = scale.transform(np.array([lo, hi]))
        upper = float(scale.inverted().transform(bounds[0] + (bounds[1]-bounds[0])/(1-fraction)))
        ax.set_ylim(lo, upper)
        if ax.get_aspect() == 1.0:
            # Preserve equal spatial units without collapsing a narrow path panel.
            slot = ax.get_subplotspec().get_position(fig)
            ratio = slot.width * fig.get_figwidth() / (slot.height * fig.get_figheight())
            left, right = ax.get_xlim()
            width = max(right-left, (upper-lo)*ratio)
            center = (left+right)/2
            ax.set_xlim(center-width/2, center+width/2)
        # Extra space is for the key, not an extension of the reported data.
        ax.yaxis.set_major_locator(FixedLocator([t for t in ticks if lo <= t <= hi]))
    fig.canvas.draw()


def annotate_results(fig, name):
    """Add numerical callouts from the plotted data, without changing curves."""
    expected = 6 if name in ['fig_control_public_tradeoff','fig_public_responses','fig_public_robustness','fig_planning_budget'] else 3
    method_names={'PG-VPS','Fixed 0.65','Fixed 0.95','PS','MPPI','iCEM','CEM','Smooth-MPPI'}
    for ax in fig.axes:
        labels=[line.get_label() for line in ax.lines if line.get_label() in method_names]
        if labels:
            assert len(labels)==expected and len(set(labels))==expected,(name,labels)
    blue = '#0072B2'
    def curves(ax):
        return {line.get_label():line for line in ax.lines if line.get_label() in
                ['PG-VPS','Fixed 0.65','Fixed 0.95','PS','MPPI','iCEM','CEM','Smooth-MPPI']}
    def callout(ax, text, xy, pos):
        ax.annotate(text, xy=xy, xycoords='data', xytext=pos, textcoords='axes fraction',
                    fontsize=8.5, color=blue, ha='left', va='center',
                    bbox=dict(boxstyle='round,pad=.23',fc='white',ec=blue,lw=.65),
                    arrowprops=dict(arrowstyle='->',color=blue,lw=.85,connectionstyle='arc3,rad=.08'))
    if name=='fig_control_ensemble':
        ax=fig.axes[2];line=curves(ax)['PG-VPS'];value=line.get_ydata()[-1]
        callout(ax,f'PG-VPS: {value:.1f}%',(line.get_xdata()[-1],value),(.62,.40))
    elif name=='fig_control_primary_effects':
        ax=fig.axes[1];lines=curves(ax);v=lines['PG-VPS'].get_ydata()[-1];b=lines['Fixed 0.65'].get_ydata()[-1]
        callout(ax,f'{100*(1-v/b):.1f}% less work\nvs Fixed 0.65',(20,v),(.37,.20))
    elif name=='fig_control_robustness':
        for ax in fig.axes:
            v=curves(ax)['PG-VPS'].get_ydata()[-1]
            ax.text(.04,.21,f'PG-VPS: {round(v*.72)}/72',transform=ax.transAxes,
                    fontsize=8.5,color=blue,bbox=dict(fc='white',ec=blue,lw=.65,pad=2))
    elif name=='fig_control_public_tradeoff':
        ax=fig.axes[3];lines=curves(ax);pg=lines['PG-VPS'].get_xdata()
        ratio=min(np.mean(l.get_xdata()) for k,l in lines.items() if k!='PG-VPS')/np.mean(pg)
        callout(ax,f'Mean work ratio\n$\\geq {int(ratio)}:1$',(pg[len(pg)//2],.5),(.38,.28))
    elif name=='fig_public_responses':
        ax=fig.axes[2];line=curves(ax)['PG-VPS']
        callout(ax,'PG-VPS: 24/24',(20,line.get_ydata()[-1]),(.55,.30))
    elif name=='fig_planning_budget':
        ax=fig.axes[0];line=curves(ax)['PG-VPS'];v=line.get_ydata()[-1]
        callout(ax,f'PG-VPS: {v:.1f} ms',(line.get_xdata()[-1],v),(.43,.22))
