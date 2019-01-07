import os
from collections import defaultdict

import numpy as np
import pandas as pd
import nibabel
import matplotlib.style
matplotlib.style.use('ggplot')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
# import seaborn as sns

subjects = pd.DataFrame({
    'id': ['101309', '108121', '102008', '102311', '107321', '121315'],
    'gender': ['M', 'F', 'M', 'F', 'F', 'M'],
})

data_path = '/Users/viovene/Downloads/antonia-data'

qs = range(1, 100, 2)

renaming_map = {
    'Marginal_sulcus_Cingulate': 'Marginal_sulcus',
    'Lateral_fissure_Lateral': 'Lateral_fissure',
    'Middle_frontal': 'Middle_frontal_sulcus',
    'Marginal_sulcus_Marginal': 'Marginal_sulcus',
    'Inferior_occipital': 'Inferior_occipital_sulcus',
    'Olfactory': 'Olfactory_sulcus',
    'Intralimbic': 'Intralimbic_sulcus',
    'Olfactory_sulcus_Olfactory': 'Olfactory_sulcus',
    'Inferior_frontal_sulcus_Inferior_frontal': 'Inferior_frontal_sulcus',
    'Orbital_H_shaped_sulcus_Orbital_H-shaped': 'Orbital_H_shaped_sulcus',
    'Inferior_temporal': 'Inferior_temporal_sulcus',
    'Intralimbic_sulcus_Cingulate': 'Intralimbic_sulcus',
    'Precentral': 'Precentral_sulcus',
    'Frontomarginal': 'Frontomarginal_sulcus',
    'Subparietal': 'Subparietal_sulcus',
    'Anterior_horizontal_ramus_lateral_fissure_Lateral': (
        'Anterior_horizontal_ramus_lateral_fissure'
    ),
    'Rhinal': 'Rhinal_sulcus',
    'Anterior_occipital': 'Anterior_occipital_sulcus',
    'Callosal_sulcus_Callosal': 'Callosal_sulcus',
    'Postcentral': 'Postcentral_sulcus',
    'Superior_frontal_sulcus_Superior_frontal': 'Superior_frontal_sulcus',
    'Lunate': 'Lunate_sulcus',
    'Occipitotemporal_sulcus_Occipitotemporal': (
        'Occipitotemporal_sulcus'
    ),
    'Intraparietal_sulcus_Intraparietal': 'Intraparietal_sulcus',
    'Inferior_frontal': 'Inferior_frontal_sulcus',
    'Angular_sulcus_Superior_temporal': 'Angular_sulcus',
    'Parieto_occipital_sulcus_Parieto-Occipital': 'Parieto_occipital_sulcus',
    'Superior_parietal_sulcus_Superior_parietal': 'Superior_parietal_sulcus',
'Inferior_temporal_sulcus_Inferior_temporal': 'Inferior_temporal_sulcus',
    'Cingulate': 'Cingulate_sulcus',
    'Retrocalcarine': 'Retrocalcarine_sulcus',
    'Superior_rostral': 'Superior_rostral_sulcus',
    'Paracentral_sulcus_Paracentral': 'Paracentral_sulcus',
    'Intermediate_primus_of_Jensen_Intermediate_primus_of': (
        'Intermediate_primus_of_Jensen'
    ),
    'Superior_frontal': 'Superior_frontal_sulcus',
    'Superior_rostral_sulcus_Superior_rostral': 'Superior_rostral_sulcus',
    'Superior_temporal_sulcus_Superior_temporal': 'Superior_temporal_sulcus',
    'Superior_occipital_sulcus_Superior_occipital': 'Superior_occipital_sulcus',
    'Intralingual': 'Intralingual_sulcus',
    'Lunate_sulcus_Lunate': 'Lunate_sulcus',
    'Superior_temporal': 'Superior_temporal_sulcus',
    'Hippocampal_sulcus_Hippocampal': 'Hippocampal_sulcus',
    'Collateral': 'Collateral_sulcus',
    'Superior_parietal': 'Superior_parietal_sulcus',
    'Cuneal_sulcus_Cuneal': 'Cuneal_sulcus',
    'Orbital_H_shaped_sulcus_Orbital': 'Orbital_H_shaped_sulcus',
    'Subparietal_sulcus_Subparietal': 'Subparietal_sulcus',
    'Anterior_vertical_ramus_lateral_fissure_Lateral': (
        'Anterior_vertical_ramus_lateral_fissure'
    ),
    'Superior_occipital': 'Superior_occipital_sulcus',
    'Cuneal': 'Cuneal_sulcus',
    'Temporopolar': 'Temporopolar_sulcus',
}

delete_sulci = {
    'Posterior_subcentral_sulcus',
    'Anterior_parolfactory_sulcus',
}


def clean_name(name):
    splits = name.split('_')
    count = len([s for s in splits if s == 'sulcus'])
    if count == 0:
        return name + '_sulcus'
    if count == 1:
        if name.endswith('_sulcus'):
            return name
        else:
            return name.split('_sulcus')[0] + '_sulcus'
        return name
    else:
        raise Exception('what')


manual_sulci = []
for _, subject in subjects.iterrows():
    for hemisphere in ('L', 'R'):
        surface_path = os.path.join(
            data_path,
            '{}.{}.pial.32k_fs_LR.surf.gii'.format(subject['id'], hemisphere)
        )
        surface = nibabel.load(surface_path)
        vertices = surface.darrays[0].data
        manual_sulci_gii_path = os.path.join(
            data_path, '{}.{}H.manual_drawings.func.gii'.format(
                subject['id'], hemisphere
            )
        )
        manual_sulci_gii = nibabel.load(manual_sulci_gii_path)
        for darray in manual_sulci_gii.darrays:
            sulcus_name = darray.meta.metadata['Name']
            sulcus_name = sulcus_name[:sulcus_name.rfind('_')]
            if sulcus_name in renaming_map:
                sulcus_name = renaming_map[sulcus_name]
            else:
                print(sulcus_name)
            if sulcus_name in delete_sulci: continue
            # sulcus_name = clean_name(sulcus_name)
            manual_sulci.append({
                'id': subject['id'],
                'gender': subject['gender'],
                'sulcus': sulcus_name,
                'hemisphere': hemisphere,
                'data': darray.data,
            })


def compute_stats(criterions):
    matching = [
        x['data'] for x in manual_sulci
        if all([x[k] == v for k, v in criterions.items()])
    ]
    if len(matching) <= 1:
        return {
            **{
                'mean': np.nan,
                'std': np.nan,
                'count': len(matching),
                'probability_map': None,
            },
            **{
                'q{}'.format(q): np.nan
                for q in qs
            }
        }
    avg = np.vstack(matching).mean(axis=0)
    return {
        **{
            'mean': avg[avg.nonzero()].mean(),
            'std': avg[avg.nonzero()].std(),
            'count': len(matching),
            'probability_map': avg,
        },
        **{
            'q{}'.format(q): (
                avg[avg > (q / 100.0)].shape[0] / avg[avg.nonzero()].shape[0]
            )
            for q in qs
        }
    }


sulci = [
    'Paracingulate_sulcus', 'Central_sulcus', 'Lateral_fissure',
    'Callosal_sulcus',
]

sulci = set(x['sulcus'] for x in manual_sulci) - {'Medial_orbital',
                                                  'Anterior_parolfactory'}


statistics = []

skipped_sulci = set()
for sulcus in sulci:
    criterions = {'sulcus': sulcus}
    print(sulcus)

    combinations = [
        ('ALL', criterions),
        ('F', {**criterions, 'gender': 'F'}),
        ('M', {**criterions, 'gender': 'M'}),
        ('L', {**criterions, 'hemisphere': 'L'}),
        ('R', {**criterions, 'hemisphere': 'R'}),
        ('FL', {**criterions, 'gender': 'F', 'hemisphere': 'L'}),
        ('FR', {**criterions, 'gender': 'F', 'hemisphere': 'R'}),
        ('ML', {**criterions, 'gender': 'M', 'hemisphere': 'L'}),
        ('MR', {**criterions, 'gender': 'M', 'hemisphere': 'R'}),
    ]

    for comb_name, comb in combinations:
        stats = compute_stats(comb)
        stats = {f'{comb_name}_{k}': v for k, v in stats.items()}
        statistics.append({
            **stats,
            **{
                'sulcus': comb['sulcus'],
                'gender': comb['gender'] if 'gender' in comb else 'ALL',
                'hemisphere': (
                    comb['hemisphere'] if 'hemisphere' in comb else 'ALL'
                ),
            }
        })

d = pd.DataFrame([{k: v for k, v in s.items()} for s in statistics])

cum_data = defaultdict(dict)
for hemisphere in ('L', 'R'):
    for sulcus in d.sulcus.unique():
        if sulcus not in {
            'Central_sulcus', 'Paracingulate_sulcus',
            'Callosal_sulcus',
        }:
            continue
        row = d.loc[(d.hemisphere == hemisphere) & (d.gender == 'ALL')
                    & (d.sulcus == sulcus)].iloc[0]
        if row[f'{hemisphere}_count'] >= 5:
            values = row[[f'{hemisphere}_q{i}' for i in qs]]
            cum_data[hemisphere][sulcus] = values.values


# fig = plt.figure(figsize=(12, 10))
# ax = fig.add_subplot(111)
# for i, (hemisphere, data) in enumerate(cum_data.items()):
    # ax.set_title(f'Percentage of probability map greater than threshold')
    # ax.set_xlabel('Threshold')
    # ax.set_ylabel('Percentage')
    # for sulcus, values in data.items():
        # ax.plot(qs, values, label=f'{hemisphere}_{sulcus}',
                # ls=':' if hemisphere == 'L' else '-')
# ax.invert_xaxis()
# fig.legend()
# fig.savefig('/tmp/plot.png')
# plt.show()


# dfs = []
# for s in statistics:
    # if (s['hemisphere'] == 'L' and s['gender'] == 'ALL' and
        # s['L_probability_map'] is not None and s['L_count'] >= 6):
        # values = s['L_probability_map']
        # values = values[values.nonzero()]
        # df = pd.DataFrame({'prob': values})
        # df['sulcus'] = s['sulcus'].split('_sulcus')[0].replace('_', ' ')
        # dfs.append(df)
# data = pd.concat(dfs)

# sns.set(style="whitegrid")
# f, ax = plt.subplots()
# sns.despine(bottom=True, left=True)
# sns.stripplot(x='prob', y='sulcus', alpha=.25, zorder=1, dodge=True,
              # jitter=True, data=data)
# f.subplots_adjust(left=0.35)

primary_sulci = {'Central_sulcus', 'Callosal_sulcus'}
secondary_sulci = {'Paracingulate_sulcus', 'Angular_sulcus'}



def violin_plot_sulci_probmap(ax, i, j, color, dataset, labels, h, cfg):
    cleaned_labels = [l.split('_sulcus')[0].replace('_', ' ') for l in labels]
    pos = range(1, len(labels) + 1)
    vp = ax.violinplot(dataset, pos, vert=False,
                       showmeans=True, showextrema=False)
    vp['cmeans'].set_color('black')
    if i == 0:
        ax.tick_params(labelbottom=False)
    for i in range(len(vp['bodies'])):
        vp['bodies'][i].set_facecolor(color)
        vp['bodies'][i].set_alpha(dataset[i].mean())
        if cfg['ordering'] == 'top':
            value = (
                dataset[i][dataset[i] > 0.99].size /
                dataset[i][dataset[i].nonzero()].size
            )
            ax.text(1.0, i + 1, '{:.3f}'.format(value))
        else:
            ax.text(
                dataset[i].mean() + 0.01, i + 1,
                '{:.3f}'.format(dataset[i].mean())
            )
    ax.set_yticks(pos)
    ax.set_yticklabels(cleaned_labels)
    if j == 1:
        ax.yaxis.tick_right()

plot_configs = [
    {'min_count': 6, 'ordering': 'top'},
    {'min_count': 2, 'ordering': 'bottom'},
]

fig = plt.figure(figsize=(20, 20))
gs = gridspec.GridSpec(2, 2)

axes = dict()

for i, cfg in enumerate(plot_configs):
    for j, h in enumerate(['L', 'R']):

        if i == 1:
            ax = fig.add_subplot(gs[i, j], sharex=axes[(0, j)])
        else:
            ax = fig.add_subplot(gs[i, j])
        axes[(i, j)] = ax

        data_and_labels = [
            (s[f'{h}_probability_map'][s[f'{h}_probability_map'] > 1 / 6],
             s['sulcus'])
            for s in statistics
            if (
                s['hemisphere'] == h and
                s['gender'] == 'ALL' and
                s[f'{h}_probability_map'] is not None and
                s[f'{h}_count'] >= cfg['min_count'] and
                s['sulcus'] in {'Parieto_occipital_sulcus'}
            )
        ]
        if cfg['ordering'] == 'top':
            data_and_labels = sorted(
                data_and_labels,
                key=lambda x: x[0][x[0] > 0.99].shape[0] / x[0].shape[0]
            )
        else:
            data_and_labels = sorted(
                data_and_labels,
                key=lambda x: x[0][x[0] > 1/6].mean()
            )
        dataset, labels = zip(*data_and_labels)
        # if cfg['ordering'] == 'top':
            # dataset = dataset[-10:]
            # labels = labels[-10:]
        # else:
            # dataset = dataset[:10]
            # labels = labels[:10]
        color = 'red' if h == 'L' else 'blue'
        violin_plot_sulci_probmap(
            ax, i, j, color, dataset, labels, h, cfg
        )
# fig.subplots_adjust(left=0.35, right=0.35, wspace=0.05)
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
fig.savefig('/tmp/figure.pdf')
