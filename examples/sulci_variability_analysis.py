import os

import numpy as np
import pandas as pd
import nibabel
import matplotlib.pyplot as plt

subjects = pd.DataFrame({
    'id': ['101309', '108121', '102008', '102311', '107321', '121315'],
    'gender': ['M', 'F', 'M', 'F', 'F', 'M'],
})

data_path = '/Users/viovene/Downloads/antonia-data'

qs = range(1, 100, 2)

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
                'count': len(matching)
            },
            **{
                'q{}'.format(q): np.nan
                for q in qs
            }
        }
        return len(matching), np.nan, np.nan
    avg = np.vstack(matching).mean(axis=0)
    return {
        **{
            'mean': avg[avg.nonzero()].mean(),
            'std': avg[avg.nonzero()].std(),
            'count': len(matching),
        },
        **{
            'q{}'.format(q): (
                avg[avg > (q / 100.0)].shape[0] / avg[avg.nonzero()].shape[0]
            )
            for q in qs
        }
    }

    return len(matching), avg[avg.nonzero()].mean(), avg[avg.nonzero()].std()


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

d = pd.DataFrame(statistics)

cum_data = {}

for gender in ('F', 'M'):
    for sulcus in d.sulcus.unique():
        row = d.loc[(d.gender == gender) & (d.hemisphere == 'ALL')
                    & (d.sulcus == sulcus)].iloc[0]
        if row[f'{gender}_count'] > 1:
            values = row[[f'{gender}_q{i}' for i in qs]]
            cum_data[sulcus] = values.values

fig = plt.figure()
ax = fig.add_subplot(111)

for sulcus, values in cum_data.items():
    ax.plot(qs, values, label=sulcus)

fig.legend()

plt.show()
