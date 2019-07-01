import os
import random
from collections import defaultdict, Counter
from statistics import mode

import numpy as np
import pandas as pd
import nibabel
from matplotlib import colors
import matplotlib.style
matplotlib.style.use('ggplot')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import matplotlib
# get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import nilearn
from nilearn import datasets, plotting, image, surface
import nibabel as nib
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

from neurolang import frontend

destrieux_dataset = datasets.fetch_atlas_destrieux_2009()
destrieux_map = nib.load(destrieux_dataset['maps'])
destrieux_dataset['labels']

subjects = pd.DataFrame({
    'id': [
        '101309', '108121', '102008', '102311', '107321', '121315', '108525',
        '116524', '127630', '140925'
    ],
    'gender': ['M', 'F', 'M', 'F', 'F', 'M', 'M', 'M', 'F', 'F'],
})

data_path = '/Users/viovene/Downloads/antonia'

medial_sulci = {
    'Callosal',
    'pericallosal',
    'subcallosal',
    'arieto_occipital',
    'Cingulate',
    'cingul_',
    'alcarine',
    'Callosomarginal',
    'Paracingulate',
    'erior_rostral',
    'aracentral',
    'ubparietal',
    'uneal',
    'cuneus',
    'Intralingual',
    'Intralimbic',
    'rectus',
    '0002',
    '0003',
    '0004',
    '010.',
    '0013',
    '0014',
    '0016',
    '0019',
    '0021',
    '0025',
    '0027',
    '0029',
    '0030',
    '0033',
    '0034',
    '0038',
    '0042',
    '0046',
    '0053',
    '0059',
    '0061',
    '0063',
    'medial_frontal',
}

lateral_sulci = {
    'Central',
    '_central',
    'ateral_fissure',
    'Lat_Fis',
    'recentral',
    'ostcentral',
    'Superior_frontal',
    'Inferior_frontal',
    'Middle_frontal',
    '_front_',
    'rontomargin',
    'frontopol',
    'subcentral',
    'Intraparietal',
    'intrapariet',
    'Jensen',
    'Superior_parietal',
    'parietal_sup',
    'Supramar',
    'Superior_occipital',
    'Inferior_occipital',
    'lateral_occipital',
    'occipital',
    'oc_sup',
    'Lingual',
    'Lunat',
    'Superior_temporal',
    'Inferior_temporal',
    'temp_sup',
    '_temporal',
    'Anterior_occipital',
    'Angular',
    'insula',
    'cent_ins',
    '0000',
    '0005',
    '0006',
    '0007',
    '0009',
    '0011',
    '0015',
    '0018',
    '0022',
    '0023',
    '0026',
    '0031',
    '0032',
    '0035',
    '0036',
    '0039',
    '0041',
    '0044',
    '0045',
    '0047',
    '0048',
    '0050',
    '0055',
    '0057',
    '0058',
    '0060',
    '0062',
}
ventral_sulci = {
    'Occipitotemporal',
    'Collateral',
    'Rhinal',
    'Parahip',
    'Hippocampal',
    'Olfactory',
    'Temporopolar',
    '0001',
    '0028',
    '0054',
    'collat',
    'oc_temp_lat',
    'rbital',
}

x_labels = [
    'medial', 'overlaps', 'during_x', 'meets', 'starts', 'finishes', 'equals',
    'lateral'
]
y_labels = [
    'anterior', 'overlaps', 'during_y', 'meets', 'starts', 'finishes',
    'equals', 'posterior'
]
z_labels = [
    'superior', 'overlaps', 'during_z', 'meets', 'starts', 'finishes',
    'equals', 'inferior'
]


def tolerance_y_3(ys_origin_sulcus, ys_target_sulcus, length):
    # we consider I the origin
    before = set()
    overlaps = set()
    during = set()
    meets = set()
    starts = set()
    finishes = set()
    equals = set()
    after = set()
    Anterior_of = set()
    Posterior_of = set()
    During = set()
    J_minus = min(ys_target_sulcus)
    J_plus = max(ys_target_sulcus) + length

    I_boxes = set(ys_origin_sulcus)
    for x in I_boxes:
        I_minus = x
        I_plus = I_minus + length

        if I_minus < I_plus < J_minus < J_plus:
            before.add(x)
        if I_minus < J_minus < I_plus < J_plus:
            overlaps.add(x)
        if J_minus < I_minus < I_plus < J_plus:
            during.add(x)
        if I_minus < I_plus == J_minus < J_plus:
            meets.add(x)
        if I_minus == J_minus < I_plus < J_plus:
            starts.add(x)
        if J_minus < I_minus < I_plus == J_plus:
            finishes.add(x)
        if I_minus == J_minus < I_plus == J_plus:
            equals.add(x)
        if J_minus < J_plus < I_minus < I_plus:
            after.add(x)
    before_pc = len(before) / len(I_boxes) * 100
    overlaps_pc = len(overlaps) / len(I_boxes) * 100
    during_pc = len(during) / len(I_boxes) * 100
    meets_pc = len(meets) / len(I_boxes) * 100
    starts_pc = len(starts) / len(I_boxes) * 100
    finishes_pc = len(finishes) / len(I_boxes) * 100
    equals_pc = len(equals) / len(I_boxes) * 100
    after_pc = len(after) / len(I_boxes) * 100

    values = [
        before_pc, overlaps_pc, during_pc, meets_pc, starts_pc, finishes_pc,
        equals_pc, after_pc
    ]

    return values


def making_dominant_sets_relative_to_primary(
    fe, primary_sulcus, labels, hemisphere_manual_sulci, axis
):

    x = fe.new_region_symbol('x')
    q = fe.query(
        x, (
            fe.symbols.anterior_of(x, fe.symbols[f'manual_{primary_sulcus}']) |
            fe.symbols.posterior_of(x,
                                    fe.symbols[f'manual_{primary_sulcus}']) |
            fe.symbols.superior_of(x, fe.symbols[f'manual_{primary_sulcus}']) |
            fe.symbols.inferior_of(x, fe.symbols[f'manual_{primary_sulcus}']) |
            fe.symbols.overlapping(x, fe.symbols[f'manual_{primary_sulcus}'])
        )
    )

    res = q.do()

    anterior = set()
    posterior = set()
    during_y = set()
    superior = set()
    inferior = set()
    during_z = set()
    medial = set()
    lateral = set()
    during_x = set()

    for r in res:
        if r.symbol_name.startswith('manual_'):

            sulcus_relativity = tolerance_y_3(
                hemisphere_manual_sulci[primary_sulcus].T[axis],
                hemisphere_manual_sulci[r.symbol_name[7:]].T[axis],
                length=.1
            )
            relations = []
            relations.append(labels[np.argmax(np.array(sulcus_relativity))])

            if mode(relations) == 'anterior':
                anterior.add(fe.symbols[r.symbol_name])
            elif mode(relations) == 'posterior':
                posterior.add(fe.symbols[r.symbol_name])
            elif mode(relations) == 'during_y':
                during_y.add(fe.symbols[r.symbol_name])
            elif mode(relations) == 'superior':
                superior.add(fe.symbols[r.symbol_name])
            elif mode(relations) == 'inferior':
                inferior.add(fe.symbols[r.symbol_name])
            elif mode(relations) == 'during_z':
                during_z.add(fe.symbols[r.symbol_name])
            elif mode(relations) == 'medial':
                medial.add(fe.symbols[r.symbol_name])
            elif mode(relations) == 'lateral':
                lateral.add(fe.symbols[r.symbol_name])
            elif mode(relations) == 'during_x':
                during_x.add(fe.symbols[r.symbol_name])
            else:
                continue

        if axis == 1:
            anterior_dominant = fe.add_region_set(
                anterior, name=f'{primary_sulcus}_anterior_dominant'
            )
            posterior_dominant = fe.add_region_set(
                posterior, name=f'{primary_sulcus}_posterior_dominant'
            )
            during_y_dominant = fe.add_region_set(
                during_y, name=f'{primary_sulcus}_during_y_dominant'
            )
        elif axis == 2:
            superior_dominant = fe.add_region_set(
                superior, name=f'{primary_sulcus}_superior_dominant'
            )
            inferior_dominant = fe.add_region_set(
                inferior, name=f'{primary_sulcus}_inferior_dominant'
            )
            during_z_dominant = fe.add_region_set(
                during_z, name=f'{primary_sulcus}_during_z_dominant'
            )
        elif axis == 0:
            medial_dominant = fe.add_region_set(
                medial, name=f'{primary_sulcus}_medial_dominant'
            )
            lateral_dominant = fe.add_region_set(
                lateral, name=f'{primary_sulcus}_lateral_dominant'
            )
            during_x_dominant = fe.add_region_set(
                during_x, name=f'{primary_sulcus}_during_x_dominant'
            )


print('loading data')
manual_sulci = []
for _, subject in subjects.iterrows():
    print('subject_id =', subject['id'])
    for hemisphere in ('L', 'R'):
        print('\themisphere', hemisphere)
        surface_path = os.path.join(
            data_path,
            '{}.{}.pial.32k_fs_LR.surf.gii'.format(subject['id'], hemisphere)
        )
        surface = nibabel.load(surface_path)
        vertices = surface.darrays[0].data
        manual_sulci_gii_path = os.path.join(
            data_path, '{}_{}_manual_segmentation.func.gii'.format(
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
                'data': vertices[darray.data.nonzero()],
            })

results = []
subject_ids = set([m['id'] for m in manual_sulci])
for subject_id in subject_ids:
    for hemisphere in ('L', 'R'):
        hemisphere_manual_sulci = {
            m['sulcus']: m['data']
            for m in manual_sulci
            if m['id'] == subject_id and m['hemisphere'] == hemisphere
        }
        fe = frontend.RegionFrontend()
        for name, points in hemisphere_manual_sulci.items():
            ijk_points = nib.affines.apply_affine(
                np.linalg.inv(destrieux_map.affine), points
            ).astype(int)
            region = frontend.ExplicitVBR(
                ijk_points, destrieux_map.affine, img_dim=destrieux_map.shape
            )
            fe.add_region(region, result_symbol_name=f'manual_{name}')
        medial_surface_sulci = set()
        lateral_surface_sulci = set()
        ventral_surface_sulci = set()
        for sulcus_name in fe.region_names:
            if any([s in sulcus_name for s in medial_sulci]):
                medial_surface_sulci.add(fe.symbols[sulcus_name])
            elif any([s in sulcus_name for s in lateral_sulci]):
                lateral_surface_sulci.add(fe.symbols[sulcus_name])
            elif any([s in sulcus_name for s in ventral_sulci]):
                ventral_surface_sulci.add(fe.symbols[sulcus_name])
            else:
                print('sulcus not in medial, lateral or ventral surface')
                print(sulcus_name)
        lateral_surface = fe.add_region_set(
            lateral_surface_sulci, name='lateral_surface'
        )
        medial_surface = fe.add_region_set(
            medial_surface_sulci, name='medial_surface'
        )
        ventral_surface = fe.add_region_set(
            ventral_surface_sulci, name='ventral_surface'
        )
        aips = fe.add_region_set(set(), name='aips')
        vertical = fe.add_region_set(set(), name='vertical')
        pias = fe.add_region_set(set(), name='pias')
        longitudinal = fe.add_region_set(set(), name='longitudinal')
        for sulcus_name in {
            'Central_sulcus',
            'Lateral_fissure',
            'Parieto_occipital_sulcus',
            'Callosal_sulcus',
            'Calcarine_sulcus',
            'Anterior_horizontal_ramus_lateral_fissure',
            'Anterior_vertical_ramus_lateral_fissure',
            'Precentral_sulcus',
            'Cingulate_sulcus',
            'Callosomarginal_sulcus',
            'Subparietal_sulcus',
            'Superior_rostral_sulcus',
            'Postcentral_sulcus',
            'Superior_temporal_sulcus',
            'Angular_sulcus',
            'Intraparietal_sulcus',
            'Intermediate_primus_of_Jensen',
            'Inferior_temporal_sulcus',
            'Superior_occipital_sulcus',
            'Inferior_occipital_sulcus',
            'Superior_frontal_sulcus',
            'Inferior_frontal_sulcus',
            'Olfactory_sulcus',
            'Orbital_H_shaped_sulcus',
            'Occipitotemporal_sulcus',
            'Collateral_sulcus',
            'Hippocampal_sulcus',
            'Paracingulate_sulcus',
            'Inferior_rostral_sulcus',
            'Paracentral_sulcus',
            'Intralimbic_sulcus',
            'Cuneal_sulcus',
            'Retrocalcarine_sulcus',
            'Frontomarginal_sulcus',
            'Middle_frontal_sulcus',
            'Anterior_occipital_sulcus',
            'lateral_occipital_sulcus',
            'Lunate_sulcus',
            'Superior_parietal_sulcus',
            'Temporopolar_sulcus',
            'Intralingual_sulcus',
        }:
            making_dominant_sets_relative_to_primary(
                fe, sulcus_name, x_labels, hemisphere_manual_sulci, axis=0
            )
            making_dominant_sets_relative_to_primary(
                fe, sulcus_name, y_labels, hemisphere_manual_sulci, axis=1
            )
            making_dominant_sets_relative_to_primary(
                fe, sulcus_name, z_labels, hemisphere_manual_sulci, axis=2
            )
        x = fe.new_region_symbol('x')
        queries = [
            fe.query(
                x, (
                    fe.symbols.isin(
                        x, fe.symbols.
                        Anterior_horizontal_ramus_lateral_fissure_anterior_dominant
                    ) & fe.symbols.isin(x, lateral_surface)
                )
            ),
            # fe.query(
            # x, (
            # fe.symbols.anatomical_anterior_of(
            # x, fe.symbols.manual_Central_sulcus
            # ) & fe.symbols.isin(x, longitudinal) &
            # fe.symbols.isin(x, lateral_surface)
            # )
            # ),
            # fe.query(
            # x, (
            # fe.symbols.anatomical_anterior_of(
            # x, fe.symbols.manual_Central_sulcus
            # ) & fe.symbols.isin(
            # x, fe.symbols.Central_sulcus_during_z_dominant
            # ) & fe.symbols.isin(x, lateral_surface)
            # )
            # ),
            # fe.query(
            # x, (
            # fe.symbols.anatomical_posterior_of(
            # x, fe.symbols.manual_lateral_fissure
            # ) & fe.symbols.anatomical_inferior_of(
            # x, fe.symbols.manual_Central_sulcus
            # ) & fe.symbols.isin(x, lateral_surface)
            # )
            # ),
            # fe.query(
            # x, (
            # fe.symbols.isin(
            # x, fe.symbols.Central_sulcus_posterior_dominant
            # ) & fe.symbols.isin(
            # x, fe.symbols.lateral_fissure_superior_dominant
            # ) & fe.symbols.isin(x, aips) &
            # fe.symbols.isin(x, lateral_surface)
            # )
            # ),
            # fe.query(
            # x, (
            # fe.symbols.anatomical_posterior_of(
            # x, fe.symbols.manual_Central_sulcus
            # ) & fe.symbols.anatomical_superior_of(
            # x, fe.symbols.manual_lateral_fissure
            # ) & fe.symbols.isin(
            # x, fe.symbols.Central_sulcus_during_x_dominant
            # ) & fe.symbols.isin(x, lateral_surface)
            # )
            # ),
            # fe.query(
            # x, (
            # fe.symbols.anatomical_inferior_of(
            # x, fe.symbols.manual_lateral_fissure
            # ) & fe.symbols.isin(
            # x, fe.symbols.lateral_fissure_during_y_dominant
            # ) & fe.symbols.isin(x, lateral_surface)
            # )
            # ),
            # fe.query(
            # x, (
            # fe.symbols.anatomical_posterior_of(
            # x, fe.symbols.manual_lateral_fissure
            # ) & fe.symbols.anatomical_superior_of(
            # x, fe.symbols.manual_lateral_fissure
            # ) & fe.symbols.isin(
            # x, fe.symbols.Central_sulcus_medial_dominant
            # ) & fe.symbols.isin(x, lateral_surface)
            # )
            # ),
            # fe.query(
            # x, (
            # fe.symbols.anatomical_posterior_of(
            # x, fe.symbols.manual_lateral_fissure
            # ) & fe.symbols.anatomical_superior_of(
            # x, fe.symbols.manual_lateral_fissure
            # ) & fe.symbols.isin(
            # x, fe.symbols.Central_sulcus_lateral_dominant
            # ) & fe.symbols.isin(x, lateral_surface)
            # )
            # ),
            # fe.query(
            # x, (
            # fe.symbols.anatomical_posterior_of(
            # x, fe.symbols.manual_lateral_fissure
            # ) & fe.symbols.isin(
            # x, fe.symbols.lateral_fissure_lateral_dominant
            # ) & fe.symbols.isin(x, lateral_surface)
            # )
            # ),
            # fe.query(
            # x, (
            # fe.symbols.anatomical_posterior_of(
            # x, fe.symbols.manual_lateral_fissure
            # ) & fe.symbols.anatomical_superior_of(
            # x, fe.symbols.manual_Calcarine_sulcus
            # ) & fe.symbols.isin(x, aips) &
            # fe.symbols.isin(x, lateral_surface)
            # )
            # ),
            # fe.query(
            # x, (
            # fe.symbols.anatomical_anterior_of(
            # x, fe.symbols.manual_Callosal_sulcus
            # ) & fe.symbols.isin(x, medial_surface)
            # )
            # ),
            # fe.query(
            # x, (
            # fe.symbols.isin(
            # x, fe.symbols.Calcarine_sulcus_during_z_dominant
            # ) & fe.symbols.isin(
            # x, fe.symbols.Callosal_sulcus_anterior_dominant
            # ) & ~fe.symbols.
            # superior_of(x, fe.symbols.manual_Callosal_sulcus) &
            # fe.symbols.isin(x, medial_surface)
            # )
            # ),
            # fe.query(
            # x, (
            # fe.symbols.superior_of(
            # x, fe.symbols.manual_Callosal_sulcus
            # ) & fe.symbols.anterior_of(
            # x, fe.symbols.manual_Callosal_sulcus
            # ) & fe.symbols.isin(
            # x, fe.symbols.Callosal_sulcus_anterior_dominant
            # ) & fe.symbols.isin(x, medial_surface)
            # )
            # ),
            # fe.query(
            # x, (
            # fe.symbols.superior_of(
            # x, fe.symbols.manual_Callosal_sulcus
            # ) & fe.symbols.isin(
            # x, fe.symbols.Callosal_sulcus_posterior_dominant
            # ) & fe.symbols.isin(x, pias) &
            # fe.symbols.isin(x, medial_surface)
            # )
            # ),
            # fe.query(
            # x, (
            # fe.symbols.isin(
            # x, fe.symbols.Callosal_sulcus_during_y_dominant
            # ) & fe.symbols.isin(x, medial_surface)
            # )
            # ),
            # fe.query(
            # x, (
            # fe.symbols.isin(
            # x, fe.symbols.Callosal_sulcus_posterior_dominant
            # ) & fe.symbols.isin(
            # x,
            # fe.symbols.Parieto_occipital_sulcus_anterior_dominant
            # ) & fe.symbols.isin(x, medial_surface)
            # )
            # ),
            # fe.query(
            # x, (
            # fe.symbols.anatomical_superior_of(
            # x, fe.symbols.manual_Parieto_occipital_sulcus
            # ) & fe.symbols.isin(x, medial_surface)
            # )
            # ),
            # fe.query(
            # x, (
            # fe.symbols.isin(
            # x, fe.symbols.Calcarine_sulcus_superior_dominant
            # ) & fe.symbols.isin(
            # x,
            # fe.symbols.Parieto_occipital_sulcus_inferior_dominant
            # ) & fe.symbols.isin(x, medial_surface)
            # )
            # ),
            # fe.query(
            # x, (
            # fe.symbols.isin(
            # x, fe.symbols.Calcarine_sulcus_inferior_dominant
            # ) & fe.symbols.anatomical_posterior_of(
            # x, fe.symbols.manual_Callosal_sulcus
            # ) & fe.symbols.isin(x, medial_surface)
            # )
            # ),
            # fe.query(
            # x, (
            # fe.symbols.isin(
            # x, fe.symbols.Calcarine_sulcus_posterior_dominant
            # ) & fe.symbols.isin(x, medial_surface)
            # )
            # ),
            # fe.query(
            # x, (
            # fe.symbols.isin(
            # x, fe.symbols.Central_sulcus_medial_dominant
            # ) & fe.symbols.isin(x, ventral_surface)
            # )
            # ),
            # fe.query(
            # x, (
            # fe.symbols.isin(
            # x, fe.symbols.lateral_fissure_medial_dominant
            # ) & fe.symbols.isin(x, ventral_surface)
            # )
            # ),
            # fe.query(
            # x, (
            # fe.symbols.isin(
            # x, fe.symbols.Callosal_sulcus_lateral_dominant
            # ) & fe.symbols.anatomical_anterior_of(
            # x, fe.symbols.manual_Callosal_sulcus
            # ) & fe.symbols.isin(x, ventral_surface)
            # )
            # ),
            # fe.query(
            # x, (
            # fe.symbols.
            # isin(x, fe.symbols.Callosal_sulcus_lateral_dominant) & ~fe.
            # symbols.anterior_of(x, fe.symbols.manual_Central_sulcus) &
            # fe.symbols.isin(x, ventral_surface)
            # )
            # ),
            # fe.query(
            # x, (
            # fe.symbols.isin(
            # x, fe.symbols.Callosal_sulcus_lateral_dominant
            # ) & fe.symbols.anatomical_posterior_of(
            # x, fe.symbols.manual_Callosal_sulcus
            # ) & fe.symbols.anatomical_anterior_of(
            # x, fe.symbols.manual_Calcarine_sulcus
            # ) & fe.symbols.isin(x, ventral_surface)
            # )
            # ),
            # fe.query(
            # x, (
            # fe.symbols.
            # isin(x, fe.symbols.Callosal_sulcus_lateral_dominant) & ~fe.
            # symbols.posterior_of(x, fe.symbols.manual_Central_sulcus) &
            # fe.symbols.isin(x, ventral_surface)
            # )
            # )
        ]
        for query_id, query in enumerate(queries):
            results.append({
                'query_id': query_id,
                'subject_id': subject_id,
                'hemisphere': hemisphere,
                'found_sulci': [s.symbol_name for s in query.do()]
            })
