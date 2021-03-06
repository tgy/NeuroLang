import logging
import os
from ..regions import region_set_from_masked_data
from pkg_resources import resource_exists, resource_filename
try:
    import neurosynth as ns
except ModuleNotFoundError:
    raise ImportError("Neurosynth not installed in the system")


class NeuroSynthHandler(object):
    def __init__(self, ns_dataset=None):
        self._dataset = ns_dataset

    def ns_region_set_from_term(
        self, terms, frequency_threshold=0.05, q=0.01,
        prior=0.5, image_type=None
    ):

        if image_type is None:
            image_type = f'association-test_z_FDR_{q}'

        if self._dataset is None:
            dataset = self.ns_load_dataset()
            self._dataset = dataset
        studies_ids = self._dataset.get_studies(
            features=terms, frequency_threshold=frequency_threshold
        )
        ma = ns.meta.MetaAnalysis(self._dataset, studies_ids, q=q, prior=prior)
        data = ma.images[image_type]
        masked_data = self._dataset.masker.unmask(data)
        affine = self._dataset.masker.get_header().get_sform()
        dim = self._dataset.masker.dims
        region_set = region_set_from_masked_data(masked_data, affine, dim)
        return region_set

    def ns_load_dataset(self):

        if resource_exists('neurolang.frontend',
                           'neurosynth_data/dataset.pkl'):
            file = resource_filename('neurolang.frontend',
                                     'neurosynth_data/dataset.pkl')
            dataset = ns.Dataset.load(file)
        else:
            path = resource_filename('neurolang.frontend', 'neurosynth_data')
            logging.info(
                f'Downloading neurosynth database'
                f' and features in path: {path}'
            )
            dataset = self.download_ns_dataset(path)

        return dataset

    @staticmethod
    def download_ns_dataset(path):
        if not os.path.exists(path):
            os.makedirs(path)
        ns.dataset.download(path=path, unpack=True)
        dataset = ns.Dataset(os.path.join(path, 'database.txt'))
        dataset.add_features(os.path.join(path, 'features.txt'))
        dataset.save(os.path.join(path, 'dataset.pkl'))
        return dataset
