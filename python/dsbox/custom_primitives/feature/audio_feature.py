import numpy as np
import typing
import json
import argparse

from bbn_primitives.time_series import *
from d3m_metadata import container, hyperparams, params
from d3m_metadata.metadata import PrimitiveMetadata
from d3m_metadata.container import ndarray
from d3m_metadata.container import List
from d3m_metadata import hyperparams, metadata as metadata_module, params, container, utils
from primitive_interfaces.transformer import TransformerPrimitiveBase
from primitive_interfaces.unsupervised_learning import UnsupervisedLearnerPrimitiveBase

#from sklearn.random_projection import johnson_lindenstrauss_min_dim, GaussianRandomProjection
from primitive_interfaces.featurization import FeaturizationPrimitiveBase, CallResult
from builtins import int

Inputs = container.List[container.DataFrame]
Outputs = container.ndarray

#######################################################################################
def extract_feats(inputs, fext_pipeline = None,
                  resampling_rate = None):
    features = List()
    i = 0
    for idx, row in inputs.iterrows():
        if row[0] == '':
            features.append(np.array([]))
            continue
        #filename = os.path.join(audio_dir, row[0])
        #print(filename)
        audio_clip = inputs['filename'].values[0][0]
        sampling_rate = resampling_rate
        if 'start' in inputs.columns and 'end' in inputs.columns:
            start = int(sampling_rate * float(inputs.loc[idx]['start']))
            end = int(sampling_rate * float(inputs.loc[idx]['end']))
            audio_clip = audio_clip[start:end]

        audio = List[ndarray]([audio_clip], {
                    'schema': metadata_module.CONTAINER_SCHEMA_VERSION,
                    'structural_type': List[ndarray],
                    'dimension': {
                        'length': 1
                    }
                    })
        audio.metadata = audio.metadata.update((metadata_module.ALL_ELEMENTS,),
                            { 'structural_type': ndarray,
                              'semantic_type': 'http://schema.org/AudioObject' })
        # sampling_rate is not supported by D3M metadata v2018.1.5
        #audio.metadata = audio.metadata.update((0,),
        #                    { 'sampling_rate': sampling_rate })

        last_output = audio
        #print(audio_clip)
        for fext_step in fext_pipeline:
            #print('xxxxxxxxxxxxxxx')
            #print(fext_step)
            product = fext_step.produce(inputs = last_output)
            last_output = product.value

        features.append(last_output[0])

        i+=1

    return features
######################################################################################
# Build the feature extraction pipeline
resampling_rate = 16000
channel_mixer = ChannelAverager(
                hyperparams = ChannelAverager.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams'].defaults()
            )
dither = SignalDither(
                hyperparams = SignalDither.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams'].defaults()
            )
framer_hyperparams = SignalFramer.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
framer_custom_hyperparams = dict()
framer_custom_hyperparams['sampling_rate'] = resampling_rate
#if args.fext_wlen is not None:
#  framer_custom_hyperparams['frame_length_s'] = args.fext_wlen
#if args.fext_rel_wshift is not None:
#  framer_custom_hyperparams['frame_shift_s'] = args.fext_rel_wshift*args.fext_wlen
framer = SignalFramer(
                hyperparams = framer_hyperparams(
                    framer_hyperparams.defaults(), **framer_custom_hyperparams
                )
            )
mfcc_hyperparams = SignalMFCC.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
mfcc_custom_hyperparams = dict()
mfcc_custom_hyperparams['sampling_rate'] = resampling_rate
#if args.fext_mfcc_ceps is not None:
#  mfcc_custom_hyperparams['num_ceps'] = args.fext_mfcc_ceps
mfcc = SignalMFCC(
                hyperparams = mfcc_hyperparams(
                    mfcc_hyperparams.defaults(), **mfcc_custom_hyperparams
                )
            )
segm = UniformSegmentation(
            hyperparams = UniformSegmentation.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams'].defaults()
        )

segm_fitter_hyperparams = SegmentCurveFitter.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
segm_fitter_custom_hyperparams = dict()
#if args.fext_poly_deg is not None:
#  segm_fitter_custom_hyperparams['deg'] = args.fext_poly_deg
segm_fitter = SegmentCurveFitter(
                hyperparams = segm_fitter_hyperparams(
                    segm_fitter_hyperparams.defaults(), **segm_fitter_custom_hyperparams
                )
            )
fext_pipeline = [ channel_mixer, dither, framer, mfcc, segm, segm_fitter ]
######################################################################################

       
class Params(params.Params):
    y_dim: int
    #projection_param: dict

class Hyperparams(hyperparams.Hyperparams):
    eps = hyperparams.Uniform(lower=0.1, upper=0.5, default=0.2)

class AudioFeaturization(FeaturizationPrimitiveBase[Inputs, Outputs, Params, Hyperparams]):
    '''
    classdocs
    '''

    metadata = PrimitiveMetadata({
        "id": "dsbox.timeseries_featurization.bbn_audio_pipeline",
        "version": "v0.1.0",
        "name": "DSBox Audio Featurization",
        "description": "Featurization of Audio Data",
        "python_path": "d3m.primitives.dsbox.Encoder",
        "primitive_family": "DATA_PREPROCESSING",
        "algorithm_types": [ "ENCODE_ONE_HOT" ], # FIXME Need algorithm type
        "source": {
            "name": 'ISI',
            "uris": [ 'git+https://github.com/usc-isi-i2/dsbox-ta2' ]
            },
        ### Automatically generated
        # "primitive_code"
        # "original_python_path"
        # "schema"
        # "structural_type"
        ### Optional
        "keywords": [ "feature_extraction",  "timeseries"],
        # "installation": [ config.INSTALLATION ],
        #"location_uris": [],
        #"precondition": [],
        #"effects": [],
        #"hyperparms_to_tune": []
        })

    def __init__(self, *, hyperparams: Hyperparams, random_seed: int = 0, 
                 docker_containers: typing.Dict[str, str] = None) -> None:
        self.hyperparams = hyperparams
        self.random_seed = random_seed
        self.docker_containers = docker_containers
        self._model = None
        self._training_data = None
        #self._x_dim = 0
        #self._y_dim = 0

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        res = extract_feats(inputs,
                        fext_pipeline = fext_pipeline,
                        resampling_rate = resampling_rate)
        return CallResult(res, True, 1)

    def set_training_data(self, *, inputs: Inputs, outputs: Outputs) -> None:
        #if len(inputs) == 0:
        #    return
        #lengths = [x.shape[0] for x in inputs]
        #is_same_length = len(set(lengths)) == 1
        #if is_same_length:
        #    self._y_dim = lengths[0]
        #else:
        #    # Truncate all time series to the shortest time series
        #    self._y_dim = min(lengths)
        #self._x_dim = len(inputs)
        #self._training_data = np.zeros((self._x_dim, self._y_dim))
        #for i, series in enumerate(inputs):
        #    self._training_data[i, :] = series.iloc[:self._y_dim, 0]
        return
        
    def fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:
        #eps = self.hyperparams['eps']
        #n_components = johnson_lindenstrauss_min_dim(n_samples=self._x_dim, eps=eps)
        #if n_components > self._x_dim: 
        #    self._model = GaussianRandomProjection(n_components=self._x_dim)
        #else:
        #    self._model = GaussianRandomProjection(eps=eps)
        #self._model.fit(self._training_data)
        return
        
    def get_params(self) -> Params:
        #if self._model:
        #    return Params(y_dim=self._y_dim,
        #                  projection_param={'': self._model.get_params()})
        #else:
        #    return Params()
        return

    def set_params(self, *, params: Params) -> None:
        #self._y_dim = params['y_dim']
        #self._model = GaussianRandomProjection()
        #self._model.set_params(params['projection_param'])
        return
