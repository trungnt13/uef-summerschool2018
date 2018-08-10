import featext.feature_processing
from featext.mfcc import Mfcc
import numpy as np
import system.gmm_em as gmm
import system.ivector as ivector
import system.backend as backend


dataFolder = 'C:\\Users\\vvestman\\Desktop\\recordings\\'

speakers = ['jackson', 'nicolas', 'theo']
n_speakers = len(speakers)
n_digits = 10   # 0 - 9
n_sessions = 50   # 0 - 49


#### Feature extraction:
# Let us train the spoken digit recognition system with speakers Jackson and Nicolas and test with Theo.

mfcc = Mfcc()
mfcc.frame_duration = 0.025
mfcc.frame_overlap_duration = 0.01
mfcc.sad_threshold = 60
mfcc.include_deltas = 1
mfcc.include_double_deltas = 1
mfcc.include_base_coeffs = 1
mfcc.include_energy = 1
mfcc.n_coeffs = 20
mfcc.rasta_coeff = 0
mfcc.pre_emphasis = 0
mfcc.cmvn = 1
mfcc.initialize()

all_features = np.empty((n_speakers, n_digits, n_sessions), dtype=object)
for speaker in range(n_speakers):
    for digit in range(n_digits):
        for session in range(n_sessions):
            filename = '{}{}_{}_{}.wav'.format(dataFolder, digit, speakers[speaker], session)
            all_features[speaker, digit, session] = featext.feature_processing.extract_features_from_file(filename, mfcc)


feature_dim = all_features[0, 0, 0].shape[0]

#### Train GMM for every digit:
n_components = 64
digit_models = []
for digit in range(n_digits):
    model = gmm.GMM(ndim=feature_dim, nmix=n_components, ds_factor=1, final_niter=10, nworkers=2)
    model.fit(np.reshape(all_features[0:2, digit, :], (-1,)))
    digit_models.append(model)

### Scoring (Based on GMM log likelihoods):
testFeatures = np.reshape(all_features[2, :, :], (-1))
n_tests = testFeatures.size
true_labels = np.repeat(np.arange(n_digits), n_sessions)

scores = np.zeros((n_digits, n_tests))
for test_segment in range(n_tests):
    for digit in range(n_digits):
        scores[digit, test_segment] = np.mean(digit_models[digit].compute_log_lik(testFeatures[test_segment]))


classifications = np.argmax(scores, axis=0)
n_correct = sum(classifications == true_labels)
print('Correct classifications: {} / {} ({:.1f} %)\n'.format(n_correct, n_tests, n_correct / n_tests * 100))

# EXERCISE: Implement GMM-based scoring with universal background model (UBM)

#### Universal background model (UBM) training:


#### GMM adaptation:


#### Scoring trials (all test files vs. all models):



###### I-vector / PLDA system
#### Sufficient statistics (Baum-Welch statistics) extraction:
# all_stats = np.empty((n_speakers, n_digits, n_sessions), dtype=object)
# for speaker in range(n_speakers):
#     for digit in range(n_digits):
#         for session in range(n_sessions):
#             N, F = ubm.compute_centered_stats(all_features[speaker, digit, session])
#             all_stats[speaker, digit, session] = (N, F)


#### Total variability matrix training:
# ivector_dim = 50;
# tMatrix = ivector.TMatrix(ivector_dim, feature_dim, n_components, niter=5, nworkers=2)
#
# tMatrix.train(np.reshape(all_stats[0:2, :, :], (-1,)), ubm)

#### I-vector extraction:

# extractor = ivector.Ivector(ivector_dim, feature_dim, n_components)
# extractor.initialize(ubm, tMatrix.Tm)
# ivectors = np.empty((ivector_dim, n_speakers, n_digits, n_sessions))
# for speaker in range(n_speakers):
#     for digit in range(n_digits):
#         for session in range(n_sessions):
#             ivectors[:, speaker, digit, session] = extractor.extract(*all_stats[speaker, digit, session])


#### I-vector processing:
# training_vectors = np.reshape(ivectors[:, 0:2, :, :], (ivector_dim, -1), order='F')
# training_labels = np.tile(np.arange(n_digits).repeat(2), n_sessions)
# model_vectors = np.reshape(np.mean(ivectors[:, 0:2, :, :], (1, 3)), (ivector_dim, -1), order='F')
# test_vectors = np.reshape(ivectors[:, 2, :, :], (ivector_dim, -1), order='F')
# true_labels = np.tile(np.arange(n_digits), n_sessions)
#
# center = backend.compute_mean(training_vectors)
# w = backend.calc_white_mat(np.cov(training_vectors))
# training_vectors = backend.preprocess(training_vectors, center, w)
# model_vectors = backend.preprocess(model_vectors, center, w)
# test_vectors = backend.preprocess(test_vectors, center, w)

#### PLDA training:
#### (probabilistic linear discriminant analysis)
# latent_dim = 40;
# plda = backend.GPLDA(ivector_dim, latent_dim, niter=20)
# plda.train_em(training_vectors, training_labels)


#### Scoring:
# scores = plda.score_trials(model_vectors, test_vectors)
# # scores = backend.cosine_similarity(modelVectors, testVectors)
# classifications = np.argmax(scores, axis=0)
# n_correct = sum(classifications == true_labels)
# print('Correct classifications: {} / {} ({:.1f} %)\n'.format(n_correct, n_tests, n_correct / n_tests * 100))

