"""
Musical key finder using the Krumhansl-Schmuckler key-finding algorithm.
Based on https://github.com/jackmcarthur/musical-key-finder
"""

import numpy as np
import librosa


class Tonal_Fragment(object):
    """
    Class that uses the librosa library to analyze the key that an audio file is in.
    
    Uses the Krumhansl-Schmuckler key-finding algorithm, which compares the chroma
    data to typical profiles of major and minor keys.
    
    Arguments:
        waveform: an audio file loaded by librosa, ideally separated out from any percussive sources
        sr: sampling rate of the audio, which can be obtained when the file is read with librosa
        tstart and tend: the range in seconds of the file to be analyzed; default to the beginning and end of file if not specified
    """
    
    def __init__(self, waveform, sr, tstart=None, tend=None):
        self.waveform = waveform
        self.sr = sr
        self.tstart = tstart
        self.tend = tend
        
        if self.tstart is not None:
            self.tstart = librosa.time_to_samples(self.tstart, sr=self.sr)
        if self.tend is not None:
            self.tend = librosa.time_to_samples(self.tend, sr=self.sr)
        self.y_segment = self.waveform[self.tstart:self.tend]
        self.chromograph = librosa.feature.chroma_cqt(y=self.y_segment, sr=self.sr, bins_per_octave=24)
        
        # chroma_vals is the amount of each pitch class present in this time interval
        self.chroma_vals = []
        for i in range(12):
            self.chroma_vals.append(np.sum(self.chromograph[i]))
        pitches = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        # dictionary relating pitch names to the associated intensity in the song
        self.keyfreqs = {pitches[i]: self.chroma_vals[i] for i in range(12)} 
        
        keys = [pitches[i] + ' major' for i in range(12)] + [pitches[i] + ' minor' for i in range(12)]

        # use of the Krumhansl-Schmuckler key-finding algorithm, which compares the chroma
        # data above to typical profiles of major and minor keys:
        maj_profile = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
        min_profile = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]

        # finds correlations between the amount of each pitch class in the time interval and the above profiles,
        # starting on each of the 12 pitches. then creates dict of the musical keys (major/minor) to the correlation
        self.min_key_corrs = []
        self.maj_key_corrs = []
        for i in range(12):
            key_test = [self.keyfreqs.get(pitches[(i + m) % 12]) for m in range(12)]
            # correlation coefficients (strengths of correlation for each key)
            self.maj_key_corrs.append(round(np.corrcoef(maj_profile, key_test)[1, 0], 3))
            self.min_key_corrs.append(round(np.corrcoef(min_profile, key_test)[1, 0], 3))

        # names of all major and minor keys
        self.key_dict = {**{keys[i]: self.maj_key_corrs[i] for i in range(12)}, 
                         **{keys[i + 12]: self.min_key_corrs[i] for i in range(12)}}
        
        # this attribute represents the key determined by the algorithm
        self.key = max(self.key_dict, key=self.key_dict.get)
        self.bestcorr = max(self.key_dict.values())
        
        # this attribute represents the second-best key determined by the algorithm,
        # if the correlation is close to that of the actual key determined
        self.altkey = None
        self.altbestcorr = None

        for key, corr in self.key_dict.items():
            if corr > self.bestcorr * 0.9 and corr != self.bestcorr:
                self.altkey = key
                self.altbestcorr = corr


def find_key(audio_file):
    """
    Find the musical key of an audio file.
    
    Args:
        audio_file: Path to audio file
        
    Returns:
        dict with 'key', 'correlation', 'alt_key', and 'alt_correlation' (if applicable)
    """
    # Load audio file
    y, sr = librosa.load(audio_file)
    
    # Separate harmonic and percussive components
    # Analysis is most accurate using only the harmonic part
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    
    # Create tonal fragment and analyze
    tonal_fragment = Tonal_Fragment(y_harmonic, sr)
    
    result = {
        'key': tonal_fragment.key,
        'correlation': tonal_fragment.bestcorr,
        'alt_key': tonal_fragment.altkey,
        'alt_correlation': tonal_fragment.altbestcorr
    }
    
    return result
