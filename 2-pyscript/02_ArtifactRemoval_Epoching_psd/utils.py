import numpy as np
import matplotlib.pyplot as plt
import mne
from mne import create_info
from mne import Epochs, find_events

def df_to_raw(df, electrodes):
    sfreq = 125
    ch_names = electrodes.copy()
    ch_names.append('Marker')
    ch_types = ['eeg'] * (len(ch_names)-1) + ['stim']
    ten_twenty_montage = mne.channels.make_standard_montage('standard_1020')

    df = df.loc[:,ch_names]
    df = df.T  #mne looks at the tranpose() format
    df[:-1] *= 1e-6  #convert from uVolts to Volts (mne assumes Volts data)

    info = create_info(ch_names=ch_names, ch_types=ch_types, sfreq=sfreq)

    raw = mne.io.RawArray(df, info)
    raw.set_montage(ten_twenty_montage)

    #try plotting the raw data of its power spectral density
    raw.plot_psd()

    return raw

def getEpochs(raw, event_id, tmin, tmax, picks):

    #epoching
    events = find_events(raw)
    
    #reject_criteria = dict(mag=4000e-15,     # 4000 fT
    #                       grad=4000e-13,    # 4000 fT/cm
    #                       eeg=100e-6,       # 150 μV
    #                       eog=250e-6)       # 250 μV

    reject_criteria = dict(eeg=100e-6)  #most voltage in this range is not brain components

    epochs = Epochs(raw, events=events, event_id=event_id, 
                    tmin=tmin, tmax=tmax, baseline=None, preload=True,verbose=False, picks=picks)  #8 channels
    print('sample drop %: ', (1 - len(epochs.events)/len(events)) * 100)

    return epochs

def get_psd(raw, fmin, fmax, filter=True):
    '''
    return log-transformed power spectra density, freq, mean and std 
    '''
    raw_copy = raw.copy()
    if(filter):
        raw_copy.filter(fmin, fmax, method='iir')
        # if drift == "drift":
        #     raw_copy.filter(fmin, fmax, method='iir')
        # else:
        #     raw_copy.filter(1, 40, method='iir')
#             raw_copy.plot_psd()     
    psd, freq = mne.time_frequency.psd_welch(raw_copy,n_fft = 96, verbose=False)
    psd =  np.log10(psd)
    mean = psd.mean(0)
    std = psd.std(0)
    return psd, freq, mean, std
#     return raw_copy

def plot_psd(raw):
    psd, freq, mean, std = get_psd(raw)
    fig, ax = plt.subplots(figsize=(10,5))
    for i in range(8):
        ax.plot(freq,psd[i] ,label=raw.info['ch_names'][i], lw=1, alpha=0.6)
    ax.fill_between(250//2, mean - std, mean + std, color='k', alpha=.5)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Amplitube (dBV)')
    ax.set_title('EEG of ')
    ax.legend()
    plt.show()
