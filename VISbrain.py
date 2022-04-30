import os
from mne import io
from visbrain.gui import Sleep
from visbrain.io import download_file, path_to_visbrain_data

download_file("sleep_brainvision.zip", unzip=True, astype='example_data')
target_path = path_to_visbrain_data(folder='example_data')

dfile = os.path.join(target_path, 'sub-02.vhdr')
hfile = os.path.join(target_path, 'sub-02.hyp')

# Read raw data using MNE-python :
raw = io.read_raw_brainvision(vhdr_fname=dfile, preload=True)

# Extract data, sampling frequency and channels names
data, sf, chan = raw._data, raw.info['sfreq'], raw.info['ch_names']

# Now, pass all the arguments to the Sleep module :
Sleep(data=data, sf=sf, channels=chan, hypno=hfile).show()