from copy import deepcopy
import pandas as pd
import pathlib
import numpy as np
import xml.etree.ElementTree as ET
import struct
import sqlite3
from datetime import datetime

def read_uint8(f):
    return struct.unpack('<B', f.read(1))[0]

def read_uint16(f):
    return struct.unpack('<H', f.read(2))[0]

def read_int16(f):
    return struct.unpack('<h', f.read(2))[0]

def read_uint32(f):
    return struct.unpack('<I', f.read(4))[0]

def read_int32(f):
    return struct.unpack('<i', f.read(4))[0]

def read_double(f):
    return struct.unpack('<d', f.read(8))[0]

# TODO: No idea if this is correct
def mu2lin(mu_bytes):
    mu_bytes = np.asarray(mu_bytes, dtype=np.uint8)

    u = np.bitwise_not(mu_bytes)

    sign = u & 0x80
    exponent = (u >> 4) & 0x07
    mantissa = u & 0x0F

    magnitude = ((mantissa << 3) + 0x84) << exponent
    pcm = magnitude - 0x84

    pcm = np.where(sign != 0, -pcm, pcm)

    return pcm.astype(np.int16)

# .NET ticks (microseconds / 10, starting from 01.01.0001)
def ticks_to_datetime(ticks):
    return np.datetime64('0001-01-01') + np.timedelta64(int(ticks) // 10, 'us')

def header_xml_to_dict(header_xml):

    def parse_properties(prop):
        d = {}
        for item in prop:
            k = item[0].text.lower()
            t = item[1].text
            v = item[2].text
            if 'Int' in t:
                v = int(v)
            d[k] = v

        return d

    header = {}
    for elem in header_xml:
        name, value = elem.tag, elem.text
        if name == 'Properties':
            header['properties'] = parse_properties(elem).copy()
        else:
            header[name.lower()] = value

    for key in ['scale', 'offset', 'samplingrate']:
        header[key] = float(header[key])
    for key in ['channelnumber']:
        header[key] = int(header[key])

    return header

def parse_header(f, length):
    # Read header. Its XML. Filter out trailing zeros.
    header_raw_bytes = [read_uint16(f) for _ in range(length//2)]
    header_raw = ''.join([chr(b) for b in header_raw_bytes if b != 0])
    header_raw = header_raw.replace('°', 'Angle') # Probably important for further parsing the XML? 
    header_xml = ET.fromstring(header_raw)
    return header_xml_to_dict(header_xml)

# Noxturnal seems to reference signals by default from raw data.
# We model this here by doing the referencing manually
COMPOSITE_CHANNELS = {
    'C3-M2': {'main': 'C3', 'type': 'EEG-C3-M2', 'refs': ['M2']},
    'F3-M2': {'main': 'F3', 'type': 'EEG-F3-M2', 'refs': ['M2']},
    'O1-M2': {'main': 'O1', 'type': 'EEG-O1-M2', 'refs': ['M2']},
    'C4-M1': {'main': 'C4', 'type': 'EEG-C4-M1', 'refs': ['M1']},
    'F4-M1': {'main': 'F4', 'type': 'EEG-F4-M1', 'refs': ['M1']},
    'O2-M1': {'main': 'O2', 'type': 'EEG-O2-M1', 'refs': ['M1']},
    'AF3-E3E4': {'main': 'AF3', 'type': 'EEG-AF3-E3E4', 'refs': ['E3', 'E4']},
    'AF4-E3E4': {'main': 'AF4', 'type': 'EEG-AF4-E3E4', 'refs': ['E3', 'E4']},
    'AF7-E3E4': {'main': 'AF7', 'type': 'EEG-AF7-E3E4', 'refs': ['E3', 'E4']},
    'AF8-E3E4': {'main': 'AF8', 'type': 'EEG-AF8-E3E4', 'refs': ['E3', 'E4']},
    'E1-E4': {'main': 'E1', 'type': 'EOG-E1-E4', 'refs': ['E4']},
    'E1-M2': {'main': 'E1', 'type': 'EOG-E1-M2', 'refs': ['M2']},
    'E2-AFZ': {'main': 'E2', 'type': 'EOG-E2-AFZ', 'refs': ['AFZ']},
    'E2-E3': {'main': 'E2', 'type': 'EOG-E2-E3', 'refs': ['E3']},
    'E2-M1': {'main': 'E2', 'type': 'EOG-E2-M1', 'refs': ['M1']},
    'E3-AFZ': {'main': 'E3', 'type': 'EOG-E3-AFZ', 'refs': ['AFZ']},
    'LA-RA': {'main': 'EKG LA', 'type': 'EKG-LA-RA', 'refs': ['EKG RA']},
    'LF-LA': {'main': 'EKG LF', 'type': 'EKG-LF-LA', 'refs': ['EKG LA']},
    'LF-RA': {'main': 'EKG LF', 'type': 'EKG-LF-RA', 'refs': ['EKG RA']},
}

class NoxReader:

    def __init__(self, path):
        self.path = pathlib.Path(path)
        self.channel_headers = {}
        self.channel_data_locations = {}

        self._read_channel_headers()
        self._load_recording_metadata()

        # Check if composite channels are present
        channel_labels = [c['label'] for _, c in self.channel_headers.items()]
        for cchannel_name, d in COMPOSITE_CHANNELS.items():
            main = d['main']
            refs = d['refs']
            t = d['type']
            if main in channel_labels and all([r in channel_labels for r in refs]):
                header = deepcopy(self.channel_headers[channel_labels.index(main)])
                header['label'] = cchannel_name
                header['type'] = t
                del header['hash']
                idx = self.n_channels
                self.channel_headers[idx] = header
                self.n_channels += 1

    def _load_recording_metadata(self):
        db_file = self.path.joinpath('Data.ndb')
        con = sqlite3.connect(db_file)
        cur = con.cursor()

        # Initialize raw data dicts
        self._recording_info = {}
        self._subject_info = {}
        self._device_info = {}
        self._technician_info = {}

        res = cur.execute('SELECT key, type, value, name FROM internal_property;').fetchall()
        for (key, datatype, value, name) in res:
            if datatype == 'Long':
                value = int(value)
            elif datatype == 'Bool':
                value = eval(value)
            elif datatype == 'Double':
                value = float(value)
            elif datatype == 'Ticks':
                value = np.datetime64('0001-01-01') + np.timedelta64(int(value) // 10, 'us')
            elif datatype == 'Text':
                value = str(value)
            else:
                raise Exception('Unknown datatype', datatype)

            if name == 'SubjectInfo':
                self._subject_info[key] = value
            elif name == 'DeviceInfo':
                self._device_info[key] = value
            elif name == 'RecordingInfo':
                self._recording_info[key] = value
            elif name == 'TechnicianInfo':
                self._technician_info[key] = value

    def _parse_ndf_header(self, path):
        header = {}
        data_chunks = {
            'filepath': path,
            'chunk_information': [],
        }
        with open(path, 'rb') as f:
            magic = f.read(4) 
            assert magic == b'NOX\x03'

            while True:
                current_position = f.tell()
                f.seek(0, 2)  # Move to end
                file_size = f.tell()
                f.seek(current_position)  # Move back

                if current_position == file_size:
                    break

                # Read block and determine action
                typ = read_uint16(f)
                length = read_uint32(f)

                if typ == 256:
                    # Get NDF header
                    ndf_header = parse_header(f, length)
                    header = ndf_header
                elif typ == 1:
                    # Get Hash
                    hash_ = ''.join([chr(read_uint16(f)) for _ in range(length//2)])
                    header['hash'] = hash_
                elif typ == 144:
                    # Set global sampling rate
                    sampling_rate = read_double(f)
                    header['samplingrate'] = float(sampling_rate)
                elif typ == 512:
                    # Start time of chunk
                    d = ''.join([chr(read_uint16(f)) for _ in range(length//2)])
                    if length == 36:
                        print('Start time length not correct. TODO')
                        start_time = datetime.strptime(d, '%Y%m%dT%H%M%S')
                    else:
                        start_time = datetime.strptime(d, '%Y%m%dT%H%M%S.%f')

                    start_time = np.datetime64(start_time, 'ns')
                    if 'start_time' not in header:
                        # First time, save for later
                        header['start_time'] = start_time

                    chunk_info = {'start_time': start_time}
                    data_chunks['chunk_information'].append(chunk_info)
                elif typ == 513:
                    # Data blocks
                    fmt = header['format']

                    current_chunk = data_chunks['chunk_information'][-1]
                    current_chunk['start_position'] = f.tell()
                    current_chunk['length'] = length
                    current_chunk['fmt'] = fmt
                    data_chunks['chunk_information'][-1] = current_chunk

                    # Done here, skip forward
                    f.seek(length, 1)
                    continue
                elif typ == 514:
                    sampling_rate = read_double(f)
                    header['samplingrate'] = float(sampling_rate)

                    current_chunk = data_chunks['chunk_information'][-1]
                    current_chunk['samplingrate'] = sampling_rate
                    data_chunks['chunk_information'][-1] = current_chunk
                else:
                    raise NotImplementedError('Unknown code', typ, 'with length', length)

        return header, data_chunks

    def _read_channel_headers(self):
        channel_paths = list(self.path.glob('*.ndf'))
        for i, channel_path in enumerate(channel_paths):
            channel_header, channel_data_location = self._parse_ndf_header(channel_path)
            self.channel_headers[i] = channel_header
            self.channel_data_locations[i] = channel_data_location
        self.n_channels = len(channel_paths)

    def getSignalHeader(self, idx):
        return self.channel_headers[idx]

    def getSignalHeaders(self):
        return list(self.channel_headers.values())

    def getSignalLabels(self):
        return [head['label'] for head in self.getSignalHeaders()]

    def getStartdatetime(self):
        return self._recording_info['RecordingStart']

    def getFileDuration(self, datarecord_duration=10.0):
        # Compatibility with EDF files: Assume a fixed block size for each data record and round accordingly
        rec_start = self._recording_info['RecordingStart']
        rec_stop = self._recording_info['RecordingStop']
        file_duration_seconds = (rec_stop - rec_start) / np.timedelta64(1, 's')
        n_datarecords = np.ceil(file_duration_seconds / datarecord_duration)
        file_duration = float(n_datarecords * datarecord_duration)

        return file_duration

    def getSampleFrequency(self, idx):
        return self.channel_headers[idx]['samplingrate']

    def getNSamples(self):
        recording_seconds = self.getFileDuration()

        def _getNSamples(idx):
            sr = float(round(self.getSampleFrequency(idx)))
            return int(recording_seconds * sr)

        return [_getNSamples(idx) for idx in range(self.n_channels)]

    def readSignal(self, idx, start=0, n=None, digital=False):
        label = self.getSignalLabels()[idx]
        if label in COMPOSITE_CHANNELS:
            # Compute composite channels
            main = COMPOSITE_CHANNELS[label]['main']
            refs = COMPOSITE_CHANNELS[label]['refs']
            ref_indcs = [self.getSignalLabels().index(r) for r in refs]
            main_idx = self.getSignalLabels().index(main)
            main_signal = self.readSignal(main_idx, start=start, n=n, digital=digital)
            ref_signals = [self.readSignal(r_idx, start=start, n=n, digital=digital) for r_idx in ref_indcs]
            ref_signal = np.stack(ref_signals, 1).mean(1)
            return main_signal - ref_signal
            
        data_chunks = self.channel_data_locations[idx]
        filepath = data_chunks['filepath']

        # Actually get the data now
        signal = []
        with open(filepath, 'rb') as f:
            for chunk_info in data_chunks['chunk_information']:
                fmt = chunk_info['fmt']
                tell = chunk_info['start_position']
                length = chunk_info['length']
                start_time = chunk_info['start_time']
                sr = chunk_info['samplingrate']

                f.seek(tell, 0)
                if fmt == 'Int16':
                    _signal = np.frombuffer(f.read(length), dtype=np.int16)
                elif fmt == 'Int32':
                    _signal = np.frombuffer(f.read(length), dtype=np.int32)
                elif fmt == 'UInt32':
                    _signal = np.frombuffer(f.read(length), dtype=np.uint32)
                elif fmt == 'Byte':
                    _signal = np.frombuffer(f.read(length), dtype=np.uint8)
                else:
                    raise NotImplementedError('Unknown data format', fmt)

                # Create corresponding timestamps
                timestamps = pd.date_range(start=start_time, periods=len(_signal), freq=pd.to_timedelta(1/sr, unit='s'))

                # Resample
                sr_rounded = float(round(self.getSignalHeader(idx)['samplingrate'])) # Just in case they are different between chunks
                period_ns = round(1_000_000_000 / sr_rounded)
                s = pd.Series(_signal, index=timestamps).resample(f'{period_ns}ns').mean().ffill()

                signal.append(s)

        if len(signal) > 1:
            series_start = signal[0].index[0]
            series_end = signal[-1].index[-1]
            freq = signal[0].index.freq # They are equal after resampling anyway
            timestamps = pd.date_range(start=series_start, end=series_end, freq=freq)
            signal = pd.concat(signal).reindex(timestamps).ffill()
        else:
            # TODO: Maybe this condition is not needed. But I'm not sure about the idempotency of timestamp magic right now
            signal = signal[0]
            timestamps = signal.index
            signal = signal.to_numpy().squeeze()

        recording_start = self.getStartdatetime()

        # TODO: Still not 100%. Sometimes off by 1, sometimes by -1, sometimes 2 
        #       Good enough for now
        pad_front = int(np.round((int(timestamps[0].to_numpy() - recording_start) / 1e9) * sr_rounded)) 
        pad_back = self.getNSamples()[idx] - pad_front - len(signal)

        signal = np.concatenate([np.zeros((pad_front)), signal, np.zeros((pad_back))])
        if not digital:
            scale = self.getSignalHeader(idx)['scale']
            offset = self.getSignalHeader(idx)['offset']
            signal = scale * signal + offset

        # Manual preprocessing to get rid of strong outliers
        if self.getSignalHeader(idx)['unit'] == 'bpm': 
            max_value = 200
            signal[np.where(signal >= max_value)] = np.nan
            signal = pd.DataFrame(signal).ffill().to_numpy().squeeze()

        if self.getSignalHeader(idx)['unit'] == '%': 
            max_value = 100
            signal[np.where(signal >= max_value)] = np.nan
            signal = pd.DataFrame(signal).ffill().to_numpy().squeeze()

        if n is None:
            end = None 
        else:
            end = start + n

        return signal[start:end]

    def getAnnotations(self, return_manual_corrections=True):

        def _apply_corrections(df_base, df_corr):
            df_base = df_base.copy()
            df_corr = df_corr.copy()

            key_cols = ['start', 'end']

            base_idx = pd.MultiIndex.from_frame(df_base[key_cols])

            # split corrections
            corr_delete = df_corr[df_corr['is_deleted']].drop(columns=['is_deleted'])
            corr_update = df_corr[~df_corr['is_deleted']].drop(columns=['is_deleted'])

            # --- 1. delete rows ---
            if not corr_delete.empty:
                delete_idx = pd.MultiIndex.from_frame(corr_delete[key_cols])
                keep_mask = ~base_idx.isin(delete_idx)
                df_base = df_base.loc[keep_mask].reset_index(drop=True)
                base_idx = pd.MultiIndex.from_frame(df_base[key_cols])

            # --- 2. update existing rows ---
            if not corr_update.empty:
                df_base = df_base.reset_index(drop=False).rename(columns={'index': 'base_id'})
                corr_update = corr_update.reset_index(drop=False).rename(columns={'index': 'corr_id'})

                # --- Rule 1: Same start, same label, different end ---
                r1 = corr_update.merge(
                    df_base,
                    on=['start', 'label'],
                    how='inner',
                    suffixes=('_corr', '_base')
                )

                r1 = r1[r1['end_corr'] != r1['end_base']]
                df_base.loc[df_base['base_id'].isin(r1['base_id']), 'end'] = (r1.drop_duplicates('base_id').set_index('base_id')['end_corr'])
                used_corr_ids = set(r1['corr_id'])
                corr_remaining = corr_update[~corr_update['corr_id'].isin(used_corr_ids)]

                # --- Rule 2: Same start, same end, different label ---
                r2 = corr_remaining.merge(
                    df_base,
                    on=['start', 'end'],
                    how='inner',
                    suffixes=('_corr', '_base')
                )

                r2 = r2[r2['label_corr'] != r2['label_base']]
                df_base.loc[df_base['base_id'].isin(r2['base_id']), 'label'] = (r2.drop_duplicates('base_id').set_index('base_id')['label_corr'])
                used_corr_ids |= set(r2['corr_id'])
                corr_remaining = corr_remaining[~corr_remaining['corr_id'].isin(used_corr_ids)]

                # --- Rule 3: Same end, same label, different start ---
                r3 = corr_remaining.merge(
                    df_base,
                    on=['end', 'label'],
                    how='inner',
                    suffixes=('_corr', '_base')
                )

                r3 = r3[r3['start_corr'] != r3['start_base']]
                df_base.loc[df_base['base_id'].isin(r3['base_id']), 'start'] = (r3.drop_duplicates('base_id').set_index('base_id')['start_corr'])
                used_corr_ids |= set(r3['corr_id'])
                corr_remaining = corr_remaining[~corr_remaining['corr_id'].isin(used_corr_ids)]

                # Append unmatched corrections
                df_base = pd.concat([df_base.drop(columns='base_id'), corr_remaining.drop(columns='corr_id')], ignore_index=True )

            df_base = df_base.sort_values('start').drop_duplicates().reset_index(drop=True)

            return df_base

        db_file = self.path.joinpath('Data.ndb')
        con = sqlite3.connect(db_file)
        cur = con.cursor()

        # Get automatic annotations first
        query = 'SELECT t1.starts_at AS start, t1.ends_at AS end, t1.type AS label FROM temporary_scoring_marker t1 JOIN temporary_scoring_key t2 ON t1.key_id = t2.id WHERE t2.type = "Automatic"'
        df = pd.read_sql_query(query, con)

        df['start'] = pd.to_datetime(df['start'].map(ticks_to_datetime))
        df['end'] = pd.to_datetime(df['end'].map(ticks_to_datetime))
        df = df.sort_values('start').drop_duplicates().reset_index(drop=True)

        if not return_manual_corrections:
            return df

        # Iterate through manual annotations
        res = cur.execute('SELECT id FROM temporary_scoring_key WHERE type = "Manual" ORDER BY id;').fetchall()
        for (manual_id,) in res:
            query = f'SELECT starts_at AS start, ends_at AS end, type AS label, is_deleted FROM temporary_scoring_marker WHERE key_id = {manual_id};'
            correction = pd.read_sql_query(query, con)
            correction['start'] = pd.to_datetime(correction['start'].map(ticks_to_datetime))
            correction['end'] = pd.to_datetime(correction['end'].map(ticks_to_datetime))
            correction['is_deleted'] = correction['is_deleted'].astype(bool)
            correction = correction.sort_values('start').drop_duplicates().reset_index(drop=True)

            df = _apply_corrections(df, correction).drop_duplicates()

        # Crop annotations to fit inside the recording time
        record_start = self.getStartdatetime()
        record_stop = record_start + np.timedelta64(int(self.getFileDuration()), 's')
        df = df[(df['start'] >= record_start) & (df['start'] <= record_stop)].reset_index(drop=True)

        return df

    def getPatientCode(self):
        return self._subject_info.get('ID', '')

    def getPatientName(self):
        first_name = self._subject_info.get('FirstName', '')
        middle_name = self._subject_info.get('MiddleName', '')
        last_name = self._subject_info.get('LastName', '')
        return ' '.join(f'{first_name} {middle_name} {last_name}'.split())

    def getPatientAdditional(self):
        return self._subject_info.copy()
