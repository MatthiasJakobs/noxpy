import numpy as np
import xml.etree.ElementTree as ET
import datetime
import struct

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
            header['properties'] = parse_properties(elem)
        else:
            header[name.lower()] = value

    for key in ['scale', 'offset', 'samplingrate']:
        header[key] = float(header[key])

    return header

def parse_header(f, length):
    # Read header. Its XML. Filter out trailing zeros.
    header_raw_bytes = [read_uint16(f) for _ in range(length//2)]
    header_raw = ''.join([chr(b) for b in header_raw_bytes if b != 0])
    header_raw = header_raw.replace('°', 'Angle') # Probably important for further parsing the XML? 
    header_xml = ET.fromstring(header_raw)
    return header_xml_to_dict(header_xml)


def parse_ndf(path, verbose=False):
    channel = {}
    with open(path, 'rb') as f:

        magic = f.read(4) # Starts with 'NOX\x03' string

        channel['start'] = []
        channel['end'] = []
        channel['gap'] = []
        channel['data'] = []
        channel['t'] = []

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
                # Get header
                header = parse_header(f, length)
                channel['header'] = header
                channel['sampling_rate'] = header['samplingrate']
                channel['offset'] = header['offset']
                channel['scale'] = header['scale']
            elif typ == 1:
                # Get Hash
                hash_ = ''.join([chr(read_uint16(f)) for _ in range(length//2)])
                # TODO: Lets do nothing with this for now
            elif typ == 512:
                # Start time
                d = ''.join([chr(read_uint16(f)) for _ in range(length//2)])
                if length == 36:
                    print('Start time length not correct. TODO')
                    start_time = datetime.datetime.strptime(d, '%Y%m%dT%H%M%S')
                else:
                    start_time = datetime.datetime.strptime(d, '%Y%m%dT%H%M%S.%f')
                channel['start'].append(start_time)
                if len(channel['start']) > 1:
                    channel['gap'].append((channel['start'][-1] - channel['end'][-1]))

            elif typ == 144:
                raise NotImplementedError('144')
            elif typ == 514:
                # Sampling rate again?
                # TODO: What is this?
                sampling_rate = read_double(f)
                if 'sampling_rate' in channel:
                    if abs(channel['sampling_rate']-sampling_rate) > 1e-5:
                        print(channel['sampling_rate'], sampling_rate)
                        raise RuntimeError('Sampling rate changed')
                else:
                    channel['sampling_rate'] = sampling_rate
            elif typ == 513:
                # Data
                if channel['header']['format'] == 'Int16':
                    raw = [read_int16(f) for _ in range(length//2)]
                    channel['data'].append(channel['scale']*np.array(raw)+channel['offset'])
                else:
                    raise NotImplementedError()

                if length > 0:
                    offsets_seconds = np.arange(length//2) / channel['sampling_rate']
                    offsets_delta = [datetime.timedelta(seconds=offset) for offset in offsets_seconds]
                    new_t = [channel['start'][-1] + offset for offset in offsets_delta]
                    channel['t'].extend(new_t)
                    channel['end'].append(new_t[-1])
            else:
                raise NotImplementedError('Unknown code', typ, 'with length', length)
    if verbose:
        print(channel)
    return channel

def main():
    import glob
    ndfs = list(glob.glob('**/*.ndf'))
    for ndf_file in ndfs:
        print(ndf_file)
        parse_ndf(ndf_file)

if __name__ == '__main__':
    main()
