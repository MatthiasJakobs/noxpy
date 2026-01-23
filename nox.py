import numpy as np
import xml.etree.ElementTree as ET
import datetime
import struct

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
                # This is the intended way as np.datetime64 cannot take a format string
                start_time = np.datetime64(start_time, 'ns')
                channel['start'].append(start_time)
                if len(channel['start']) > 1:
                    channel['gap'].append((channel['start'][-1] - channel['end'][-1]))

            elif typ == 144:
                raise NotImplementedError('144')
            elif typ == 514:
                # Sampling rate again?
                sampling_rate = read_double(f)
                channel['sampling_rate'] = sampling_rate
                # TODO: I'm pretty sure the sampling rate is in Hz anyway.
                #       Maybe it makes more sense to round to an integer.
                #       That way they would not change over time

            elif typ == 513:
                # Data
                fmt = channel['header']['format']
                if fmt == 'Int16':
                    raw = np.frombuffer(f.read(length), dtype=np.int16)
                    channel['data'].append(channel['scale']*np.array(raw)+channel['offset'])
                elif fmt == 'Byte':
                    raw = np.frombuffer(f.read(length), dtype=np.uint8)
                    raw = mu2lin(raw)
                    channel['data'].append(channel['scale']*np.array(raw)+channel['offset'])
                else:
                    raise NotImplementedError('Unknown format', channel['header']['format'])

                if length > 0:
                    start = np.datetime64(channel['start'][-1], 'ns')
                    offset_ns = (np.arange(length // 2, dtype=np.int64)*1000000000) // channel['sampling_rate']
                    new_t = start + offset_ns.astype('timedelta64[ns]')
                    channel['t'].append(new_t)
                    channel['end'].append(new_t[-1])
            else:
                raise NotImplementedError('Unknown code', typ, 'with length', length)
    if verbose:
        print(channel)
    return channel

def main():
    import glob
    import tqdm
    ndfs = list(glob.glob('**/*.ndf'))
    n = len(ndfs)
    print(n)
    correct = 0
    for ndf_file in tqdm.tqdm(ndfs):
        try:
            parse_ndf(ndf_file)
            correct += 1
        except:
            print(ndf_file)
    print(correct, n)



if __name__ == '__main__':
    main()
