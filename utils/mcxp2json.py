#!/usr/bin/env python3
"""Convert an MCX Studio 1 project (.mcxp) into mcx JSON inputs.

A .mcxp is an INI file with one section per session, holding what the old
GUI's fields contained rather than what mcx reads.  MCX Studio 2 has no
format of its own -- a simulation is a .json and nothing else -- so a
project with six sessions in it becomes six .json files.

    mcxp2json.py mcx_demo.mcxp            # writes ./<session>.json each
    mcxp2json.py mcx_demo.mcxp -o out/    # ...into out/
    mcxp2json.py mcx_demo.mcxp -s cube60  # just that one, to stdout

What does not survive is what was never part of the simulation: the thread
and block counts, which GPU to use, the workload split, the remote-execution
settings.  Those are how a run is launched, not what it computes; mcx takes
them on the command line and MCX Studio 2 keeps them beside the document
rather than in it.  They are reported on stderr so nothing disappears
silently.
"""

import argparse
import configparser
import csv
import io
import json
import os
import sys

# .mcxp key -> JSON path.  Booleans are written as the 0/1 the old GUI
# stored, because that is what mcx's own examples use and what its parser
# has always accepted.
SCALARS = {
    'PhotonNum':    'Session.Photons',
    'Seed':         'Session.RNGSeed',
    'DoReflect':    'Session.DoMismatch',
    'DoSave':       'Session.DoSaveVolume',
    'DoNormalize':  'Session.DoNormalize',
    'Autopilot':    'Session.DoAutoThread',
    'SaveDetector': 'Session.DoPartialPath',
    'DetectedNum':  'Session.MaxDetPhoton',
    'DoSaveRef':    'Session.DoSaveRef',
    'DoSaveExit':   'Session.DoSaveExit',
    'DoSaveSeed':   'Session.DoSaveSeed',
    'DoSpecular':   'Session.DoSpecular',
    'UnitInMM':     'Domain.LengthUnit',
    'DoSrcFrom0':   'Domain.OriginType',
}

# Kept out on purpose; see the module docstring.
NOT_SIMULATION = (
    'ThreadNum', 'ThreadBlock', 'GPUID', 'Workload', 'RespinNum',
    'ArrayOrder', 'BubbleSize', 'GateNum', 'MCProgram', 'RemoteCmd',
    'DoRemote', 'DoSharedFS', 'MoreParam', 'UseAtomic', 'BasicOrder',
    'DebugPhoton', 'DoSkipVoid', 'InputFile',
)

# The old GUI stored the three debug switches as a bit per character.
DEBUG_LETTERS = 'RMP'


def put(doc, path, value):
    """Writes value at a dotted path, building the objects on the way."""
    node = doc
    parts = path.split('.')
    for key in parts[:-1]:
        node = node.setdefault(key, {})
    node[parts[-1]] = value


def as_number(text):
    """A number if it reads as one, otherwise the string itself.

    '1e8' has to come back as 100000000 rather than as a float, because a
    photon count is counted; everything else keeps whatever it was written
    as, so 5e-9 stays 5e-09 and not 0.000000005.
    """
    try:
        value = int(text)
        return value
    except ValueError:
        pass
    try:
        value = float(text)
        if value.is_integer() and abs(value) < 1e15:
            return int(value)
        return value
    except ValueError:
        return text


def split_rows(text):
    """The old GUI's '|' separated list of 'a,b,c' rows."""
    return [row for row in text.split('|') if row.strip()]


def parse_csv_row(row):
    """One row of MCXConfig: Section, Key, Value, with Pascal's quoting.

    A row holding a value that itself contains commas was written wrapped in
    quotes as a whole, with its inner quotes doubled, so it parses first to
    one field and has to be read again to become three.
    """
    fields = next(csv.reader(io.StringIO(row)))
    if len(fields) == 1:
        fields = next(csv.reader(io.StringIO(fields[0])))
    return fields


def convert_media(text):
    """'0,0,1,1|0.005,1,0.01,1.37' -> a list of media objects."""
    media = []
    for row in split_rows(text):
        parts = [as_number(p.strip()) for p in row.split(',')]
        parts += [0] * (4 - len(parts))
        media.append({'mua': parts[0], 'mus': parts[1],
                      'g': parts[2], 'n': parts[3]})
    return media


def convert_detectors(text):
    """'29,19,0,1|...' -> a list of detector objects."""
    dets = []
    for row in split_rows(text):
        parts = [as_number(p.strip()) for p in row.split(',')]
        parts += [0] * (4 - len(parts))
        dets.append({'Pos': parts[0:3], 'R': parts[3]})
    return dets


def convert_config(doc, text, warn):
    """MCXConfig: the rows of the old GUI's property grid.

    Each is Section, Key, Value -- 'Optode.Source', 'Pos', '[29, 29, 0]' --
    and a row whose section is blank was a spacer in the grid.
    """
    for row in split_rows(text):
        try:
            fields = parse_csv_row(row)
        except (csv.Error, StopIteration):
            warn('could not read a config row: %s' % row)
            continue
        if len(fields) < 3:
            continue
        section, key, value = fields[0].strip(), fields[1].strip(), fields[2]
        if not section or not key or section.strip() == '':
            continue
        value = value.strip()
        if not value:
            continue
        # The volume designer wrote a placeholder here rather than a path.
        if value.startswith('See '):
            continue
        if value.startswith('['):
            try:
                value = json.loads(value)
            except ValueError:
                warn('could not read the array in %s.%s: %s'
                     % (section, key, value))
                continue
        else:
            value = as_number(value)
        put(doc, '%s.%s' % (section, key), value)


def convert_session(cfg, name, warn):
    """One .mcxp section -> one mcx input document."""
    doc = {}
    put(doc, 'Session.ID', name)

    items = dict(cfg.items(name))
    for key, value in items.items():
        value = value.strip()
        if not value:
            continue
        # configparser lower-cases keys; the table is written as the file is.
        canon = {k.lower(): k for k in list(SCALARS) + list(NOT_SIMULATION)}
        real = canon.get(key, key)

        if real in NOT_SIMULATION:
            continue
        if real in SCALARS:
            put(doc, SCALARS[real], as_number(value))
        elif real.lower() == 'mediacfg':
            put(doc, 'Domain.Media', convert_media(value))
        elif real.lower() == 'detectorcfg':
            put(doc, 'Optode.Detector', convert_detectors(value))
        elif real.lower() == 'mcxconfig':
            convert_config(doc, value, warn)
        elif real.lower() == 'shapecfg':
            try:
                shapes = json.loads(value)
            except ValueError:
                warn('could not read ShapeCfg')
                continue
            if isinstance(shapes, dict) and 'Shapes' in shapes:
                doc['Shapes'] = shapes['Shapes']
            else:
                doc['Shapes'] = shapes
        elif real.lower() == 'debugflags':
            letters = ''.join(DEBUG_LETTERS[i]
                              for i, c in enumerate(value[:3]) if c == '1')
            if letters:
                put(doc, 'Session.DebugFlag', letters)
        elif real.lower() == 'dosavemask':
            if as_number(value):
                put(doc, 'Session.SaveDataMask', 'DSPMXVW')
        elif real.lower() in ('doreplay', 'replaydet'):
            if as_number(value):
                put(doc, 'Session.ReplayDet', as_number(items.get('replaydet', 0)))
        else:
            warn('%s: no home for %s=%s' % (name, real, value))

    # OriginType is what mcx calls srcfrom0, and it is 1 or 2 rather than a
    # flag: 1 puts the first voxel's corner at the origin.
    origin = doc.get('Domain', {}).get('OriginType')
    if origin is not None:
        doc['Domain']['OriginType'] = 1 if origin else 2
    return doc


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('project', help='the .mcxp file to read')
    ap.add_argument('-o', '--outdir', default='.',
                    help='where to write the .json files (default: here)')
    ap.add_argument('-s', '--session',
                    help='convert only this session, and write it to stdout')
    ap.add_argument('-q', '--quiet', action='store_true',
                    help='do not report what was left behind')
    args = ap.parse_args(argv)

    def warn(text):
        if not args.quiet:
            print('mcxp2json: ' + text, file=sys.stderr)

    cfg = configparser.ConfigParser()
    # Values hold '%' and ':' and must not be interpolated or re-split.
    cfg = configparser.ConfigParser(interpolation=None, delimiters=('=',))
    cfg.optionxform = str
    with open(args.project, 'r') as handle:
        cfg.read_file(handle)

    names = cfg.sections()
    if args.session:
        if args.session not in names:
            print('mcxp2json: no session named %s; the file has: %s'
                  % (args.session, ', '.join(names)), file=sys.stderr)
            return 1
        names = [args.session]

    for name in names:
        doc = convert_session(cfg, name, warn)
        text = json.dumps(doc, indent=4)
        if args.session:
            print(text)
        else:
            os.makedirs(args.outdir, exist_ok=True)
            path = os.path.join(args.outdir, name + '.json')
            with open(path, 'w') as handle:
                handle.write(text + '\n')
            print(path)
    return 0


if __name__ == '__main__':
    sys.exit(main())
