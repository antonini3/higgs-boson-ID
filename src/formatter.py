from os.path import dirname, join, realpath
from random import shuffle
import logging
logging.basicConfig(format="[%(asctime)s]\t%(msg)s", level=logging.INFO)

import json

class Formatter(object):

    def __init__(self, verbose=False, delta_r=True):
        self.rootpath = dirname(dirname(realpath(__file__)))

        self.datapath = join(self.rootpath, "data")
        self.rawpath = join(self.datapath, "raw")
        self.processedpath = join(self.datapath, "processed")
        self.verbose = verbose
        self.delta_r = delta_r


    def log(self, *args, **kwargs):
        if self.verbose: 
            logging.info(*args, **kwargs)


    def _read_data(self, filename, is_higgs):
        data = []
        for i, r in enumerate(open(join(self.rawpath, filename))):
            l = {}
            row = r.split()

            # Metadata
            l['name'] =             '%d%d' % (i, is_higgs)
            l['is_higgs'] =         int(is_higgs)

            # Pull info
            l['pull'] =             [float(elem) for elem in row[0:2]]   # pull1, pull2
            l['pull_fine'] =        [float(elem) for elem in row[2:4]]   # pull1_fine, pull2_fine 
            l['pull_nopix'] =       [float(elem) for elem in row[4:6]]   # pull1_nopix, pull2_nopix
            
            # Mass and jet info
            l['mass'] =             float(row[6])   # fTLeadingM
            l['lead_jet_fine'] =    float(row[7])   # leading_jet_fine.m()
            l['lead_jet_nopix'] =   float(row[8])   # leading_jet_nopix.m()

            if self.delta_r:
                # Delta R
                l['delta_r'] =          float(row[-1])

                # Image info
                l['image'] =            [float(elem) for elem in row[9:-1]]
            else:
                # Image info
                l['image'] =            [float(elem) for elem in row[9:]]

            data.append(l)

        return data


    def format(self, files, output, limit=None):
        """
        Formats raw data into processed JSON data
        Args:
            files:      [(first_filename, is_higgs), ...]
            output:     output_filename
        """

        self.log('-- FORMATTER --')
        self.log('  Reading all of the data...')
        all_data = [d for filename, is_higgs in files for d in self._read_data(filename, is_higgs)]
        if limit is not None:
            shuffle(all_data)
            all_data = all_data[:limit]
        self.log('  Storing to file...')
        with open(join(self.processedpath, '%s.json' % output), 'w') as outfile:
            json.dump(all_data, outfile, ensure_ascii=False)

        self.log('  Done!')


if __name__ == '__main__':
    formatter = Formatter(verbose=True, delta_r=True)
    formatter.format([('Singlet_Rotated_withDR.txt', True), ('Octet_Rotated_withDR.txt', False)], 'rotated_sample', limit=100)

    # formatter = Formatter(verbose=True, delta_r=False)
    # formatter.format([('Singlet_Rotated.txt', True), ('Octet_Rotated.txt', False)], 'rotated', delta_r=False)

