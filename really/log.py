'''
Created on Sep 10, 2018

@author: enerve
'''

import logging
import sys
import time

from really import util

def configure_logger(logger, name):
    # create file handler which logs even debug messages
    log_filename = (util.pre_outputdir if util.pre_outputdir else '') + \
                    'log/output' + \
                    '_%s' % name + \
                    '_%s' % str(int(round(time.time()) % 1000000)) + \
                    '.log'
    print("Logging to %s" % log_filename)
    # TODO: create directories if they don't already exist
    fh = logging.FileHandler(log_filename)
    fh.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    # create formatter and add it to the handlers
    formatter = logging.Formatter(
        '(%(name)s) %(levelname)s: %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    util.init_logger()
