'''
Created on Sep 10, 2018

@author: enerve
'''
import argparse

def parse_args():
    parser = argparse.ArgumentParser()  
    #parser.add_argument('file', help='path to data file')
    parser.add_argument('--output_dir', help='path to store output files',
                        default='/Users/erw/MLData/RL/output/')
    parser.add_argument('--nn', action='store_true')
    return parser.parse_args()
