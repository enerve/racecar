'''
Created on 20 Feb 2019

@author: erwin
'''

if __name__ == '__main__':
    import pstats
    p = pstats.Stats('timing')
    p.strip_dirs().sort_stats('tottime').print_stats('\\.py', .1)