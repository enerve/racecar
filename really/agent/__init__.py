'''
Created on Oct 30, 2018

@author: enerve
'''

from .driver import Driver
from .s_a_driver import SADriver
from .q_driver import QDriver
from .q_lambda_driver import QLambdaDriver
from .sarsa_driver import SarsaDriver
from .sarsa_lambda_driver import SarsaLambdaDriver
from .sarsa_fa_driver import SarsaFADriver
from .sarsa_fa_student import SarsaFAStudent
from .sarsa_lambda_fa_driver import SarsaLambdaFADriver
from .sarsa_lambda_fa_student import SarsaLambdaFAStudent
from .q_fa_driver import QFADriver
from .q_fa_student import QFAStudent
from .q_lambda_fa_driver import QLambdaFADriver
from .q_lambda_fa_student import QLambdaFAStudent
from .manual_driver import ManualDriver

__all__ = ["QDriver",               # deprecated
           "QLambdaDriver",         # deprecated
           "SarsaDriver",           # deprecated
           "SarsaLambdaDriver",     # deprecated
           "SarsaFADriver",
           "SarsaFAStudent",
           "SarsaLambdaFADriver",
           "SarsaLambdaFAStudent",
           "QFADriver",
           "QFAStudent",
           "QLambdaFADriver",
           "QLambdaFAStudent",
           "ManualDriver"]