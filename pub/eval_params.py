"""
Evaluation parameter sets used in experiments.
"""
import numpy as np

TTENV_EVAL_SET_0 = [{
        'lin_dist_range':(5.0, 10.0),
        'ang_dist_range_target':(-0.5*np.pi, 0.5*np.pi),
        'ang_dist_range_belief':(-0.25*np.pi, 0.25*np.pi),
        'blocked':False
        },
        {
        'lin_dist_range':(10.0, 15.0),
        'ang_dist_range_target':(-0.5*np.pi, 0.5*np.pi),
        'ang_dist_range_belief':(-0.25*np.pi, 0.25*np.pi),
        'blocked':True
        },
        { # target and beleif in the opposite direction
        'lin_dist_range':(5.0, 10.0),
        'ang_dist_range_target':(0.5*np.pi, -0.5*np.pi),
        'ang_dist_range_belief':(-0.25*np.pi, 0.25*np.pi),
        'blocked':False
        },
        { # target and beleif in the opposite direction
        'lin_dist_range':(10.0, 15.0),
        'ang_dist_range_target':(0.5*np.pi, -0.5*np.pi),
        'ang_dist_range_belief':(-0.25*np.pi, 0.25*np.pi),
        'blocked':True
        },
        { # target in the opposite direction but belief in the same direction
        'lin_dist_range':(5.0, 10.0),
        'ang_dist_range_target':(0.5*np.pi, -0.5*np.pi),
        'ang_dist_range_belief':(0.75*np.pi, -0.75*np.pi),
        'blocked':False
        },
        { #target in the opposite direction but belief in the same direction
        'lin_dist_range':(10.0, 15.0),
        'ang_dist_range_target':(0.5*np.pi, -0.5*np.pi),
        'ang_dist_range_belief':(0.75*np.pi, -0.75*np.pi),
        'blocked':True
        },
]

# Zone A, B, C
TTENV_EVAL_SET_1 = [{
        'lin_dist_range_a2b':(5.0, 15.0),
        'ang_dist_range_a2b':(-np.pi, np.pi),
        'lin_dist_range_b2t':(0.0, 5.0),
        'ang_dist_range_b2t':(-np.pi/2, np.pi/2),
        'blocked':False
        },
        {
        'lin_dist_range_a2b':(5.0, 15.0),
        'ang_dist_range_a2b':(-np.pi, np.pi),
        'lin_dist_range_b2t':(0.0, 5.0),
        'ang_dist_range_b2t':(-np.pi/2, np.pi/2),
        'blocked':True
        },
        {
        'lin_dist_range_a2b':(5.0, 15.0),
        'ang_dist_range_a2b':(-np.pi, np.pi),
        'lin_dist_range_b2t':(10.0, 15.0),
        'ang_dist_range_b2t':(-np.pi/2, np.pi/2),
        'blocked':False
        },
]
#0226-0229
TTENV_EVAL_SET_2 = [
        { # Tracking
        'lin_dist_range_a2b':(3.0, 10.0),
        'ang_dist_range_a2b':(-np.pi, np.pi),
        'lin_dist_range_b2t':(0.0, 3.0),
        'ang_dist_range_b2t':(-np.pi/2, np.pi/2),
        'blocked':False,
        'target_speed_limit': 3.0,
        'const_q':1.0,
        },
        { # Discovery
        'lin_dist_range_a2b':(3.0, 10.0),
        'ang_dist_range_a2b':(-np.pi, np.pi),
        'lin_dist_range_b2t':(10.0, 15.0),
        'ang_dist_range_b2t':(-np.pi/2, np.pi/2),
        'blocked':None,
        'target_speed_limit': 2.0,
        'const_q': 0.2,
        },
        { # Navigation
        'lin_dist_range_a2b':(10.0, 20.0),
        'ang_dist_range_a2b':(-np.pi, np.pi),
        'lin_dist_range_b2t':(0.0, 3.0),
        'ang_dist_range_b2t':(-np.pi/2, np.pi/2),
        'blocked':True,
        'target_speed_limit': 2.0,
        'const_q': 0.2,
        },
]

TTENV_EVAL_SET = [
        { # Tracking
        'lin_dist_range_a2b':(3.0, 10.0),
        'ang_dist_range_a2b':(-np.pi, np.pi),
        'lin_dist_range_b2t':(0.0, 3.0),
        'ang_dist_range_b2t':(-np.pi/2, np.pi/2),
        'blocked':False,
        'target_speed_limit': 3.2,
        'const_q':0.5,
        },
        { # Discovery
        'lin_dist_range_a2b':(3.0, 10.0),
        'ang_dist_range_a2b':(-np.pi, np.pi),
        'lin_dist_range_b2t':(10.0, 15.0),
        'ang_dist_range_b2t':(-np.pi/2, np.pi/2),
        'blocked':None,
        'target_speed_limit': 2.0,
        'const_q': 0.2,
        },
        { # Navigation
        'lin_dist_range_a2b':(10.0, 20.0),
        'ang_dist_range_a2b':(-np.pi, np.pi),
        'lin_dist_range_b2t':(0.0, 3.0),
        'ang_dist_range_b2t':(-np.pi/2, np.pi/2),
        'blocked':True,
        'target_speed_limit': 2.0,
        'const_q': 0.2,
        },
]

TTENV_EVAL_MULTI_SET = [
        {
        'lin_dist_range_a2b':(3.0, 10.0),
        'ang_dist_range_a2b':(-np.pi, np.pi),
        'lin_dist_range_b2t':(0.0, 3.0),
        'ang_dist_range_b2t':(-np.pi/2, np.pi/2),
        'blocked':None,
        'target_speed_limit':1.0,
        'const_q':0.02,
        },
        # {
        # 'lin_dist_range_a2b':(3.0, 10.0),
        # 'ang_dist_range_a2b':(-np.pi, np.pi),
        # 'lin_dist_range_b2t':(0.0, 3.0),
        # 'ang_dist_range_b2t':(-np.pi/2, np.pi/2),
        # 'blocked':None,
        # 'target_speed_limit':2.0,
        # 'const_q':0.01,
        # },
        # {
        # 'lin_dist_range_a2b':(3.0, 10.0),
        # 'ang_dist_range_a2b':(-np.pi, np.pi),
        # 'lin_dist_range_b2t':(0.0, 3.0),
        # 'ang_dist_range_b2t':(-np.pi/2, np.pi/2),
        # 'blocked':None,
        # 'target_speed_limit':2.0,
        # 'const_q':0.5,
        # },
]

TTENV_TEST_SET = [
        { # Tracking v_t = v_a
        'lin_dist_range_a2b':(3.0, 10.0),
        'ang_dist_range_a2b':(-np.pi, np.pi),
        'lin_dist_range_b2t':(0.0, 3.0),
        'ang_dist_range_b2t':(-np.pi/2, np.pi/2),
        'blocked':False,
        'target_speed_limit': 3.0,
        'const_q':0.2,
        },
        { # Tracking v_t > v_a
        'lin_dist_range_a2b':(3.0, 10.0),
        'ang_dist_range_a2b':(-np.pi, np.pi),
        'lin_dist_range_b2t':(0.0, 3.0),
        'ang_dist_range_b2t':(-np.pi/2, np.pi/2),
        'blocked':False,
        'target_speed_limit': 3.2,
        'const_q':0.2,
        },
        { # Tracking v_t >> v_a
        'lin_dist_range_a2b':(3.0, 10.0),
        'ang_dist_range_a2b':(-np.pi, np.pi),
        'lin_dist_range_b2t':(0.0, 3.0),
        'ang_dist_range_b2t':(-np.pi/2, np.pi/2),
        'blocked':False,
        'target_speed_limit': 3.5,
        'const_q':0.2,
        },
        { # Tracking v_t > v_a, large q
        'lin_dist_range_a2b':(3.0, 10.0),
        'ang_dist_range_a2b':(-np.pi, np.pi),
        'lin_dist_range_b2t':(0.0, 3.0),
        'ang_dist_range_b2t':(-np.pi/2, np.pi/2),
        'blocked':False,
        'target_speed_limit': 3.0,
        'const_q':1.0,
        },
        { # Discovery
        'lin_dist_range_a2b':(3.0, 10.0),
        'ang_dist_range_a2b':(-np.pi, np.pi),
        'lin_dist_range_b2t':(15.0, 20.0),
        'ang_dist_range_b2t':(-np.pi/2, np.pi/2),
        'blocked':None,
        'target_speed_limit': 3.0,
        'const_q': 0.2,
        },
        { # Navigation
        'lin_dist_range_a2b':(15.0, 20.0),
        'ang_dist_range_a2b':(-np.pi, np.pi),
        'lin_dist_range_b2t':(0.0, 3.0),
        'ang_dist_range_b2t':(-np.pi/2, np.pi/2),
        'blocked':True,
        'target_speed_limit': 3.0,
        'const_q': 0.2,
        },
]

# TTENV_MULTI_TEST_SET = [
#         {
#         'lin_dist_range_a2b':(3.0, 10.0),
#         'ang_dist_range_a2b':(-np.pi, np.pi),
#         'lin_dist_range_b2t':(0.0, 10.0),
#         'ang_dist_range_b2t':(-np.pi/2, np.pi/2),
#         'blocked':None,
#         'target_speed_limit': 1.0,
#         'const_q':0.2,
#         },
#         # {
#         # 'lin_dist_range_a2b':(3.0, 10.0),
#         # 'ang_dist_range_a2b':(-np.pi, np.pi),
#         # 'lin_dist_range_b2t':(5.0, 10.0),
#         # 'ang_dist_range_b2t':(-np.pi/2, np.pi/2),
#         # 'blocked':None,
#         # 'target_speed_limit': 1.0,
#         # 'const_q':0.2,
#         # }
# ]

TTENV_MULTI_TEST_SET = [
        {
        'lin_dist_range_a2b':(3.0, 10.0),
        'ang_dist_range_a2b':(-np.pi, np.pi),
        'lin_dist_range_b2t':(0.0, 3.0),
        'ang_dist_range_b2t':(-np.pi/2, np.pi/2),
        'blocked':None,
        'target_speed_limit': 1.0,
        'const_q':0.002,
        },
        {
        'lin_dist_range_a2b':(3.0, 10.0),
        'ang_dist_range_a2b':(-np.pi, np.pi),
        'lin_dist_range_b2t':(0.0, 3.0),
        'ang_dist_range_b2t':(-np.pi/2, np.pi/2),
        'blocked':None,
        'target_speed_limit': 1.0,
        'const_q':0.02,
        },
        {
        'lin_dist_range_a2b':(3.0, 10.0),
        'ang_dist_range_a2b':(-np.pi, np.pi),
        'lin_dist_range_b2t':(0.0, 3.0),
        'ang_dist_range_b2t':(-np.pi/2, np.pi/2),
        'blocked':None,
        'target_speed_limit': 1.0,
        'const_q':0.2,
        },
]

TTENV_MULTI_TEST_SET_2 = [
        {
        'lin_dist_range_a2b':(3.0, 10.0),
        'ang_dist_range_a2b':(-np.pi, np.pi),
        'lin_dist_range_b2t':(10.0, 15.0),
        'ang_dist_range_b2t':(-np.pi/2, np.pi/2),
        'blocked':None,
        'target_speed_limit': 1.0,
        'const_q':0.002,
        },
        {
        'lin_dist_range_a2b':(3.0, 10.0),
        'ang_dist_range_a2b':(-np.pi, np.pi),
        'lin_dist_range_b2t':(10.0, 15.0),
        'ang_dist_range_b2t':(-np.pi/2, np.pi/2),
        'blocked':None,
        'target_speed_limit': 1.0,
        'const_q':0.02,
        },
        {
        'lin_dist_range_a2b':(3.0, 10.0),
        'ang_dist_range_a2b':(-np.pi, np.pi),
        'lin_dist_range_b2t':(10.0, 15.0),
        'ang_dist_range_b2t':(-np.pi/2, np.pi/2),
        'blocked':None,
        'target_speed_limit': 1.0,
        'const_q':0.2,
        },
]

# More extensive testing.
TTENV_TEST_SET_PUB = []
for q in [0.02, 0.2, 2.0]:
    for v_max in [3.0, 3.25, 3.5]:
        TTENV_TEST_SET_PUB.append(
        {
        'lin_dist_range_a2b':(3.0, 10.0),
        'ang_dist_range_a2b':(-np.pi, np.pi),
        'lin_dist_range_b2t':(0.0, 3.0),
        'ang_dist_range_b2t':(-np.pi/2, np.pi/2),
        'blocked':False,
        'target_speed_limit': v_max,
        'const_q':q,
        })
TTENV_TEST_SET_PUB.append(
        { # Discovery
        'lin_dist_range_a2b':(3.0, 10.0),
        'ang_dist_range_a2b':(-np.pi, np.pi),
        'lin_dist_range_b2t':(15.0, 20.0),
        'ang_dist_range_b2t':(-np.pi/2, np.pi/2),
        'blocked':None,
        'target_speed_limit': 3.0,
        'const_q': 0.2,
        })
TTENV_TEST_SET_PUB.append(
        { # Navigation
        'lin_dist_range_a2b':(15.0, 20.0),
        'ang_dist_range_a2b':(-np.pi, np.pi),
        'lin_dist_range_b2t':(0.0, 3.0),
        'ang_dist_range_b2t':(-np.pi/2, np.pi/2),
        'blocked':True,
        'target_speed_limit': 3.0,
        'const_q': 0.2,
        })

TTENV_TEST_SET_PUB_MORE = []
for q in [0.2]:
    for v_max in [2.5, 2.75]:
        TTENV_TEST_SET_PUB_MORE.append(
        {
        'lin_dist_range_a2b':(3.0, 10.0),
        'ang_dist_range_a2b':(-np.pi, np.pi),
        'lin_dist_range_b2t':(0.0, 3.0),
        'ang_dist_range_b2t':(-np.pi/2, np.pi/2),
        'blocked':False,
        'target_speed_limit': v_max,
        'const_q':q,
        })

for q in [0.1, 1.0]:
    for v_max in [3.0]:
        TTENV_TEST_SET_PUB_MORE.append(
        {
        'lin_dist_range_a2b':(3.0, 10.0),
        'ang_dist_range_a2b':(-np.pi, np.pi),
        'lin_dist_range_b2t':(0.0, 3.0),
        'ang_dist_range_b2t':(-np.pi/2, np.pi/2),
        'blocked':False,
        'target_speed_limit': v_max,
        'const_q':q,
        })
