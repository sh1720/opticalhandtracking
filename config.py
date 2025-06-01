RIGHT_MANO_PATH = 'mano\MANO_RIGHT.pkl'
LEFT_MANO_PATH = 'mano\MANO_LEFT.pkl'

DETNET_SKELETON = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),       # Index
    (0, 9), (9,10), (10,11), (11,12),    # Middle
    (0,13), (13,14), (14,15), (15,16),   # Ring
    (0,17), (17,18), (18,19), (19,20)    # Pinky
]

MANO_SKELETON = [
    0,      # wrist
    1, 2, 3,        # index
    4, 5, 6,        # middle
    7, 8, 9,        # pinky
    10, 11, 12,     # ring
    13, 14, 15, 16  # thumb
]

MANO_PARENTS = [
    None, 0, 1, 
    0, 4, 5, 
    0, 10, 11,
    0, 7, 8, 
    0, 13, 14
]

