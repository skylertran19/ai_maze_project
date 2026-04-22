# Distinct colors for each hazard symbol
# Colors sampled directly from MAZE_1.png using analyze_colors.py
# Both the source pad and destination pad of each teleport pair must map
# to the same TP color string so maze_loader pairs them correctly.

DISTINCT_COLORS = {
    # Confusion pad - brown/tan tones
    'confusion': [
        (156, 117, 60),
        (199, 154, 65),
    ],

    # Death pit - red/pink tones
    'deathpit': [
        (247, 140, 129),
        (250,  98,  86),
        (250, 161, 149),
        (254, 226, 225),
        (254, 228, 227),
        (254, 249, 249),
    ],

    # Green teleport SOURCE pad
    'greentp': [
        (91, 220, 148),
    ],

    # Green teleport DESTINATION pad
    'greentpdest': [
        (65, 158, 112),
    ],

    # Orange teleport SOURCE pad
    # Original reference colors (kept for other mazes):
    'orangetpdest': [
        (255, 113, 59),
        (255, 148, 75),
        (255, 148, 80),
        (255, 149, 75),
        (255, 149, 78),
        (255, 149, 80),
        (255, 150, 81),
        (255, 151, 80),
        (255, 151, 81),
        # MAZE_1 source pad (7,30) actual pixel colors:
        (229, 155, 90),
        (233, 158, 93),
        (234, 155, 96),
        (234, 160, 99),
        (236, 150, 101),
        (237, 148,  88),
        (239, 154,  99),
        (241, 151,  88),
        (241, 157,  84),
        (247, 158,  92),
        (248, 143,  75),
        (251, 154, 101),
        (251, 162,  82),
        (252, 153,  86),
        (254, 154,  68),
        (255, 154,  95),
        (255, 157,  87),
        (255, 161,  90),
        (255, 165,  97),
        (255, 167,  79),
        (255, 167,  99),
        (255, 178,  95),
    ],

    # Purple teleport SOURCE pad
    'purpletp': [
        (135, 105, 193),
        (141, 119, 187),
    ],

    # Purple teleport DESTINATION pad
    'purpletpdest': [
        (143, 101, 227),
        (181, 153, 234),
    ],

    # Yellow/orange teleport DESTINATION pad
    # Original reference colors (kept for other mazes):
    'yellowtp': [
        (220, 176, 108),
        (222, 174,  78),
        (224, 183, 123),
        (230, 181,  70),
        (231, 193, 118),
        (240, 192,  76),
        # MAZE_1 destination pad (59,55) actual pixel colors:
        (205, 169,  75),
        (206, 169,  62),
        (207, 164,  49),
        (208, 164,  57),
        (208, 172,  86),
        (209, 158,  69),
        (212, 172,  58),
        (215, 166,  71),
        (218, 162,  67),
        (218, 171,  57),
        (218, 175,  62),
        (220, 171,  66),
        (222, 161,  70),
        (222, 171,  66),
        (223, 178,  61),
        (224, 166,  69),
        (224, 171,  69),
        (224, 179,  60),
        (224, 182,  61),
        (226, 167,  75),
        (226, 186,  64),
        (226, 189,  59),
        (228, 174,  68),
        (229, 163,  66),
        (229, 165,  75),
        (230, 167,  72),
        (230, 178,  76),
        (230, 191,  64),
        (231, 179,  69),
        (233, 171,  72),
        (234, 192,  71),
        (234, 196,  69),
        (238, 181,  76),
        (238, 205,  76),
        (240, 182,  75),
        (241, 175,  78),
        (241, 208,  77),
    ],
}