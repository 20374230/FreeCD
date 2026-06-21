"""Dataset class names and color palettes for inference."""

DATASETS = {
    'SECOND': {
        'name_path': 'SECOND',
        'cls_file': 'cls_SECOND.txt',
        'num_classes': 6,
        'is_ori': False,
        'classes': (
            'background', 'low vegetation', 'nvg surface', 'tree',
            'water', 'building', 'playground',
        ),
        'palette': [
            [255, 255, 255], [0, 128, 0], [128, 128, 128],
            [0, 255, 0], [0, 0, 255], [128, 0, 0], [255, 0, 0],
        ],
    },
    'HIUCD': {
        'name_path': 'HIUCD',
        'cls_file': 'cls_HIUCD.txt',
        'num_classes': 8,
        'is_ori': False,
        'classes': (
            'background', 'low vegetation', 'nvg surface', 'tree',
            'water', 'building', 'playground',
        ),
        'palette': [
            [255, 255, 255], [0, 153, 255], [202, 255, 122], [230, 0, 0],
            [230, 0, 255], [255, 230, 0], [255, 181, 197], [175, 122, 255],
            [26, 255, 0],
        ],
    },
    'HRSCD': {
        'name_path': 'HRSCD',
        'cls_file': 'cls_HRSCD.txt',
        'num_classes': 5,
        'is_ori': False,
        'classes': (
            'background', 'low vegetation', 'nvg surface', 'tree',
            'water', 'building', 'playground',
        ),
        'palette': [
            [255, 255, 255], [0, 0, 128], [0, 128, 0],
            [0, 255, 0], [128, 0, 0], [255, 0, 0],
        ],
    },
    'JLSCD': {
        'name_path': 'JLSCD',
        'cls_file': 'cls_JLSCD.txt',
        'num_classes': 4,
        'is_ori': True,
        'classes': (
            'background', 'low vegetation', 'nvg surface', 'tree',
            'water', 'building', 'playground',
        ),
        'palette': [
            [255, 255, 255], [0, 128, 0], [0, 255, 255],
            [0, 255, 0], [0, 0, 128],
        ],
    },
    'LEVIR': {
        'name_path': 'WHU',
        'cls_file': 'cls_WHU.txt',
        'num_classes': 1,
        'is_ori': True,
        'classes': ('background', 'change'),
        'palette': [[0, 0, 0], [255, 255, 255]],
    },
    'WHU': {
        'name_path': 'WHU',
        'cls_file': 'cls_WHU.txt',
        'num_classes': 1,
        'is_ori': True,
        'classes': ('background', 'change'),
        'palette': [[0, 0, 0], [255, 255, 255]],
    },
    'LEVIRSCD': {
        'name_path': 'LEVIRSCD',
        'cls_file': 'cls_LEVIRSCD.txt',
        'num_classes': 15,
        'is_ori': False,
        'classes': (
            'background', 'low vegetation', 'nvg surface', 'tree',
            'water', 'building', 'playground',
        ),
        'palette': [
            [255, 255, 255], [97, 101, 63], [238, 238, 217], [197, 196, 123],
            [214, 203, 201], [98, 180, 252], [194, 206, 218], [139, 115, 227],
            [206, 232, 255], [115, 83, 73], [202, 198, 215], [250, 228, 220],
            [210, 209, 197], [72, 57, 113], [52, 91, 121], [234, 165, 140],
        ],
    },
}


def get_dataset_meta(name):
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset '{name}'. Choose from: {list(DATASETS)}")
    cfg = DATASETS[name]
    return {
        'classes': cfg['classes'],
        'palette': cfg['palette'],
    }, cfg
