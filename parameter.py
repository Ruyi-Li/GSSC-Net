# Dataset-specific public configuration.
# 0=Houston2013, 1=Augsburg City, 2=Beijing (legacy), 3=Berlin.
# Patch size and epoch count are kept consistent with the manuscript setup.
_DATASET_HYPERPARAMS = {
    0: {'windowSize': 13, 'epoch_nums': 300},
    1: {'windowSize': 13, 'epoch_nums': 100},
    2: {'windowSize': 11, 'epoch_nums': 100},
    3: {'windowSize': 19, 'epoch_nums': 100},
}


def _init():
    global parameter

    default_dataset = 3
    cfg = _DATASET_HYPERPARAMS[default_dataset]

    parameter = {
        'data_type': default_dataset,

        # network/data
        'channels': 35,
        'windowSize': cfg['windowSize'],
        'ms_channels': 4,
        'radar_channels': 2,
        'out_features': [15, 13, 13, 13],

        # training
        'device': 'cuda0',
        'cuda': 'cuda0',
        'lr': 1e-4,
        'epoch_nums': cfg['epoch_nums'],
        'batch_size': 128,
        'num_workers': 0,
        'random_seed': 6,
        'visualization': False,
        'report': True,
        'tsne': False,
        'test_base': 1,

        # paths
        'model_savepath': [
            '../model/Houston2013_model.pth',
            '../model/Augsburg_city.pth',
            '../model/Beijing_model.pth',
            '../model/Berlin_model.pth',
        ],
        'log_path': [
            '../log/Houston2013_log.txt',
            '../log/Augsburg_city.txt',
            '../log/Beijing_log.txt',
            '../log/Berlin_log.txt',
        ],
        'report_path': [
            '../report/Houston2013_report.txt',
            '../report/Augsburg_city.txt',
            '../report/Beijing_report.txt',
            '../report/Berlin_report.txt',
        ],
        'image_path': [
            '../pic/Houston2013.png',
            '../pic/Augsburg_city.png',
            '../pic/Beijing.png',
            '../pic/Berlin.png',
        ],
        'save_max_acc': [1.0, 1.0, 1.0, 1.0],
    }


def configure_dataset(dataset_type):
    if dataset_type not in _DATASET_HYPERPARAMS:
        raise ValueError('Unsupported dataset_type: {}'.format(dataset_type))

    cfg = _DATASET_HYPERPARAMS[dataset_type]
    parameter['data_type'] = dataset_type
    parameter['windowSize'] = cfg['windowSize']
    parameter['epoch_nums'] = cfg['epoch_nums']


def set_value(key, value):
    parameter[key] = value


def get_value(key):
    try:
        return parameter[key]
    except Exception:
        print('Failed to read parameter: {}'.format(key))
        return None


def get_taskInfo():
    return (
        '-----------------------taskInfo-----------------------\n'
        ' dataset_type:\t{}\n'
        ' lr:\t{}\n'
        ' epoch_nums:\t{}\n'
        ' batch_size:\t{}\n'
        ' window_size:\t{}\n'
        ' channels:\t{}\n'
        ' ms_channels:\t{}\n'
        ' radar_channels:\t{}\n'
        '------------------------------------------------------'
    ).format(
        parameter['data_type'],
        parameter['lr'],
        parameter['epoch_nums'],
        parameter['batch_size'],
        parameter['windowSize'],
        parameter['channels'],
        parameter['ms_channels'],
        parameter['radar_channels'],
    )
