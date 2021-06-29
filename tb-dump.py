#!/usr/bin/env python
#
# Dumps the contents of a TensorBoard event log.

import argparse

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def dump(log_path, max_scalars):
    tf_size_guidance = {
        'images': 1,
        'scalars': max_scalars,
        'histograms': 1,
        'compressedHistograms': 1,
    }

    events = EventAccumulator(log_path, tf_size_guidance)
    events.Reload()

    print('Logged variables:')
    for tag, vars in events.Tags().items():
        if isinstance(vars, list):
            vars = ', '.join(vars)
        print(f'  {tag}: {vars}')
    print()

    vars = events.Tags()['scalars']
    for var in vars:
        print(f'{var}:')
        for event in events.Scalars(var):
            print(f'  {event.step}: {event.value}')
        print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('log_path', metavar='LOG-FILE', type=str,
                        help='a TensorBoard log file to read')
    parser.add_argument('--max-scalars', metavar='N', type=int, default=0,
                        help='keep at most N scalars in memory (default is 0, meaning unlimited)')
    args = parser.parse_args()

    dump(args.log_path, max_scalars=args.max_scalars)
