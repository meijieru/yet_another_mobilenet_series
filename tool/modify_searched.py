import copy
import argparse
import yaml
from models.mobilenet_base import _make_divisible

def _make_int(val):
    return round(int(val))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', type=str)
    parser.add_argument('action', type=str)
    args = parser.parse_args()

    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)

    res = copy.deepcopy(config)
    inverted_residual_setting = []
    for c, n, s, ks, hiddens, expand in res['inverted_residual_setting']:
        if args.action == 'mixnet':
            hidden = int(sum(hiddens) / len(hiddens))
            hiddens = [hidden for _ in hiddens]
        elif args.action == 'all_3x3':
            ks = [3]
            hiddens = [sum(hiddens)]
        elif args.action == 'all_7x7':
            ks = [7]
            hiddens = [sum(hiddens)]
        elif args.action.startswith('multiply'):
            width_multi = float(args.action[len('multiply'):])
            hiddens = [round(int(width_multi * hidden)) for hidden in hiddens]
            # c = _make_divisible(c * width_multi, 8)
            c = _make_int(c * width_multi)
        else:
            raise ValueError()
        inverted_residual_setting.append([c, n, s, ks, hiddens, expand])
    res['inverted_residual_setting'] = inverted_residual_setting

    if args.action.startswith('multiply'):
        width_multi = float(args.action[len('multiply'):])
        assert width_multi > 1.0
        for key in ['input_channel', 'last_channel']:
            res[key] = _make_int(res[key] * width_multi)
        res['inverted_residual_setting'][0][4] = [res['input_channel']]
    print(res)
