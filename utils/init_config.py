
import argparse
import yaml


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)      # 用 safe_load,别用 load


def getConfig(path):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',default=path,type=str)

    
    yaml_cfg = load_yaml(parser.parse_args().config)

    for key,value in yaml_cfg.items():
        arg_type = type(value) if value is not None else str  # 推断类型
        parser.add_argument(f'--{key}',default=value,type = arg_type)
        print(f'{key}={value}')



    args = parser.parse_args()


    return args

if __name__ == '__main__':

    args = getConfig('../config.yaml')
    


