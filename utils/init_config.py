
import argparse
import yaml


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)      # 用 safe_load,别用 load


def getConfig(path):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--config',
        default=path,
        type=str
    )

    # 第一次解析，只为了拿 config
    config_args, _ = parser.parse_known_args()

    yaml_cfg = load_yaml(config_args.config)

    # 根据 yaml 添加参数
    for key, value in yaml_cfg.items():
        arg_type = type(value) if value is not None else str

        parser.add_argument(
            f'--{key}',
            default=value,
            type=arg_type
        )

    # 第二次解析，得到最终参数
    args = parser.parse_args()

    return args

if __name__ == '__main__':

    args = getConfig('../config.yaml')
    


