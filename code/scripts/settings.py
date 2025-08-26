from pathlib import Path
import yaml


# 主程序目录
DIR_OF_MAIN_PROG = Path(__file__).resolve().parent.parent.parent

# 读取 config.yaml
config = []
with open(str(DIR_OF_MAIN_PROG.joinpath('code', 'scripts', 'config.yaml')), encoding='utf-8') as cfg:
    config = yaml.safe_load(cfg)

# log 相关参数：绝对目录和文件名
log_path = str(DIR_OF_MAIN_PROG.joinpath(config['Log']['path']))
