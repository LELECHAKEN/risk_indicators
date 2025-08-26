# @Time : 2022/4/19 12:00
# @Author : for wangp
# @File : update_bond_index_dura.py
# @Software: PyCharm

from scripts.risk_alert_additional import PortfolioWarning_add


if __name__ == '__main__':
    t = '2022-08-24'
    rw_add = PortfolioWarning_add(t)
    rw_add.calc_bond_index_dura_all()