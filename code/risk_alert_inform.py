#!/usr/bin/env python
# !-*-coding:utf-8 -*-
# !@Time   022/12/14 11:09
# !@File   : risk_alert_inform.py
# !@Author : shiyue

import os
import xlwings as xw
from xlwings import constants as con
from scripts.utils.log_utils import logger
from typing import Optional, Union


t = "2022-12-21"

folder_path = r'\\shaoafile01\RiskManagement\27. RiskQuant\Data\RiskIndicators\日频数据\%s\\'%t.replace('-', '')
app = xw.App(visible=True, add_book=False)

leader_list = ["吴玮", "杨凡颖"]
exempt_fms = ["杨勇"] + leader_list
sma_fms = ["李松", "杨彦煜", "方睿彦", "刘懿"]
risk_l2 = ["回撤风险", "销售平台要求：中短债胜率监控", "可转债风险", "全天候固收+：组合收益监控",
           "全天候固收+：资产收益监控", "全天候固收+：配置偏离监控", "投决会授权限额：久期监控", "投决会授权限额：剩余期限监控"]
risk_l1 = ["一、市场风险", "二、信用风险", "三、流动性风险", "四、合规内控风险"]


def get_type_range(sht: xw.Sheet.api, risk_type: str):
    '''
    返回某个风险类型下相应的取值区域
    :param sht: xw.Sheet (VBA中的Sheet)
    :param risk_type: str, 风险类型
    :return: rng: xw.Range (VBA中的Range)
    '''
    type_cell = sht.Cells.Find(risk_type)
    add_row = 2 if risk_type in risk_l1 else 1
    bg_row = type_cell.Row + add_row
    rows = sht.Range("B" + str(bg_row)).End(con.Direction.xlDown).Row
    cols = sht.Range("B" + str(bg_row)).End(con.Direction.xlToRight).Column
    type_rng = sht.Range(sht.Cells(bg_row, 2), sht.Cells(rows, cols))
    return type_rng


def get_managers_in_range(sht, risk_type):
    '''
    获取某风险类型下的基金经理名单
    :param sht: xw.Sheet
    :param risk_type: str, 风险类型
    :return: list, 基金经理名单列表
    '''
    type_rng = get_type_range(sht, risk_type)
    fm_col = type_rng.Find("基金经理").Column
    fm_rng = sht.Range(sht.Cells(type_rng.Row+1, fm_col), sht.Cells(type_rng.Row+type_rng.Rows.Count-1, fm_col))  # 基金经理一列
    mg_list = []
    if type(fm_rng.Value) == str:  # 如果区域内仅有一个单元格
        mg_list = [fm_rng.Value]
    else:
        for fm_cell in fm_rng.Value:
            mg_list += fm_cell[0].split("、") if "、" in fm_cell[0] else fm_cell[0].split("，")
    return list(set(mg_list))


def get_distinct_value(sht: xw.Sheet.api, rng: xw.Range.api, col_name: str, split_str=False):
    '''
    获取指定区域内，某一列的数值
    :param sht: xw.Sheet.api
    :param rng: xw.Range.api
    :param col_name: str, user-defined column name
    :param split_str: bool, if split string or not
    :return: list
    '''
    spec_col = rng.Find(col_name).Column  # 指定列的列数
    if spec_col is None:  # 如果区域内没有包含制定的列）
        return []
    else:
        spec_rng = sht.Range(sht.Cells(rng.Row+1, spec_col), sht.Cells(rng.Row+rng.Rows.Count-1, spec_col))  # 指定列区域
        value_list = []
        if spec_rng.Rows.Count == 1:  # 如果区域内仅有一个单元格
            value_list = [spec_rng.Value]
        else:
            for cell_value in spec_rng.Value:
                if split_str and type(cell_value[0]) == str:
                    value_list += cell_value[0].split("、") if "、" in cell_value[0] else cell_value[0].split("，")
                else:
                    value_list += cell_value
        return list(set(value_list))


def norm_format(sht: xw.Sheet.api, risk_type: str):
    rng = get_type_range(sht, risk_type)
    rng.UnMerge()  # 取消合并单元格

    for border_id in range(7, 13):  # 设置单元格边框
        rng.Borders(border_id).LineStyle = con.LineStyle.xlContinuous
        rng.Borders(border_id).Weight = con.BorderWeight.xlThin

    rng_blank = rng.SpecialCells(con.CellType.xlCellTypeBlanks)  # 取出选定区域中的空白单元格
    for rng_area in rng_blank:
        rng_area.Cells(1, 1).Offset(0, 1).Resize(2, 1).FillDown()  # 用空白单元格的前置对其进行填充


def prepare_work(sht: xw.Sheet.api):
    '''生成预警文件中所包含的经理列表，风险类型列表'''
    manager_list = []
    type_list = []
    for type in risk_l1:
        type_cell = sht.Cells.Find(type)
        if type_cell:
            if sht.Cells(type_cell.Row + 1, type_cell.Column).Value != "无" and \
                    sht.Cells(type_cell.Row + 2, type_cell.Column).Value not in risk_l2:
                manager_list += get_managers_in_range(sht, type)
                type_list.append(type)

    for type in risk_l2:
        type_cell = sht.Cells.Find(type)
        if type_cell:  # 出现了该风险类型
            manager_list += get_managers_in_range(sht, type)
            type_list.append(type)

    return list(set(manager_list)), type_list


def merge_same_cells(sht: xw.Sheet.api, rng: xw.Range.api, spec_cols: Optional[list] = None):
    '''
    按列合并内容相同的单元格
    :param sht: xw.Sheet.api
    :param rng: xw.Range.api
    :param spec_cols: Optional[list] = None 仅对特定的列进行合并
    :return:
    '''
    if spec_cols is None:
        col_list = list(range(rng.Column, rng.Column+rng.Columns.Count))
    else:
        col_list = [rng.Find(col_name).Column for col_name in spec_cols]

    for col in col_list:
        row_list = [rng.Row]
        for row in range(rng.Row+1, rng.Row+rng.Rows.Count):
            if sht.Cells(row, col).Value != sht.Cells(row-1, col).Value:
                row_list.append(row)

        for idx in range(len(row_list)):
            bg_row = row_list[idx]
            end_row = row_list[idx+1] if idx < len(row_list)-1 else rng.Row+rng.Rows.Count
            if bg_row == end_row-1:
                pass
            else:
                sht.Range(sht.Cells(bg_row+1, col), sht.Cells(end_row-1, col)).ClearContents()
                sht.Range(sht.Cells(bg_row, col), sht.Cells(end_row-1, col)).Merge()


def delete_spec_entirerows(sht: xw.Sheet.api, rng: xw.Range.api, col_name: str,
                           keep_list: Optional[list] = None, delete_list: Optional[list] = None):
    '''
    删除不包含指定内容或包含了禁止内容的行
    :param sht: xw.Sheet.api
    :param rng: xw.Range.api
    :param col_name: str, user-defined column name
    :param keep_list: Optional[list] = None
    :param delete_list: Optional[list] = None
    '''
    value_list = get_distinct_value(sht, rng, col_name, split_str=True)
    if len(value_list) == 0:  # 如果区域内没有包含指定列
        pass
    else:
        spec_col = rng.Find(col_name).Column
        last_row = rng.Row + rng.Rows.Count - 1
        for row_num in range(last_row, rng.Row, -1):  # 从后往前删
            cell_value = sht.Cells(row_num, spec_col).Value
            if (keep_list is not None and cell_value not in keep_list) or \
                    (delete_list is not None and cell_value in delete_list):  # 取值不是指定内容；或取值是禁止内容时，删掉整行
                sht.Cells(row_num, 2).EntireRow.Delete()


def match_manager(sht: xw.Sheet.api, fm_name: str, type_list: list):
    '''
    为指定的经理匹配不同风险类型下的产品
    :param sht: xw.Sheet
    :param fm_name: str, 经理名字
    :param type_list: list, 风险类型列表
    '''
    new_type_list = []
    for risk_type in type_list:
        type_rng = get_type_range(sht, risk_type)
        type_fms = get_managers_in_range(sht, risk_type)
        if fm_name in type_fms:
            new_type_list.append(type_list)
            fm_col = type_rng.Find("基金经理").Column
            last_row = type_rng.Row + type_rng.Rows.Count - 1
            for row_num in range(last_row, type_rng.Row, -1):  # 从后往前删
                name = sht.Cells(row_num, fm_col).Value
                if fm_name not in name:
                    sht.Cells(row_num, 2).EntireRow.Delete()
        else:
            if risk_type in risk_l1:
                sht.Cells(type_rng.Row-1, 2).Value = "无"
                type_rng.EntireRow.Delete()
            else:
                sht.Range(sht.Cells(type_rng.Row-2, 2), sht.Cells(type_rng.Row+type_rng.Rows.Count-1, 2)).EntireRow.Delete()

    if "回撤风险" in new_type_list:  # 对回撤风险部分进行格式调整
        merge_same_cells(sht, get_type_range(sht, "回撤风险"), ["产品类型", "组合类型", "基金经理", "回撤风险预算", "行业平均\n当前回撤"])


def split_risk_alert(orig_sheet: xw.Sheet.api, manager_list: list, exempt_fms:  list, type_list: list):
    '''拆分风险预警信息'''
    logger.info("开始拆分风险预警信息")

    fm_list = [i for i in manager_list if i not in exempt_fms]
    for fm_name in fm_list:
        if fm_name in sma_fms:  # 全天候固收+产品的汇总文件名字是特殊的
            fm_book = xw.books.open(folder_path + fm_name + "\【Risk Monitor】风险指标监控-全天候固收+(专户)-%s.xlsx" % fm_name)
            first_sheet = fm_book.sheets("全天候固收+").api
        else:
            fm_book = xw.books.open(folder_path + fm_name + "\风险管理部-组合风险指标-固收-%s.xlsx" % fm_name)
            first_sheet = fm_book.sheets("组合风险指标").api
        if "组合风险提示" in fm_book.sheet_names:
            fm_book.sheets("组合风险提示").delete()
        orig_sheet.Copy(Before=first_sheet)
        match_manager(fm_book.sheets("组合风险提示").api, fm_name, type_list)
        fm_book.save()
        fm_book.close()
        logger.info(fm_name + "已拆分完成")


def inv_cmte_risk_alert(orig_sheet: xw.Sheet.api, type_list: list):
    logger.info("开始生成投决会风险预警监控")
    ic_book = app.books.add()
    orig_sheet.Copy(Before=ic_book.sheets("Sheet1").api)
    ic_sheet = ic_book.sheets("组合风险提示").api

    # 目前仅包括回撤风险、中短债胜率
    for risk_type in type_list:
        type_rng = get_type_range(ic_sheet, risk_type)
        if risk_type == "回撤风险":
            fundtype_col = type_rng.Find("产品类型").Column
            threshold_col = type_rng.Find("回撤风险预算").Column
            dd_col = type_rng.Find("当前回撤").Column
            last_row = type_rng.Row + type_rng.Rows.Count - 1
            for row_num in range(last_row, type_rng.Row, -1):  # 从后往前删
                if ic_sheet.Cells(row_num, fundtype_col).Value != "公募":
                    ic_sheet.Cells(row_num, 2).EntireRow.Delete()   # 删除公募产品
                elif abs(ic_sheet.Cells(row_num, threshold_col).Value) > abs(ic_sheet.Cells(row_num, dd_col).Value):
                    ic_sheet.Cells(row_num, 2).EntireRow.Delete()  # 删除当前回撤没有超出阈值的产品
                else:
                    continue
        elif risk_type in ["销售平台要求：中短债胜率监控"]:
            pass  # 原样保留
        elif risk_type in risk_l1:  # 一级风险
            ic_sheet.Cells(type_rng.Row-1, 2).Value = "无"
            type_rng.EntireRow.Delete()
        else:   # 没进投决会的二级风险
            ic_sheet.Range(ic_sheet.Cells(type_rng.Row-2, 2), ic_sheet.Cells(type_rng.Row+type_rng.Rows.Count-1, 2)).EntireRow.Delete()

    if "回撤风险" in type_list:  # 对回撤风险部分进行格式调整
        merge_same_cells(ic_sheet, get_type_range(ic_sheet, "回撤风险"),
                         ["产品类型", "组合类型", "基金经理", "回撤风险预算", "行业平均\n当前回撤"])
    ic_book.sheets("Sheet1").delete()
    if os.path.exists(folder_path + "投决会风险预警监控.xlsx"):
        ic_book.save(folder_path + "投决会风险预警监控(1).xlsx")
    else:
        ic_book.save(folder_path + "投决会风险预警监控.xlsx")
    ic_book.close()
    logger.info("投决会风险预警监控已生成")


def diy_for_leader(ori_sheet: xw.Sheet.api, leader_list: list, type_list: list):
    logger.info("开始拆分管理层风险预警信息")

    for leader in leader_list:
        if leader == "吴玮":
            gs_book = xw.books.open(folder_path + "\风险管理部-组合风险指标-固收.xlsx")
            if "组合风险提示" in gs_book.sheet_names:
                gs_book.sheets("组合风险提示").delete()
            ori_sheet.Copy(Before=gs_book.sheets("组合风险指标").api)
            gs_sheet = gs_book.sheets("组合风险提示").api
            for risk_type in type_list:
                type_rng = get_type_range(gs_sheet, risk_type)
                delete_spec_entirerows(gs_sheet, type_rng, "产品类型", keep_list=["公募"])
                delete_spec_entirerows(gs_sheet, type_rng, "组合类型", delete_list=["全天候固收+"])
                if type_rng.Rows.Count == 1:
                    if risk_type in risk_l1:
                        gs_sheet.Cells(type_rng.Row - 1, 2).Value = "无"
                    else:
                        gs_sheet.Range(gs_sheet.Cells(type_rng.Row - 2, 2),
                                       gs_sheet.Cells(type_rng.Row - 1, 2)).EntireRow.Delete()
                    type_rng.EntireRow.Delete()
            if "回撤风险" in type_list:  # 对回撤风险部分进行格式调整
                merge_same_cells(gs_sheet, get_type_range(gs_sheet, "回撤风险"),
                                 ["产品类型", "组合类型", "基金经理", "回撤风险预算", "行业平均\n当前回撤"])
            gs_book.save()
            gs_book.close()
        elif leader == "杨凡颖":
            xy_book = xw.books.open(folder_path + "\风险管理部-组合风险指标-信用.xlsx")
            if "组合风险提示" in xy_book.sheet_names:
                xy_book.sheets("组合风险提示").delete()
            ori_sheet.Copy(Before=xy_book.sheets("组合风险指标").api)
            xy_sheet = xy_book.sheets("组合风险提示").api
            for risk_type in type_list:
                type_rng = get_type_range(xy_sheet, risk_type)
                delete_spec_entirerows(xy_sheet, type_rng, "组合类型", keep_list=["信用债型"])
                delete_spec_entirerows(xy_sheet, type_rng, "产品类型", keep_list=["公募"])
                delete_spec_entirerows(xy_sheet, type_rng, "基金经理", delete_list=["吴玮", "钱布克"])
                if type_rng.Rows.Count == 1:
                    if risk_type in risk_l1:
                        xy_sheet.Cells(type_rng.Row - 1, 2).Value = "无"
                    else:
                        xy_sheet.Range(xy_sheet.Cells(type_rng.Row - 2, 2),
                                       xy_sheet.Cells(type_rng.Row - 1, 2)).EntireRow.Delete()
                    type_rng.EntireRow.Delete()
            if "回撤风险" in type_list:  # 对回撤风险部分进行格式调整
                merge_same_cells(xy_sheet, get_type_range(xy_sheet, "回撤风险"),
                                 ["产品类型", "组合类型", "基金经理", "回撤风险预算", "行业平均\n当前回撤"])
            xy_book.save()
            xy_book.close()
        else:
            pass

    logger.info("管理层风险预警信息拆分已完成")


# step1: 格式化管理层版本风控指标中的回撤风险部分
mg_book = xw.books.open(folder_path + '【Risk Monitor】风险指标监控-管理层版.xlsx')
mg_sheet = mg_book.sheets("组合风险提示").api
norm_format(mg_sheet,  "回撤风险")
manager_list, type_list = prepare_work(mg_sheet)

# step2: 生成投决会风险监控
# inv_cmte_risk_alert(mg_sheet, type_list)

# step3: 拆分至对应的基金经理
diy_for_leader(mg_sheet, leader_list, type_list)
split_risk_alert(mg_sheet, manager_list, exempt_fms, type_list)

# last_step：关闭管理层版本的excel
mg_book.close()













