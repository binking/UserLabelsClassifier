# -*- coding: utf8 -*-
import sys
import xlrd
import json
import MySQLdb as mdb
from collections import Counter

reload(sys)
sys.setdefaultencoding('utf-8')

def gen_label_dict():
    label_id_dict = {}
    book = xlrd.open_workbook('事件分类列表-用于机器学习训练集.xlsx')
    sheet = book.sheet_by_index(0)
    tag_id = 0
    for i in range(1, sheet.nrows):
        la = sheet.cell_value(i, 1).strip()
        if la not in label_id_dict:
            label_id_dict[la] = tag_id
            tag_id += 1
    print 'there are %d labels' % len(label_id_dict)  # 257
    with open('tag_labels_ids.json', 'w') as fw:
        json.dump(label_id_dict, fw)
    # return label_id_dict

def count_labels():
    labels = []
    book = xlrd.open_workbook('事件分类列表-用于机器学习训练集.xlsx')
    sheet = book.sheet_by_index(0)
    for i in range(1, sheet.nrows):
        la = sheet.cell_value(i, 1)
        labels.append(la)
    # import ipdb; ipdb.set_trace()
    c = Counter(labels)
    for tup in c.most_common(20):
    	print '%s has: %d' % (tup[0], tup[1])
    return c

if __name__=='__main__':
    # gen_label_dict()
    count_labels()