# -*- coding: utf8 -*-
import sys
import xlrd
import json
import csv
import traceback
import MySQLdb as mdb
from collections import Counter

reload(sys)
sys.setdefaultencoding('utf-8')


OUTER_MYSQL = {
    'host': '582af38b773d1.bj.cdb.myqcloud.com',
    'port': 14811,
    'db': 'webcrawler',
    'user': 'web',
    'passwd': "Crawler20161231",
    'charset': 'utf8',
    'connect_timeout': 20,
}

def load_train_dataset():
    topics = []
    labels = []
    label_id_dict = json.load(open('tag_labels_ids.json', 'r'))
    book = xlrd.open_workbook('事件分类列表-用于机器学习训练集.xlsx')
    sheet = book.sheet_by_index(0)
    for i in range(1, sheet.nrows):
        topic = sheet.cell_value(i, 0)
        label = sheet.cell_value(i, 1)
        topics.append(topic)
        labels.append(label_id_dict[label])
    # traceback.print_exc()
    # import ipdb; ipdb.set_trace()
    return topics, labels

def get_content_by_title(title):
    conn = mdb.connect(**OUTER_MYSQL)
    cursor = conn.cursor()
    select_topic_sql = """
        SELECT cvt.content
        FROM topicinfo t JOIN cache_v_topiccontent AS cvt
        ON cvt.id=t.id 
        WHERE t.title='{}' limit 1
    """.format(title)
    status = cursor.execute(select_topic_sql)
    # import ipdb; ipdb.set_trace()
    record = cursor.fetchone()
    if status:
        return record[0]
    cursor.close()
    conn.close()
    return ''

def save_dataset_as_csv(filename, list_of_records):
    with open(filename, 'w') as fw:
    	writer = csv.writer(fw, delimiter='|')
    	writer.writerow(['Title', 'Content', 'Label'])
    	for record in list_of_records:
    	    writer.writerow(record)


def main():
    dataset = []
    train_tags, labels = load_train_dataset()
    size = len(train_tags)
    try:
	    for i in range(size):
	    	print train_tags[i], 
	        content = get_content_by_title(train_tags[i])
	        dataset.append([train_tags[i], content, labels[i]])
	        print len(content)
	except (Exception, KeyboardInterrupt) as e:
		traceback.print_exc()
    save_dataset_as_csv('init_dataset.csv', dataset)
    

if __name__ == '__main__':
    main()