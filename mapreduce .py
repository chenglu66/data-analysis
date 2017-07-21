# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 09:59:37 2017

@author: Lenovo-Y430p
"""
import sys  
from operator import itemgetter     
# 调用标准输入流    
for line in sys.stdin:    
    # 读取文本内容     
    line = line.strip()    
    # 对文本内容分词，形成一个列表   
    words = line.split()    
    # 读取列表中每一个元素的值    
    for word in words:    
        # map函数输出，key为word，下一步将进行shuffle过程，将按照key排序，输出，这两步为map阶段工作为，在本地节点进行    
        print ('%s\t%s' % (word, 1))    
current_word = None    
current_count = 0    
word = None    
# input comes from STDIN    
for line in sys.stdin:    
    # remove leading and trailing whitespace    
    line = line.strip()    
    
    # parse the input we got from mapper.py    
    word, count = line.split('\t', 1)    
    
    # convert count (currently a string) to int    
    try:    
        count = int(count)    
    except ValueError:    
        # count was not a number, so silently    
        # ignore/discard this line    
        continue    
    
    # this IF-switch only works because Hadoop sorts map output    
    # by key (here: word) before it is passed to the reducer    
    if current_word == word:    
        current_count += count    
    else:    
        if current_word:    
            # write result to STDOUT    
            print ('%s\t%s' % (current_word, current_count))    
        current_count = count    
        current_word = word    
    
# do not forget to output the last word if needed!    
if current_word == word:    
    print ('%s\t%s' % (current_word, current_count) )

