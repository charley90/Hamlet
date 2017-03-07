#!/usr/bin/python
# -*- coding:utf-8 -*-




from gensim.models import Word2Vec

model = Word2Vec(sentences, size=200, min_count=5, workers=8)
# 词向量维度为200，丢弃出现次数少于5次的词,workers是默认与附件8个词有相关度
model.save('news.model')

intrested_words = ('中国', '手机', '学习', '公安局', '大学')
print '特征向量：'
for word in intrested_words:
    print word, len(model[word]), model[word]
for word in intrested_words:
    result = model.most_similar(word)
    print '与', word, '最相近的词：'
    for w, s in result:
        print '\t', w, s

words = ('中国', '祖国', '毛泽东', '人民')
for i in range(len(words)):
    w1 = words[i]
    for j in range(i + 1, len(words)):
        w2 = words[j]
        print '%s 和 %s 的相似度为：%.6f' % (w1, w2, model.similarity(w1, w2))

print '========================'
opposites = ((['中国', '城市'], ['学生']),
             (['男', '工作'], ['女']),
             (['俄罗斯', '美国', '英国'], ['日本']))
for positive, negative in opposites:
    result = model.most_similar(positive=positive, negative=negative)
    print_list(positive)
    print '-',
    print_list(negative)
    print '：'
    for word, similar in result:
        print '\t', word, similar

print '========================'
words_list = ('苹果 三星 美的 海尔', '中国 日本 韩国 美国 北京',
              '医院 手术 护士 医生 感染 福利', '爸爸 妈妈 舅舅 爷爷 叔叔 阿姨 老婆')
for words in words_list:
    print words, '离群词：', model.doesnt_match(words.split(' '))