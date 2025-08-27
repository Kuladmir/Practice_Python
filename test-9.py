import jieba
s = "这是一个字符串，看到了请返回Hello World"
print(jieba.lcut(s))
print(jieba.lcut(s, cut_all=True))
print(jieba.lcut_for_search(s))
jieba.add_word('Hel')
print(jieba.lcut(s))