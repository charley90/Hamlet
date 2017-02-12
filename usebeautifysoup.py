#!/usr/bin/python2.6
# -*- coding: utf-8 -*-






#
# import  urllib2
# response=urllib2.urlopen('http://www.csrc.gov.cn/pub/newsite/zjhxwfb/xwdd/201604/t20160401_295316.html')
# print response.read()
#




import urllib2
# from beautifulsoup4 import *
from bs4 import BeautifulSoup

def has_hz(text):
    hz_yes = False
    for ch in text:
        #if isinstance(ch, unicode):
        if ch >= u'/u4e00' and ch<=u'/u9fa5':
                hz_yes = True
                break
        else:
            continue

    return hz_yes



spr='http://www.csrc.gov.cn/pub/newsite/zjhxwfb/xwdd/./201605/t20160506_297035.html'
request=urllib2.Request(spr)
try:
    urllib2.urlopen(request)
except urllib2.URLError, e:
    if hasattr(e,"code"):
        print e.code
    if hasattr(e,"reason"):
        print e.reason
else:
    print "*****"
    print "****"
    print "***"

response=urllib2.urlopen(request)
html=response.read()
soup = BeautifulSoup(html,"lxml")
tags = soup('p')  #p为正文部分
txt=''
for tag in tags:
    #if has_hz(tag.string):
        print tag
        txt=txt+'/n'+str(tag.contents[0].text)
    #else:
        #continue

print 'TAG:',txt





# # print soup.prettify()
# tags = soup('a')
# for tag in tags:
#     print 'TAG:',tag
#     print 'URL:',spr+tag.get('href', None)
#     print 'Content:',tag.contents[0]
#     print 'Attrs:',tag.attrs
#



#
# from  sqlalchemy  import *
#
# from sqlalchemy import Column
# from sqlalchemy.types import CHAR, Integer, String
# from sqlalchemy.ext.declarative import declarative_base
# from sqlalchemy.orm import sessionmaker
#
# engine = create_engine('mysql://root:123456.a@localhost:3306/work')
# print engine
#
# session=sessionmaker(bind=engine)
# print session
# # BaseModel = declarative_base()
# # def init_db():
# #     BaseModel.metadata.create_all(engine)
# # def drop_db():
# #     BaseModel.metadata.drop_all(engine)
# class url():
#     # 表的名字:
#     __tablename__ = 'url'
#
#     # 表的结构:
#     title = Column(String(200))
#     url = Column(String(200))
# # init_db()
#
# u=url()
# u.title='1'
# u.url='1'
#
# # session.add(u)










#
# import urllib
# import re
#
# url = 'http://www.csrc.gov.cn/pub/newsite/'
# html = urllib.urlopen(url).read()
# links = re.findall('href="(http://.*?)"', html)
# for link in links:
#     print link





#
#
# import urllib2
# import urllib
#
# values = {"username":"chenc2","password":"123456.a"}
# data = urllib.urlencode(values)
# url = "https://www.jisilu.cn/login/"
# request = urllib2.Request(url,data)
# try:
#     urllib2.urlopen(request)
# except urllib2.URLError, e:
#     if hasattr(e,"code"):
#         print e.code
#     if hasattr(e,"reason"):
#         print e.reason
# else:
#     print "OK"
#
# response = urllib2.urlopen(request)
# print response.read()



#
# import urllib
# import urllib2
# import cookielib
#
# filename = 'cookie.txt'
# #声明一个MozillaCookieJar对象实例来保存cookie，之后写入文件
# cookie = cookielib.MozillaCookieJar(filename)
# opener = urllib2.build_opener(urllib2.HTTPCookieProcessor(cookie))
# postdata = urllib.urlencode({
#             'stuid':'201200131012',
#             'pwd':'23342321'
#         })
# #登录教务系统的URL
# loginUrl = 'http://jwxt.sdu.edu.cn:7890/pls/wwwbks/bks_login2.login'
# #模拟登录，并把cookie保存到变量
# result = opener.open(loginUrl,postdata)
# #保存cookie到cookie.txt中
# cookie.save(ignore_discard=True, ignore_expires=True)
# #利用cookie请求访问另一个网址，此网址是成绩查询网址
# gradeUrl = 'http://jwxt.sdu.edu.cn:7890/pls/wwwbks/bkscjcx.curscopre'
# #请求访问成绩查询网址
# result = opener.open(gradeUrl)
# print result.read()



## 结构化查询
soup.title
# <title>The Dormouse's story</title>
soup.title.name
# u'title'
soup.title.string
# u'The Dormouse's story'
soup.title.parent.name
# u'head'
soup.p
# <p class="title"><b>The Dormouse's story</b></p>
soup.p['class']
# u'title'
soup.a
# <a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>
soup.find_all('a')
# [<a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>,
#  <a class="sister" href="http://example.com/lacie" id="link2">Lacie</a>,
#  <a class="sister" href="http://example.com/tillie" id="link3">Tillie</a>]
soup.find(id="link3")
# <a class="sister" href="http://example.com/tillie" id="link3">Tillie</a>


## 查找<a>标签的连接
for link in soup.find_all('a'):
    print(link.get('href'))
    # http://example.com/elsie
    # http://example.com/lacie
    # http://example.com/tillie


## tag 用法
tag.name
# u'b'
tag.attrs
# {u'class': u'boldest'}
tag['class']
# u'boldest'

##NavigableString 如果想在Beautiful Soup之外使用 NavigableString 对象,需要调用 unicode() 方法,将该对象转换成普通的Unicode字符串
tag.string
# u'Extremely bold'
type(tag.string)
# <class 'bs4.element.NavigableString'>
unicode_string = unicode(tag.string)
unicode_string
# u'Extremely bold'
type(unicode_string)
# <type 'unicode'>

#档转换成字符串,Unicode编码会被编码成UTF-8.这样就无法正确显示HTML特殊字符了
soup = BeautifulSoup("&ldquo;Dammit!&rdquo; he said.")
unicode(soup)
# u'<html><head></head><body>\u201cDammit!\u201d he said.</body></html>'
str(soup)
# '<html><head></head><body>\xe2\x80\x9cDammit!\xe2\x80\x9d he said.</body></html>'

#如果只想得到tag中包含的文本内容,那么可以嗲用 get_text() 方法,这个方法获取到tag中包含的所有文版内容包括子孙tag中的内容,并将结果作为Unicode字符串返回:
markup = '<a href="http://example.com/">\nI linked to <i>example.com</i>\n</a>'
soup = BeautifulSoup(markup)
soup.get_text()
#u'\nI linked to example.com\n'
soup.i.get_text()
#u'example.com'
soup.get_text("|", strip=True)  #指定分割符,去掉空格
#u'I linked to|example.com'




##BeautifulSoup
#此处respHtml是GB2312编码的，所以要指定该编码类型，BeautifulSoup才能解析出对应的soup
htmlCharset = "GB2312";
soup = BeautifulSoup(respHtml, fromEncoding=htmlCharset);




##遍历文档树
###  为什么需要 研究父节点或者子孙节点? 因为有得模式可能在上级或者下级是比较容易发现的,通过层级定位从而更快的找到想要的.


doc="'<td style="padding-left:0" width="60%"><label>November</label>
<input type="Hidden" id="cboMonth1" name="cboMonth1" value="11">
</td><td style="padding-right:0;" width="40%">
<label>2012</label>
<input type="Hidden" id="cboYear1" name="cboYear1" value="2012">
</td>"'

soup = BeautifulSoup(''.join(doc))

# 发现同级节点有比较好的识别方式,再通过parent进行转化 寻找月份和年份.
foundCboMonth = eachMonthHeader.find("input", {"id":re.compile("cboMonth\d+")});
tdMonth = foundCboMonth.parent;
tdMonthLabel = tdMonth.label;
monthStr = tdMonthLabel.string;
print "monthStr=",monthStr;

foundCboYear = eachMonthHeader.find("input", {"id":re.compile("cboYear\d+")});
tdYear = foundCboYear.parent;
tdYearLabel = tdYear.label;
yearStr = tdYearLabel.string;
print "yearStr=",yearStr;






### 子节点
soup.body.b
# <b>The Dormouse's story</b>
#可以在文档树的tag中多次调用这个方法.下面的代码可以获取<body>标签中的第一个<b>标签:通过点取属性的方式只能获得当前名字的第一个tag
soup.find_all('a')
# [<a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>,
#  <a class="sister" href="http://example.com/lacie" id="link2">Lacie</a>,
#  <a class="sister" href="http://example.com/tillie" id="link3">Tillie</a>]
title_tag = head_tag.contents[0]
title_tag
# <title>The Dormouse's story</title>
title_tag.contents
# [u'The Dormouse's story']
len(soup.contents)  #可以用len看有多少节点
# 1

head_tag.contents
# [<title>The Dormouse's story</title>]
head_tag.string   # 如果一个tag仅有一个子节点,string的 方法可以过滤掉一层子节点
# u'The Dormouse's story'
for string in soup.stripped_strings: #如果有多个节点,string方法会报none错误,但可以用循环,stripped_strings去掉空格的了
    print(repr(string))
    # u"The Dormouse's story"
    # u"The Dormouse's story"
    # u'Once upon a time there were three little sisters; and their names were'
    # u'Elsie'
    # u','
    # u'Lacie'
    # u'and'
    # u'Tillie'
    # u';\nand they lived at the bottom of a well.'
    # u'...'



for child in title_tag.children:  #子节点 直接
    print(child)
    # The Dormouse's story
for child in head_tag.descendants: #子孙节点,包括子节点和子节点的子节点
    print(child)
    # <title>The Dormouse's story</title>
    # The Dormouse's story

### 父节点
title_tag
# <title>The Dormouse's story</title>
title_tag.parent
# <head><title>The Dormouse's story</title></head>
link = soup.a
link
# <a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>
for parent in link.parents:
    if parent is None:  #beautifysoup的父节点是none
        print(parent)
    else:
        print(parent.name)

### 兄弟节点
sibling_soup.b.next_sibling
# <c>text2</c>
sibling_soup.c.previous_sibling
# <b>text1</b>

last_a_tag = soup.find("a", id="link3")
last_a_tag
# <a class="sister" href="http://example.com/tillie" id="link3">Tillie</a>
last_a_tag.next_sibling #结果是一个字符串,因为当前的解析过程 [2] 因为当前的解析过程因为遇到了<a>标签而中断了
# '; and they lived at the bottom of a well.'
last_a_tag.next_element
# u'Tillie'

## 搜索文档树
### find-all() 过滤器
soup.find_all('b')
soup.find_all(["a", "b"])  #列表
# [<b>The Dormouse's story</b>,
#  <a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>,
#  <a class="sister" href="http://example.com/lacie" id="link2">Lacie</a>,
#  <a class="sister" href="http://example.com/tillie" id="link3">Tillie</a>]
for tag in soup.find_all(re.compile("^b")):print(tag.name) #正则表达式
# body
# b
def has_class_but_no_id(tag):
    return tag.has_attr('class') and not tag.has_attr('id')
soup.find_all(has_class_but_no_id)   #方法如果没有合适过滤器,那么还可以定义一个方法,方法只接受一个元素参数
# [<p class="title"><b>The Dormouse's story</b></p>,
#  <p class="story">Once upon a time there were...</p>,
#  <p class="story">...</p>]



### find_all( name , attrs , recursive , text , limit,**kwargs )
#等价关系
soup.find_all("a")
soup("a")
soup.title.find_all(text=True)
soup.title(text=True)

#name
soup.find_all("title")

# attrs
# [<title>The Dormouse's story</title>]
soup.find_all(href=re.compile("elsie"))
# [<a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>]
data_soup.find_all(attrs={"data-foo": "value"})
# [<div data-foo="value">foo!</div>]
sopu.findAll("div", attrs={"aria-lable": True}); #用BeautifulSoup查找未知属性值，但是已知属性的名字的标签
#<div aria-lable="xxx">



#Beautiful Soup会检索当前tag的所有子孙节点,如果只想搜索tag的直接子节点,可以使用参数 recursive=False .
soup.html.find_all("title")
# [<title>The Dormouse's story</title>]
soup.html.find_all("title", recursive=False)
# []

#通过 text 参数可以搜搜文档中的字符串内容  使用正则表达式!!! re.compile(r"")
soup.find_all(text=re.compile("Dormouse"))
#[u"The Dormouse's story", u"The Dormouse's story"]




soup.find_all("a", limit=2)
# [<a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>,
#  <a class="sister" href="http://example.com/lacie" id="link2">Lacie</a>]


###select 使用css语法
soup.select("body a")
# [<a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>,
#  <a class="sister" href="http://example.com/lacie"  id="link2">Lacie</a>,
#  <a class="sister" href="http://example.com/tillie" id="link3">Tillie</a>]




##修改文档树
### extract 方法将当前tag移除文档树,并作为方法结果返回:
arkup = '<a href="http://example.com/">I linked to <i>example.com</i></a>'
soup = BeautifulSoup(markup)
a_tag = soup.a
i_tag = soup.i.extract()
a_tag
# <a href="http://example.com/">I linked to</a>
i_tag
# <i>example.com</i>
print(i_tag.parent)
None

###wrap() 方法可以对指定的tag元素进行包装 unwrap() 方法与 wrap() 方法相反.将移除tag内的所有tag标签,该方法常被用来进行标记的解包
soup = BeautifulSoup("<p>I wish I was bold.</p>")
soup.p.string.wrap(soup.new_tag("b"))
# <b>I wish I was bold.</b>
soup.p.wrap(soup.new_tag("div"))
# <div><p><b>I wish I was bold.</b></p></div>
markup = '<a href="http://example.com/">I linked to <i>example.com</i></a>'
soup = BeautifulSoup(markup)
a_tag = soup.a
a_tag.i.unwrap()
a_tag
# <a href="http://example.com/">I linked to example.com</a>