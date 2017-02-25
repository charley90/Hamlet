#!/usr/bin/python
# -*- coding:utf-8 -*-



###Base
## IO
from bokeh.io import output_notebook, show,output_file
output_notebook()
show(layout) #在notebook中显示
output_file("lines.html")  #在浏览器上显示
show(p)

## 工具 图例旁边的工具栏
TOOLS = "pan,wheel_zoom,box_zoom,reset,save,box_select,lasso_select"

## 布局 复杂的布局设置
from bokeh.layouts import gridplot,raw,column
row1 = [p1,p2]
row2 = [p3]
layout = gridplot([[p1,p2],[p3]])
layout = gridplot([[s1, s2, s3]], toolbar_location=None)



### Model
## column data sourse   cds 在数据交互中非常有用处  不用这个仅仅是图表层面的共享交互 可以是数据列表的列,也可以是映射关系             #cds 数据交互的最底层
from bokeh.models import ColumDataSource
source=ColumnDataSource(df)
new_data={’x’:[],’y’:[]}  #推荐用这种形式来生成数据
source.date=new_date
cds_df = ColumnDataSource(df)

##Hover
rom bokeh.models import HoverTool

source = ColumnDataSource(data=dict(
            x=[1, 2, 3, 4, 5],
            y=[2, 5, 8, 2, 7],
            desc=['A', 'b', 'C', 'd', 'E'],
            )
         )

hover = HoverTool(tooltips=[                                                                                            #也可以导入各种html的样式文件
            ("index", "$index"),
            ("(x,y)", "($x, $y)"),
            ("desc", "@desc"),
            ]
        )

p = figure(plot_width=300, plot_height=300, tools=[hover], title="Mouse over the dots")

p.circle('x', 'y', size=20, source=source)

# Also show custom hover
from utils import get_custom_hover

show(gridplot([[p, get_custom_hover()]]))


## 映射点的大小  在数量型可以显示不同的大小或者颜色深浅,那目前可以可以可视化的应该是5维的数据,加上stream的时间轴
from bokeh.models import LinearInterpolator
size_mapper = LinearInterpolator(
    x=[autompg.hp.min(), autompg.hp.max()],
    y=[3, 30]
)
p = figure(height=400, width=800, x_axis_label='year', y_axis_label='mpg')
p.circle(x='yr', y=autompg.mpg, alpha=0.6, size={'field': 'hp', 'transform': size_mapper}, source=source)               #这种映射数据的技巧可以学习
show(p)


##annotations  指示添加到图层
#横线
from bokeh.models.annotations import Span
upper = Span(location=1, dimension='width', line_color='olive', line_width=4)
p.add_layout(upper)

# 区域色块
lower = BoxAnnotation(top=-1, fill_alpha=0.1, fill_color='firebrick')
p.add_layout(lower)

#标签
from bokeh.models.annotations import Label
label = Label(x=5, y=7, x_offset=12, text="Second Point", text_baseline="middle")  #x_offset 离点的x距离
p.add_layout(label)


#标签字段集合显示
from bokeh.models import ColumnDataSource, LabelSet
source = ColumnDataSource(data=dict(
    temp=[166, 171, 172, 168, 174, 162],
    pressure=[165, 189, 220, 141, 260, 174],
    names=['A', 'B', 'C', 'D', 'E', 'F']))
p = figure(x_range=(160, 175))
p.scatter(x='temp', y='pressure', size=8, source=source)
p.xaxis.axis_label = 'Temperature (C)'
p.yaxis.axis_label = 'Pressure (lbs)'
labels = LabelSet(x='temp', y='pressure', text='names', level='glyph',                                                  #构建标签集的方法
                  x_offset=5, y_offset=5, source=source, render_mode='canvas')
p.add_layout(labels)
show(p)

# 箭头
from bokeh.models.annotations import Arrow
from bokeh.models.arrow_heads import OpenHead, NormalHead, VeeHead
p.add_layout(Arrow(end=OpenHead(line_color="firebrick", line_width=4),
                   x_start=0, y_start=0, x_end=1, y_end=0))

## 颜色
from bokeh.models import LinearColorMapper, ColorBar
from bokeh.palettes import Viridis256
color_mapper = LinearColorMapper(palette=Viridis256, low=autompg.weight.min(), high=autompg.weight.max())               #类似于前面的点大小的map
p.circle(x='yr', y='mpg', color={'field': 'weight', 'transform': color_mapper}, size=20, alpha=0.6, source=source)      #根据'weight'字段显示颜色
color_bar = ColorBar(color_mapper=color_mapper, label_standoff=12, location=(0,0), title='Weight')
p.add_layout(color_bar, 'right') #显示颜色指示




###Plot
##figure
from bokeh.plotting import  figure
p1 =  figure(plot_width=300, tools='pan,box_zoom ’, title=‘name’ , x_axis_type="datetime")                              # x轴是时间
p2 =  figure(plot_width=300, plot_height=300,x_range=(0, 8), y_range=(0, 8))
p3 =  figure(x_axis_label='year', y_axis_label='mpg',toolbar_location='above)



plot_options = dict(width=250, plot_height=250, tools='pan,wheel_zoom')                                                 #多个图例公用相同配置
s1 = figure(**plot_options)
s2 = figure(x_range=s1.x_range, y_range=s1.y_range, **plot_options)

p.outline_line_width = 7
p.outline_line_alpha = 0.3
p.outline_line_color = "navy"

## glyphs
p.circle(radius=1)                                                                                                      #半径为1的圆 区别于size 是像素的大小
p.circle_x(x,y,color=‘red’,alpha=0.6,fill_color="white" )                                                               # 正例样本
p.x( size=[arrary ])                                                                                                    #负例样本
p.cross(source=source) #十字坐标
p.diamond(
                    # set visual properties for selected glyphs                                                         #选择和没有选择的样本区别显示
                    selection_color="firebrick",

                    # set visual properties for non-selected glyphs
                    nonselection_fill_alpha=0.2,
                    nonselection_fill_color="grey",
                    nonselection_line_color="firebrick",
                    nonselection_line_alpha=1.0)
)# *符
p.line([1,2,3],[4,5,6],line_dash=1,line_width=2)                                                                        #line_dash 线段连续
p.square

r.glyph.size = 50                                                                                                       #对glpyh属性进行设置
r.glyph.fill_alpha = 0.2
r.glyph.line_color = "firebrick"
r.glyph.line_dash = [5, 1]
r.glyph.line_width = 2
r.selection_glyph = Circle(fill_alpha=1, fill_color="firebrick", line_color=None)                                       #选中和没有选中的区别对待
r.nonselection_glyph = Circle(fill_alpha=0.2, fill_color="grey", line_color=None)

##grids 网格线
p.xgrid.grid_line_color = None
p.ygrid.grid_line_dash = [6, 4]# 网格线用虚线表示
p.ygrid.band_fill_alpha = 0.1
p.ygrid.band_fill_color = "navy"                                                                                        #网格块 金融中常用


##legends 图例
p.legend.location = "top_left"                                                                                          #图例位置

##axis坐标轴
p.xaxis.major_label_orientation = pi/4
p.yaxis.major_label_orientation = "vertical"                                                                            #调整坐标轴字体方向


# change just some things about the x-axes
p.xaxis.axis_label = "Temp"
p.xaxis.axis_line_width = 3
p.xaxis.axis_line_color = "red"

# change just some things about the y-axes
p.yaxis.axis_label = "Pressure"
p.yaxis.major_label_text_color = "orange"
p.yaxis.major_label_orientation = "vertical"

# change things on all axes
p.axis.minor_tick_in = -3
p.axis.minor_tick_out = 6

p.axis.visable=False                                                                                                    #去掉坐标轴


###Charts
from bokeh.charts import Area,Bar,BoxPlot,Donut,Dot ,HeatMap,Histogram,Horizon,Line,Scatter
#目前有12种图表
from bokeh.palettes import Spectral8   #配色方案
from bokeh.charts import defaults                                                                                       #使用默认配置


from bokeh.models import HoverTool                                                                                      #从底层Models 中导入点上去显示的注解
scatter=Scatter(autompg,x=‘mpg’,y=‘hp’,color=‘origin’)#根据’origin’的类别来匹配颜色
scatter.add_tools(HoverTool(tooltips=[(‘x’,$name’),(‘y’,’@age’)]))                                                  #当鼠标点上去的时候显示出这个点’name’ ‘age’
show(scartter)

p = Bar(autompg, label='yr', values='mpg', agg='median', group='origin',  title="Median MPG by YR, grouped by ORIGIN",  #group by
        legend='top_left', tools='crosshair')                                                                           #agg 函数

                                                                                                                        #group 可以换成stack


hist = Histogram(df, values='value', bins=30)#直方图 bins特有属性


show(BoxPlot(autompg,values=‘mpg’,label=[‘cyl’,’origin’]))                                                        #根据’cyl’和’origin’类别的组合来显示盒装图

show(Donut(autompg.cyl.astype(’str’),palette=Spectral8,legend=‘top_right’)) #饼图 图例表示方法                            #显示配色方案


from bokeh.charts.attributes import cat                                                                                 #attributs 常用的 cat,color,maker
show(Bar(make_counts,label=cat(columns=‘make’,sort=False),values=‘count’ ,agg=‘mean’))
# 降序排序,使用函数均值



area=Area(glucose,stack=True)                                                                                           #累计区域图
area.y_range.start=0                                                                                                    #设置y轴的起点为0 将负数的去掉
show(area)


from bokeh.charts.operations import blend                                                                               #operations 常用blend stack
b=blend(’1964’,’1964’,’1984’,name=‘life_expectency’,labels_name=‘year’)                                                 #blend长格式转短格式
#使用堆叠功能 将一个国家的几年的数据堆叠陈一个变量.方便下面使用,类似长格式转短格式,这样相当类别数据可以标颜色了
m国家n年份的二维表,变成了 国家-年份-数据的短格式



show(Dot(decades,values=b,label=‘country’,line_color=‘year’)) #点图 和散点图有区别                                         #点图是对离散数据使用的
#点图是对类别数据使用的, 散点是对连续数据



## 绘制K线图
inc = df.close > df.open
dec = df.open > df.close
w = 12*60*60*1000 # half day in ms

TOOLS = "pan,wheel_zoom,box_zoom,reset,save"

p = figure(x_axis_type="datetime", tools=TOOLS, plot_width=1000, title = "MSFT Candlestick")
p.xaxis.major_label_orientation = pi/4
p.grid.grid_line_alpha=0.3

p.segment(df.date, df.high, df.date, df.low, color="black")
p.vbar(df.date[inc], w, df.open[inc], df.close[inc], fill_color="#D5E1DD", line_color="black")
p.vbar(df.date[dec], w, df.open[dec], df.close[dec], fill_color="#F2583E", line_color="black")

output_file("candlestick.html", title="candlestick.py example")


##绘制热力图

source = ColumnDataSource(
    data=dict(month=month, year=year, color=color, rate=rate)
)

TOOLS = "resize,hover,save,pan,box_zoom,wheel_zoom"

p = figure(title="US Unemployment (1948 - 2013)",
           x_range=years, y_range=list(reversed(months)),
           x_axis_location="above", plot_width=900, plot_height=400,
           toolbar_location="left", tools=TOOLS)

p.grid.grid_line_color = None
p.axis.axis_line_color = None
p.axis.major_tick_line_color = None
p.axis.major_label_text_font_size = "5pt"
p.axis.major_label_standoff = 0
p.xaxis.major_label_orientation = pi/3

p.rect("year", "month", 1, 1, source=source,
       color="color", line_color=None)

p.select_one(HoverTool).tooltips = [
    ('date', '@month @year'),
    ('rate', '@rate'),
]



###server
## 互动操作
import numpy as np
from bokeh.io import push_notebook

x = np.linspace(0, 2*np.pi, 2000)
y = np.sin(x)

source = ColumnDataSource(data=dict(x=x, y=y))
p = figure(title="simple line example", plot_height=300, plot_width=600, y_range=(-5, 5))
p.line(x, y, color="#2222aa", alpha=0.5, line_width=2, source=source, name="foo")

#回调函数
def update(f, w=1, A=1, phi=0):
    if   f == "sin": func = np.sin
    elif f == "cos": func = np.cos
    elif f == "tan": func = np.tan
    source.data['y'] = A * func(w * x + phi)
    push_notebook() #一旦更新里面同步到notebook中

show(p, notebook_handle=True)


##hollo 示例
from bokeh.io import curdoc
from bokeh.layouts import column
from bokeh.models.widgets import TextInput, Button, Paragraph

# 创建组件
button = Button(label="Say HI")
input = TextInput(value="Bokeh")
output = Paragraph()
# 添加回调函数
def update():
    output.text = "Hello, " + input.value
    button.on_click(update)   #触发事件
# 排列组件
layout = column(button, input, output)
# 输入文档
curdoc().add_root(layout)


##更加复杂的交互式操作
import numpy as np
from bokeh.plotting import Figure
from bokeh.models import ColumnDataSource,HBox,VBoxForm
from bokeh.models.widgets import Slider,TextInput #滑条
from bokeh.io import curdoc

#主显示区域
N=200
x=np.linspace(0,4*np.pi,N)
y=np.sin(x)
source=ColumnDataSource(date=dict(x=x,y=y))                                                                             #cds可以是列数据也可以是映射关系

plot=Figure(plot_height=400,plot_width=400,title=‘my sine’,tools=‘croosshair,pan,reset,resize,save’ x_range=[0,4*np.pi],y_range=[-2.5,2.5])
plot.line(‘x’,’y’,,source=source)

#交互工具栏组件
text=TextInput(title=’title’,value=‘my sin’)
offset=Slider(title=‘offset’,value=0.0,start=-5.0,end=5.0,step=0.1)
phase=Slider(title=‘phase’,value=0.0,start=0.0,end=2*np.pi )
freq=Slider(title=‘freq’,value=1.0,start=0.1,end=5.1 )


#回调函数
def update_title(attrname,old,now):
	plot.title=text.value

text.on_change(‘value’,update_title) #自身的显示修改

def update_date(attrname,old,now):
	a=amplitude.value
	b=offset.value
	w=phase.value
	k=freq.value
	x=np.linspace(0,4*np.pi,N)
	y=a*np.sin(k*x+w)+b

#组件布局
inputs=VBoxForm(children=[text,offset,amllitude,phase,freq])
curdoce().add_root(HBox(children=[input,plot]),width=800)


#开启服务器
bokeh serve -—show sliders.py  #打开远程服务
—allow -websocket -origin foo.com #远程服务


##stream 动态效果
from math import cos, sin

from bokeh.io import curdoc
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure

p = figure(x_range=(-1.1, 1.1), y_range=(-1.1, 1.1))
p.circle(x=0, y=0, radius=1, fill_color=None, line_width=2) #画一个圆形

source = ColumnDataSource(data=dict(x=[1], y=[0])) #数据转换成cds格式
p.circle(x='x', y='y', size=12, fill_color='white', source=source) #创建移动点

def update():
    x, y = source.data['x'][-1], source.data['y'][-1] #每次取最后的一行数据

    # construct the new values for all columns, and pass to stream
    new_data = dict(x=[x*cos(0.1) - y*sin(0.1)], y=[x*sin(0.1) + y*cos(0.1)])  #原来的一行做做函数映射
    source.stream(new_data, rollover=8)  #每次stream 一批又8个数据行

curdoc().add_periodic_callback(update, 150) #每150ms调用一次update
curdoc().add_root(p)

##Datashader  图形批处理系统
https://anaconda.org/jbednar/notebooks





###参考
github.com/bokeh/bokeh-demos/tree/master/presentations/2016-03-pydata_strata
github.com/bokeh/bokeh-notebooks/tree/master/tutorial
