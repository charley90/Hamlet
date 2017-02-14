#!/usr/bin/python
# -*- coding:utf-8 -*-



###Base
## IO
from bokeh.io import output_notebook, show,output_file
output_notebook()
show(layout) #在notebook中显示
output_file("lines.html")  #在浏览器上显示
show(p)

## 工具
TOOLS = "pan,wheel_zoom,box_zoom,reset,save,box_select,lasso_select"

## 布局
from bokeh.layouts import gridplot
row1 = [p1,p2]
row2 = [p3]
layout = gridplot([[p1,p2],[p3]])
layout = gridplot([[s1, s2, s3]], toolbar_location=None)



### Model
## column data sourse   cds 在数据交互中非常有用处  不用这个仅仅是图表层面的共享交互 可以是数据列表的列,也可以死映射关系
from bokeh.models import ColumDataSource
source=ColumnDataSource(df)
new_data={’x’:[],’y’:[]}  #推荐用这种形式来生成数据
source.date=new_date
cds_df = ColumnDataSource(df)

##Hover
rom bokeh.models import HoverTool

source = ColumnDataSource(
        data=dict(
            x=[1, 2, 3, 4, 5],
            y=[2, 5, 8, 2, 7],
            desc=['A', 'b', 'C', 'd', 'E'],
            )
         )

hover = HoverTool(
        tooltips=[
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


## 映射点的大小
from bokeh.models import LinearInterpolator
size_mapper = LinearInterpolator(
    x=[autompg.hp.min(), autompg.hp.max()],
    y=[3, 30]
)
p = figure(height=400, width=800, x_axis_label='year', y_axis_label='mpg')
p.circle(x='yr', y=autompg.mpg, alpha=0.6, size={'field': 'hp', 'transform': size_mapper}, source=source)
show(p)



###Plot
##figure
from bokeh.plotting import  figure
p1 =  figure(plot_width=300, tools='pan,box_zoom’,title=‘what name ’)
p2 =  figure(plot_width=300, plot_height=300,x_range=(0, 8), y_range=(0, 8))
p3 =  figure(x_axis_label='year', y_axis_label='mpg')



plot_options = dict(width=250, plot_height=250, tools='pan,wheel_zoom')
s1 = figure(**plot_options)
s2 = figure(x_range=s1.x_range, y_range=s1.y_range, **plot_options)

p.outline_line_width = 7
p.outline_line_alpha = 0.3
p.outline_line_color = "navy"

## glyphs
p.circle_x(x,y,color=‘red’,alpha=0.6,fill_color="white" ) # 正例样本
p.x( size=[arrary ]) #负例样本
p.cross(source=source) #十字坐标
p.diamond(
  # set visual properties for selected glyphs
                    selection_color="firebrick",

                    # set visual properties for non-selected glyphs
                    nonselection_fill_alpha=0.2,
                    nonselection_fill_color="grey",
                    nonselection_line_color="firebrick",
                    nonselection_line_alpha=1.0)
)# *符
p.line([1,2,3],[4,5,6],line_dash=1,line_width=2) #line_dash 线段连续
p.square

r.glyph.size = 50
r.glyph.fill_alpha = 0.2
r.glyph.line_color = "firebrick"
r.glyph.line_dash = [5, 1]
r.glyph.line_width = 2
r.selection_glyph = Circle(fill_alpha=1, fill_color="firebrick", line_color=None) #选中和没有选中的区别对待
r.nonselection_glyph = Circle(fill_alpha=0.2, fill_color="grey", line_color=None)

##grids 网格线
p.xgrid.grid_line_color = None
p.ygrid.grid_line_dash = [6, 4]# 网格线用虚线表示
p.ygrid.band_fill_alpha = 0.1
p.ygrid.band_fill_color = "navy" #网格块


##legends 图例


##axis坐标轴
p.xaxis.major_label_orientation = pi/4
p.yaxis.major_label_orientation = "vertical" #调整字体方向


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




###Charts
from bokeh.charts import Area,Bar,BoxPlot,Donut,Dot ,HeatMap,Histogram,Horizon,Line,Scatter
#目前有12种图表
from bokeh.palettes import Spectral8   #配色方案
from bokeh.charts import defaults #使用默认配置


from bokeh.models import HoverTool #从底层Models 中导入点上去显示的注解
scatter=Scatter(autompg,x=‘mpg’,y=‘hp’,color=‘origin’)#根据’origin’的类别来匹配颜色
scatter.add_tools(HoverTool(tooltips=[(‘x’,$name’),(‘y’,’@age’)])) #当鼠标点上去的时候显示出这个点’name’ ‘age’
show(scartter)

p = Bar(autompg, label='yr', values='mpg', agg='median', group='origin',  title="Median MPG by YR, grouped by ORIGIN",
        legend='top_left', tools='crosshair')
#group 可以换成stack

hist = Histogram(df, values='value', bins=30)#直方图 bins特有属性


show(BoxPlot(autompg,values=‘mpg’,label=[‘cyl’,’origin’])) #根据’cyl’和’origin’类别的组合来显示盒装图

show(Donut(autompg.cyl.astype(’str’),palette=Spectral8,legend=‘top_right’)) #饼图 图例表示方法


from bokeh.charts.attributes import cat  #attributs 常用的 cat,color,maker
show(Bar(make_counts,label=cat(columns=‘make’,sort=False),values=‘count’ ,agg=‘mean’))
# 降序排序,使用函数均值



area=Area(glucose,stack=True)#累计区域图
area.y_range.start=0 #设置y轴的起点为0 将负数的去掉
show(area)


from bokeh.charts.operations import blend #operations 常用blend stack
b=blend(’1964’,’1964’,’1984’,name=‘life_expectency’,labels_name=‘year’)
#使用堆叠功能 将一个国家的几年的数据堆叠陈一个变量.方便下面使用,类似长格式转短格式,这样相当类别数据可以标颜色了
m国家n年份的二维表,变成了 国家-年份-数据的短格式



show(Dot(decades,values=b,label=‘country’,line_color=‘year’)) #点图 和散点图有区别
#点图是对类别数据使用的, 散点是对连续数据


###server
##streaming data 更加复杂的交互式操作
import numpy as np
from bokeh.plotting import Figure
from bokeh.models import ColumnDataSource,HBox,VBoxForm
from bokeh.models.widgets import Slider,TextInput
from bokeh.io import curdoc

N=200
x=np.linspace(0,4*np.pi,N)
y=np.sin(x)
source=ColumnDataSource(date=dict(x=x,y=y))

plot=Figure(plot_height=400,plot_width=400,title=‘my sine’,tools=‘croosshair,pan,reset,resize,save’ x_range=[0,4*np.pi],y_range=[-2.5,2.5])
plot.line(‘x’,’y’,,source=source)


text=TextInput(title=’title’,value=‘my sin’)
offset=Slider(title=‘offset’,value=0.0,start=-5.0,end=5.0,step=0.1)
phase=Slider(title=‘phase’,value=0.0,start=0.0,end=2*np.pi )
freq=Slider(title=‘freq’,value=1.0,start=0.1,end=5.1 )

def update_title(attrname,old,now):
	plot.title=text.value

text.on_change(‘value’,update_title)

def update_date(attrname,old,now):
	a=amplitude.value
	b=offset.value
	w=phase.value
	k=freq.value
	x=np.linspace(0,4*np.pi,N)
	y=a*np.sin(k*x+w)+b

inputs=VBoxForm(children=[text,offset,amllitude,phase,freq])
curdoce().add_root(HBox(children=[input,plot]),width=800)

bokeh serve —show sliders.py
—allow -websocket -origin foo.com #远程服务

##Datashader  图形批处理系统
https://anaconda.org/jbednar/notebooks





## 互动操作
import numpy as np
from bokeh.io import push_notebook

x = np.linspace(0, 2*np.pi, 2000)
y = np.sin(x)

source = ColumnDataSource(data=dict(x=x, y=y))

p = figure(title="simple line example", plot_height=300, plot_width=600, y_range=(-5, 5))
p.line(x, y, color="#2222aa", alpha=0.5, line_width=2, source=source, name="foo")

def update(f, w=1, A=1, phi=0):
    if   f == "sin": func = np.sin
    elif f == "cos": func = np.cos
    elif f == "tan": func = np.tan
    source.data['y'] = A * func(w * x + phi)
    push_notebook()

show(p, notebook_handle=True)

###参考
github.com/bokeh/bokeh-demos/tree/master/presentations/2016-03-pydata_strata
github.com/bokeh/bokeh-notebooks/tree/master/tutorial
