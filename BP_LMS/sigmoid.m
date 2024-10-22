x=-10.0:0.01:10.0;%定义⾃变量的取值
%y=(1-exp(-1.0*x))./(1.0+exp(-1.0*x));%sigmoid函数函数⾥⼀定要⽤点除（./）,因为是矩阵运算，所以要把纬度保持⼀致。
x0=3;
y=heaviside(x-x0);
%y=sign(x);
plot(x,y,'k',LineWidth=1.5)%绘制图形
xlabel('x')%添加横轴名称
ylabel('y')%添加纵轴名称
axis([-10 10 -0.5 1.5])
%legend('T=0')%添加曲线标记符
title('双极性Sigmoid函数')%给图像添加标题
