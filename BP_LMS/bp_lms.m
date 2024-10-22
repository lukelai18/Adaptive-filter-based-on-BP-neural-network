clear;
clc;
close all;

L=2048;    %信号长度
a=1;       %原始信号幅度
%k=128       %网络迭代次数
t=1:L;      
dn=a*sin(0.05*pi*t);%原始正弦信号
xn=awgn(dn,1);   %添加信噪比5dB的白高斯噪声

input_train=xn(1:1536);   %定义输入训练数据
output_train=dn(1:1536);  %定义输出训练数据
input_test=xn(1537:2048); %定义输入测试数据
output_test=dn(1537:2048); %定义输出测试数据
%训练数据归一化
[inputn,inputps]=mapminmax(input_train);
[outputn,outputps]=mapminmax(output_train);

%% 求收敛常数u
%fe = max(eig(dn*dn.'));%求解输入xn的自相关矩阵的最大特征值fe,A = eig(B),意为将矩阵B的特征值组成向量A
%u = 2*(1/fe);
u=0.45;

%bp神经网络构建
net=newff(xn,dn,4);
net.trainParam.epochs=L;
net.trainParam.lr=u;
net.trainParam.goal=0.001;
%bp网络训练
net_train=train(net,inputn,outputn);
%bp网络预测数据归一化
inputn_test=mapminmax('apply',input_test,inputps);
%bp神经网络预测输出
an=sim(net,inputn_test);
BPoutput=mapminmax('reverse',an,outputps);

subplot(411);plot(dn);axis([0,L,-a-1,a+1]);
title('原始信号s时域波形');

subplot(412);plot(xn);axis([0,L,-a-1,a+1]);
title('信号加高斯白噪声后的时域波形');

subplot(413);plot(BPoutput);axis([0,L,-a-1,a+1]);
title('BP神经网络-LMS算法自适应滤波后的预测输出');

subplot(414);plot(output_test);axis([0,L,-a-1,a+1]);
title('BP神经网络-LMS算法自适应滤波后的期望输出');

%subplot(415);plot(error);axis([0,L,-a-1,a+1]);
%title('BP神经网络-LMS算法自适应滤波后与原始信号误差');