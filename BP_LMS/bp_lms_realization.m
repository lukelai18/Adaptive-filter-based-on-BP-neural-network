clear;
clc;
close all;

L=2048;    %信号长度
a=1;       %原始信号幅度
%k=128       %网络迭代次数
t=1:L;      
dn=a*sin(0.05*pi*t);%原始正弦信号

xn=awgn(dn,1);   %添加信噪比5dB的白高斯噪声

%input_train=xn(1:1536);
%output_train=dn(1:1536);
%input_test=xn(1537:2048);
%onput_test=dn(1537:2048);

%% 求收敛常数u
fe = max(eig(dn*dn.'));%求解输入xn的自相关矩阵的最大特征值fe,A = eig(B),意为将矩阵B的特征值组成向量A
u = 2*(1/fe);

%bp神经网络构建
%net=newff(xn,dn,127);
%net.trainParam.epochs=k;
%net.trainParam.lr=u;
%net.trainParam.goal=0.01;
%网络训练
%net_train=train(net,xn,dn);
figure(1)
subplot(211);plot(dn,'k');axis([0,L,-a-1,a+1]);
xlabel('输入信号数量/n');ylabel('信号幅度/y');
title('原始信号s时域波形');
subplot(212);plot(xn,'k');axis([0,L,-a-1,a+1]);
xlabel('输入信号数量/n');ylabel('信号幅度/y');
title('信号加高斯白噪声后的时域波形');

[w1,b1,w2,b2,e,y2] = my_BP_LMS(xn,dn);%调用滤波算法

figure(2)
subplot(211);plot(y2,'k');axis([0,L,-a-1,a+1]);
xlabel('输入信号数量/n');ylabel('信号幅度/y');
title('BP神经网络自适应滤波后的输出时域波形');

subplot(212);plot(e,'k');axis([0,L,-a-1,a+1]);
xlabel('输入信号数量/n');ylabel('误差幅度/y');
title('BP神经网络自适应滤波后与原始信号误差');