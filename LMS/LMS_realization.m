clear;
clc;
close all;

L=2048;    %信号长度
a=1;       %原始信号幅度
t=1:L;      
dn=a*sin(0.05*pi*t);%原始正弦信号

xn=awgn(dn,1);   %添加信噪比5dB的白高斯噪声

figure(1)
subplot(211);plot(dn,'k');axis([0,L,-a-1,a+1]);
xlabel('输入信号数量/n');ylabel('信号幅度/y');
title('原始正弦信号波形');
subplot(212);plot(xn,'k');axis([0,L,-a-1,a+1]);
xlabel('输入信号数量/n');ylabel('信号幅度/y');
title('正弦信号加高斯白噪声后的波形');

[w,e,yn] = my_LMS(xn,dn);%调用滤波算法

figure(2)
subplot(211);plot(yn,'k');axis([0,L,-a-1,a+1]);
xlabel('输入信号数量/n');ylabel('信号幅度/y');
title('LMS算法自适应滤波后的输出波形');

subplot(212);plot(e,'k');axis([0,L,-a-1,a+1]);
xlabel('输入信号数量/n');ylabel('误差幅度/y');
title('LMS算法自适应滤波后与原始信号误差');
