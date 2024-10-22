L=2048;    %信号长度
a=1;       %原始信号幅度
t=1:L;      
dn=a*sin(0.05*pi*t);%原始正弦信号
plot(dn,'k');axis([0,L,-a-1,a+1]);
xlabel('输入信号数量/n');ylabel('信号幅度/y');
title('原始正弦信号波形');