function [w1,b1,w2,b2,e,y2] = my_BP_LMS(xn,dn)
%% BP神经网络滤波器实现程序
%   输入：
% xn   输入信号       
% dn   理想信号          
% L    迭代次数      
% k    滤波器的阶数   
%   输出：
% w    滤波器的系数矩阵    大小为 k×L  每一列代表一次迭代后的系数
% e    误差信号            大小为 L×1  每一行代表一次迭代后产生的误差
% yn   滤波器的输出信号    

%% 参数配置
innum=32;           %输入层神经元个数
midnum=4;           %隐含层神经元个数
outnum=1;           %输出层神经元个数
L=length(xn);       %迭代次数=输入信号长度

%% 初始化
y1=zeros(1,midnum);      %初始化隐含层输出信号
y2=zeros(1,L);      %初始化输出层输出信号
w1=rands(midnum,innum);       %初始化隐含层神经元权重
w2=rands(midnum,outnum);       %初始化输出层神经元权重
e=zeros(1,L);       %初始化误差
b1=rands(midnum,1);      %初始化隐含层神经元偏置值
b2=rands(1,outnum);      %初始化输出层神经元偏置值

%% 求收敛常数u
%fe = max(eig(xn*xn.'));%求解输入xn的自相关矩阵的最大特征值fe,A = eig(B),意为将矩阵B的特征值组成向量A
%u = 2*(1/fe);
u=0.45;
%% 迭代更新滤波器的参数
for i=(innum+1):L    %要保证输入延时后的信号有效，所以实际的迭代次数只有（L-innum）次，
    XN=xn((i-innum):(i-1));   %将输入信号延迟，使得滤波器的每个抽头都有输入
    %隐含层输出
    for j=1:midnum
    I(j)=XN*w1(j,:)'+b1(j);   
    y1(j)=1./(1+exp(-I(j))); %使用sigmoid函数，此处为隐含层输出值
    end
    
    %输出层输出
    y2(i)=b2+y1*w2;
    
    %预测误差
    e(i)=dn(i)-y2(i);     %得出误差信号
    
    %计算w2,b2调整量
    dw2=e(i)*y1;
    db2=e(i);

    %计算w1,b1调整量
    %for j=1:midnum
   %     S=1./(1+exp(-I(j)));
   %    FI(j)=S*(1-S);
  % end
   FI=y1.*(ones(1,midnum)-y1);%定义：隐含层输出信号乘以（1-隐含层输出信号）
   
    for ii=1:innum
        for jj=1:midnum
            dw1(ii,jj)=FI(jj)*XN(ii)*(w2(1)*e(i)+w2(2)*e(i)+w2(3)*e(i)+w2(4)*e(i));
            db1(jj)=FI(jj)*(w2(1)*e(i)+w2(2)*e(i)+w2(3)*e(i)+w2(4)*e(i));
        end
    end
    
    %权值阈值更新
    w1=w1+u*dw1';
    b1=b1+u*db1';
    w2=w2+u*dw2';
    b2=b2+u*db2';
    
    %结果保存
    %w1_1=w1;
    %w2_1=w2;
    %b1_1=b1;
    %b2_1=b2;
end

end
