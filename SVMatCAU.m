clc;clear;

%read data
data1 = xlsread('D:\课件\毕设\实验数据\model\shuju.xlsx');
data = data1';
 
figure;
plot(data);
title('original data');
xlabel('波长(nm)')
ylabel('反射率')
 
 
%SNV标准正态变化
[m,n] = size(data);
Ym = mean(data,2);
dY = data - repmat(Ym,1,n);
Ysnv = dY./repmat(sqrt(sum(dY.^2,2)/(n-1)),1,n);
 
figure;
plot(Ysnv);
title('SNV');
xlabel('Wavelength(nm)')
ylabel('Reflectance')
 
%MSC多元散射校正
 me=mean(data);
 [m,n]=size(data);
 for i=1:m,
     p=polyfit(me,data(i,:),1);
     Xmsc(i,:)=(data(i,:)-p(2)*ones(1,n))./(p(1)*ones(1,n));
 end
 
 figure;
 
 plot(Xmsc);
 title('MSC');
 xlabel('Wavelength(nm)')
 ylabel('Reflectance')
 
 %导数1d
 X1st = (diff(data1,1))';% 一阶导数
 
 figure;
 hold on;
 plot(X1st);
 title('1st');
 xlabel('Wavelength(nm)')
 ylabel('Reflectance')
 
  %导数2d
 X2st = (diff(data1,2))';% 二阶导数
 
 figure;
 plot(X2st);
 title('2st');
 xlabel('Wavelength(nm)')
 ylabel('Reflectance')
 
 %平滑滤波
 %Xsmooth=smooth(data,30,'lowess'); %平滑滤波
 %Xsmooth = smoothdata(data,'sgolay',5);
 Xsmooth = sgolayfilt(data,4,99);
 
 figure;
 plot(Xsmooth);
 title('smooth');
 xlabel('Wavelength(nm)')
 ylabel('Reflectance')
 
%SVM模型建立：
Y = xlsread('D:\课件\毕设\实验数据\model\Y.xlsx');
X1 = xlsread('D:\课件\毕设\实验数据\model\xmsc.xlsx');
 
label = Y;
X = X1';       %原始数据不需要转置，其余都需要
 
Ytest = xlsread('D:\课件\毕设\实验数据\model\Y.xlsx');
Xtest = xlsread('D:\课件\毕设\实验数据\model\xmsc.xlsx');
 
%[m,~] = KS(X,128);
% m = RS(X,round(length(label)/3*2));
[m,~] = KS(X,round(length(label)/3*2));
c=1:length(label); 
n=setdiff(c,m);
Xtrain=X(m,:);
Xtest=X(n,:);       %因为有负数，所以索引超出矩阵维度
Ytrain=label(m,:);
Ytest=label(n,:);
[ss1,sss]=size(Xtrain);
[tt1,ttt]=size(Xtest);
 
[bestacc,bestc,bestg] = SVMcg(Ytrain,Xtrain,-8,8,-25,8,8,1.0,1.0,1.5);
%[bestacc,bestc,bestg] = SVMcg(Ytrain,Xtrain,-8,8,-8,20,5,1.0,0.8,1.2);
 
cmd = ['-c ',num2str(bestc),' -g ',num2str(bestg)];
%或许参数问题
 
model2 = libsvmtrain(Ytrain,Xtrain,cmd);
[predict_label_1,accuracy_1]=libsvmpredict(Ytrain,Xtrain,model2);
[predict_label_2,accuracy_2]=libsvmpredict(Ytest,Xtest,model2);