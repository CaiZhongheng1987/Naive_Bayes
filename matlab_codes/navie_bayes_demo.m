%{
name   :  navie bayes demo
author :  CaiZhongheng

date           version          record
2018.07.14     v1.0             init
%}

clc;
clear;
close all;
%% setting
lamda_laplace  = 1; % 0: naive bayes; 1: naive bayes and laplace smoothing
%% training data
x_feature_num  = 2; % 特征数目
% line1: x1 特征1 取值范围：1 2 3 
X1_num   = [1;2;3];
% line2: x2 特征2 取值范围：0:S, 1:M, 2:L
X2_num   = [0;1;2];
% line3: Y  分类  -1 1
Y_num    = [-1;1];
training_data_x     = [1 1 1 1 1 2 2 2 2 2 3 3 3 3 3;...
                       0 1 1 0 0 0 1 1 2 2 2 1 1 2 2;];
training_data_class = [-1 -1 1 1 -1 -1 -1 1 1 1 1 1 1 1 -1];

test_data = [2,0]';

max_x_num         = max(length(X1_num),length(X2_num));
P_test_data_class = zeros(length(Y_num),1);%初始化后验概率矩阵
P_class           = zeros(length(Y_num),1);%初始化分类概率矩阵
P_bayes           = zeros(size(training_data_x,1)*length(Y_num),max_x_num);

% create the bayes matrix
% P_class = [P(Y=-1); P(Y=1)];
% P_bayes = [P(x1=1|Y=-1), P(x1=2|Y=-1), P(x1=3|Y=-1);
%            P(x2=S|Y=-1), P(x2=M|Y=-1), P(x2=L|Y=-1);
%            P(x1=1|Y=1),  P(x1=2|Y=1),  P(x1=3|Y=1);
%            P(x2=S|Y=1),  P(x2=M|Y=1),  P(x2=L|Y=1);];

if(length(X1_num)~=length(X2_num))
    error('Please check the X1_num and X2_num!!!');
else
end
%% calc P matrix 
for y_idx=1:length(Y_num)
    P_class(y_idx)             = (length(find(training_data_class==Y_num(y_idx)))+lamda_laplace)...
        /(length(training_data_class)+length(Y_num)*lamda_laplace); % P(Y=-1) or P(Y=1)
end

for y_idx=0:(length(Y_num)*x_feature_num-1)
    for x_idx=1:max_x_num
        feature_idx            = mod(y_idx,2)+1;% 特征编号
        class_idx              = floor(y_idx/2)+1;% 分类编号
        tmp_feature_num        = eval(['X' num2str(feature_idx,'%d') '_num']);
        P_bayes(y_idx+1,x_idx) = (length(intersect(find(training_data_x(feature_idx,:)==tmp_feature_num(x_idx)),find(training_data_class==Y_num(class_idx))))+lamda_laplace)...
            /(length(find(training_data_class==Y_num(class_idx)))+length(tmp_feature_num)*lamda_laplace);
    end
end

% using test data to calc the Possibility
% if test data is in Y=-1, then the P(Y=-1|X=test_data) = arg max P(Y)*prod(P(xi=test_data(i)|Y=-1))
for y_idx=1:length(Y_num)
    x1_idx                     = find(X1_num==test_data(1));
    x2_idx                     = find(X2_num==test_data(2));
    P_test_data_class(y_idx)   = P_class(y_idx)*P_bayes((y_idx-1)*size(training_data_x,1)+1,x1_idx)*P_bayes((y_idx-1)*size(training_data_x,1)+2,x2_idx);
end

[P_out, class_out] = max(P_test_data_class);
fprintf('The class of test data is Y = %d.\n', Y_num(class_out));
fprintf('The max Prossibility of test data is %f.\n', P_out);


